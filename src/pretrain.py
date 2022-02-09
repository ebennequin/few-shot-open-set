"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
from loguru import logger
import torch
from typing import Dict
from pathlib import Path
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from torch import Tensor
from tqdm import tqdm

from src.utils.utils import (
    set_random_seed, load_model, main_process, setup, cleanup, find_free_port
)
from src.few_shot_methods import ALL_FEW_SHOT_CLASSIFIERS
from src.utils.data_fetchers import get_task_loader, get_classic_loader
from src.detectors import ALL_DETECTORS
from collections import defaultdict
from .inference_features import detect_outliers
from .losses import _CrossEntropy
from .plot import main as plot_fn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> argparse.Namespace:

    # Data
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mini_imagenet")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--pool", action='store_true')
    parser.add_argument("--image_size", type=int, default=84)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max_updates_per_epoch", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--milestones", type=int, nargs='+', default=[75, 150, 180])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.2)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")

    # Method
    parser.add_argument("--inference_method", type=str, default="SimpleShot")
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument("--prepool_transforms", type=str, nargs='+', default=['trivial'])
    parser.add_argument("--postpool_transforms", nargs='+', type=str, default=['l2_norm'])

    # Multiprocessing
    parser.add_argument("--gpus", type=int, nargs='+', default=[0])

    # Logging / Saving results

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--general_hparams", type=str, nargs='+',
                        default=['backbone', 'dataset', 'outlier_detectors', 'inference_method',
                                 'n_way', 'n_shot', 'prepool_transforms', 'postpool_transforms'],
                        )
    parser.add_argument("--simu_hparams", type=str, nargs='*', default=[])
    parser.add_argument("--override", action='store_true')

    # Misc
    parser.add_argument("--debug", type=str2bool)

    args = parser.parse_args()

    if args.debug:
        args.n_tasks = 5
        args.epochs = 3
        args.max_updates_per_epoch = 5
    return args


def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:

    logger.info(f"Running on rank {rank}")
    setup(args, rank, world_size)
    set_random_seed(args.random_seed + rank)

    if main_process(args):
        save_dir = Path('results') / 'training' / args.exp_name
        save_dir.mkdir(exist_ok=True, parents=True)

    if main_process(args):
        logger.info(f"Dropping config file at {save_dir / 'config.json'}")
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f)

    logger.info("Building model...")
    feature_extractor = load_model(args=args,
                                   backbone=args.backbone,
                                   weights=None,
                                   dataset_name=args.dataset,
                                   device=rank)

    feature_extractor = nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
    feature_extractor = DDP(feature_extractor, device_ids=[rank])

    logger.info("Defining optimizer and schedulers")
    optimizer = torch.optim.SGD(feature_extractor.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    logger.info("Creating data loaders  ...")
    train_dataset, train_sampler, train_loader = get_classic_loader(args, args.dataset, shuffle=True, training=True,
                                                                    split='train', batch_size=args.batch_size,
                                                                    world_size=world_size)
    num_classes = len(np.unique(train_dataset.labels))
    if main_process(args):
        val_loader = get_task_loader(args, "val", args.dataset, args.n_way, args.n_shot,
                                     args.n_query, args.n_tasks, args.n_workers)

        logger.info("Creating few-shot classifier for validation ...")
        few_shot_classifier = [
            class_
            for class_ in ALL_FEW_SHOT_CLASSIFIERS
            if class_.__name__ == args.inference_method
        ][0].from_cli_args(args, defaultdict(list), defaultdict(list))

        logger.info("Creating outlier detector for validation  ...")
        detector_sequence = [ALL_DETECTORS['knn'](n_neighbors=3, method='mean')]
        final_detector = ALL_DETECTORS['aggregator'](detector_sequence)

        logger.info("Defining metrics ...")
        updates_per_epoch = min(len(train_loader), args.max_updates_per_epoch)
        logs_per_epoch = (updates_per_epoch // args.log_freq + 1)
        metrics: Dict[str, Tensor] = {"train_loss": torch.zeros(args.epochs * logs_per_epoch,
                                                                dtype=torch.float32),
                                      "val_acc": torch.zeros(args.epochs,
                                                             dtype=torch.float32),
                                      "val_auc": torch.zeros(args.epochs,
                                                             dtype=torch.float32)}

    layer = feature_extractor.module.last_layer_name
    loss_fn = _CrossEntropy(args, num_classes)
    best_val_acc = 0.

    logger.info("Starting training ...")

    for i in tqdm(range(args.epochs)):

        feature_extractor.train()
        if args.distributed:
            train_sampler.set_epoch(i)  # necessary per https://pytorch.org/docs/stable/data.html

        for j, (images, labels) in enumerate(tqdm(train_loader, unit="batch")):

            images, labels = images.to(rank), labels.to(rank)
            feat = feature_extractor(images, layers=layer)[layer].view(images.size(0), -1)
            logits = feature_extractor.module.fc(feat)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()  # Even if distributed, DDP takes care of averaging gradients across processes
            optimizer.step()

            if args.distributed:
                dist.reduce(loss, 0)
            if main_process(args):
                if j % args.log_freq == 0:
                    metrics['train_loss'][logs_per_epoch * i + j // args.log_freq] = loss.item() / world_size
                if j == updates_per_epoch - 1:
                    break

        scheduler.step()

        if main_process(args):
            feature_extractor.eval()
            with torch.no_grad():
                val_metrics = detect_outliers([layer], few_shot_classifier, final_detector, val_loader,
                                              args.n_way, args.n_query, False, feature_extractor)
            metrics['val_acc'][i] = val_metrics['acc'].mean().item()
            metrics['val_auc'][i] = val_metrics['auc'].mean().item()

            logger.info("Epoch {}: Val acc = {:.2f}  Val AUC = {:.2f}".format(
                i, 100 * metrics['val_acc'][i], 100 * metrics['val_auc'][i]))

            for k, array in metrics.items():
                np.save(save_dir / f'{k}.npy', array)

            if metrics['val_acc'][i] > best_val_acc:
                best_val_acc = metrics['val_acc'][i]
                metrics['val_acc'][i]
                torch.save(feature_extractor.state_dict(), save_dir / 'model_best.pth')
            plot_fn(folder=save_dir)

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    args.port = find_free_port()
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)