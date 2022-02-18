"""
Utils for quick fetching of Dataset or DataLoader objects.
"""
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional

from easyfsl.data_tools import TaskSampler
from numpy import ndarray
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
from src.datasets import FewShotCIFAR100, MiniImageNet, CUB, FeaturesDataset, TieredImageNet, \
                         FeatTieredImageNet, ImageNet, Aircraft
from src.open_query_sampler import OpenQuerySamplerOnFeatures, OpenQuerySampler


def create_dataloader(dataset: Dataset, sampler: TaskSampler, n_workers: int):
    """
    Create a torch dataloader of tasks from the input dataset sampled according
    to the input tensor.
    Args:
        dataset: dataset from which to sample tasks
        sampler: task sampler, must implement an episodic_collate_fn method
        n_workers: number of workers of the dataloader

    Returns:
        a dataloader of tasks
    """
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )


def get_cifar_set(args, split, training):
    return FewShotCIFAR100(
        root=Path(args.data_dir) / 'cifar',
        args=args,
        split=split,
        training=training,
    )


def get_mini_imagenet_set(args, split, training):
    return MiniImageNet(
        root=Path(args.data_dir) / 'mini_imagenet',
        args=args,
        split=split,
        training=training,
    )


def get_aircraft_set(args, split, training):
    return Aircraft(
        root=Path(args.data_dir) / 'fgvc-aircraft-2013b' / 'data',
        args=args,
        split=split,
        training=training,
    )


def get_imagenet_set(args, split, training):
    return ImageNet(
        root=Path(args.data_dir) / 'ilsvrc_2012',
        args=args,
        split=split,
        training=training,
    )


def get_tiered_imagenet_set(args, split, training):
    if args.model_source == 'feat':
        logger.warning("Return FEAT version of Tiered-ImageNet ! ")
        return FeatTieredImageNet(
                root=Path(args.data_dir) / 'tiered_imagenet',
                args=args,
                split=split,
                training=training,
                )
    else:
        return TieredImageNet(
            root=Path(args.data_dir) / 'tiered_imagenet',
            args=args,
            split=split,
            training=training,
        )


def get_cub_set(args, split, training):
    return CUB(
        root=Path(args.data_dir) / 'cub',
        args=args,
        split=split,
        training=training,
    )


def get_dataset(dataset_name, args, split, training):
    if dataset_name == "cifar":
        dataset = get_cifar_set(args, split, training)
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set(args, split, training)
    elif dataset_name == "imagenet":
        dataset = get_imagenet_set(args, split, training)
    elif dataset_name == "tiered_imagenet":
        dataset = get_tiered_imagenet_set(args, split, training)
    elif dataset_name == "cub":
        dataset = get_cub_set(args, split, training)
    elif dataset_name == "aircraft":
        dataset = get_aircraft_set(args, split, training)
    else:
        raise NotImplementedError(f"I don't know this dataset {dataset_name}.")
    return dataset


def get_classic_loader(args, dataset_name, training=False, shuffle=False, split="train", batch_size=1024, world_size=1, n_workers=6):

    dataset = get_dataset(dataset_name, args, split, training)
    sampler = DistributedSampler(dataset, shuffle=True) if (world_size > 1) else None
    batch_size = int(args.batch_size / world_size) if (world_size > 1) else batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                             sampler=sampler, pin_memory=True, shuffle=shuffle and (sampler is None))
    return dataset, sampler, data_loader


def get_task_loader(args, split, dataset_name, n_way,
                    n_shot, n_query, n_tasks, n_workers,
                    features_dict=None, training: bool = False):

    if features_dict is not None:
        dataset = FeaturesDataset(features_dict)
        sampler = OpenQuerySamplerOnFeatures(
            dataset=dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks,
            balanced=args.balanced_tasks,
            alpha=args.alpha,
        )
    else:
        dataset = get_dataset(dataset_name, args, split, training)
        sampler = OpenQuerySampler(
            dataset=dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks,
            balanced=args.balanced_tasks,
            alpha=args.alpha
        )
    return create_dataloader(dataset=dataset, sampler=sampler, n_workers=n_workers)


def get_train_features(backbone, dataset, training_method, layer, path: Optional[Path] = None):

    pickle_basename = f"{backbone}_{dataset}_{training_method}_{layer}.pickle"
    avg_train_features_path = Path("data/features") / dataset / "train" / pickle_basename

    # We also load features from the train set to center query features on the average train set
    # feature vector
    with open(avg_train_features_path, "rb") as stream:
        train_features = pickle.load(stream)
        assert len(train_features) == 2
        average_train_features = train_features[0].unsqueeze(0)
        std_train_features = train_features[1].unsqueeze(0)
    return average_train_features, std_train_features


def get_test_features(backbone, src_dataset, tgt_dataset, training_method, model_source,
                      layer, path: Optional[Path] = None) -> Tuple[Dict, Dict, ndarray]:
    pickle_basename = f"{backbone}_{src_dataset}_{model_source}_{layer}.pickle"
    features_path = Path("data/features") / src_dataset / tgt_dataset / "test" / training_method / pickle_basename
    avg_train_features_path = Path("data/features") / src_dataset / tgt_dataset / "train" / training_method / pickle_basename
    logger.info(f"Loading features from {features_path}")

    with open(features_path, "rb") as stream:
        features = pickle.load(stream)

    # We also load features from the train set to center query features on the average train set
    # feature vector
    with open(avg_train_features_path, "rb") as stream:
        train_features = pickle.load(stream)
        assert len(train_features) == 2
        average_train_features = train_features[0].unsqueeze(0)
        std_train_features = train_features[1].unsqueeze(0)
    return features, train_features, average_train_features, std_train_features
