import random
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from loguru import logger
from numpy import ndarray
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from src.constants import (
    BACKBONES,
)
import os
from types import SimpleNamespace
from src.utils.data_fetchers import get_classic_loader
from collections import OrderedDict, defaultdict
import argparse
import torch.distributed as dist


def set_random_seed(seed: int):
    """
    Set random, numpy and torch random seed, for reproducibility of the training
    Args:
        seed: defined random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_product(left_tensor: torch.Tensor, right_tensor: torch.Tensor):
    """
    Args:
        left_tensor: shape (n, l)
        right_tensor: shape (m, l)

    Returns:
        their tensor product of shape (n, m, l).
        result[i,j,k] = left_tensor[i, k] * right_tensor[j, k]
    """
    return torch.matmul(
        left_tensor.T.unsqueeze(2), right_tensor.T.unsqueeze(1)
    ).permute((1, 2, 0))


def merge_from_dict(args, dict_: Dict):
    for key, value in dict_.items():
        if isinstance(value, dict) and not any([isinstance(key, int) for key in value.keys()]):
            setattr(args, key, SimpleNamespace())
            merge_from_dict(getattr(args, key), value)
        else:
            setattr(args, key, value)


def compute_features(feature_extractor: nn.Module, loader: DataLoader, split: str, layers, device="cuda") -> Tuple[ndarray, ndarray]:
    with torch.no_grad():
        if split == 'val' or split == 'test':
            all_features = defaultdict(list)
            all_labels = []
            for images, labels in tqdm(loader, unit="batch"):
                feat = feature_extractor(images.to(device), layers=layers)
                for layer in feat:
                    all_features[layer].append(feat[layer].cpu())
                all_labels.append(labels)

            for layer in layers:
                all_features[layer] = torch.cat(all_features[layer], dim=0)
            return (
                all_features,
                torch.cat(all_labels, dim=0),

            )
        else:
            mean = defaultdict(float)
            var = defaultdict(float)
            N = 1.
            for images, labels in tqdm(loader, unit="batch"):
                feats = feature_extractor(images.to(device), layers=layers)
                for layer in layers:
                    for new_sample in feats[layer].cpu():
                        if N == 1:
                            mean[layer] = new_sample
                        else:
                            var[layer] = incremental_var(var[layer], mean[layer], new_sample, N)  # [d,]
                            mean[layer] = incremental_mean(mean[layer], new_sample, N)  # [d,]
                        N += 1
            train_feats = {}
            for layer in layers:
                train_feats[layer] = torch.stack([mean[layer], var[layer]], 0)  # [2, d]
            return train_feats, None


def incremental_mean(old_mean: Tensor, new_sample: Tensor, n: int):
    new_mean = 1 / n * (new_sample + (n-1) * old_mean)
    return new_mean


def incremental_var(old_var: Tensor, old_mean: Tensor, new_sample: Tensor, n: int):
    new_var = (n - 2) / (n - 1) * (old_var) + 1 / n * (new_sample - old_mean) ** 2
    return new_var


def strip_prefix(state_dict: OrderedDict, prefix: str):
    return OrderedDict(
        [
            (k[len(prefix):] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )


def load_model(args, backbone: str, weights: Optional[Path], dataset_name, device: torch.device,
               num_classes: int = None):
    logger.info("Fetching data...")
    if num_classes is None:
        train_dataset, _, _ = get_classic_loader(args, dataset_name, split='train', batch_size=10)
        num_classes = len(np.unique(train_dataset.labels))

    logger.info("Building model...")
    feature_extractor = BACKBONES[backbone](num_classes=num_classes).to(device)

    if weights is not None:
        state_dict = torch.load(weights, map_location=device)
        if "state_dict" in state_dict:
            state_dict = strip_prefix(state_dict["state_dict"], "module.")
        elif "params" in state_dict:
            state_dict = strip_prefix(state_dict["params"], "encoder.")
        else:
            state_dict = strip_prefix(state_dict, "backbone.")

        missing_keys, unexpected = feature_extractor.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights}")
        print(f"Missing keys {missing_keys}")
        print(f"Unexpected keys {unexpected}")
        feature_extractor.eval()

    return feature_extractor


def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(args: argparse.Namespace,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
