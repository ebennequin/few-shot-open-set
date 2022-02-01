import random
from typing import Tuple, Dict
import numpy as np
import torch
from loguru import logger
from numpy import ndarray
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def compute_features(feature_extractor: nn.Module, loader: DataLoader, split: str, layer: int, device="cuda") -> Tuple[ndarray, ndarray]:
    with torch.no_grad():
        if split == 'val' or split == 'test':
            all_features = []
            all_labels = []
            for images, labels in tqdm(loader, unit="batch"):
                feat = feature_extractor(images.to(device), layer=layer).cpu()
                all_features.append(feat)
                all_labels.append(labels)

            return (
                torch.cat(all_features, dim=0),
                torch.cat(all_labels, dim=0),

            )
        else:
            mean = 0.
            var = 0.
            N = 1.
            for images, labels in tqdm(loader, unit="batch"):
                feats = feature_extractor(images.to(device), layer=layer).mean((-2, -1))
                for new_sample in feats:
                    if N == 1:
                        mean = new_sample
                        var = 0.
                    else:
                        var = incremental_var(var, mean, new_sample, N)  # [d,]
                        mean = incremental_mean(mean, new_sample, N)  # [d,]
                    N += 1
            feats = torch.stack([mean, var], 0)  # [2, d]
            return feats.cpu(), None


def incremental_mean(old_mean: Tensor, new_sample: Tensor, n: int):
    new_mean = 1 / n * (new_sample + (n-1) * old_mean)
    return new_mean


def incremental_var(old_var: Tensor, old_mean: Tensor, new_sample: Tensor, n: int):
    new_var = (n - 2) / (n - 1) * (old_var) + 1 / n * (new_sample - old_mean) ** 2
    return new_var
