import random
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from numpy import ndarray
from torch import nn
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


def compute_features(
    feature_extractor: nn.Module, loader: DataLoader, device="cuda"
) -> Tuple[ndarray, ndarray]:
    with torch.no_grad():
        all_features = []
        all_labels = []
        for images, labels in tqdm(loader, unit="batch"):
            all_features.append(feature_extractor(images.to(device)).data.cpu())
            all_labels.append(labels)

    return (
        torch.cat(all_features, dim=0).cpu().numpy(),
        torch.cat(all_labels, dim=0).cpu().numpy(),
    )
