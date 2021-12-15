"""
Utils for quick fetching of Dataset or DataLoader objects.
"""
import pickle
from pathlib import Path
from typing import Tuple, Dict

import torch
from easyfsl.data_tools import TaskSampler
from numpy import ndarray
from torch.utils.data import Dataset, DataLoader

from src.constants import (
    CIFAR_ROOT_DIR,
    CIFAR_SPECS_DIR,
    MINI_IMAGENET_ROOT_DIR,
    MINI_IMAGENET_SPECS_DIR,
)
from src.datasets import FewShotCIFAR100, MiniImageNet, FeaturesDataset
from src.open_query_sampler import OpenQuerySampler


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


def get_cifar_set(split):
    return FewShotCIFAR100(
        root=CIFAR_ROOT_DIR,
        specs_file=CIFAR_SPECS_DIR / f"{split}.json",
        training=False,
    )


def get_mini_imagenet_set(split):
    return MiniImageNet(
        root=MINI_IMAGENET_ROOT_DIR,
        specs_file=MINI_IMAGENET_SPECS_DIR / f"{split}_images.csv",
        training=False,
    )


def get_task_loader(
    dataset_name, n_way, n_shot, n_query, n_tasks, split="test", n_workers=12
):
    if dataset_name == "cifar":
        dataset = get_cifar_set(split)
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set(split)
    else:
        raise NotImplementedError("I don't know this dataset.")

    sampler = OpenQuerySampler(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    return create_dataloader(dataset, sampler, n_workers)


def get_classic_loader(dataset_name, split="train", batch_size=1024, n_workers=12):
    if dataset_name == "cifar":
        train_set = get_cifar_set(split)
    elif dataset_name == "mini_imagenet":
        train_set = get_mini_imagenet_set(split)
    else:
        raise NotImplementedError("I don't know this dataset.")

    return DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )


def get_features_data_loader(
    features_dict, features_to_center_on, n_way, n_shot, n_query, n_tasks, n_workers
):
    dataset = FeaturesDataset(
        features_dict, features_to_center_on=features_to_center_on
    )
    sampler = OpenQuerySampler(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    return create_dataloader(dataset=dataset, sampler=sampler, n_workers=n_workers)


def get_test_features(backbone, dataset, training_method) -> Tuple[Dict, ndarray]:
    pickle_basename = f"{backbone}_{dataset}_{training_method}.pickle"
    features_path = Path("data/features") / dataset / "test" / pickle_basename
    train_features_path = Path("data/features") / dataset / "train" / pickle_basename

    with open(features_path, "rb") as stream:
        features = pickle.load(stream)

    # We also load features from the train set to center query features on the average train set
    # feature vector
    with open(train_features_path, "rb") as stream:
        train_features = pickle.load(stream)
        average_train_features = torch.cat(
            [
                torch.from_numpy(features_per_label)
                for features_per_label in train_features.values()
            ],
            dim=0,
        ).mean(0)

    return features, average_train_features
