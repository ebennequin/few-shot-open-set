"""
Utils for quick fetching of Dataset or DataLoader objects.
"""
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional

from easyfsl.data_tools import TaskSampler
from numpy import ndarray
from torch.utils.data import Dataset, DataLoader

from src.constants import (
    CIFAR_ROOT_DIR,
    CIFAR_SPECS_DIR,
    MINI_IMAGENET_ROOT_DIR,
    MINI_IMAGENET_SPECS_DIR,
    TIERED_IMAGENET_ROOT_DIR,
    TIERED_IMAGENET_SPECS_DIR,
)
from src.datasets import FewShotCIFAR100, MiniImageNet, FeaturesDataset, TieredImageNet
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


def get_tiered_imagenet_set(split):
    return TieredImageNet(
        root=TIERED_IMAGENET_ROOT_DIR,
        specs_file=TIERED_IMAGENET_SPECS_DIR / f"{split}.json",
        training=False,
    )


def get_task_loader(
    dataset_name, n_way, n_shot, n_query, n_tasks, split="test", n_workers=12
):
    if dataset_name == "cifar":
        dataset = get_cifar_set(split)
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set(split)
    elif dataset_name == "tiered_imagenet":
        dataset = get_tiered_imagenet_set(split)
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


def get_dataset(dataset_name, split):
    if dataset_name == "cifar":
        dataset = get_cifar_set(split)
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set(split)
    elif dataset_name == "tiered_imagenet":
        dataset = get_tiered_imagenet_set(split)
    else:
        raise NotImplementedError("I don't know this dataset.")
    return dataset


def get_classic_loader(dataset_name, split="train", batch_size=1024, n_workers=6):

    dataset = get_dataset(dataset_name, split)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        shuffle=False,
    )


def get_features_data_loader(
    features_dict, n_way, n_shot, n_query, n_tasks, n_workers
):
    dataset = FeaturesDataset(
        features_dict,
    )
    sampler = OpenQuerySamplerOnFeatures(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    return create_dataloader(dataset=dataset, sampler=sampler, n_workers=n_workers)


def get_test_features(backbone, dataset, training_method, layer, path: Optional[Path] = None) -> Tuple[Dict, Dict, ndarray]:
    pickle_basename = f"{backbone}_{dataset}_{training_method}_{layer}.pickle"
    features_path = Path("data/features") / dataset / "test" / pickle_basename
    avg_train_features_path = Path("data/features") / dataset / "train" / pickle_basename

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
