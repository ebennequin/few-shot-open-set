"""
Utils for quick fetching of Dataset or DataLoader objects.
"""

from easyfsl.data_tools import TaskSampler
from torch.utils.data import Dataset, DataLoader

from src.constants import (
    CIFAR_ROOT_DIR,
    CIFAR_SPECS_DIR,
    MINI_IMAGENET_ROOT_DIR,
    MINI_IMAGENET_SPECS_DIR,
)
from src.datasets import FewShotCIFAR100, MiniImageNet
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
