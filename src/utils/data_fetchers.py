"""
Utils for quick fetching of Dataset or DataLoader objects.
"""
import pickle
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
from src.datasets import (
    FewShotCIFAR100,
    MiniImageNet,
    CUB,
    Fungi,
    FeaturesDataset,
    TieredImageNet,
    ImageNet,
    Aircraft,
)
from src.datasets.imagenet_val import ImageNetVal
from src.sampler import OpenQuerySamplerOnFeatures, TaskSampler


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
        root=Path(args.data_dir) / "cifar",
        args=args,
        split=split,
        training=training,
    )


def get_mini_imagenet_set(args, split, training, bis=False):
    root = Path(args.data_dir) / "mini_imagenet"
    if bis:
        root = root / "bis"
    return MiniImageNet(
        root=root,
        args=args,
        split=split,
        training=training,
    )


def get_aircraft_set(args, split, training):
    return Aircraft(
        root=Path(args.data_dir) / "fgvc-aircraft-2013b" / "data",
        args=args,
        split=split,
        training=training,
    )


def get_fungi_set(args, split, training):
    return Fungi(
        root=Path(args.data_dir) / "fungi",
        args=args,
        split=split,
        training=training,
    )


def get_imagenet_val_set(args):
    return ImageNetVal(
        root=Path(args.data_dir) / "ILSVRC2015",
        args=args,
    )


def get_imagenet_set(args, split, training):
    return ImageNet(
        root=Path(args.data_dir) / "ilsvrc_2012",
        args=args,
        split=split,
        training=training,
    )


def get_tiered_imagenet_set(args, split, training, bis=False):
    root = Path(args.data_dir) / "tiered_imagenet"
    if bis:
        root = root / "bis"
    # if args.model_source == "feat":
    # logger.warning("Return FEAT version of Tiered-ImageNet ! ")
    return TieredImageNet(
        root=root,
        args=args,
        split=split,
        training=training,
    )
    # else:
    #     return TieredImageNet(
    #         root=root,
    #         args=args,
    #         split=split,
    #         training=training,
    #     )


def get_cub_set(args, split, training):
    return CUB(
        root=Path(args.data_dir) / "cub",
        args=args,
        split=split,
        training=training,
    )


def get_dataset(dataset_name, args, split, training):
    if dataset_name == "cifar":
        dataset = get_cifar_set(args, split, training)
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set(args, split, training)
    elif dataset_name == "mini_imagenet_bis":
        dataset = get_mini_imagenet_set(args, split, training, bis=True)
    elif dataset_name == "imagenet":
        dataset = get_imagenet_set(args, split, training)
    elif dataset_name == "tiered_imagenet":
        dataset = get_tiered_imagenet_set(args, split, training)
    elif dataset_name == "tiered_imagenet_bis":
        dataset = get_tiered_imagenet_set(args, split, training, bis=True)
    elif dataset_name == "cub":
        dataset = get_cub_set(args, split, training)
    elif dataset_name == "aircraft":
        dataset = get_aircraft_set(args, split, training)
    elif dataset_name == "fungi":
        dataset = get_fungi_set(args, split, training)
    elif dataset_name == "imagenet_val":
        dataset = get_imagenet_val_set(args)
    else:
        raise NotImplementedError(f"I don't know this dataset {dataset_name}.")
    return dataset


def get_classic_loader(
    args,
    dataset_name,
    training=False,
    shuffle=False,
    split="train",
    batch_size=1024,
    world_size=1,
    n_workers=6,
):
    dataset = get_dataset(dataset_name, args, split, training)
    sampler = DistributedSampler(dataset, shuffle=True) if (world_size > 1) else None
    batch_size = int(args.batch_size / world_size) if (world_size > 1) else batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        pin_memory=True,
        shuffle=shuffle and (sampler is None),
    )
    return dataset, sampler, data_loader


def get_task_loader(
    n_way: int,
    n_shot: int,
    n_id_query: int,
    n_ood_query: int,
    n_tasks: int,
    n_workers: int,
    features_dict=None,
    broad_open_set=False,
):
    assert features_dict is not None
    dataset = FeaturesDataset(features_dict)
    sampler = OpenQuerySamplerOnFeatures(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_id_query=n_id_query,
        n_ood_query=n_ood_query,
        n_tasks=n_tasks,
        broad_open_set=broad_open_set,
    )
    return create_dataloader(dataset=dataset, sampler=sampler, n_workers=n_workers)


def get_test_features(
    data_dir,
    backbone,
    src_dataset,
    tgt_dataset,
    training_method,
    model_source,
    split: str = "test",
    path: Optional[Path] = None,
):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    pickle_basename = f"{backbone}_{src_dataset}_{model_source}.pickle"
    features_path = (
        data_dir
        / "features"
        / src_dataset
        / tgt_dataset
        / split
        / training_method
        / pickle_basename
    )
    avg_train_features_path = (
        data_dir
        / "features"
        / src_dataset
        / src_dataset
        / "train"
        / training_method
        / pickle_basename
    )
    logger.info(f"Loading train features from {avg_train_features_path}")
    logger.info(f"Loading test features from {features_path}")

    with open(features_path, "rb") as stream:
        features = pickle.load(stream)

    # We also load features from the train set to center query features on the average train set
    # feature vector
    with open(avg_train_features_path, "rb") as stream:
        train_features = pickle.load(stream)
        assert len(train_features) == 2
        average_train_features = train_features[0].unsqueeze(0)
        std_train_features = train_features[1].unsqueeze(0)
    return (
        features,
        train_features,
        average_train_features,
        std_train_features,
        features_path,
        avg_train_features_path,
    )
