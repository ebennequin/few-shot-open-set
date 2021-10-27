import itertools
import random
from functools import partial
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional

import torchvision
from easyfsl.data_tools import EasySet, TaskSampler
from easyfsl.methods import AbstractMetaLearner
import networkx as nx
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.constants import BACKBONES, FEW_SHOT_METHODS


def plot_dag(dag: nx.DiGraph):
    """
    Utility function to quickly draw a Directed Acyclic Graph.
    Root is at the top, leaves are on the bottom.
    Args:
        dag: input directed acyclic graph
    """
    pos = graphviz_layout(dag, prog="dot")
    nx.draw(dag, pos, with_labels=False, node_size=10, arrows=False)
    plt.show()


def get_median_distance(labels: List[int], distances: np.ndarray) -> float:
    """
    From a list of labels and a matrix of pair-wise distances, compute the median
    distance of all possible pairs from the list.
    Args:
        labels: integer labels in range(len(distances))
        distances: square symmetric matrix

    Returns:
        median distance
    """
    return median(
        [
            distances[label_a, label_b]
            for label_a, label_b in itertools.combinations(labels, 2)
        ]
    )


def get_distance_std(labels: List[int], distances: np.ndarray):
    """
    From a list of labels and a matrix of pair-wise distances, compute the standard deviation
    of distances of all possible pairs from the list.
    Args:
        labels: integer labels in range(len(distances))
        distances: square symmetric matrix

    Returns:
        median distance
    """
    return stdev(
        [
            distances[label_a, label_b]
            for label_a, label_b in itertools.combinations(labels, 2)
        ]
    )


def get_pseudo_variance(labels: List[int], distances: np.ndarray) -> float:
    """
    From a list of labels and a matrix of pair-wise distances, compute the pseudo-variance
    distance of all possible pairs from the list, i.e. the mean of all square distances.
    Args:
        labels: integer labels in range(len(distances))
        distances: square symmetric matrix

    Returns:
        pseudo-variance
    """
    return mean(
        [
            (distances[label_a, label_b] ** 2)
            for label_a, label_b in itertools.combinations(labels, 2)
        ]
    )


def get_intra_class_distances(training_tasks_record: List, distances: np.ndarray):
    df = pd.DataFrame(training_tasks_record)
    return df.join(
        df.true_class_ids.apply(
            [
                partial(get_median_distance, distances=distances),
                partial(get_distance_std, distances=distances),
            ]
        ).rename(
            columns={
                "get_median_distance": "median_distance",
                "get_distance_std": "std_distance",
            }
        )
    )


def get_training_confusion_for_single_task(row, n_classes):
    indices = []
    values = []

    for (local_label1, true_label1) in enumerate(row["true_class_ids"]):
        for (local_label2, true_label2) in enumerate(row["true_class_ids"]):
            indices.append([true_label1, true_label2])
            values.append(row["task_confusion_matrix"][local_label1, local_label2])

    return torch.sparse_coo_tensor(
        torch.tensor(indices).T, values, (n_classes, n_classes)
    )


def get_training_confusion(df, n_classes):
    return torch.sparse.sum(
        torch.stack(
            [
                get_training_confusion_for_single_task(row, n_classes)
                for _, row in df.iterrows()
            ]
        ),
        dim=0,
    ).to_dense()


def get_sampled_together_for_single_task(row, n_classes):
    indices = []
    values = []

    for label1, label2 in itertools.combinations(row["true_class_ids"], 2):
        indices.append([min(label1, label2), max(label1, label2)])
        values.append(1)

    return torch.sparse_coo_tensor(
        torch.tensor(indices).T, values, (n_classes, n_classes)
    )


def get_sampled_together(df, n_classes):
    return torch.sparse.sum(
        torch.stack(
            [
                get_sampled_together_for_single_task(row, n_classes)
                for _, row in df.iterrows()
            ]
        ),
        dim=0,
    ).to_dense()


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
    logger.info(f"Random seed : {seed}")


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


def build_model(
    backbone: str,
    feature_dimension: int,
    method: str,
    device: str,
    pretrained_weights: Optional[Path] = None,
) -> AbstractMetaLearner:
    """
    Build a meta-learner and cast it on the appropriate device
    Args:
        backbone: backbone of the model to build. Must be a key of constants.BACKBONES.
        feature_dimension: dimension of the feature space
        method: few-shot learning method to use
        device: device on which to put the model
        pretrained_weights: if you want to use pretrained_weights for the backbone

    Returns:
        a PrototypicalNetworks
    """
    convolutional_network = BACKBONES[backbone](
        pretrained=False, num_classes=feature_dimension
    )

    model = FEW_SHOT_METHODS[method](convolutional_network).to(device)

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    return model


def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()
