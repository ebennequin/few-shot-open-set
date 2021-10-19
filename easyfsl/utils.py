"""
General utilities
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn


def plot_images(images: torch.Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
    """
    Compute the dimension of the feature space defined by a feature extractor.
    Args:
        backbone: feature extractor

    Returns:
        shape of the feature vector computed by the feature extractor for an instance

    """
    input_images = torch.ones((4, 3, 32, 32))
    output = backbone(input_images)

    return tuple(output.shape[1:])


def compute_biconfusion_matrix(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    The biconfusion matrix is typically used to measure the hardness of the discrimination task between two classes.
        Element (i,j) corresponds to the number of missclassifications between classes i and j, regardless of the
        direction (i to j or j to i).
    Args:
        confusion_matrix: a 2-dimentional square matrix
    Returns:
        a 2-dimentional symmetric square matrix of the same shape as confusion_matrix.
    """
    return fill_diagonal(confusion_matrix + confusion_matrix.T, 0)


def compute_prototypes(
    support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )


def fill_diagonal(square_tensor: torch.Tensor, scalar: float) -> torch.Tensor:
    """
    Fill the input tensor diagonal with a scalar value.
    Args:
        square_tensor: input tensor. Must be 2-dim and square
        scalar: value with which to fill the diagonal

    Returns:
        input tensor with diagonal filled with the scalar value

    Raises:
        ValueError: if the input tensor is not 2-dim or not square
    """
    if (
        len(square_tensor.shape) != 2
        or square_tensor.shape[0] != square_tensor.shape[1]
    ):
        raise ValueError("Input tensor must be a 2-dim square tensor.")

    return square_tensor.masked_fill(
        torch.eye(square_tensor.shape[0], dtype=torch.bool), scalar
    )


def get_task_perf(
    task_id: int,
    classification_scores: torch.Tensor,
    labels: torch.Tensor,
    class_ids: List[int],
):
    """
    Records the classification results for each query instance.
    Args:
        task_id: index of the task
        classification_scores: predicted classification scores
        labels: ground truth labels in [0, n_way]
        class_ids: indices (in the full dataset) of the classes composing the current
            classification task
    Returns:
        pd.DataFrame: for each couple (query, class_id), gives classification score, class_id,
            ground truth query label, current task id
    """
    classification_scores = classification_scores.cpu()

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "task_id": task_id,
                    "image_id": i,
                    "true_label": class_ids[labels[i]],
                    "predicted_label": [
                        class_ids[label]
                        for label in range(classification_scores.shape[1])
                    ],
                    "score": classification_scores[i],
                }
            )
            for i in range(labels.shape[0])
        ]
    )


def sort_items_per_label(labels: List[int]) -> Dict[int, List[int]]:
    """
    From a list of integer labels, returns for each unique label the list of indices of occurences.
    Args:
        labels: list of integers

    Returns:
        a dictionary where the keys are the unique values of the input list, and the values are
            the lists of indices where the corresponding key occurs in the input list.
    """

    items_per_label = {}
    for item, label in enumerate(labels):
        if label in items_per_label.keys():
            items_per_label[label].append(item)
        else:
            items_per_label[label] = [item]

    return items_per_label


def get_accuracies(results: pd.DataFrame) -> pd.Series:
    return (
        results.sort_values("score", ascending=False)
        .drop_duplicates(["task_id", "image_id"])
        .sort_values(["task_id", "image_id"])
        .reset_index(drop=True)
        .assign(accuracy=lambda df: df.true_label == df.predicted_label)
        .groupby("task_id")
        .accuracy.mean()
    )
