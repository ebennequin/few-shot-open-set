"""
Chenghao Liu, Zhihao Wang, Doyen Sahoo, Yuan Fang, Kun Zhang, Steven C.H. Hoi
"Adaptive Task Sampling for Meta-Learning" (2020)
https://arxiv.org/abs/2007.08735
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from easyfsl.data_tools.samplers import AbstractTaskSampler
from easyfsl.data_tools.samplers.utils import sample_label_from_potential
from easyfsl.utils import compute_biconfusion_matrix, fill_diagonal


class AdaptiveTaskSampler(AbstractTaskSampler):
    """
    Implements Adaptive Task Sampling.
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.

    Classes are sampled so that classes that were previously confused by the model during training
    are sampled in a same task with a higher probability.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        hardness: float,
        forgetting: float,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
            hardness: parameter influencing the "hardness" of the potential matrix update
            forgetting: parameter influencing the decay of previous potential matrix weights during
                update
        """
        super().__init__(
            dataset=dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks,
        )
        self.hardness = hardness
        self.forgetting = forgetting

        self.total_number_of_classes = np.max(list(self.items_per_label.keys())) + 1

        self.potential_matrix = fill_diagonal(
            torch.ones((self.total_number_of_classes, self.total_number_of_classes)), 0
        )

    def update(self, confusion_matrix: torch.Tensor, **kwargs):
        """
        Updates the potential matrix using the new confusion matrix, and store the last-to-date
        confusion matrix
        Args:
            confusion_matrix: confusion matrix updated from the last episodes
        """
        normalized_confusion_matrix = nn.functional.normalize(
            confusion_matrix, p=1, dim=1
        )

        self.potential_matrix = self.potential_matrix.pow(self.forgetting) * torch.exp(
            self.hardness * compute_biconfusion_matrix(normalized_confusion_matrix)
        )

    def _sample_labels(self) -> torch.Tensor:
        """
        Iteratively sample task labels with a probability proportional to their potential given
        previously sampled labels, in a greedy fashion.
        First label is sampled with probability proportional to its weight in the potential matrix.
        Second label is sampled with probability proportional to its potential given first label.
        Third label is sampled with probability proportional to its actualized potential,
            which is obtained by element-wise multiplication between the previous potential, and
            the potential given the second label.
        Etc...
        Returns:
            1-dim tensor of sampled labels
        """
        potential = self.potential_matrix.sum(dim=1)
        to_yield = [sample_label_from_potential(potential)]

        potential = self.potential_matrix[to_yield[0]]

        for _ in range(1, self.n_way):
            to_yield.append(sample_label_from_potential(potential))
            potential = potential * self.potential_matrix[to_yield[-1]]

        # pylint: disable=not-callable
        return torch.tensor(to_yield)
        # pylint: enable=not-callable
