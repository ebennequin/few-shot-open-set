from typing import Callable

import numpy as np
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from easyfsl.data_tools.samplers import AbstractTaskSampler
from easyfsl.data_tools.samplers.utils import sample_label_from_potential
from easyfsl.utils import fill_diagonal

STRATEGIES = {
    "constant": lambda alpha, epoch: alpha,
    "exponential": lambda alpha, epoch: alpha * epoch,
    "linear": lambda alpha, epoch: alpha * np.log(epoch + 1),
    "weibull": lambda alpha, epoch: alpha * (1 - np.exp(-np.power(epoch * 0.01, 5))),
}


class SemanticTaskSampler(AbstractTaskSampler):
    """
    Implements Semantic Task Sampling.
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.

    Classes are sampled so that classes that are semantically close have a higher probability of
    being sampled in a same task.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        semantic_distances_csv: Path,
        alpha: float,
        strategy: str,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
            semantic_distances_csv: path to a csv file containing pair-wise semantic distances
                between classes
            alpha: float factor weighting the importance of semantic distances in the sampling
            strategy: defines the curriculum strategy to update alpha at each learning step.
                Must be a key of STRATEGIES.
        """
        super().__init__(
            dataset=dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks,
        )

        self.distances = torch.tensor(
            pd.read_csv(semantic_distances_csv, header=None).values
        )

        self.alpha = alpha
        self.alpha_fn = self._get_alpha_fn(strategy)
        self.learning_step = 0
        self.potential_matrix = self._compute_potential_matrix()

    def _sample_labels(self) -> torch.Tensor:
        """
        Sample a first label uniformly, then sample other labels with a probability proportional to
        their potential given previously sampled labels, in a greedy fashion.
        Returns:
            1-dim tensor of sampled labels
        """
        to_yield = random.sample(self.items_per_label.keys(), 1)

        potential = self.potential_matrix[to_yield[0]]

        for _ in range(1, self.n_way):
            to_yield.append(sample_label_from_potential(potential))
            potential = potential * self.potential_matrix[to_yield[-1]]

        # pylint: disable=not-callable
        return torch.tensor(to_yield)
        # pylint: enable=not-callable

    def update(self, **kwargs):
        """
        Increment the learning step and update the potential matrix. This implements curriculum
        semantic sampling.
        """
        self.learning_step += 1
        self.potential_matrix = self._compute_potential_matrix()

    @staticmethod
    def _get_alpha_fn(strategy: str) -> Callable[[float, int], float]:
        # This is a quick and dirty way to parameterize the Weibull curriculum without passing
        # additional parameters in the pipeline.
        # Strategy "weibull-0.02-8" now parameterizes with gamma = 0.02 and beta = 8.
        # To make the "weibull" strategy still work, we keep the previous behaviour in this case.
        if strategy.startswith("weibull") & (strategy != "weibull"):
            _, gamma_str, beta_str = strategy.split("-")
            gamma = float(gamma_str)
            beta = float(beta_str)
            return lambda alpha, epoch: alpha * (
                1 - np.exp(-np.power(epoch * gamma, beta))
            )

        if strategy not in STRATEGIES.keys():
            raise ValueError(
                f"{strategy} is not a valid strategy. Valid strategies are: {', '.join(STRATEGIES.keys())}."
            )
        return STRATEGIES[strategy]

    def _compute_potential_matrix(self) -> torch.Tensor:
        """
        Compute the potential matrix depending on the initial alpha, the learning step and the
        function (alpha_fn) corresponding to the chosen curriculum strategy.
        Returns:
            the potential matrix
        """
        return fill_diagonal(
            torch.exp(-self.distances * self.alpha_fn(self.alpha, self.learning_step)),
            0,
        )
