import random

import torch

from easyfsl.data_tools.samplers import AbstractTaskSampler


class UniformTaskSampler(AbstractTaskSampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

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
        return torch.tensor(random.sample(self.items_per_label.keys(), self.n_way))
