import random
from abc import abstractmethod
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from easyfsl.utils import sort_items_per_label


class AbstractTaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self, dataset: Dataset, n_way: int, n_shot: int, n_query: int, n_tasks: int
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        assert hasattr(
            dataset, "labels"
        ), "Task samplers need a dataset with a field 'label' containing the labels of all images."
        self.items_per_label = sort_items_per_label(dataset.labels)

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    self._sample_items_from_label(int(label))
                    for label in self._sample_labels()
                ]
            )

    @abstractmethod
    def _sample_labels(self) -> torch.Tensor:
        """
        Specific to each sampler.
        Returns:
            1-dim tensor of sampled labels (integers)
        """

    def _sample_items_from_label(self, label: int) -> torch.Tensor:
        """
        Sample images with a defined label.
        Args:
            label: label from which to sample items. Must be a key of items_per_label.

        Returns:
            n_shot + n_query randomly sampled items
        """
        # pylint: disable=not-callable
        return torch.tensor(
            random.sample(self.items_per_label[label], self.n_shot + self.n_query)
        )
        # pylint: enable=not-callable

    def update(self, **kwargs):
        pass

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
