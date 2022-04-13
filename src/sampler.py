import random
from typing import List, Tuple

import torch
from easyfsl.data_tools import TaskSampler
from loguru import logger
import random
from typing import List, Tuple
import math
import torch
from torch.utils.data import Sampler, Dataset
import numpy as np


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_id_query: int,
        n_ood_query: int,
        n_tasks: int,
        balanced: bool,
        alpha: float,
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
        self.n_id_query = n_id_query
        self.n_ood_query = n_ood_query
        self.n_tasks = n_tasks
        self.balanced = balanced
        self.alpha = alpha

        self.items_per_label = {}
        assert hasattr(
            dataset, "labels"
        ), "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        raise NotImplementedError

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        raise NotImplementedError


class OpenQuerySamplerOnFeatures(TaskSampler):
    def __iter__(self):
        for _ in range(self.n_tasks):
            # TODO: allow customizable shape of the open query task
            all_labels = random.sample(self.items_per_label.keys(), self.n_way * 2)
            support_labels = all_labels[: self.n_way]
            open_set_labels = all_labels[self.n_way :]
            if self.balanced:
                id_samples_per_class = [self.n_id_query] * self.n_way
                ood_samples_per_class = [self.n_ood_query] * self.n_way
            else:
                query_samples_per_class = get_dirichlet_proportion(
                    [self.alpha] * self.n_way * 2,
                    1,
                    2 * self.n_way,
                    self.n_way * (self.n_id_query + self.n_ood_query),
                )[0]
                id_samples_per_class = query_samples_per_class[: self.n_way]
                ood_samples_per_class = query_samples_per_class[self.n_way :]

            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_shot)
                    )
                    for i, label in enumerate(support_labels)
                ]
                + [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], id_samples_per_class[i]
                        )
                    )
                    for i, label in enumerate(support_labels)
                ]
                + [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], ood_samples_per_class[i]
                        )
                    )
                    for i, label in enumerate(open_set_labels)
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[dict, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Overwrite episodic_collate_fn from TaskSampler.
        Args:
            input_data: each element is a tuple containing:
                - an image (or feature vector) as a  torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """
        layers = list(input_data[0][0].keys())
        true_class_ids = list(
            dict.fromkeys([x[1] for x in input_data])
        )  # This way we keep class orders

        support_data = input_data[: self.n_way * self.n_shot]
        in_set_labels = set([x[1] for x in support_data])
        id_query = [
            x for x in input_data[self.n_way * self.n_shot :] if x[1] in in_set_labels
        ]
        ood_query = [
            x
            for x in input_data[self.n_way * self.n_shot :]
            if x[1] not in in_set_labels
        ]

        support_images = {}
        query_images = {}

        # Preparing labels
        support_labels = torch.Tensor(
            [true_class_ids.index(x[1]) for x in support_data]
        )
        id_query_labels = torch.Tensor([true_class_ids.index(x[1]) for x in id_query])
        ood_query_labels = torch.Tensor([true_class_ids.index(x[1]) for x in ood_query])
        query_labels = torch.cat([id_query_labels, ood_query_labels])

        # Preparing outlier gt
        outliers = torch.cat([torch.zeros(len(id_query)), torch.ones(len(ood_query))])

        # Preparing features
        for layer in layers:
            support_images[layer] = torch.stack(
                [x[0][layer] for x in support_data], 0
            )  # [Ns, d_layer]
            query_images[layer] = torch.stack(
                [x[0][layer] for x in id_query], 0
            )  # [Ns, d_layer]
            if len(ood_query):
                query_images[layer] = torch.cat(
                    [
                        query_images[layer],
                        torch.stack([x[0][layer] for x in ood_query], 0),
                    ]
                )

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            outliers,
        )


class OpenQuerySampler(TaskSampler):
    def __iter__(self):
        for _ in range(self.n_tasks):
            # TODO: allow customizable shape of the open query task
            all_labels = random.sample(self.items_per_label.keys(), self.n_way * 2)
            support_labels = all_labels[: self.n_way]
            open_set_labels = all_labels[self.n_way :]
            if self.balanced:
                id_samples_per_class = [self.n_id_query] * self.n_way
                ood_samples_per_class = [self.n_ood_query] * self.n_way
            else:
                query_samples_per_class = get_dirichlet_proportion(
                    [self.alpha] * self.n_way * 2,
                    1,
                    2 * self.n_way,
                    self.n_way * (self.n_id_query + self.n_ood_query),
                )[0]
                id_samples_per_class = query_samples_per_class[: self.n_way]
                ood_samples_per_class = query_samples_per_class[self.n_way :]

            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_shot)
                    )
                    for i, label in enumerate(support_labels)
                ]
                + [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], id_samples_per_class[i]
                        )
                    )
                    for i, label in enumerate(support_labels)
                ]
                + [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], ood_samples_per_class[i]
                        )
                    )
                    for i, label in enumerate(open_set_labels)
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Overwrite episodic_collate_fn from TaskSampler.
        Args:
            input_data: each element is a tuple containing:
                - an image (or feature vector) as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """
        true_class_ids = list(
            dict.fromkeys([x[1] for x in input_data])
        )  # This way we keep class orders

        support_data = input_data[: self.n_way * self.n_shot]
        in_set_labels = set([x[1] for x in support_data])
        id_query = [
            x for x in input_data[self.n_way * self.n_shot :] if x[1] in in_set_labels
        ]
        ood_query = [
            x
            for x in input_data[self.n_way * self.n_shot :]
            if x[1] not in in_set_labels
        ]

        support_images = []
        query_images = []

        # Preparing labels
        support_labels = torch.Tensor(
            [true_class_ids.index(x[1]) for x in support_data]
        )
        id_query_labels = torch.Tensor([true_class_ids.index(x[1]) for x in id_query])
        ood_query_labels = torch.Tensor([true_class_ids.index(x[1]) for x in ood_query])
        query_labels = torch.cat([id_query_labels, ood_query_labels])

        # Preparing outlier gt
        outliers = torch.cat([torch.zeros(len(id_query)), torch.ones(len(ood_query))])

        # Preparing features
        support_images = [x[0] for x in support_data]  # [Ns, d_layer]
        query_images = [x[0] for x in id_query]  # [Ns, d_layer]
        if len(ood_query):
            query_images += [x[0] for x in ood_query]

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            outliers,
        )


def get_dirichlet_proportion(alpha, n_tasks, n_ways, total_samples):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    total_samples = (total_samples * prob_dist).round().astype(np.int32)
    return total_samples
