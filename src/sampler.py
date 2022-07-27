from collections import Counter
import random
from typing import List, Tuple
import torch
from torch.utils.data import Sampler, Dataset


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
        broad_open_set: bool = False,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_id_query: number of closed-set query images for each class in one task
            n_ood_query: number of open-set query images for each open-set class in one task
            n_tasks: number of tasks to sample
            broad_open_set: whether to use all remaining test classes for open-set.
                If False, we randomly sample n_way open-set classes and n_ood_query instances per open-set class
                If True, we randomly sample n_ood_query * n_way open-set instances from all open-set classes.
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_id_query = n_id_query
        self.n_ood_query = n_ood_query
        self.n_tasks = n_tasks
        self.items_per_label = {}
        self.broad_open_set = broad_open_set

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
            all_labels = random.sample(
                self.items_per_label.keys(), len(self.items_per_label)
            )
            support_labels = all_labels[: self.n_way]
            id_samples_per_class = [self.n_id_query] * self.n_way
            if self.broad_open_set:
                open_set_labels_with_replacement = random.choices(
                    list(set(all_labels).difference(set(support_labels))),
                    k=self.n_ood_query * self.n_way,
                )
                occurences_per_class = Counter(open_set_labels_with_replacement)
                open_set_labels = list(occurences_per_class.keys())
                ood_samples_per_class = list(occurences_per_class.values())
            else:
                open_set_labels = all_labels[self.n_way : self.n_way * 2]
                ood_samples_per_class = [self.n_ood_query] * self.n_way

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overwrite episodic_collate_fn from TaskSampler.
        Args:
            input_data: each element is a tuple containing:
                - an image (or feature vector) as a  torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, Tensor): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - for each query instance, 0 if it is an inlier, 1 if it is an outlier
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
        support_images = torch.stack([x[0] for x in support_data], 0)  # [Ns, d_layer]
        query_images = torch.stack([x[0] for x in id_query], 0)  # [Ns, d_layer]
        if len(ood_query):
            query_images = torch.cat(
                [
                    query_images,
                    torch.stack([x[0] for x in ood_query], 0),
                ]
            )

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            outliers,
        )
