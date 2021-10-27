import random
from typing import List, Tuple

import torch
from easyfsl.data_tools import TaskSampler
from torch.utils.data import Sampler, Dataset

#TODO: This is a v0 of an OpenQuerySampler, for fast iteration.
class OpenQuerySampler(TaskSampler):
    def __iter__(self):
        for _ in range(self.n_tasks):
            # TODO: make it customizable
            all_labels = random.sample(self.items_per_label.keys(), self.n_way * 2)
            support_labels = all_labels[: self.n_way]
            open_set_labels = all_labels[self.n_way :]
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in support_labels
                ]
                + [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_query)
                    )
                    # pylint: enable=not-callable
                    for label in open_set_labels
                ]
            )

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
        true_class_ids = list(
            dict.fromkeys([x[1] for x in input_data])
        )  # This way we keep class orders

        in_set_data = input_data[: self.n_way * (self.n_shot + self.n_query)]
        open_set_data = input_data[self.n_way * (self.n_shot + self.n_query) :]

        in_set_images = torch.cat([x[0].unsqueeze(0) for x in in_set_data])
        in_set_images = in_set_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *in_set_images.shape[1:])
        )
        open_set_images = torch.cat([x[0].unsqueeze(0) for x in open_set_data])

        # pylint: disable=not-callable
        in_set_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in in_set_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        open_set_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in open_set_data]
        )
        # pylint: enable=not-callable

        support_images = in_set_images[:, : self.n_shot].reshape(
            (-1, *in_set_images.shape[2:])
        )
        query_images = torch.cat(
            [
                in_set_images[:, self.n_shot :].reshape((-1, *in_set_images.shape[2:])),
                open_set_images,
            ]
        )
        support_labels = in_set_labels[:, : self.n_shot].flatten()
        query_labels = torch.cat(
            [in_set_labels[:, self.n_shot :].flatten(), open_set_labels]
        )

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
