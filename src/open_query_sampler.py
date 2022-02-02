import random
from typing import List, Tuple

import torch
from easyfsl.data_tools import TaskSampler

import random
from typing import List, Tuple

import torch
from easyfsl.data_tools import TaskSampler


class OpenQuerySamplerOnFeatures(TaskSampler):
    def __iter__(self):
        for _ in range(self.n_tasks):
            # TODO: allow customizable shape of the open query task
            all_labels = random.sample(self.items_per_label.keys(), self.n_way * 2)
            support_labels = all_labels[: self.n_way]
            open_set_labels = all_labels[self.n_way:]
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    for label in support_labels
                ]
                + [
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_query)
                    )
                    for label in open_set_labels
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

        in_set_data = input_data[: self.n_way * (self.n_shot + self.n_query)]
        open_set_data = input_data[self.n_way * (self.n_shot + self.n_query):]

        support_images = {}
        query_images = {}

        # Preparing labels
        in_set_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in in_set_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        open_set_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in open_set_data]
        )
        support_labels = in_set_labels[:, : self.n_shot].flatten()
        query_labels = torch.cat(
            [in_set_labels[:, self.n_shot:].flatten(), open_set_labels]
        )

        # Preparing features
        for layer in layers:
            in_set_images = torch.stack([x[0][layer] for x in in_set_data], 0)
            in_set_images = in_set_images.reshape(
                (self.n_way, self.n_shot + self.n_query, *in_set_images.shape[1:])
            )
            open_set_images = torch.stack([x[0][layer] for x in open_set_data], 0)

            support_images[layer] = in_set_images[:, : self.n_shot].reshape(
                                (-1, *in_set_images.shape[2:])
                                )
            query_images[layer] = torch.cat(
                                [
                                    in_set_images[:, self.n_shot :].reshape((-1, *in_set_images.shape[2:])),
                                    open_set_images,
                                ]
                                )

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )


class OpenQuerySampler(TaskSampler):
    def __iter__(self):
        for _ in range(self.n_tasks):
            # TODO: allow customizable shape of the open query task
            all_labels = random.sample(self.items_per_label.keys(), self.n_way * 2)
            support_labels = all_labels[: self.n_way]
            open_set_labels = all_labels[self.n_way :]
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    for label in support_labels
                ]
                + [
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_query)
                    )
                    for label in open_set_labels
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

        in_set_data = input_data[: self.n_way * (self.n_shot + self.n_query)]
        open_set_data = input_data[self.n_way * (self.n_shot + self.n_query) :]

        in_set_images = torch.cat([x[0].unsqueeze(0) for x in in_set_data])
        in_set_images = in_set_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *in_set_images.shape[1:])
        )
        open_set_images = torch.cat([x[0].unsqueeze(0) for x in open_set_data])

        in_set_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in in_set_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        open_set_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in open_set_data]
        )

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
