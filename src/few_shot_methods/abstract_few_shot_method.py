import argparse

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AbstractFewShotMethod(nn.Module):
    """
    Abstract class for few-shot methods
    """

    def __init__(self, softmax_temperature: float = 1.0):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.prototypes: Tensor

    def forward(
        self, feat_s: Tensor, feat_q: Tensor, y_s: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feat_s: support features
            feat_q: query features
            y_s: support labels

        Returns:
            support_soft_predictions: Tensor of shape [n_query, K], where K is the number of classes
                in the task, representing the soft predictions of the method for support samples.
            query_soft_predictions: Tensor of shape [n_query, K], where K is the number of classes
                in the task, representing the soft predictions of the method for query samples.
        """
        raise NotImplementedError

    def get_logits_from_euclidean_distances_to_prototypes(self, samples):
        return -self.softmax_temperature * torch.cdist(samples, self.prototypes)

    def get_logits_from_cosine_distances_to_prototypes(self, samples):
        return (
            self.softmax_temperature
            * nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )
