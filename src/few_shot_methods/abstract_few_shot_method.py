import argparse
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Dict


class AbstractFewShotMethod(nn.Module):
    """
    Abstract class for few-shot methods
    """

    def __init__(
        self, softmax_temperature: float = 1.0, normalize_features: bool = False
    ):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.normalize_features = normalize_features
        self.prototypes: Tensor

    def forward(
        self, support_features: Tensor, query_features: Tensor, support_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            support_features: support features
            query_features: query features
            support_labels: support labels

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
            * F.normalize(samples, dim=1)
            @ F.normalize(self.prototypes, dim=1).T
        )

    def normalize_features_if_specified(self, features):
        return F.normalize(features, dim=-1) if self.normalize_features else features

    @classmethod
    def from_cli_args(cls, args):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()}
        )

    @classmethod
    def from_args(cls, args: Dict):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args.items() if k in signature.parameters.keys()}
        )
