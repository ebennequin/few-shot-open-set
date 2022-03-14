import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, List, Dict


class AllInOne(nn.Module):
    """
    Abstract class for few-shot methods
    """

    def __init__(self, *args, **kwargs):
        self.type == 'all_in_one'
        super().__init__()

    def forward(
        self, support_features: Tensor, query_features: Tensor, support_labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
            outlier_scores: Tensor of shape [n_query,], where K is the number of classes
                in the task, representing the soft predictions of the method for query samples.
        """
        raise NotImplementedError

    @classmethod
    def from_cli_args(cls, args):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()},
        )
