import argparse
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AbstractFewShotMethod(nn.Module):
    """
    Abstract class for few-shot methods
    """

    def __init__(self, args: argparse.Namespace):
        super(AbstractFewShotMethod, self).__init__()

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
