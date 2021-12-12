import argparse
from typing import Optional, Tuple

from torch import Tensor

from .method import FSmethod
from easyfsl.utils import compute_prototypes
import torch.nn.functional as F


class SimpleShot(FSmethod):
    """
    Implementation of SimpleShot method https://arxiv.org/abs/1911.04623
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.softmax_temp = args.softmax_temp

    def get_logits(self, samples: Tensor) -> Tensor:
        """
        inputs:
            samples : tensor of shape [shot, feature_dim]

        returns :
            logits : tensor of shape [shot, num_class]
        """
        logits = (samples.matmul(self.weights.t())
                  - 1 / 2 * (self.weights**2).sum(1).view(1, -1)
                  - 1 / 2 * (samples**2).sum(1).view(-1, 1))

        return self.softmax_temp * logits

    def forward(self,
                feat_s: Tensor,
                feat_q: Tensor,
                y_s: Tensor,
                **kwargs) -> Tuple[Tensor, Tensor]:

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)  # [S, d]
        feat_q = F.normalize(feat_q, dim=-1)  # [Q, d]

        # Initialize weights
        self.weights = compute_prototypes(feat_s, y_s)  # []
        P_q = self.get_logits(feat_q).softmax(-1)
        P_s = self.get_logits(feat_s).softmax(-1)
        return P_s, P_q
