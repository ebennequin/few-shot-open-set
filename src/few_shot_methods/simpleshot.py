import argparse
from typing import Optional, Tuple

import torch
from torch import Tensor

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes
import torch.nn.functional as F


class SimpleShot(AbstractFewShotMethod):
    """
    Implementation of SimpleShot method https://arxiv.org/abs/1911.04623
    This is an inductive method.
    In this fashion, it comes down to Prototypical Networks.
    """

    def forward(
        self, feat_s: Tensor, feat_q: Tensor, y_s: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)  # [S, d]
        feat_q = F.normalize(feat_q, dim=-1)  # [Q, d]

        self.prototypes = compute_prototypes(feat_s, y_s)

        return (
            self.get_logits_from_euclidean_distances_to_prototypes(feat_s).softmax(-1),
            self.get_logits_from_euclidean_distances_to_prototypes(feat_q).softmax(-1),
        )
