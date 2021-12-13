import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes


class Finetune(AbstractFewShotMethod):
    """
    Implementation of Finetune (or Baseline method) (ICLR 2019) https://arxiv.org/abs/1904.04232
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.inference_steps = args.inference_steps
        self.lr = args.inference_lr

    def forward(
        self,
        feat_s: Tensor,
        feat_q: Tensor,
        y_s: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        num_classes = y_s.unique().size(0)
        y_s_one_hot = F.one_hot(y_s, num_classes)

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)
        feat_q = F.normalize(feat_q, dim=-1)

        # Initialize prototypes
        self.prototypes = compute_prototypes(feat_s, y_s)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        for i in range(self.inference_steps):

            logits_s = self.get_logits_from_euclidean_distances_to_prototypes(feat_s)
            ce = -(y_s_one_hot * logits_s.log_softmax(1)).sum(1).mean(0)
            optimizer.zero_grad()
            ce.backward()
            optimizer.step()

        probs_q = self.get_logits_from_euclidean_distances_to_prototypes(
            feat_q
        ).softmax(-1)
        probs_s = self.get_logits_from_euclidean_distances_to_prototypes(
            feat_s
        ).softmax(-1)

        return probs_s.detach(), probs_q.detach()
