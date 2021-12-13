import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes


class TIM(AbstractFewShotMethod):
    """Implementation of TIM method (NeurIPS 2020) https://arxiv.org/abs/2008.11297"""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.softmax_temperature = args.softmax_temperature
        self.loss_weights = [1.0, 1.0, 0.1]
        self.inference_steps = args.inference_steps
        self.prototypes: Tensor  # Will be init at the first forward

    def get_logits(self, feats: Tensor) -> Tensor:
        """
        inputs:
            feats : Tensor of shape [shot, feature_dim]

        returns :
            logits : Tensor of shape [shot, num_class]
        """
        logits = self.softmax_temperature * (
            feats.matmul(self.prototypes.t())
            - 1 / 2 * (self.prototypes ** 2).sum(1).view(1, -1)
            - 1 / 2 * (feats ** 2).sum(1).view(-1, 1)
        )

        return logits


class TIM_GD(TIM):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.lr = args.inference_lr

    def forward(
        self,
        feat_s: Tensor,
        feat_q: Tensor,
        y_s: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # Metric dic
        num_classes = y_s.unique().size(0)
        y_s_one_hot = F.one_hot(y_s, num_classes)

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)
        feat_q = F.normalize(feat_q, dim=-1)

        # Initialize weights
        self.prototypes = compute_prototypes(feat_s, y_s)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)

        for i in range(self.inference_steps):
            logits_s = self.get_logits(feat_s)
            logits_q = self.get_logits(feat_q)

            ce = -(y_s_one_hot * logits_s.log_softmax(1)).sum(1).mean(0)
            q_probs = logits_q.softmax(1)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(1).mean(0)
            marginal_y = q_probs.mean(0)
            q_ent = -(marginal_y * torch.log(marginal_y)).sum(0)

            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return logits_s.softmax(-1).detach(), logits_q.softmax(-1).detach()
