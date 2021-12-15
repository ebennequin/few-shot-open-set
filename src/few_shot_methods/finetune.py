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
    This is an inductive method.
    """

    def __init__(
        self,
        softmax_temperature: float = 1.0,
        inference_steps: int = 10,
        inference_lr: float = 1e-3,
    ):
        super().__init__(softmax_temperature)
        self.inference_steps = inference_steps
        self.lr = inference_lr

    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        num_classes = support_labels.unique().size(0)
        support_labels_one_hot = F.one_hot(support_labels, num_classes)

        # Perform required normalizations
        support_features = F.normalize(support_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)

        # Initialize prototypes
        self.prototypes = compute_prototypes(support_features, support_labels)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        for i in range(self.inference_steps):

            logits_s = self.get_logits_from_euclidean_distances_to_prototypes(
                support_features
            )
            ce = -(support_labels_one_hot * logits_s.log_softmax(1)).sum(1).mean(0)
            optimizer.zero_grad()
            ce.backward()
            optimizer.step()

        probs_q = self.get_logits_from_euclidean_distances_to_prototypes(
            query_features
        ).softmax(-1)
        probs_s = self.get_logits_from_euclidean_distances_to_prototypes(
            support_features
        ).softmax(-1)

        return probs_s.detach(), probs_q.detach()
