from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes


class Finetune(AbstractFewShotMethod):
    """
    Implementation of Finetune (or Baseline method) (ICLR 2019) https://arxiv.org/abs/1904.04232
    This is an inductive method.
    """

    def __init__(
        self,
        inference_steps: int = 10,
        inference_lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_steps = inference_steps
        self.lr = inference_lr

    def classify_support_and_queries(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        num_classes = support_labels.unique().size(0)

        # Initialize prototypes
        self.prototypes = compute_prototypes(support_features, support_labels)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        for i in range(self.inference_steps):

            logits_s = self.get_logits_from_euclidean_distances_to_prototypes(
                support_features
            )
            ce = nn.functional.cross_entropy(logits_s, support_labels)
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
