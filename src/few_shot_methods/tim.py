from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes


class AbstractTIM(AbstractFewShotMethod):
    """
    Implementation of TIM method (NeurIPS 2020) https://arxiv.org/abs/2008.11297
    This is an abstract class.
    TIM is a transductive method.
    """

    def __init__(
        self,
        inference_steps: int = 10,
        inference_lr: float = 1e-3,
        loss_weights: List[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_weights = [1.0, 1.0, 0.1] if loss_weights is None else loss_weights
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr


class TIM_GD(AbstractTIM):
    def classify_support_and_queries(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # Metric dic
        num_classes = support_labels.unique().size(0)
        support_labels_one_hot = F.one_hot(support_labels, num_classes)

        # Initialize weights
        self.prototypes = compute_prototypes(support_features, support_labels)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.inference_lr)

        for i in range(self.inference_steps):
            logits_s = self.get_logits_from_euclidean_distances_to_prototypes(
                support_features
            )
            logits_q = self.get_logits_from_euclidean_distances_to_prototypes(
                query_features
            )

            ce = -(support_labels_one_hot * logits_s.log_softmax(1)).sum(1).mean(0)
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
