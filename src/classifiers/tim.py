from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes


class AbstractTIM(FewShotMethod):
    """
    Implementation of TIM method (NeurIPS 2020) https://arxiv.org/abs/2008.11297
    This is an abstract class.
    TIM is a transductive method.
    """

    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        loss_weights: List[float] = None,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature


class TIM_GD(AbstractTIM):
    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        if kwargs["use_transductively"] is not None:
            unlabelled_data = query_features[kwargs["use_transductively"]]
        else:
            unlabelled_data = query_features

        # Metric dic
        num_classes = support_labels.unique().size(0)
        support_labels_one_hot = F.one_hot(support_labels, num_classes)

        # Initialize weights
        self.prototypes = compute_prototypes(support_features, support_labels)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.inference_lr)

        q_cond_ent_values = []
        q_ent_values = []
        ce_values = []
        acc_values = []

        for i in range(self.inference_steps):
            logits_s = self.get_logits_from_cosine_distances_to_prototypes(
                support_features
            )
            logits_q = self.get_logits_from_cosine_distances_to_prototypes(
                unlabelled_data
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

            with torch.no_grad():
                q_probs = self.get_logits_from_cosine_distances_to_prototypes(
                    query_features
                )
                q_cond_ent_values.append(q_cond_ent.item())
                q_ent_values.append(q_ent.item())
                ce_values.append(ce.item())
                inliers = ~kwargs["outliers"].bool()
                acc_values.append(
                    (q_probs.argmax(-1) == kwargs["query_labels"])[inliers]
                    .float()
                    .mean()
                    .item()
                )

        kwargs["intra_task_metrics"]["classifier_losses"]["cond_ent"].append(
            q_cond_ent_values
        )
        kwargs["intra_task_metrics"]["classifier_losses"]["marg_ent"].append(
            q_ent_values
        )
        kwargs["intra_task_metrics"]["classifier_losses"]["ce"].append(ce_values)
        kwargs["intra_task_metrics"]["classifier_metrics"]["acc"].append(acc_values)

        with torch.no_grad():
            probas_s = self.get_logits_from_cosine_distances_to_prototypes(
                support_features
            ).softmax(-1)
            probas_q = self.get_logits_from_cosine_distances_to_prototypes(
                query_features
            ).softmax(-1)

        return probas_s, probas_q
