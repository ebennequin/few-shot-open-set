from typing import Tuple

import torch.nn.functional as F
from torch import Tensor

from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes


class BDCSPN(FewShotMethod):

    """
    Implementation of BD-CSPN (ECCV 2020) https://arxiv.org/abs/1911.10713
    This is a transductive method.
    """

    def rectify_prototypes(
        self, support_features: Tensor, query_features: Tensor, support_labels: Tensor
    ) -> None:
        Kes = support_labels.unique().size(0)
        one_hot_s = F.one_hot(support_labels, Kes)  # [shot_s, K]
        eta = support_features.mean(0, keepdim=True) - query_features.mean(
            0, keepdim=True
        )  # [1, feature_dim]
        query_features = query_features + eta

        logits_s = self.get_logits_from_cosine_distances_to_prototypes(
            support_features
        ).exp()  # [shot_s, K]
        logits_q = self.get_logits_from_cosine_distances_to_prototypes(
            query_features
        ).exp()  # [shot_q, K]

        preds_q = logits_q.argmax(-1)
        one_hot_q = F.one_hot(preds_q, Kes)

        normalization = (
            (one_hot_s * logits_s).sum(0) + (one_hot_q * logits_q).sum(0)
        ).unsqueeze(
            0
        )  # [1, K]
        w_s = (one_hot_s * logits_s) / normalization  # [shot_s, K]
        w_q = (one_hot_q * logits_q) / normalization  # [shot_q, K]

        self.prototypes = (w_s * one_hot_s).t().matmul(support_features) + (
            w_q * one_hot_q
        ).t().matmul(query_features)

    def forward(
        self, support_features: Tensor, query_features: Tensor, support_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # Initialize prototypes
        self.prototypes = compute_prototypes(support_features, support_labels)  # [K, d]
        self.rectify_prototypes(
            support_features=support_features,
            support_labels=support_labels,
            query_features=query_features,
        )
        probs_s = self.get_logits_from_cosine_distances_to_prototypes(
            support_features
        ).softmax(-1)
        probs_q = self.get_logits_from_cosine_distances_to_prototypes(
            query_features
        ).softmax(-1)
        return probs_s, probs_q
