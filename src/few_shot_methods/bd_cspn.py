import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes


class BDCSPN(AbstractFewShotMethod):

    """
    Implementation of BD-CSPN (ECCV 2020) https://arxiv.org/abs/1911.10713
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.temp = args.softmax_temp

    def rectify_prototypes(self, feat_s: Tensor, feat_q: Tensor, y_s: Tensor) -> None:
        Kes = y_s.unique().size(0)
        one_hot_s = F.one_hot(y_s, Kes)  # [shot_s, K]
        eta = feat_s.mean(0, keepdim=True) - feat_q.mean(
            0, keepdim=True
        )  # [1, feature_dim]
        feat_q = feat_q + eta

        logits_s = self.get_logits(feat_s).exp()  # [shot_s, K]
        logits_q = self.get_logits(feat_q).exp()  # [shot_q, K]

        preds_q = logits_q.argmax(-1)
        one_hot_q = F.one_hot(preds_q, Kes)

        normalization = (
            (one_hot_s * logits_s).sum(0) + (one_hot_q * logits_q).sum(0)
        ).unsqueeze(
            0
        )  # [1, K]
        w_s = (one_hot_s * logits_s) / normalization  # [shot_s, K]
        w_q = (one_hot_q * logits_q) / normalization  # [shot_q, K]

        self.prototypes = (w_s * one_hot_s).t().matmul(feat_s) + (
            w_q * one_hot_q
        ).t().matmul(feat_q)

    def get_logits(self, feats: Tensor):
        """
        inputs:
            samples : tensor of shape [shot, feature_dim]

        returns :
            logits : tensor of shape [shot, K]
        """
        cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        logits = cosine(feats[:, None, :], self.prototypes[None, :, :])
        assert logits.max() <= self.temp and logits.min() >= -self.temp, (
            logits.min(),
            logits.max(),
        )
        return self.temp * logits

    def forward(
        self, feat_s: Tensor, feat_q: Tensor, y_s: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=-1)
        feat_q = F.normalize(feat_q, dim=-1)

        # Initialize prototypes
        self.prototypes = compute_prototypes(feat_s, y_s)  # [K, d]
        self.rectify_prototypes(feat_s=feat_s, y_s=y_s, feat_q=feat_q)
        probs_s = self.get_logits(feat_s).softmax(-1)
        probs_q = self.get_logits(feat_q).softmax(-1)
        return probs_s, probs_q
