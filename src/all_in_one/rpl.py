from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from sklearn.metrics import precision_recall_curve
from skimage.filters import threshold_otsu
import math
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
import numpy as np
import torch.nn as nn


class RPL(AllInOne):
    def __init__(
        self,
        inference_lr: float,
        inference_steps: int,
        num_rp_per_cls: int,
        gamma: float,
        lamb: float,
    ):
        super().__init__()
        self.inference_lr = inference_lr
        self.inference_steps = inference_steps
        self.num_rp_per_cls = num_rp_per_cls
        self.gamma = gamma
        self.lamb = lamb
        self.divide = False

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        num_classes = len(support_labels.unique())

        self.reciprocal_points = nn.Parameter(
            torch.zeros((num_classes * self.num_rp_per_cls, support_features.size(1)))
        )
        nn.init.normal_(self.reciprocal_points)
        self.R = nn.Parameter(torch.zeros((num_classes,)))

        optimizer = torch.optim.Adam(
            [self.reciprocal_points, self.R], lr=self.inference_lr
        )

        for iter_ in range(self.inference_steps):
            loss, _, _ = self.compute_rpl_loss(support_features, support_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits_q, dist_to_rp = self.compute_rpl_logits(query_features, num_classes)

            probs_q = logits_q.softmax(-1)
            outlier_scores = -dist_to_rp.mean(-1).max(-1).values

            return None, probs_q.detach().cpu(), outlier_scores.detach().cpu()

    def compute_rpl_logits(self, features, num_classes):
        # Calculate L2 distance to reciprocal points
        raw_dist_to_rp = -self.cosine(features, self.reciprocal_points)  # [N, M]

        # expects each distance to be squared (squared euclidean distance)
        # dist_to_rp = raw_dist_to_rp ** 2
        dist_to_rp = raw_dist_to_rp
        dist_to_rp = torch.reshape(
            dist_to_rp, (dist_to_rp.shape[0], num_classes, self.num_rp_per_cls)
        )  # [N, K, M // K]
        # output should be batch_size x num_classes
        logits = self.gamma * torch.mean(dist_to_rp, dim=2)

        return logits, dist_to_rp

    def cosine(self, X, Y):
        return F.normalize(X, dim=1) @ F.normalize(Y, dim=1).T

    def compute_rpl_loss(self, support_features, support_labels):
        criterion = nn.CrossEntropyLoss()
        latent_size = support_features.size(1)

        open_loss = torch.tensor(0.0)

        num_classes = len(support_labels.unique())
        logits, dist_to_rp = self.compute_rpl_logits(support_features, num_classes)

        for i in range(0, support_labels.shape[0]):
            curr_label = support_labels[i].item()
            if self.divide:
                dist_to_cls_rp_vector = dist_to_rp[i, curr_label] / latent_size
            else:
                dist_to_cls_rp_vector = dist_to_rp[i, curr_label]
            open_loss += torch.mean(
                (dist_to_cls_rp_vector - self.R[curr_label]) ** 2
            )  # [M // K]

        # this criterion is just cross entropy
        closed_loss = criterion(logits, support_labels).sum()

        open_loss = self.lamb * open_loss
        loss = closed_loss  # + open_loss

        return loss, open_loss, closed_loss
