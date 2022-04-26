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
import torch.nn as nn


class PROSER(AllInOne):
    def __init__(
        self,
        beta: float,
        inference_lr: float,
        inference_steps: int,
        n_dummies: float
    ):
        super().__init__()
        self.beta = beta
        self.inference_lr = inference_lr
        self.inference_steps = inference_steps
        self.n_dummies = n_dummies

    def cosine(self, X, Y):
        return F.normalize(X, dim=1) @ F.normalize(Y, dim=1).T

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:

        criterion = nn.CrossEntropyLoss()

        # Metric dic
        num_classes = support_labels.unique().size(0)

        self.cls = nn.Linear(support_features.size(1), num_classes + self.n_dummies)

        # Run adaptation
        optimizer = torch.optim.Adam(self.cls.parameters(), lr=self.inference_lr)

        for i in range(self.inference_steps):

            logits = self.cls(support_features)

            closed_set_loss = criterion(logits, support_labels)

            maxdummy = logits[:, num_classes:].max(-1, keepdim=True).values  # [Nq, 1]
            dummpyoutputs = torch.cat([logits[:, :num_classes], maxdummy], dim=1)
            for i in range(len(dummpyoutputs)):
                dummpyoutputs[i, support_labels[i]] = -1e9
            dummytargets = torch.ones_like(support_labels) * num_classes
            assert dummpyoutputs.size(1) == num_classes + 1, dummpyoutputs.size(1)
            open_set_loss = criterion(dummpyoutputs, dummytargets)

            loss = closed_set_loss + self.beta * open_set_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits_q = self.cls(query_features)
            logits_q[:, num_classes] = logits_q[:, num_classes:].max(-1).values # [Nq, 1]
            return (
                None,
                logits_q[:, :-1].softmax(-1),
                logits_q[:, -1],
            )




