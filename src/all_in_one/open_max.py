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
from scipy.stats import weibull_min
import numpy as np


class OpenMax(AllInOne):
    def __init__(
        self,
    ):
        super().__init__()

    def cosine(self, X, Y):
        return F.normalize(X, dim=1) @ F.normalize(Y, dim=1).T

    def get_logits(self, prototypes, query_features, bias=True):

        logits = self.cosine(query_features, prototypes)  # [Nq, Ns]
        if self.use_extra_class:
            logits = torch.cat(
                [logits, -logits.mean(-1, keepdim=True)], dim=1
            )  # [Nq, Ns]
        if bias:
            return self.softmax_temperature * logits - self.biases
        else:
            return self.softmax_temperature * logits

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:

        n_shots = (support_labels == 0).sum()
        n_classes = len(support_labels.unique())
        N_q = query_features.size(0)

        # Fit distributions
        self.prototypes = compute_prototypes(support_features, support_labels)
        if n_shots > 1:
            self.fit_weibull(support_features)

        # Compute closed-set activations
        activations = - torch.cdist(query_features, self.prototypes)

        # Compute augmented logits

        weights = torch.zeros(query_features.size(0), n_classes)
        if n_shots > 1:
            top_k_class_values, top_k_class_indexes = activations.topk(k=self.k, largest=True, dim=-1)  # [k * Nq]
            top_k_class_values, top_k_class_indexes = top_k_class_values.ravel(), top_k_class_indexes.ravel()
            relevant_models = self.models[top_k_class_indexes]  # [k*Nq, 3]
            new_weights = []
            for mod, activation in zip(relevant_models, top_k_class_values):
                dist = - activation  # Because we use cosine distance
                new_weights.append(weibull_min.cdf(*mod.numpy(), dist))
            new_weights = torch.Tensor(new_weights)

            sample_index = torch.range(0, query_features.size(0)).repeat_interleave(self.k)
            weights[sample_index, top_k_class_indexes] = new_weights

        augmented_activations = weights * activations
        augmented_probs = augmented_activations.softmax(-1)

        return None, augmented_probs[:, :-1], augmented_probs[:, -1]

    def fit_weibull(self, support_features, support_labels):
        """
        support_features [N, d]

        Builds:
            models : Shape [K, 3]
        """
        classes = support_labels.unique()
        distances = torch.cdist(support_features, self.prototypes)
        self.models = []
        for class_ in classes:
            within_class_distances = distances[support_labels == class_, class_]
            c, loc, scale = weibull_min.fit(within_class_distances)
            self.models.append([c, loc, scale])
        self.models = np.stack(self.models, 0)
