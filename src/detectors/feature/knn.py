from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import inspect
from .abstract import FeatureDetector


class kNNDetector(FeatureDetector):
    """
    Abstract class for an outlier detector
    """
    def __init__(self, distance: str, n_neighbors: int, method: str):

        self.distance = distance
        self.n_neighbors = n_neighbors
        self.method = method

    def __call__(self, support_features, query_features, **kwargs):
        """
        support_probas: [Ns, K]
        query_probas: [Nq, K]
        """
        distance_fn = eval(self.distance)
        distances = distance_fn(query_features, support_features)  # [Nq, Ns]
        closest_distances = distances.topk(k=self.n_neighbors, largest=False, dim=-1).values  # [Nq, knn]

        if self.method == 'mean':
            outlier_scores = closest_distances.mean(-1)

        elif self.method == 'largest':
            outlier_scores = closest_distances[:, -1]

        return outlier_scores.squeeze()


def l2(feat_a, feat_b):
    return torch.cdist(feat_a, feat_b)