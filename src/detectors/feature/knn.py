from typing import Dict, List, Optional
import torch
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
        closest_distances = distances.topk(
            k=self.n_neighbors, largest=False, dim=-1
        ).values  # [Nq, knn]

        if self.method == "mean":
            outlier_scores = closest_distances.mean(-1)

        elif self.method == "largest":
            outlier_scores = closest_distances[:, -1]

        return outlier_scores.squeeze()

    def standardize(self, scores):
        """
        L2 distance on normalized features can vary between 0 and 4
        """
        return scores / 4


def l2(a, b):
    return torch.cdist(a, b)
