from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import inspect
from .abstract import ProbaDetector


class kNNDetector(ProbaDetector):
    """
    Abstract class for an outlier detector
    """

    def __init__(self, distance: str, n_neighbors: int, method: str):
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.method = method

    def __call__(self, support_probas, query_probas, **kwargs):
        """
        support_probas: [Ns, K]
        query_probas: [Nq, K]
        """
        distance_fn = eval(self.distance)
        distances = distance_fn(
            query_probas[:, None, :], support_probas[None, :, :]
        )  # [Nq, Ns]
        closest_distances = distances.topk(
            k=self.n_neighbors, largest=False, dim=-1
        ).values  # [Nq, knn]

        if self.method == "mean":
            outlier_scores = closest_distances.mean(-1)

        elif self.method == "largest":
            outlier_scores = closest_distances[:, -1]

        return outlier_scores.squeeze()


def kl(prob_a, prob_b):
    return (prob_a * torch.log(prob_a / prob_b)).sum(-1)


def reverse_kl(prob_a, prob_b):
    return (prob_b * torch.log(prob_b / prob_a)).sum(-1)


def bc(prob_a, prob_b):
    return (prob_a * prob_b).sqrt().sum(-1)
