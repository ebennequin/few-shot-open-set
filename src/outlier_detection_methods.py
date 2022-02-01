import math
from typing import Dict

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LocalOutlierFactor
import torch
from torch import Tensor
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
from functools import partial
from sklearn.metrics import roc_curve, auc
from src.few_shot_methods import AbstractFewShotMethod
from .distances import __dict__ as ALL_DISTANCES
from pyod.utils.utility import standardizer


class AbstractOutlierDetector(nn.Module):
    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        raise NotImplementedError


class local_knn(object):

    def __init__(self, n_neighbors: str, method: str, distance: str):
        self.n_neighbors = n_neighbors
        self.method = method
        self.distance = ALL_DISTANCES[distance]
        self.downsample_support = True

    def fit(self, support_features):
        if self.downsample_support:
            self.support_features = F.adaptive_avg_pool2d(support_features, (1, 1))
        else:
            self.support_features = support_features

    def decision_function(self, query_features):
        """
        support_features: [ns, d, h, w]
        query_features: [nq, d, h, w]

        """
        query_features = F.adaptive_avg_pool2d(query_features, (1, 1))
        self.support_features = F.normalize(self.support_features, dim=1)
        query_features = F.normalize(query_features, dim=1)
        # print(self.support_features.size(), query_features.size())
        distances = self.distance(self.support_features, query_features).t()  # [n_q, n_s]
        neighbors = min(self.n_neighbors, distances.size(1))
        k_smallest_distances = torch.topk(distances, axis=-1, k=neighbors, sorted=True, largest=False).values  # [n_q, k]
        if self.method == 'largest':
            scores = k_smallest_distances[:, -1]  # [n_q]
        elif self.method == 'mean':
            scores = k_smallest_distances.mean(-1)  # [n_q]
        return scores


class NaiveAggregator(object):

    def __init__(self, detectors):
        self.detectors = detectors
        self.standardize = False

    def fit(self, support_features):
        for detector in self.detectors:
            detector.fit(support_features)

    def decision_function(self, support_features, query_features):
        n_clf = len(self.detectors)
        test_scores = np.zeros([query_features.shape[0], n_clf])  # [Q, n_clf]
        for i, detector in enumerate(self.detectors):
            # train_scores[:, i] = detector.decision_scores_  # [Q, ]
            test_scores[:, i] = detector.decision_function(query_features)  # [Q, ]

        test_scores_norm = test_scores

        # Combine
        outlier_scores = test_scores_norm.mean(axis=-1)
        return outlier_scores


class FewShotDetector(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod, detector):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier
        self.detector = detector

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:

        # Transforming features
        support_features, query_features = self.few_shot_classifier.transform_features(support_features.copy(), query_features.copy())
        # Doing OOD detection
        self.detector.fit(support_features)
        outlier_scores = torch.from_numpy(self.detector.decision_function(support_features, query_features))  # [?,]

        # Obtaining predictions from few-shot classifier
        if not self.few_shot_classifier.pool:
            support_features = support_features.mean((-2, -1))
            query_features = query_features.mean((-2, -1))

        _, probs_q = self.few_shot_classifier(support_features, query_features, support_labels)
        predictions = probs_q.argmax(-1)
        return outlier_scores, predictions