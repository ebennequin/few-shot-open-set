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
from src.utils.outlier_detectors import (
    get_pseudo_renyi_entropy,
    get_shannon_entropy,
    compute_outlier_scores_with_renyi_divergence,
)


class AbstractOutlierDetector(nn.Module):
    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        raise NotImplementedError


class RenyiEntropyOutlierDetector(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        # Transforming features
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)

        _, query_predictions = self.few_shot_classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
        )

        return get_pseudo_renyi_entropy(query_predictions), query_predictions.argmax(-1)


class ShannonEntropyOutlierDetector(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        # Transforming features
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)

        _, query_predictions = self.few_shot_classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
        )

        return get_shannon_entropy(query_predictions), query_predictions.argmax(-1)


class RenyiDivergenceOutlierDetector(AbstractOutlierDetector):
    def __init__(
        self,
        few_shot_classifier: AbstractFewShotMethod,
        alpha: int = 2,
        method: str = "topk",
        k: int = 3,
    ):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier
        self.alpha = alpha
        self.method = method
        self.k = k

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        # Transforming features
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)

        support_predictions, query_predictions = self.few_shot_classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
        )

        return compute_outlier_scores_with_renyi_divergence(
            soft_predictions=query_predictions,
            soft_support_predictions=support_predictions,
            alpha=self.alpha,
            method=self.method,
            k=self.k,
        ), query_predictions.argmax(-1)


class AbstractOutlierDetectorOnFeatures(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier

    def fit_detector(self, known: Tensor):
        raise NotImplementedError

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:

        # Transforming features
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)

        # Doing OOD detection

        detector = self.fit_detector(support_features)
        outlier_scores = torch.from_numpy(1 - detector.decision_function(query_features))
        
        # Obtaining predictions from few-shot classifier

        _, probs_q = self.few_shot_classifier(support_features, query_features, support_labels)
        predictions = probs_q.argmax(-1)
        return outlier_scores, predictions


class LOFOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(
        self,
        few_shot_classifier: AbstractFewShotMethod,
        n_neighbors: int = 3,
        metric: str = "euclidean",
    ):
        super().__init__(few_shot_classifier)
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit_detector(self, known: Tensor):
        return LocalOutlierFactor(
            n_neighbors=self.n_neighbors, novelty=True, metric=self.metric
        )


class IForestOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(self, few_shot_classifier, n_estimators: int = 100):
        super().__init__(few_shot_classifier)
        self.n_estimators = n_estimators

    def fit_detector(self, known: Tensor):
        return IForest(n_estimators=self.n_estimators, n_jobs=-1)


class KNNOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(
        self,
        few_shot_classifier,
        n_neighbors: int = 3,
        method: str = "mean",
    ):
        super().__init__(few_shot_classifier)
        self.n_neighbors = n_neighbors
        self.method = method

    def fit_detector(self, known: Tensor):
        detector = KNN(n_neighbors=self.n_neighbors, method=self.method, n_jobs=-1)
        detector.fit(known)
        return detector


class IterativeDetector(KNNOutlierDetector):

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)
        known = support_features
        unknown = query_features

        n_query = query_features.size(0)

        n_neighbors = self.n_neighbors

        for index in range(20):

            # Compute those with very low outlier scores

            detector = KNN(n_neighbors=n_neighbors, method=self.method, n_jobs=-1)
            detector.fit(known)
            unknown_score = torch.from_numpy(detector.decision_function(unknown))
            k = 1
            kth_score = unknown_score.topk(k=k, largest=False).values[k-1]
            new_known = unknown_score <= kth_score

            # Update sets

            known = torch.cat([known, unknown[new_known]], 0)
            unknown = unknown[~new_known]

            # Monitoring 

            full_outlier = torch.from_numpy(detector.decision_function(query_features))
            fp_rate, tp_rate, _ = roc_curve([False] * (n_query // 2) + [True] * (n_query // 2), full_outlier)
            area = auc(fp_rate, tp_rate)

            print(f"Iteration {index} : auc={area} \t |K|={len(known)} \t |U|={len(unknown)}")

        outlier_scores = torch.from_numpy(1 - detector.decision_function(query_features))

        return outlier_scores, torch.ones_like(query_labels)


class MultiDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(
        self,
        few_shot_classifier,
        detectors,
    ):
        super().__init__(few_shot_classifier)
        self.detectors = detectors
        # print(self.detectors)
        self.few_shot_classifier = few_shot_classifier

    def fit_detectors(self, known: Tensor):
        for detector in self.detectors:
            detector.fit(known)

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:

        # Transforming features
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)

        # Fit detectors
        self.fit_detectors(support_features)

        # Doing OOD detection
        outlier_scores = []
        for detector in self.detectors:
            out = torch.from_numpy(1 - detector.decision_function(query_features))
            out = out - out.min() / (out.max() - out.min())
            outlier_scores.append(out)
        outlier_scores = torch.stack(outlier_scores, 0).mean(0)

        # Obtaining predictions from few-shot classifier

        _, probs_q = self.few_shot_classifier(support_features, query_features, support_labels)
        predictions = probs_q.argmax(-1)
        return outlier_scores, predictions


DETECTORS = {f'knn_{i}': KNN(n_neighbors=i, method='mean') for i in range(20)}
