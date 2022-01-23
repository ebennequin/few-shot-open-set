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
from pyod.utils.utility import standardizer


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


class NaiveAggregator(object):

    def __init__(self, detectors):
        self.detectors = detectors
        self.standardize = False

    def fit(self, support_features):
        for detector in self.detectors:
            detector.fit(support_features)

    def decision_function(self, support_features, query_features):
        n_clf = len(self.detectors)
        train_scores = np.zeros([support_features.shape[0], n_clf])  # [Q, n_clf]
        test_scores = np.zeros([query_features.shape[0], n_clf])  # [Q, n_clf]
        for i, detector in enumerate(self.detectors):
            train_scores[:, i] = detector.decision_scores_  # [Q, ]
            test_scores[:, i] = detector.decision_function(query_features)  # [Q, ]

        # Normalize scores
        # if self.standardize:
        #     train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
        # else:
        test_scores_norm = test_scores

        # Combine
        # outlier_scores = torch.from_numpy(test_scores_norm).max(-1).values
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
        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)
        d = support_features.size(1)

        # Doing OOD detection
        outlier_scores = []
        # for i in range(10):
        # mask = torch.FloatTensor(1, d).uniform_() > 0.8  # remove 80 % of features
        # dropped_support_features = support_features * mask
        # dropped_query_features = query_features * mask
        self.detector.fit(support_features.numpy())
        outlier_scores = torch.from_numpy(self.detector.decision_function(support_features.numpy(), query_features.numpy()))
        # outlier_scores.append(torch.from_numpy(self.detector.decision_function(dropped_support_features.numpy(), dropped_query_features.numpy())))
        # outlier_scores = torch.stack(outlier_scores, 0).mean(0)

        # Obtaining predictions from few-shot classifier

        _, probs_q = self.few_shot_classifier(support_features, query_features, support_labels)
        predictions = probs_q.argmax(-1)
        return outlier_scores, predictions