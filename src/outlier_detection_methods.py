from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LocalOutlierFactor
import torch
from torch import nn

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
        _, query_predictions = self.few_shot_classifier(
            feat_s=support_features, feat_q=query_features, y_s=support_labels
        )

        return get_pseudo_renyi_entropy(query_predictions)


class ShannonEntropyOutlierDetector(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        _, query_predictions = self.few_shot_classifier(
            feat_s=support_features, feat_q=query_features, y_s=support_labels
        )

        return get_shannon_entropy(query_predictions)


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
        support_predictions, query_predictions = self.few_shot_classifier(
            feat_s=support_features, feat_q=query_features, y_s=support_labels
        )

        return compute_outlier_scores_with_renyi_divergence(
            soft_predictions=query_predictions,
            soft_support_predictions=support_predictions,
            alpha=-5,
        )


class AbstractOutlierDetectorOnFeatures(AbstractOutlierDetector):
    def initialize_detector(self):
        raise NotImplementedError

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        detector = self.initialize_detector()
        detector.fit(support_features)
        return torch.from_numpy(1 - detector.decision_function(query_features))


class LOFOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(self, n_neighbors=3, metric="euclidean"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric

    def initialize_detector(self):
        return LocalOutlierFactor(
            n_neighbors=self.n_neighbors, novelty=True, metric=self.metric
        )


class IForestOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.n_estimators = n_estimators

    def initialize_detector(self):
        return IForest(n_estimators=self.n_estimators, n_jobs=-1)


class KNNOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(self, n_neighbors=3, method="mean"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.method = method

    def initialize_detector(self):
        return KNN(n_neighbors=self.n_neighbors, method=self.method, n_jobs=-1)


ALL_OUTLIER_DETECTORS = [
    RenyiEntropyOutlierDetector,
    RenyiDivergenceOutlierDetector,
    ShannonEntropyOutlierDetector,
    LOFOutlierDetector,
    IForestOutlierDetector,
    KNNOutlierDetector,
]
