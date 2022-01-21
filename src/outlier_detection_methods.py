import inspect
import math
from typing import Dict

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LocalOutlierFactor
import torch
from sklearn.svm import SVC
from torch import nn, Tensor
from torch.nn import functional as F

from src.feature_transforms import AbstractFeatureTransformer, IdentityTransformer
from src.few_shot_methods import AbstractFewShotMethod
from src.utils.outlier_detectors import (
    get_pseudo_renyi_entropy,
    get_shannon_entropy,
    compute_outlier_scores_with_renyi_divergence,
)
from src.utils.utils import pool_features


class AbstractOutlierDetector(nn.Module):
    def __init__(
        self,
        prepool_feature_transformer: AbstractFeatureTransformer = None,
        postpool_feature_transformer: AbstractFeatureTransformer = None,
    ):
        super().__init__()
        self.prepool_feature_transformer = (
            prepool_feature_transformer
            if prepool_feature_transformer
            else IdentityTransformer()
        )
        self.postpool_feature_transformer = (
            postpool_feature_transformer
            if postpool_feature_transformer
            else IdentityTransformer()
        )

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        support_features, query_features = self.transform_features(
            support_features, query_features
        )
        return self.detect_outliers(
            support_features, support_labels, query_features, query_labels
        )

    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        raise NotImplementedError(
            "All outlier detectors must implement detect_outliers()."
        )

    def transform_features(self, support_features: Tensor, query_features: Tensor):
        """
        Performs an (optional) normalization of feature maps or feature vectors,
        then average pooling to obtain feature vectors in all cases,
        then another (optional) normalization.
        """

        # Pre-pooling transforms
        support_features, query_features = self.prepool_feature_transformer(
            support_features, query_features
        )

        # Average pooling
        query_features, support_features = pool_features(
            query_features, support_features
        )

        # Post-pooling transforms
        support_features, query_features = self.postpool_feature_transformer(
            support_features, query_features
        )

        return support_features, query_features

    @classmethod
    def from_args(cls, args: Dict):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args.items() if k in signature.parameters.keys()}
        )


class RenyiEntropyOutlierDetector(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod, **kwargs):
        super().__init__(**kwargs)
        self.few_shot_classifier = few_shot_classifier

    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        _, query_predictions = self.few_shot_classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
        )

        return get_pseudo_renyi_entropy(query_predictions)


class ShannonEntropyOutlierDetector(AbstractOutlierDetector):
    def __init__(self, few_shot_classifier: AbstractFewShotMethod):
        super().__init__()
        self.few_shot_classifier = few_shot_classifier

    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        _, query_predictions = self.few_shot_classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
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

    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
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
        )


class AbstractOutlierDetectorOnFeatures(AbstractOutlierDetector):
    def initialize_detector(self):
        raise NotImplementedError(
            "All outlier detectors on features must implement initialize_detectors."
        )

    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        detector = self.initialize_detector()
        detector.fit(support_features)
        return torch.from_numpy(1 - detector.decision_function(query_features))


class LOFOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(
        self,
        n_neighbors: int = 3,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.metric = metric

    def initialize_detector(self):
        return LocalOutlierFactor(
            n_neighbors=self.n_neighbors, novelty=True, metric=self.metric
        )


class IForestOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators

    def initialize_detector(self):
        return IForest(n_estimators=self.n_estimators, n_jobs=-1)


class KNNOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(
        self,
        n_neighbors: int = 3,
        method: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.method = method

    def initialize_detector(self):
        return KNN(n_neighbors=self.n_neighbors, method=self.method, n_jobs=-1)


class SupervisedOutlierDetector(AbstractOutlierDetectorOnFeatures):
    """
    Deprecated.
    """

    def __init__(
        self,
        predict_class_by_class: bool = True,
        base_features: Dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if base_features is None:
            raise ValueError("Missing base set features")
        self.base_features = torch.from_numpy(
            np.concatenate(list(base_features.values()))
        )
        # Deprecated because we now have several possible normalization methods
        # if normalize_features:
        #     self.base_features = F.normalize(self.base_features, dim=-1)
        self.predict_class_by_class = predict_class_by_class

    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        if self.predict_class_by_class:
            predictions = torch.zeros((len(query_labels), 5))
            for i in range(5):
                base_sample = self.base_features[
                    torch.ones(len(self.base_features)).multinomial(num_samples=25)
                ]
                X = torch.cat([support_features, base_sample])
                Y = torch.cat(
                    [
                        1 - torch.eq(support_labels, i).int(),
                        +torch.tensor(len(base_sample) * [1]),
                    ]
                )
                # st.write(Y)
                classifier = SVC(probability=True, class_weight="balanced")
                classifier.fit(X, Y)
                predictions[:, i] = torch.from_numpy(
                    classifier.predict_proba(query_features)[:, 0]
                )
            return predictions.max(dim=1)[0]

        else:
            base_sample = self.base_features[
                torch.ones(len(self.base_features)).multinomial(num_samples=25)
            ]
            X = torch.cat([support_features, base_sample])
            Y = torch.tensor(len(support_features) * [0] + len(base_sample) * [1])
            classifier = SVC(probability=True, class_weight="balanced")
            classifier.fit(X, Y)
            return classifier.predict_proba(query_features)[:, 0]


class KNNWithDispatchedClusters(KNNOutlierDetector):
    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        dispatcher = nn.Linear(
            support_features.shape[1], support_features.shape[1], bias=False
        )
        nn.init.eye_(dispatcher.weight)

        optimizer = torch.optim.Adam(dispatcher.parameters(), lr=0.001)

        repeated_labels = support_labels.repeat((len(support_labels), 1))
        mask = ((~repeated_labels.eq(repeated_labels.T) * 2.0) - 1.0).triu(diagonal=1)
        mask = mask / mask.sum()
        for i in range(10):
            optimizer.zero_grad()
            transported_support = dispatcher(support_features)
            transported_support = F.normalize(transported_support, dim=-1)
            loss = (transported_support.mm(transported_support.T) * mask).sum()
            loss.backward()
            optimizer.step()

        dispatched_support = dispatcher(support_features).detach()
        dispatched_queries = dispatcher(query_features).detach()

        detector = self.initialize_detector()
        detector.fit(dispatched_support)
        return torch.from_numpy(1 - detector.decision_function(dispatched_queries))

    # adapted from https://github.com/Yunseong-Jeong/n_sphere/blob/master/n_sphere/n_sphere.py
    @staticmethod
    def to_spherical(tensor):
        result = []
        for line in tensor:
            r = 0
            for i in range(0, len(line)):
                r += line[i] * line[i]
            r = math.sqrt(r)
            convert = [r]
            for i in range(0, len(line) - 2):
                convert.append(round(math.acos(line[i] / r), 6))
                r = math.sqrt(r * r - line[i] * line[i])
            if line[-2] >= 0:
                convert.append(math.acos(line[-2] / r))
            else:
                convert.append(2 * math.pi - math.acos(line[-2] / r))
            convert = torch.tensor(convert)
            result += [convert]

        result = torch.stack(result)

        return result[:, 1:]

    @staticmethod
    def to_rectangular(tensor):
        result = []
        for line in tensor:
            r = 1.0
            multi_sin = 1
            convert = []
            for i in range(1, len(line) - 1):
                convert.append(r * multi_sin * math.cos(line[i]))
                multi_sin *= math.sin(line[i])
            convert.append(r * multi_sin * math.cos(line[-1]))
            convert.append(r * multi_sin * math.sin(line[-1]))
            convert = torch.tensor(convert)
            result += [convert]

        result = torch.stack(result)

        return result


class NearestNeighborRatio(AbstractOutlierDetectorOnFeatures):
    def detect_outliers(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        support_features_by_class = support_features.view(
            (len(support_labels.unique()), -1, support_features.shape[1])
        )
        distances = torch.cdist(query_features, support_features_by_class)

        query_to_classwise_nn = distances.min(dim=-1)[0].T

        top2_query_to_classwise_nn = query_to_classwise_nn.topk(
            k=2, dim=1, largest=False
        )[0]

        return top2_query_to_classwise_nn[:, 1] / top2_query_to_classwise_nn[:, 0] ** 2

    def initialize_detector(self):
        pass


ALL_OUTLIER_DETECTORS = [
    RenyiEntropyOutlierDetector,
    RenyiDivergenceOutlierDetector,
    ShannonEntropyOutlierDetector,
    LOFOutlierDetector,
    IForestOutlierDetector,
    KNNOutlierDetector,
    SupervisedOutlierDetector,
    KNNWithDispatchedClusters,
    NearestNeighborRatio,
]
