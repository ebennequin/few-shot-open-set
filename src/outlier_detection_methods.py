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
        


class SupervisedOutlierDetector(AbstractOutlierDetectorOnFeatures):
    def __init__(
        self,
        few_shot_classifier,
        predict_class_by_class: bool = True,
        base_features: Dict = None,
    ):
        super().__init__(few_shot_classifier)
        if base_features is None:
            raise ValueError("Missing base set features")
        self.base_features = torch.from_numpy(
            np.concatenate(list(base_features.values()))
        )
        if normalize_features:
            self.base_features = F.normalize(self.base_features, dim=-1)
        self.predict_class_by_class = predict_class_by_class

    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:
        if self.normalize_features:
            support_features = F.normalize(support_features, dim=-1, p=2)
            query_features = F.normalize(query_features, dim=-1, p=2)

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
    def forward(
        self, support_features, support_labels, query_features, query_labels
    ) -> torch.Tensor:

        support_features, query_features = self.few_shot_classifier.transform_features(support_features, query_features)

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

        detector = self.fit_detector()
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
    def forward(
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

    def fit_detector(self, known: Tensor):
        pass


# ALL_OUTLIER_DETECTORS = [
#     RenyiEntropyOutlierDetector,
#     RenyiDivergenceOutlierDetector,
#     ShannonEntropyOutlierDetector,
#     LOFOutlierDetector,
#     IForestOutlierDetector,
#     KNNOutlierDetector,
#     SupervisedOutlierDetector,
#     KNNWithDispatchedClusters,
#     NearestNeighborRatio,
# ]
