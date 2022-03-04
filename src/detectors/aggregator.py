import numpy as np
from src.detectors import AbstractDetector
import inspect
from typing import List


class NaiveAggregator(AbstractDetector):

    def __init__(self, detectors: List[AbstractDetector]):
        assert isinstance(detectors, List), detectors
        self.detectors = detectors
        self.standardize = False

    def fit(self, support_features, support_labels):
        for detector in self.detectors:
            if "support_labels" in inspect.getfullargspec(detector.fit).args:
                detector.fit(support_features, support_labels=support_labels)
            else:
                detector.fit(support_features)

    def decision_function(self, support_features, query_features):
        n_clf = len(self.detectors)
        test_scores = np.zeros([query_features.shape[0], n_clf])  # [Q, n_clf]
        for i, detector in enumerate(self.detectors):
            # train_scores[:, i] = detector.decision_scores_  # [Q, ]
            detector_scores = detector.decision_function(query_features)  # [Q, ]
            test_scores[:, i] = detector_scores
            # test_scores[:, i] = detector_scores - detector_scores.min() / (detector_scores.max() - detector_scores.min())  # [Q, ]

        test_scores_norm = test_scores

        # Combine
        outlier_scores = test_scores_norm.mean(axis=-1)
        return outlier_scores