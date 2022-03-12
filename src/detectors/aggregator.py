import numpy as np
from src.detectors import AbstractDetector
import inspect
from typing import List
from loguru import logger


class NaiveAggregator(AbstractDetector):

    def __init__(self, detectors: List[AbstractDetector]):
        assert isinstance(detectors, List), detectors
        self.detectors = detectors

    def __str__(self):
        return str(self.detectors)

    def __repr__(self):
        return repr(self.detectors)

    def fit(self, support_features, **kwargs):
        for detector in self.detectors:
            if "kwargs" in inspect.getfullargspec(detector.fit).args:
                detector.fit(support_features, **kwargs)
            else:
                detector.fit(support_features)

    def decision_function(self, query_features, **kwargs):
        n_clf = len(self.detectors)
        test_scores = np.zeros([query_features.shape[0], n_clf])  # [Q, n_clf]
        for i, detector in enumerate(self.detectors):
            if inspect.getfullargspec(detector.decision_function).varkw == 'kwargs':
                detector_scores = detector.decision_function(query_features, **kwargs)
            else:
                detector.decision_function(query_features)
            test_scores[:, i] = detector_scores
        test_scores_norm = test_scores
        outlier_scores = test_scores_norm.mean(axis=-1)
        return outlier_scores