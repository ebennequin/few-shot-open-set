import numpy as np
from .abstract import FeatureDetector
import inspect
from typing import List
from loguru import logger


class NaiveAggregator(FeatureDetector):
    def __init__(self, detectors: List[FeatureDetector]):
        assert isinstance(detectors, List), detectors
        self.detectors = detectors

    def __str__(self):
        return str(self.detectors)

    def __repr__(self):
        return repr(self.detectors)

    def __call__(self, support_features, query_features, **kwargs):
        n_clf = len(self.detectors)
        test_scores = np.zeros([query_features.shape[0], n_clf])  # [Q, n_clf]
        for i, detector in enumerate(self.detectors):
            detector_scores = detector.__call__(
                support_features, query_features, **kwargs
            )
            test_scores[:, i] = detector_scores
        test_scores_norm = test_scores
        outlier_scores = test_scores_norm.mean(axis=-1)
        return outlier_scores
