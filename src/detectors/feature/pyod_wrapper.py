from .abstract import FeatureDetector
from pyod.models.base import BaseDetector
import torch
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from loguru import logger
import inspect


class PyodWrapper(FeatureDetector):
    """
    Abstract class for an outlier detector
    """

    def __init__(self, pyod_detector: str, **kwargs):
        self.pyod_detector = pyod_detector
        self.kwargs = kwargs
        self.detector = eval(pyod_detector)(**kwargs)
        # assert isinstance(pyod_detector, BaseDetector)

    def __call__(self, support_features, query_features, **kwargs):

        self.detector.fit(support_features.cpu().numpy())
        return torch.from_numpy(
            self.detector.decision_function(query_features.cpu().numpy())
        )

    def __str__(self):
        arg_names = list(inspect.signature(self.detector.__init__).parameters)
        if len(arg_names):
            args = [f"{k}={self.kwargs[k]}" for k in arg_names if k in self.kwargs]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def __repr__(self):
        arg_names = list(inspect.signature(self.detector.__init__).parameters)
        if len(arg_names):
            args = [f"{k}={self.kwargs[k]}" for k in arg_names if k in self.kwargs]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__
