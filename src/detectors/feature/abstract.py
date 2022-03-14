from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import inspect


class FeatureDetector:
    """
    Abstract class for an outlier detector
    """
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, support_features):
        raise NotImplementedError

    def decision_function(self, support_features, query_features):
        raise NotImplementedError

    def __str__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def __repr__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__