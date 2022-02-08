from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn


class AbstractDetector:
    """
    Abstract class for an outlier detector
    """
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, support_features):
        raise NotImplementedError

    def decision_function(self, support_features, query_features):
        raise NotImplementedError