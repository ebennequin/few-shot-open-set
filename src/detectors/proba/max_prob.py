from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import inspect
from .abstract import ProbaDetector


class MaxProbDetector(ProbaDetector):
    """
    Abstract class for an outlier detector
    """

    def __call__(self, support_probas, query_probas, **kwargs):
        """
        support_probas: [Ns, K]
        query_probas: [Nq, K]
        """
        return -query_probas.max(-1).values
