from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import inspect
from .abstract import ProbaDetector


class EntropyDetector(ProbaDetector):
    """
    Abstract class for an outlier detector
    """
    def __init__(self, *args, **kwargs):
        pass

    def decision_function(self, support_probas, query_probas):
        """
        support_probas: [Ns, K]
        query_probas: [Nq, K]
        """
        entropy = - (query_probas * torch.log(query_probas + 1e-10)).sum(-1)
        return entropy