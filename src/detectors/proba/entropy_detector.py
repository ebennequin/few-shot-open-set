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

    def __call__(self, support_probas, query_probas, **kwargs):
        """
        support_probas: [Ns, K]
        query_probas: [Nq, K]
        """
        entropy = -(query_probas * torch.log(query_probas + 1e-6)).sum(-1)
        assert not torch.any(torch.isnan(entropy)), query_probas.min()
        return entropy
