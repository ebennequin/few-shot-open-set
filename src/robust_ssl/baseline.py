import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import inspect
from src.models import BACKBONE_CONFIGS
from loguru import logger
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
from .abstract import SSLMethod


class FixMatch(SSLMethod):

    def select_inliers(self, weak_feat_q, **kwargs):
        """
        Returns a mask, shape [Nq] that says whether samples are believed to be inliers or outliers.
        """
        # return torch.ones(weak_feat_q.size(0)).bool().to(self.device)
        return ~ (kwargs['outliers'].bool().to(self.device))

    def get_outlier_scores(self, weak_feat_q, **kwargs):
        """
        Returns a mask, shape [Nq] that says whether samples are believed to be inliers or outliers.
        """
        mask = torch.zeros(weak_feat_q.size(0)).to(self.device)
        mask[kwargs['outliers'].bool().to(self.device)] = 1.0
        return mask

    def update_detector(self, **kwargs):
        return None, None