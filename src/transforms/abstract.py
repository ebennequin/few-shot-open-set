import inspect
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
import torch
from easyfsl.utils import compute_prototypes
import torch.nn as nn
import numpy as np
from tarjan import tarjan
import networkx as nx
from sklearn.cluster import DBSCAN, OPTICS
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as auc_fn
import pyod
from sklearn.neighbors import NearestNeighbors


class FeatureTransform:
    def __init__(self):
        pass

    @classmethod
    def from_cli_args(cls, args):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()},
        )

    def __call__(self, raw_feat_s, raw_feat_q, **kwargs):
        raise NotImplementedError

    def __str__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def compute_auc(self, outlierness, **kwargs):
        fp_rate, tp_rate, thresholds = roc_curve(
            kwargs["outliers"].numpy(), outlierness.cpu().numpy()
        )
        return auc_fn(fp_rate, tp_rate)

    def __repr__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__


class SequentialTransform(FeatureTransform):
    def __init__(self, transform_list: List[FeatureTransform]):
        self.transform_list = transform_list

    def __str__(self):
        return str(self.transform_list)

    def __repr__(self):
        return repr(self.transform_list)

    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        for transf in self.transform_list:
            raw_feat_s, raw_feat_q = transf(raw_feat_s, raw_feat_q, **kwargs)
        return raw_feat_s, raw_feat_q


class Power(FeatureTransform):
    def __init__(self, beta: float):
        self.beta = beta
        self.name = "PowerTransform"

    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        # assert torch.all(raw_feat_s > 0) and torch.all(raw_feat_q > 0), (raw_feat_s.min(), raw_feat_q.min())
        return torch.pow(raw_feat_s.relu() + 1e-6, self.beta), torch.pow(
            raw_feat_q.relu() + 1e-6, self.beta
        )


class QRreduction(FeatureTransform):

    name = "QRreduction"

    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        all_features = torch.cat([raw_feat_s, raw_feat_q], 0)
        all_features = torch.qr(all_features.t()).R
        all_features = all_features.t()
        return all_features[: raw_feat_s.size(0)], all_features[raw_feat_s.size(0) :]


class Trivial(FeatureTransform):

    name = "Trivial"

    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        return raw_feat_s, raw_feat_q


class BaseCentering(FeatureTransform):

    name = "Base Centering"

    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        train_mean: Tensor = kwargs["train_mean"]
        return (raw_feat_s - train_mean), (raw_feat_q - train_mean)


class L2norm(FeatureTransform):
    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """

        return F.normalize(raw_feat_s, dim=1), F.normalize(raw_feat_q, dim=1)


class Pool(FeatureTransform):
    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        if len(raw_feat_s.size()) > 2:
            return raw_feat_s.mean((-2, -1)), raw_feat_q.mean((-2, -1))
        else:
            return raw_feat_s, raw_feat_q


class MeanCentering(FeatureTransform):
    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        # all_feats = torch.cat([feat_s, feat_q], 0)
        mean = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
        assert len(mean.size()) == 2, mean.size()
        return raw_feat_s - mean, raw_feat_q - mean


class TransductiveBatchNorm(FeatureTransform):
    def __call__(self, raw_feat_s: Tensor, raw_feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        mean = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
        std = torch.cat([raw_feat_s, raw_feat_q], 0).std(0, unbiased=False, keepdim=True)
        assert len(mean.size()) == 2, mean.size()
        return (raw_feat_s - mean) / (std + 1e-10), (raw_feat_q - mean) / (std + 1e-10)
