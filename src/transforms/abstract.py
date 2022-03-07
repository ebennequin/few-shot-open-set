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


class FeatureTransform:

    def __init__(self):
        pass

    @classmethod
    def from_cli_args(cls, args):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()},
        )

    def __call__(self, feat_s, feat_q, **kwargs):
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


class SequentialTransform(FeatureTransform):

    def __init__(self, transform_list: List[FeatureTransform]):
        self.transform_list = transform_list

    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        for transf in self.transform_list:
            feat_s, feat_q = transf(feat_s, feat_q, **kwargs)
        return feat_s, feat_q


class AlternateCentering(FeatureTransform):

    def __init__(self, lambda_: float, lr: float, n_iter: int, init: str):
        super().__init__()
        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter
        self.init = init

    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        if self.init == 'base':
            mu = kwargs['train_mean'].squeeze()
        elif self.init == 'zero':
            mu = torch.zeros(1, feat_s.size(-1))
        elif self.init == 'prototype':
            prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
            mu = prototypes.mean(0, keepdim=True)
        mu.requires_grad_()
        optimizer = torch.optim.SGD([mu], lr=self.lr)
        # cos = torch.nn.CosineSimilarity(dim=-1)

        raw_feat_s = feat_s.clone()
        raw_feat_q = feat_q.clone()

        loss_values = []

        for i in range(10):

            # 1 --- Find potential outliers

            feat_s = F.normalize(raw_feat_s - mu, dim=1)
            feat_q = F.normalize(raw_feat_q - mu, dim=1)
            # prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]

            similarities = feat_q @ feat_s.mean(0, keepdim=True).t()  # [N, 1]
            outlierness = (-self.lambda_ * similarities).detach().sigmoid()  # [N, 1]

            # 2 --- Update mu
            loss = (outlierness * similarities).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

        kwargs['intra_task_metrics']['loss'].append(loss_values)
        return raw_feat_s - mu.detach(), raw_feat_q - mu.detach()


class BaseCentering(FeatureTransform):

    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        train_mean: Tensor = kwargs['train_mean']
        # train_mean = train_mean.unsqueeze(0)
        if len(train_mean.size()) > len(feat_s.size()):
            mean = train_mean.squeeze(-1).squeeze(-1)
        elif len(train_mean.size()) < len(feat_s.size()):
            mean = train_mean.unsqueeze(-1).unsqueeze(-1)
        else:
            mean = train_mean
        return (feat_s - mean), (feat_q - mean)


class L2norm(FeatureTransform):

    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        return F.normalize(feat_s, dim=1), F.normalize(feat_q, dim=1)


class Pool(FeatureTransform):

    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        return feat_s.mean((-2, -1)), feat_q.mean((-2, -1))


class DebiasedCentering(FeatureTransform):

    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        # all_feats = torch.cat([feat_s, feat_q], 0)
        prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
        nodes_degrees = torch.cdist(F.normalize(feat_q, dim=1), F.normalize(prototypes, dim=1)).sum(-1, keepdim=True)  # [N]
        farthest_points = nodes_degrees.topk(dim=0, k=min(feat_q.size(0), max(feat_s.size(0), feat_q.size(0) // self.ratio))).indices.squeeze()
        mean = torch.cat([prototypes, feat_q[farthest_points]], 0).mean(0, keepdim=True)
        assert len(mean.size()) == 2, mean.size()
        return feat_s - mean, feat_q - mean