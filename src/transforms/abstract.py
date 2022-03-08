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

        for i in range(self.n_iter):

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


class EntropicCentering(FeatureTransform):

    def __init__(self, lambda_: float, lr: float, n_iter: int, n_neighbors: int, init: str):
        super().__init__()
        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter
        self.init = init
        self.n_neighbors = n_neighbors

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
        elif self.init == 'debiased':
            mu = DebiasedCentering(1 / 8).compute_mean(feat_s, feat_q, **kwargs)
        mu.requires_grad_()
        optimizer = torch.optim.Adam([mu], lr=self.lr)
        # cos = torch.nn.CosineSimilarity(dim=-1)

        raw_feat_s = feat_s.clone()
        raw_feat_q = feat_q.clone()

        loss_values = []
        aucs = []

        for i in range(self.n_iter):

            feat_s = F.normalize(raw_feat_s - mu, dim=1)
            feat_q = F.normalize(raw_feat_q - mu, dim=1)

            # Get affinities
            with torch.no_grad():
                dist = torch.cdist(feat_q, feat_s)  # [Nq, Ns]
                n_neighbors = min(self.n_neighbors, feat_s.size(0))

                knn_index = dist.topk(n_neighbors, dim=-1, largest=False).indices  # [N, knn]

                W = torch.zeros(feat_q.size(0), feat_s.size(0))
                W.scatter_(dim=-1, index=knn_index, value=1.0)  # [Nq, Ns]

            similarities = ((feat_q @ feat_s.t()) * W).sum(-1, keepdim=True) / W.sum(-1, keepdim=True)

            # similarities = feat_q @ (feat_s.mean(0, keepdim=True).t())  # [N, 1]
            outlierness_prob = (-self.lambda_ * similarities - bias).sigmoid()  # [N, 1]
            # if i == 0:
            p_outlier = torch.cat([outlierness_prob, 1 - outlierness_prob], dim=1) 
            entropy = - (p_outlier * torch.log(p_outlier + 1e-10)).sum(-1).mean()

            # 2 --- Update mu
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

            with torch.no_grad():
                loss_values.append(entropy.item())
                # outlier_scores = torch.cdist(feat_q, feat_s).topk(k=5, largest=False).values.mean(-1).numpy()
                # detector = pyod.models.knn.KNN(method='mean', n_neighbors=4, n_jobs=None)
                # detector.fit(feat_s.numpy())
                # outlier_scores = detector.decision_function(feat_q.numpy())
                # knn = NearestNeighbors()
                # knn.fit(X=feat_s.numpy())
                # distances, _ = knn.kneighbors(X=feat_q.numpy(), n_neighbors=5, return_distance=True)
                # # outlier_scores_2 = distances.mean(-1)
                # assert np.all(outlier_scores == outlier_scores_2), (outlier_scores, outlier_scores_2)
                fp_rate, tp_rate, thresholds = roc_curve(kwargs['outliers'].numpy(), -similarities.numpy())
                aucs.append(auc_fn(fp_rate, tp_rate))

        logger.warning((outlierness_prob, entropy))
        kwargs['intra_task_metrics']['loss'].append(loss_values)
        kwargs['intra_task_metrics']['auc'].append(aucs)
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

    def compute_mean(self, feat_s, feat_q, **kwargs):
        prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
        nodes_degrees = torch.cdist(F.normalize(feat_q, dim=1), F.normalize(prototypes, dim=1)).sum(-1, keepdim=True)  # [N]
        farthest_points = nodes_degrees.topk(dim=0, k=min(feat_q.size(0), max(feat_s.size(0), feat_q.size(0) // self.ratio))).indices.squeeze()
        mean = torch.cat([prototypes, feat_q[farthest_points]], 0).mean(0, keepdim=True)
        return mean


    def __call__(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        # all_feats = torch.cat([feat_s, feat_q], 0)
        mean = self.compute_mean(feat_s, feat_q, **kwargs)
        assert len(mean.size()) == 2, mean.size()
        return feat_s - mean, feat_q - mean