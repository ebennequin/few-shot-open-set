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


class FeatureTransform(nn.Module):

    @classmethod
    def from_cli_args(cls, args, average_train_features, std_train_features):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()},
            average_train_features=average_train_features,
            std_train_features=std_train_features
        )


class AlternateCentering(FeatureTransform):

    def __init__(self, lambda_: float, lr: float, n_iter: int):
        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter

    def forward(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        # mu = kwargs['average_train_features'].squeeze()
        mu = torch.zeros(1, feat_s.size(-1))
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

    def forward(feat_s: Tensor, feat_q: Tensor, average_train_features: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        # average_train_features = average_train_features.unsqueeze(0)
        if len(average_train_features.size()) > len(feat_s.size()):
            mean = average_train_features.squeeze(-1).squeeze(-1)
        elif len(average_train_features.size()) < len(feat_s.size()):
            mean = average_train_features.unsqueeze(-1).unsqueeze(-1)
        else:
            mean = average_train_features
        return (feat_s - mean), (feat_q - mean)

class L2norm(FeatureTransform):

    def forward(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        return F.normalize(feat_s, dim=1), F.normalize(feat_q, dim=1)


class Pool(FeatureTransform):

    def forward(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        return feat_s.mean((-2, -1)), feat_q.mean((-2, -1))


class DebiasedCentering(FeatureTransform):

    def __init__(self, ratio: float):
        self.ratio = ratio

    def forward(self, feat_s: Tensor, feat_q: Tensor, **kwargs):
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