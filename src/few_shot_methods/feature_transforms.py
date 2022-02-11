import torch.nn.functional as F
from torch import Tensor
from loguru import logger
import torch
from easyfsl.utils import compute_prototypes
import torch.nn as nn


def trivial(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    return feat_s, feat_q


def l2_norm(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    return F.normalize(feat_s, dim=1), F.normalize(feat_q, dim=1)


def debiased_bn(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """

    # Assessing if this is even achievable
    # all_feats = torch.cat([feat_s, feat_q], 0)
    # all_labels = torch.cat([kwargs['support_labels'], kwargs['query_labels']], 0)
    # prototypes = compute_prototypes(all_feats, all_labels)  # [K, d]
    # mean = prototypes.mean(0, keepdim=True)  # [1, d]

    # Attempt
    # all_feats = torch.cat([feat_s, feat_q], 0)
    # all_feats = F.normalize(all_feats, dim=1)
    # nodes_degrees = torch.cdist(all_feats, all_feats).mean(-1, keepdim=True)  # [N]
    # logger.warning((nodes_degrees.min(), nodes_degrees.max().values))
    # normalized_degrees = nodes_degrees / nodes_degrees.sum()  # [N, 1]
    # mean = (all_feats * normalized_degrees).sum(0)

    # all_feats = torch.cat([feat_s, feat_q], 0)
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    nodes_degrees = torch.cdist(F.normalize(feat_q, dim=1), F.normalize(prototypes, dim=1)).sum(-1, keepdim=True)  # [N]
    # logger.info(nodes_degrees.size())
    farthest_points = nodes_degrees.topk(dim=0, k=50).indices.squeeze()
    # logger.warning(farthest_points)
    mean = torch.cat([feat_s, feat_q[farthest_points]], 0).mean(0, keepdim=True)
    assert len(mean.size()) == 2, mean.size()
    return feat_s - mean, feat_q - mean


def max_norm(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    norm_term = feat_s.max(dim=0, keepdim=True).values
    return feat_s / norm_term, feat_q / norm_term


def layer_norm(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, h, w]
    """
    dims = (1, 2, 3)
    mean_s = torch.mean(feat_s, dim=dims, keepdim=True)
    var_s = torch.var(feat_s, dim=dims, unbiased=False, keepdim=True)
    mean_q = torch.mean(feat_q, dim=dims, keepdim=True)
    var_q = torch.var(feat_q, dim=dims, unbiased=False, keepdim=True)
    return (feat_s - mean_s) / (var_s.sqrt() + 1e-10), (feat_q - mean_q) / (var_q.sqrt() + 1e-10)


def inductive_batch_norm(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, h, w]
    """
    assert len(feat_s.size()) >= 4
    dims = (0, 2, 3)
    mean = torch.mean(feat_s, dim=dims, keepdim=True)
    var = torch.var(feat_s, dim=dims, unbiased=False, keepdim=True)
    return (feat_s - mean) / (var.sqrt() + 1e-10), (feat_q - mean) / (var.sqrt() + 1e-10)


def instance_norm(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, h, w]
    """
    assert len(feat_s.size()) >= 4
    dims = (2, 3)
    mean_s = torch.mean(feat_s, dim=dims, keepdim=True)
    var_s = torch.var(feat_s, dim=dims, unbiased=False, keepdim=True)
    mean_q = torch.mean(feat_q, dim=dims, keepdim=True)
    var_q = torch.var(feat_q, dim=dims, unbiased=False, keepdim=True)
    return (feat_s - mean_s) / (var_s.sqrt() + 1e-10), (feat_q - mean_q) / (var_q.sqrt() + 1e-10)


def transductive_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, h, w]
    """
    if len(feat_s.size()) == 4:
        dims = (0, 2, 3)  # we normalize over the batch, as well as spatial dims
    elif len(feat_s.size()) == 2:
        dims = (0,)
    else:
        raise ValueError("Problem with size of features.")
    cat_feat = torch.cat([feat_s, feat_q], 0)
    mean = torch.mean(cat_feat, dim=dims, keepdim=True)
    return (feat_s - mean), (feat_q - mean)


def transductive_batch_norm(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, h, w]
    """
    if len(feat_s.size()) == 4:
        dims = (0, 2, 3) # we normalize over the batch, as well as spatial dims
    elif len(feat_s.size()) == 2:
        dims = (0,)
    else:
        raise ValueError("Problem with size of features.")
    cat_feat = torch.cat([feat_s, feat_q], 0)
    mean = torch.mean(cat_feat, dim=dims, keepdim=True)
    var = torch.var(cat_feat, dim=dims, unbiased=False, keepdim=True)
    return (feat_s - mean) / (var.sqrt() + 1e-10), (feat_q - mean) / (var.sqrt() + 1e-10)


def power(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    beta = 0.5
    feat_s = torch.pow(feat_s + 1e-10, beta)
    feat_q = torch.pow(feat_q + 1e-10, beta)

    return feat_s, feat_q


def base_bn(feat_s: Tensor, feat_q: Tensor, average_train_features: Tensor,
            std_train_features: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    # print(feat_s.size(), feat_q.size(), average_train_features.size())
    if len(average_train_features.size()) > len(feat_s.size()):
        average_train_features = average_train_features.squeeze(-1).squeeze(-1)
        std_train_features = std_train_features.squeeze(-1).squeeze(-1)
    elif len(average_train_features.size()) < len(feat_s.size()):
        average_train_features = average_train_features.unsqueeze(-1).unsqueeze(-1)
        std_train_features = std_train_features.unsqueeze(-1).unsqueeze(-1)
    return (feat_s - average_train_features) / (std_train_features + 1e-10).sqrt(), \
           (feat_q - average_train_features) / (std_train_features + 1e-10).sqrt()


def base_centering(feat_s: Tensor, feat_q: Tensor, average_train_features: Tensor, **kwargs):
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