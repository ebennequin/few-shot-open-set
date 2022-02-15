import torch.nn.functional as F
from torch import Tensor
from loguru import logger
import torch
from easyfsl.utils import compute_prototypes
import torch.nn as nn
import numpy as np


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


def debiased_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """

    # Assessing if this is even achievable
    # all_feats = torch.cat([feat_s, feat_q], 0)
    # all_labels = torch.cat([kwargs['support_labels'], kwargs['query_labels']], 0)
    # prototypes = compute_prototypes(all_feats, all_labels)  # [K, d]
    # mean = prototypes.mean(0, keepdim=True)  # [1, d]

    # all_feats = torch.cat([feat_s, feat_q], 0)
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    nodes_degrees = torch.cdist(F.normalize(feat_q, dim=1), F.normalize(prototypes, dim=1)).sum(-1, keepdim=True)  # [N]
    farthest_points = nodes_degrees.topk(dim=0, k=min(feat_q.size(0), max(feat_s.size(0), feat_q.size(0) // 2))).indices.squeeze()
    mean = torch.cat([prototypes, feat_q[farthest_points]], 0).mean(0, keepdim=True)
    assert len(mean.size()) == 2, mean.size()
    return feat_s - mean, feat_q - mean


# def kcenter_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
#     """
#     feat: Tensor shape [N, hidden_dim, *]
#     """
#     random_starts = np.random.choice(np.arange(feat_q.size(0)), size=1, replace=False)
#     centers = []
#     for init_index in random_starts:
#         centers.append(k_center(feat_q, init_index, 75))
#     centers = torch.cat(centers, 0)
#     mean = torch.cat([feat_s, centers], 0).mean(0, keepdim=True)
#     assert len(mean.size()) == 2, mean.size()
#     return feat_s - mean, feat_q - mean

def kcenter_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    all_feats = torch.cat([feat_s, feat_q], 0)
    centers = k_center(all_feats, np.arange(feat_s.size(0)), 75)
    mean = centers.mean(0, keepdim=True)
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


# def base_centering(feat_s: Tensor, feat_q: Tensor, average_train_features: Tensor, **kwargs):
#     """
#     feat: Tensor shape [N, hidden_dim, *]
#     """
#     # average_train_features = average_train_features.unsqueeze(0)
#     if len(average_train_features.size()) > len(feat_s.size()):
#         mean = average_train_features.squeeze(-1).squeeze(-1)
#     elif len(average_train_features.size()) < len(feat_s.size()):
#         mean = average_train_features.unsqueeze(-1).unsqueeze(-1)
#     else:
#         mean = average_train_features
#     return (feat_s - mean), (feat_q - mean)

def protorect_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    # average_train_features = average_train_features.unsqueeze(0)
    ksi = feat_s.mean(0, keepdim=True) - feat_q.mean(0, keepdim=True)
    return feat_s, feat_q + ksi


def k_center(feats: Tensor, init_indexes: np.ndarray, k: int):
    """
    feats : [N, d]
    init_point: [d]

    Runs K-center algorithm. At each iteration, find the farthest node from the set.
    """

    centers_locations = torch.zeros(feats.size(0)).bool()
    for x in init_indexes:
        centers_locations[x] = True

    for i in range(k):
        centers = feats[centers_locations]
        # distances = torch.cdist(F.normalize(feats, dim=1), F.normalize(centers, dim=1))  # [N, N_centers]
        distances = torch.cdist(feats, centers)  # [N, N_centers]
        point_to_set_distances = distances.min(-1).values  # [N,]
        farthest_point = point_to_set_distances.argmax()  # [,]
        centers_locations[farthest_point] = True

    assert centers_locations.sum().item() == (k + len(init_indexes)), centers_locations.sum().item()
    return feats[centers_locations]
