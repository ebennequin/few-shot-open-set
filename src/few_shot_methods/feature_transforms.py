import torch.nn.functional as F
from torch import Tensor
from loguru import logger
import torch
from easyfsl.utils import compute_prototypes
import torch.nn as nn
import numpy as np
from tarjan import tarjan
import networkx as nx
import matplotlib.pyplot as plt


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
    farthest_points = nodes_degrees.topk(dim=0, k=min(feat_q.size(0), max(feat_s.size(0), feat_q.size(0) // 4))).indices.squeeze()
    mean = torch.cat([prototypes, feat_q[farthest_points]], 0).mean(0, keepdim=True)
    assert len(mean.size()) == 2, mean.size()
    return feat_s - mean, feat_q - mean


def oracle_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    prototypes = compute_prototypes(torch.cat([feat_s, feat_q], 0),
                                    torch.cat([kwargs["support_labels"], kwargs["query_labels"]], 0))  # [K, d]
    mean = prototypes.mean(0, keepdim=True)
    assert len(mean.size()) == 2, mean.size()
    return feat_s - mean, feat_q - mean

# def oracle_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
#     """
#     feat: Tensor shape [N, hidden_dim, *]
#     """
#     mean_id = torch.cat([feat_s, feat_q[~ kwargs['outliers'].bool()]], 0).mean(0, keepdim=True)
#     mean_ood = feat_q[kwargs['outliers'].bool()].mean(0, keepdim=True)
#     mean = (mean_ood + mean_id) / 2
#     assert len(mean.size()) == 2, mean.size()
#     return feat_s - mean, feat_q - mean


# def kcenter_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
#     """
#     feat: Tensor shape [N, hidden_dim, *]
#     """
#     random_starts = np.random.choice(np.arange(feat_q.size(0)), size=5, replace=False)
#     centers = []
#     for init_index in random_starts:
#         centers.append(k_center(feat_q, [init_index], 10))
#     centers = torch.cat(centers, 0)
#     mean = torch.cat([feat_s, centers], 0).mean(0, keepdim=True)
#     assert len(mean.size()) == 2, mean.size()
#     return feat_s - mean, feat_q - mean

def kcenter_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    all_feats = torch.cat([prototypes, feat_q], 0)
    centers = k_center(all_feats, np.arange(prototypes.size(0)), prototypes.size(0))
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
        # point_to_set_distances = distances.min(-1).values  # [N,]
        point_to_set_distances = distances.mean(-1)
        point_to_set_distances[centers_locations] = 0.
        farthest_point = point_to_set_distances.argmax()  # [,]
        centers_locations[farthest_point] = True

    assert centers_locations.sum().item() == (k + len(init_indexes)), centers_locations.sum().item()
    return feats[centers_locations]


def tarjan_centering(feat_s: Tensor, feat_q: Tensor, knn: int = 1, **kwargs):

    # Create affinity matrix
    
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    all_feats = torch.cat([prototypes, feat_q])
    N = all_feats.size(0)
    dist = torch.cdist(all_feats, all_feats)  # [N, N]
    # dist = torch.cdist(F.normalize(all_feats, dim=1), F.normalize(all_feats, dim=1))  # [N, N]

    # Using nearest neighbors

    n_neighbors = min(knn + 1, N)
    knn_indexes = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]
    # trimmed_indexes = []
    # dist_limit = dist.mean()
    # for i, row in enumerate(knn_indexes):
    #     acceptable_indexes = dist[i, row] < dist_limit
    #     trimmed_indexes.append(row[acceptable_indexes])
    # logger.info(trimmed_indexes)

    # Strongly connected components

    # graph = {i: array.tolist() for i, array in enumerate(knn_indexes)}
    # connected_components = tarjan(graph)

    # Weakly connected components

    # logger.info(kwargs['query_labels'].unique(return_counts=True)[-1])
    G = nx.DiGraph()
    for i, children in enumerate(knn_indexes):
        G.add_edges_from([(i, k.item()) for k in children])
    connected_components = [list(x) for x in nx.weakly_connected_components(G)]

    centers = [prototypes]
    for group in connected_components:
        if any([i in group for i in range(len(prototypes))]):  # we remove components that already contain prototypes
            continue
        else:
            centers.append(all_feats[group].mean(0, keepdim=True))
    centers = torch.cat(centers, 0)
    mean = centers.mean(0, keepdim=True)

    # Draw the graph
    # fig = plt.Figure((10, 10), dpi=200)
    # labels = torch.cat([kwargs["support_labels"], kwargs["query_labels"]])
    # pos = nx.spring_layout(G, k=0.2, iterations=50, scale=2)
    # # k controls the distance between the nodes and varies between 0 and 1
    # # iterations is the number of times simulated annealing is run
    # # default k=0.1 and iterations=50
    # colors = labels.numpy()[G.nodes]
    # colors[len(prototypes):][kwargs['outliers'].bool().numpy()] = 6
    # nx.drawing.nx_pylab.draw_networkx(G, ax=fig.gca(), pos=pos,
    #                                   node_color=colors,
    #                                   node_size=500, cmap='Set1')
    # kwargs['figures']['network'] = fig
    return feat_s - mean, feat_q - mean

















