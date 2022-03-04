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


def alternate_centering(feat_s: Tensor, feat_q: Tensor, lambda_=1.0, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    # mu = kwargs['average_train_features'].squeeze()
    mu = torch.zeros(1, feat_s.size(-1))
    mu.requires_grad_()
    optimizer = torch.optim.SGD([mu], lr=1.0)
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
        outlierness = (-lambda_ * similarities).detach().sigmoid()  # [N, 1]

        # 2 --- Update mu
        loss = (outlierness * similarities).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

    # kwargs['intra_task_metrics']['outlier_prec'].append(outlier_prec)
    # kwargs['intra_task_metrics']['inlier_prec'].append(inlier_prec)
    kwargs['intra_task_metrics']['loss'].append(loss_values)
    return raw_feat_s - mu.detach(), raw_feat_q - mu.detach()


def sgd_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    mu = kwargs['average_train_features'].squeeze()
    mu.requires_grad_()
    optimizer = torch.optim.SGD([mu], lr=0.1)
    # cos = torch.nn.CosineSimilarity(dim=-1)

    raw_feat_s = feat_s.clone()
    raw_feat_q = feat_q.clone()

    outlier_prec = []
    inlier_prec = []
    loss_values = []

    for i in range(10):

        # 1 --- Find potential outliers

        feat_s = F.normalize(raw_feat_s - mu, dim=1)
        feat_q = F.normalize(raw_feat_q - mu, dim=1)
        prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]

        with torch.no_grad():
            similarity = (2 - torch.cdist(feat_q, prototypes)) / 2  # [N, K]
            potentiel_outliers = similarity.max(-1).values < 0.5  # [N]
            potentiel_inliers = similarity.max(-1).values > 0.7  # [N]
        if potentiel_outliers.sum():
            outlier_prop = kwargs['outliers'][potentiel_outliers].float().mean().item()
            outlier_prec.append(outlier_prop)
        else:
            outlier_prec.append(255)
        if potentiel_inliers.sum():
            inlier_prop = (1 - kwargs['outliers'][potentiel_inliers]).float().mean().item()
            inlier_prec.append(inlier_prop)
        else:
            inlier_prec.append(255)

        # 2 --- Update mu
        # inliers = torch.cat([feat_s, feat_q[potentiel_inliers]], 0)
        if potentiel_outliers.sum():
            loss = (feat_q[potentiel_outliers] @ feat_s.t()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
        else:
            loss_values.append(255)


    kwargs['intra_task_metrics']['outlier_prec'].append(outlier_prec)
    kwargs['intra_task_metrics']['inlier_prec'].append(inlier_prec)
    kwargs['intra_task_metrics']['loss'].append(loss_values)
    return raw_feat_s - mu.detach(), raw_feat_q - mu.detach()


def debiased_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
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


def custom_gradient(sup, raw_feat_i, raw_feat_o, feat_i, feat_o, mu):
    d = feat_i.size(-1)
    all_feats = torch.cat([feat_i, feat_o], 0)
    all_raw_feats = torch.cat([raw_feat_i, raw_feat_o], 0)
    sign_mask = torch.cat([torch.ones(feat_i.size(0)), - torch.ones(feat_o.size(0))])  # [n,]

    eye = torch.eye(d).unsqueeze(0)  # [1, d, d]
    gram = all_feats[:, :, None] @ all_feats[:, None, :]
    centering_matrix = eye - gram  # [n, d, d]
    norms = torch.cdist(all_raw_feats, mu, p=2).unsqueeze(-1)  # [n, d] , [1, d] = [n, 1] -> [n, 1, 1]
    grad = (sign_mask[:, None, None] * centering_matrix / norms).sum(0)   # [d, d]
    grad = sup.matmul(grad)  # [1, d] x [d, d] = [1, d]
    assert grad.size() == torch.Size([1, d]), grad.size()
    return grad


def cheat_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim]
    """
    raw_feat_i = feat_q[~kwargs['outliers'].bool()]
    raw_feat_o = feat_q[kwargs['outliers'].bool()]

    mu = torch.zeros(1, raw_feat_i.size(1), requires_grad=True)
    optimizer = torch.optim.SGD([mu], lr=0.1)
    d = raw_feat_i.size(-1)

    for i in range(100):
        # with torch.no_grad():
        unorm_feat_i = raw_feat_i - mu
        feat_i = F.normalize(unorm_feat_i, dim=1)  # [?, d]

        feat_o = raw_feat_o - mu
        feat_o = F.normalize(feat_o, dim=1)  # [?, d]
        
        sup = feat_s - mu
        sup = F.normalize(sup, dim=1)
        sup = sup.mean(0, keepdim=True)  # [1, d]

        loss = (feat_o @ sup.t()).sum() #- (feat_i @ sup.t()).sum()

        # ======== Check derivative w.r.t z_i : OK ========
        # expected_grad_zi = - sup  # [N, d]
        # # feat_i.register_hook(lambda grad: logger.warning(((grad - expected_grad_zi) ** 2).sum()))

        # ======== Check derivative w.r.t x_i : OK ========
        # eye = torch.eye(d).unsqueeze(0)  # [1, d, d]
        # gram = feat_i[:, :, None] @ feat_i[:, None, :]
        # norms = torch.cdist(raw_feat_i, mu, p=2).squeeze()  # [n, d] , [1, d] = [n, 1] -> [n]
        # expected_grad_xi = ((eye - gram) / norms[:, None, None])  # [n, d, d]
        # expected_grad_xi = (expected_grad_xi @ expected_grad_zi[:, :, None]).squeeze()  # [n, d]
        # # unorm_feat_i.register_hook(lambda grad: logger.warning(((grad - expected_grad_xi) ** 2).sum()))

        # ======== Check derivative w.r.t mu: OK ========
        # expected_grad_mu = expected_grad_xi.sum(0, keepdim=True)
        # mu.register_hook(lambda grad: logger.warning(((grad - expected_grad_mu) ** 2).sum()))


        optimizer.zero_grad()
        loss.backward()

        # ======== Checking global derivative w.r.t mu: OK ========
        # real_grad = mu.grad
        # analytical_grad = custom_gradient(sup, raw_feat_i, raw_feat_o, feat_i, feat_o, mu)
        # assert real_grad.size() == analytical_grad.size()
        # logger.warning(f"{i}: {((real_grad - analytical_grad) ** 2).sum()}")
        optimizer.step()
        # logger.info(f"{i}: {loss.item()}")


    mu = mu.detach()

    return feat_s - mu, feat_q - mu


def kcenter_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    all_feats = torch.cat([prototypes, feat_q], 0)
    centers = k_center(all_feats, np.arange(prototypes.size(0)), feat_q.size(0) // 4)
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


def inductive_centering(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, h, w]
    """
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    mean = prototypes.mean(0, keepdim=True)
    return F.normalize(feat_s - mean, dim=1), F.normalize(feat_q - mean, dim=1)


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


def dbscan_centering(feat_s: Tensor, feat_q: Tensor, knn: int = 1, **kwargs):
    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]

    intra_class_distances = []
    multi_shot = feat_s.size(0) > 5
    if multi_shot:
        for i, label in enumerate(kwargs["support_labels"].unique()):
            assert i == label, (i, label)
            relevant_feats = feat_s[kwargs["support_labels"] == label]
            intra_dist = torch.cdist(prototypes[i].unsqueeze(0), relevant_feats).squeeze()
            intra_class_distances.append(intra_dist)
        intra_class_distances = torch.cat(intra_class_distances)
        eps = intra_class_distances.mean().item()
    else:
        inter_class_distances = torch.cdist(feat_s, feat_s)
        # eps = inter_class_distances.min().item() / 2
        n_terms = feat_s.size(0) * (feat_s.size(0) - 1) / 2
        eps = inter_class_distances.triu(diagonal=1).sum().item() / n_terms

    # logger.warning(eps)
    all_feats = torch.cat([feat_q], 0)
    db = DBSCAN(max_eps=eps, min_samples=5).fit(all_feats.numpy())
    labels = db.labels_
    centers = [prototypes]
    logger.warning(labels)
    for label in np.unique(labels):
        centers.append(all_feats[labels == label].mean(0, keepdim=True))
    mean = torch.cat(centers, axis=0).mean(0, keepdim=True)
    return feat_s - mean, feat_q - mean


def tarjan_centering(feat_s: Tensor, feat_q: Tensor, knn: int = 1, **kwargs):

    # Create affinity matrix
    # feat_s = feat_s.cuda()
    # feat_q = feat_q.cuda()

    prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
    all_feats = torch.cat([prototypes, feat_q])
    N = all_feats.size(0)
    dist = torch.cdist(all_feats, all_feats)  # [N, N]
    # dist = torch.cdist(F.normalize(all_feats, dim=1), F.normalize(all_feats, dim=1))  # [N, N]

    # Using nearest neighbors

    n_neighbors = min(knn + 1, N)
    knn_indexes = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

    # Trimming graph

    intra_class_distances = []
    multi_shot = feat_s.size(0) > 5
    if multi_shot:
        for i, label in enumerate(kwargs["support_labels"].unique()):
            assert i == label, (i, label)
            relevant_feats = feat_s[kwargs["support_labels"] == label]
            intra_dist = torch.cdist(prototypes[i].unsqueeze(0), relevant_feats).squeeze()
            intra_class_distances.append(intra_dist)
        intra_class_distances = torch.cat(intra_class_distances)
        dist_limit = intra_class_distances.mean()
    else:
        inter_class_distances = torch.cdist(feat_s, feat_s)
        n_terms = feat_s.size(0) * (feat_s.size(0) - 1) / 2
        dist_limit = inter_class_distances.triu(diagonal=1).sum() / n_terms

    trimmed_indexes = []
    for i, row in enumerate(knn_indexes):
        acceptable_indexes = dist[i, row] < dist_limit
        trimmed_indexes.append(row[acceptable_indexes])
    knn_indexes = trimmed_indexes

    # Create and fill graph

    G = nx.DiGraph()
    for i in range(all_feats.size(0)):
        G.add_node(i)

    for i, children in enumerate(knn_indexes):
        G.add_edges_from([(i, k.item()) for k in children])
    connected_components = [list(x) for x in nx.weakly_connected_components(G)]

    centers = [prototypes]
    supposed_inliers = []
    supposed_labels = []
    for group in connected_components:
        if any([i in group for i in range(len(prototypes))]):  # we remove components that already contain prototypes
            supposed_inliers.append(all_feats[group])
            support_labels.append([])
        else:
            centers.append(all_feats[group].mean(0, keepdim=True))
    centers = torch.cat(centers, 0)
    mean = centers.mean(0, keepdim=True)

    # Draw the graph
    # fig = plt.Figure((10, 10), dpi=200)
    # pos = nx.spring_layout(G, k=0.2, iterations=50, scale=2)
    # # k controls the distance between the nodes and varies between 0 and 1
    # # iterations is the number of times simulated annealing is run
    # # default k=0.1 and iterations=50
    # labels = torch.cat([kwargs["support_labels"].unique(), kwargs["query_labels"]])
    # colors = labels.numpy()[G.nodes]
    # colors[len(prototypes):][kwargs['outliers'].bool().numpy()] = 6
    # nx.drawing.nx_pylab.draw_networkx(G, ax=fig.gca(), pos=pos,
    #                                   node_color=colors,
    #                                   node_size=500, cmap='Set1')
    # kwargs['figures']['network'] = fig
    return feat_s - mean, feat_q - mean

















