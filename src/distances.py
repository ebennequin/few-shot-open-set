import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def min_distance(support: Tensor, query: Tensor):
    """
    arr1: [n1, d, h, w]
    arr2: [n2, d, h, w]

    returns:

    [n1, n2]
    """
    device = support.device
    ns, d, hs, ws = support.size()
    nq, d, hq, wq = query.size()

    support = support.flatten(start_dim=2).permute(0, 2, 1)  # [n_s, d, h, w] -> [n_s, d, h * w] -> [n_s, h * w, d]
    query = query.flatten(start_dim=2).permute(0, 2, 1)
    cross_distances = torch.zeros(ns, nq, hs * ws, hq * wq).to(device)
    for i in range(ns):
        for j in range(nq):
            cross_distances[i, j] = torch.cdist(support[i], query[j])
    # cross_distances = cross_distances.min(dim=-1).values.mean(dim=-1)  # [ns, nq]
    # cross_distances = cross_distances.min(-1).values.min(-1).values  # [ns, nq]
    cross_distances = cross_distances.mean((-2, -1))  # [ns, nq]

    return cross_distances


def sanity_check(support: Tensor, query: Tensor):
    """
    arr1: [n1, d, h, w]
    arr2: [n2, d, h, w]

    returns:

    [n1, n2]
    """
    support = support.mean((-2, -1))
    query = query.mean((-2, -1))

    support = F.normalize(support, dim=-1)
    query = F.normalize(query, dim=-1)

    cross_distances = torch.cdist(support, query)  # [n1, n2]
    return cross_distances