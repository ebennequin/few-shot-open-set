import torch
from torch import Tensor


def concat(support_features: Tensor, query_features: dict):
    all_support_features = list(support_features.values())
    all_query_features = list(query_features.values())
    base_size = all_support_features[0].size()
    if len(base_size) == 4:
        spatial_resolution = base_size[-2:]
        assert torch.all([x.size()[-2:] == spatial_resolution for x in all_support_features])
        assert torch.all([x.size()[-2:] == spatial_resolution for x in all_query_features])

    return torch.cat(all_support_features, dim=1), torch.cat(all_query_features, dim=1)


def l2_bar(support_features: Tensor, query_features: dict):
    all_support_features = list(support_features.values())
    all_query_features = list(query_features.values())
    return torch.stack(all_support_features, 0).mean(0), torch.stack(all_query_features, 0).mean(0)