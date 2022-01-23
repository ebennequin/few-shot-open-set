import torch.nn.functional as F
from torch import Tensor
import torch


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


def react(feat_s: Tensor, feat_q: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    c = torch.quantile(feat_s, 0.95)
    return feat_s.clamp(max=c), feat_q.clamp(max=c)


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
    feat_s = torch.pow(feat_s + 1e-6, beta)
    feat_q = torch.pow(feat_q + 1e-6, beta)

    return feat_s, feat_q


def base_centering(feat_s: Tensor, feat_q: Tensor, average_train_features: Tensor, **kwargs):
    """
    feat: Tensor shape [N, hidden_dim, *]
    """
    # print(feat_s.size(), average_train_features.size())
    if len(average_train_features.size()) != len(feat_s.size()):
        average_train_features = average_train_features.squeeze(-1).squeeze(-1)
    return (feat_s - average_train_features), (feat_q - average_train_features)