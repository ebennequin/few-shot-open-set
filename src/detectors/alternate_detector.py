import torch
from .abstract_detector import AbstractDetector
from easyfsl.utils import compute_prototypes
from src.constants import MISC_MODULES
from loguru import logger
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
from .abstract_detector import AbstractDetector


class AlternateDetector(AbstractDetector):

    def __init__(self, lambda_: float, lr: float, n_iter: int, init: str, n_neighbors: int):
        super().__init__()
        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter
        self.init = init
        self.n_neighbors = n_neighbors
        self.name = 'AlternateDetector'

    def fit(self, support_features, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        self.raw_feat_s = support_features

    def compute_auc(self, outlierness, **kwargs):
        fp_rate, tp_rate, thresholds = roc_curve(kwargs['outliers'].numpy(), outlierness.cpu().numpy())
        return auc_fn(fp_rate, tp_rate)

    def decision_function(self, raw_feat_q, **kwargs):

        loss_values = []
        aucs = []
        marg_entropy = []
        diff_with_oracle = []
        raw_feat_s = self.raw_feat_s.cuda()
        raw_feat_q = raw_feat_q.cuda()

        if self.init == 'base':
            mu = kwargs['train_mean'].squeeze().cuda()
        elif self.init == 'zero':
            mu = torch.zeros(1, raw_feat_s.size(-1)).cuda()
        elif self.init == 'mean':
            mu = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
        mu.requires_grad_()
        optimizer = torch.optim.SGD([mu], lr=self.lr)

        for i in range(self.n_iter):

            # 1 --- Find potential outliers

            feat_s = F.normalize(raw_feat_s - mu, dim=1)
            feat_q = F.normalize(raw_feat_q - mu, dim=1)
            # prototypes = compute_prototypes(feat_s, kwargs["support_labels"])  # [K, d]
            with torch.no_grad():
                dist = torch.cdist(feat_q, feat_s)  # [Nq, Ns]
                n_neighbors = min(self.n_neighbors, feat_s.size(0))
                knn_index = dist.topk(n_neighbors, dim=-1, largest=False).indices  # [N, knn]

                W = torch.zeros(feat_q.size(0), feat_s.size(0)).cuda()
                W.scatter_(dim=-1, index=knn_index, value=1.0)  # [Nq, Ns]

            similarities = ((feat_q @ feat_s.t()) * W).sum(-1, keepdim=True) / W.sum(-1, keepdim=True)  # [N, 1]
            support_self_similarity = ((feat_s @ feat_s.t())).mean()  # [Ns, Ns]
            outlierness = (-self.lambda_ * similarities).detach().sigmoid()  # [N, 1]

            # 2 --- Update mu

            loss = (outlierness * similarities).mean() - support_self_similarity  #- ((1 - outlierness) * similarities).mean() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step()

            with torch.no_grad():
                loss_values.append(loss.item())
                marg_probas = torch.cat([outlierness, 1 - outlierness], dim=1).mean(0)
                marg_entropy.append(- (marg_probas * torch.log(marg_probas)).sum().item())
                diff_with_oracle.append(abs(marg_probas[0].item() - (kwargs['outliers'].sum() / kwargs['outliers'].size(0)).item()))
                aucs.append(self.compute_auc(outlierness, **kwargs))
        kwargs['intra_task_metrics']['loss'].append(loss_values)
        kwargs['intra_task_metrics']['auc'].append(aucs)
        kwargs['intra_task_metrics']['marg_entropy'].append(marg_entropy)
        kwargs['intra_task_metrics']['diff_with_oracle'].append(diff_with_oracle)
        return outlierness.cpu().numpy().squeeze()
