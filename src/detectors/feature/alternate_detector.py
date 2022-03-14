import torch
from .abstract_detector import FeatureDetector
from easyfsl.utils import compute_prototypes
from src.constants import MISC_MODULES
from loguru import logger
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
from .abstract import FeatureDetector


class AlternateDetector(FeatureDetector):

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
        inlier_entropy = []
        outlier_entropy = []
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
                probas = torch.cat([outlierness, 1 - outlierness], dim=1)
                marg_entropy.append(- (probas.mean(0) * torch.log(probas.mean(0))).sum().item())
                entropy = -(probas * torch.log(probas)).sum(-1)
                diff_with_oracle.append(abs(probas.mean(0)[0].item() - (kwargs['outliers'].sum() / kwargs['outliers'].size(0)).item()))
                outlier_entropy.append(entropy[kwargs['outliers'].bool()].mean().item())
                inlier_entropy.append(entropy[~ kwargs['outliers'].bool()].mean().item())
                aucs.append(self.compute_auc(outlierness, **kwargs))
        kwargs['intra_task_metrics']['main_loss']['main'].append(loss_values)
        kwargs['intra_task_metrics']['secondary_loss']['inlier_entropy'].append(inlier_entropy)
        kwargs['intra_task_metrics']['secondary_loss']['outlier_entropy'].append(outlier_entropy)
        kwargs['intra_task_metrics']['main_metrics']['auc'].append(aucs)
        kwargs['intra_task_metrics']['secondary_metrics']['marg_entropy'].append(marg_entropy)
        kwargs['intra_task_metrics']['secondary_metrics']['marg_diff_oracle'].append(diff_with_oracle)
        return outlierness.cpu().numpy().squeeze()
