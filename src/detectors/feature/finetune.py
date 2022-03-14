import torch
from .abstract_detector import FeatureDetector
from easyfsl.utils import compute_prototypes
from src.constants import MISC_MODULES
from loguru import logger
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
from .abstract import FeatureDetector
from copy import deepcopy


class FinetuneDetector(FeatureDetector):

    def __init__(self, lambda_: float, lr: float, n_iter: int, init: str, n_neighbors: int, optimizer_name: str):
        super().__init__()
        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter
        self.init = init
        self.n_neighbors = n_neighbors
        self.name = 'FinetuneDetector'
        self.optimizer_name = optimizer_name

    def fit(self, support_features, **kwargs):
        """
        feat: Tensor shape [N, hidden_dim, *]
        """
        self.raw_feat_s = support_features

    def compute_auc(self, outlierness, **kwargs):
        fp_rate, tp_rate, thresholds = roc_curve(kwargs['outliers'].numpy(), outlierness.cpu().numpy())
        return auc_fn(fp_rate, tp_rate)

    def compute_probas(self, feat_s, X, W):
        assert W.size(0) == X.size(0) and W.size(1) == feat_s.size(0)
        similarities = ((X @ feat_s.t()) * W).sum(-1, keepdim=True) / W.sum(-1, keepdim=True)  # [N, 1]
        outlierness = (-self.lambda_ * similarities).sigmoid()  # [N, 1]
        return torch.cat([outlierness, 1 - outlierness], dim=1)

    def decision_function(self, raw_feat_q, **kwargs):

        loss_values = []
        aucs = []
        marg_entropy = []
        diff_with_oracle = []
        inlier_entropy = []
        outlier_entropy = []
        entropies = []
        ces = []
        kls = []
        raw_feat_s = self.raw_feat_s.cuda()
        raw_feat_q = raw_feat_q.cuda()

        if self.init == 'base':
            mu = kwargs['train_mean'].squeeze().cuda()
        elif self.init == 'zero':
            mu = torch.zeros(1, raw_feat_s.size(-1)).cuda()
        elif self.init == 'mean':
            mu = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
        mu.requires_grad_()
        optimizer = eval(f'torch.optim.{self.optimizer_name}([mu], lr=self.lr)')

        for i in range(self.n_iter):

            # 1 --- Find potential outliers

            feat_s = F.normalize(raw_feat_s - mu, dim=1)
            feat_q = F.normalize(raw_feat_q - mu, dim=1)

            # Compute nearest-neighbor
            with torch.no_grad():
                dist = torch.cdist(feat_q, feat_s)  # [Nq, Ns]
                n_neighbors = min(self.n_neighbors, feat_s.size(0))
                knn_index = dist.topk(n_neighbors, dim=-1, largest=False).indices  # [N, knn]

                W_q = torch.zeros(feat_q.size(0), feat_s.size(0)).cuda()
                W_q.scatter_(dim=-1, index=knn_index, value=1.0)  # [Nq, Ns]

                dist = torch.cdist(feat_s, feat_s)  # [Nq, Ns]
                n_neighbors = min(self.n_neighbors + 1, feat_s.size(0))
                knn_index = dist.topk(n_neighbors, dim=-1, largest=False).indices[:, 1:]  # [N, knn]

                W_s = torch.zeros(feat_s.size(0), feat_s.size(0)).cuda()
                W_s.scatter_(dim=-1, index=knn_index, value=1.0)  # [Nq, Ns]

            # 2 --- Update mu
            probas_q = self.compute_probas(feat_s, feat_q, W_q)
            probas_s = self.compute_probas(feat_s, feat_s, W_s)
            ce = - torch.log(probas_s[:, 1])
            entropy = -(probas_q * torch.log(probas_q)).sum(-1)
            # logger.warning(kl)

            loss = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step()

            with torch.no_grad():
                loss_values.append(loss.item())
                marg_entropy.append(- (probas_q.mean(0) * torch.log(probas_q.mean(0))).sum().item())
                diff_with_oracle.append(abs(probas_q.mean(0)[0].item() - (kwargs['outliers'].sum() / kwargs['outliers'].size(0)).item()))
                outlier_entropy.append(entropy[kwargs['outliers'].bool()].mean().item())
                entropies.append(entropy.mean().item())
                inlier_entropy.append(entropy[~ kwargs['outliers'].bool()].mean().item())
                ces.append(ce.mean().item())
                aucs.append(self.compute_auc(probas_q[:, 0], **kwargs))
        kwargs['intra_task_metrics']['main_losses']['ce'].append(ces)
        kwargs['intra_task_metrics']['main_losses']['kl'].append(kls)
        kwargs['intra_task_metrics']['main_losses']['entropy'].append(entropies)
        kwargs['intra_task_metrics']['secondary_loss']['inlier_entropy'].append(inlier_entropy)
        kwargs['intra_task_metrics']['secondary_loss']['outlier_entropy'].append(outlier_entropy)
        kwargs['intra_task_metrics']['main_metrics']['auc'].append(aucs)
        kwargs['intra_task_metrics']['secondary_metrics']['marg_diff_oracle'].append(diff_with_oracle)
        return probas_q[:, 0].detach().cpu().numpy().squeeze()
