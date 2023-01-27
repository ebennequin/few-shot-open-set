import torch
from easyfsl.utils import compute_prototypes
from loguru import logger
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
from .abstract import FeatureDetector
from copy import deepcopy
from skimage.filters import threshold_otsu


class RepriDetector(FeatureDetector):
    def __init__(
        self,
        lambda_: float,
        lr: float,
        n_iter: int,
        init: str,
        n_neighbors: int,
        optimizer_name: str,
        weight=1.0,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter
        self.init = init
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.name = "AlternateDetector"
        self.optimizer_name = optimizer_name

    def compute_auc(self, outlierness, **kwargs):
        fp_rate, tp_rate, thresholds = roc_curve(
            kwargs["outliers"].numpy(), outlierness.cpu().numpy()
        )
        return auc_fn(fp_rate, tp_rate)

    def compute_probas(self, feat_s, X, W):
        assert W.size(0) == X.size(0) and W.size(1) == feat_s.size(0)
        similarities = ((X @ feat_s.t()) * W).sum(-1, keepdim=True) / W.sum(
            -1, keepdim=True
        )  # [N, 1]
        outlierness = (-self.lambda_ * similarities).sigmoid()  # [N, 1]
        return -self.lambda_ * similarities, torch.cat(
            [outlierness, 1 - outlierness], dim=1
        )

    def do_iter(self, raw_feat_s, raw_feat_q, mu):
        feat_s = F.normalize(raw_feat_s - mu, dim=1)
        feat_q = F.normalize(raw_feat_q - mu, dim=1)

        # Compute nearest-neighbor
        with torch.no_grad():
            dist = torch.cdist(feat_q, feat_s)  # [Nq, Ns]
            n_neighbors = min(self.n_neighbors, feat_s.size(0))
            knn_index = dist.topk(
                n_neighbors, dim=-1, largest=False
            ).indices  # [N, knn]

            W_q = torch.zeros(feat_q.size(0), feat_s.size(0)).cuda()
            W_q.scatter_(dim=-1, index=knn_index, value=1.0)  # [Nq, Ns]

            dist = torch.cdist(feat_s, feat_s)  # [Nq, Ns]
            n_neighbors = min(self.n_neighbors + 1, feat_s.size(0))
            knn_index = dist.topk(n_neighbors, dim=-1, largest=False).indices[
                :, 1:
            ]  # [N, knn]

            W_s = torch.zeros(feat_s.size(0), feat_s.size(0)).cuda()
            W_s.scatter_(dim=-1, index=knn_index, value=1.0)  # [Nq, Ns]

        # 2 --- Update mu
        logits_q, probas_q = self.compute_probas(feat_s, feat_q, W_q)
        logits_s, probas_s = self.compute_probas(feat_s, feat_s, W_s)

        return logits_s, probas_s, logits_q, probas_q

    def standardize(self, scores):
        return scores

    def __call__(self, support_features, query_features, **kwargs):
        loss_values = []
        aucs = []
        marg_entropy = []
        diff_with_oracle = []
        inlier_inlierness = []
        outlier_inlierness = []
        entropies = []
        pi_diff_with_oracle = []
        ces = []
        kls = []
        raw_feat_s = support_features.cuda()
        raw_feat_q = query_features.cuda()

        if self.init == "base":
            mu = kwargs["train_mean"].squeeze().cuda()
        elif self.init == "zero":
            mu = torch.zeros(1, raw_feat_s.size(-1)).cuda()
        elif self.init == "mean":
            mu = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
        mu.requires_grad_()
        optimizer = eval(f"torch.optim.{self.optimizer_name}([mu], lr=self.lr)")

        for i in range(self.n_iter):
            if i == 0:
                with torch.no_grad():
                    # mu_init = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
                    mu_init = mu
                    _, _, logits_q, _ = self.do_iter(raw_feat_s, raw_feat_q, mu_init)
                    thresh = threshold_otsu(logits_q.detach().cpu().numpy())
                    pi = (logits_q > thresh).sum() / logits_q.size(0)
                    prior_prob = torch.Tensor([pi, 1 - pi]).cuda()
                    # logger.warning(f"Threshold found {thresh:.2f}. Prior found {pi:.2f}")

            _, probas_s, logits_q, probas_q = self.do_iter(raw_feat_s, raw_feat_q, mu)
            ce = -torch.log(probas_s[:, 1])
            entropy = -(probas_q * torch.log(probas_q)).sum(-1)
            # kl = kl_div(probas_q.mean(0) - prior_prob)
            kl = probas_q.mean(0)[0] - prior_prob[0]

            loss = ce.mean() + entropy.mean() + self.weight * kl
            # loss = kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step()

            with torch.no_grad():
                loss_values.append(loss.item())
                kls.append(kl.item())
                marg_entropy.append(
                    -(probas_q.mean(0) * torch.log(probas_q.mean(0))).sum().item()
                )
                diff_with_oracle.append(
                    abs(
                        probas_q.mean(0)[0].item()
                        - (kwargs["outliers"].sum() / kwargs["outliers"].size(0)).item()
                    )
                )
                pi_diff_with_oracle.append(
                    abs(
                        pi.item()
                        - (kwargs["outliers"].sum() / kwargs["outliers"].size(0)).item()
                    )
                )
                outlier_inlierness.append(
                    probas_q[kwargs["outliers"].bool()][:, 1].mean().item()
                )
                entropies.append(entropy.mean().item())
                inlier_inlierness.append(
                    probas_q[~kwargs["outliers"].bool()][:, 1].mean().item()
                )
                ces.append(ce.mean().item())
                aucs.append(self.compute_auc(logits_q, **kwargs))
        kwargs["intra_task_metrics"]["main_losses"]["ce"].append(ces)
        kwargs["intra_task_metrics"]["main_losses"]["kl"].append(kls)
        kwargs["intra_task_metrics"]["main_losses"]["entropy"].append(entropies)
        kwargs["intra_task_metrics"]["secondary_loss"]["inlier_inlierness"].append(
            inlier_inlierness
        )
        kwargs["intra_task_metrics"]["secondary_loss"]["outlier_inlierness"].append(
            outlier_inlierness
        )
        kwargs["intra_task_metrics"]["main_metrics"]["auc"].append(aucs)
        # kwargs['intra_task_metrics']['secondary_metrics']['marg_entropy'].append(marg_entropy)
        kwargs["intra_task_metrics"]["secondary_metrics"]["marg_diff_oracle"].append(
            diff_with_oracle
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["pi_diff_oracle"].append(
            pi_diff_with_oracle
        )
        return logits_q.detach().cpu().squeeze()


def kl_div(pa, pb):
    return (pa * torch.log(pa / pb + 1e-10)).sum(-1)
