from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from sklearn.metrics import precision_recall_curve
from skimage.filters import threshold_otsu
import math
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes

class OOD_TIM(AllInOne):
    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        init: str,
        params2adapt: str,
        lambda_ce: float,
        lambda_marg: float,
        lambda_em: float,
        use_extra_class: bool,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_em = lambda_em
        self.lambda_marg = lambda_marg
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature
        self.params2adapt = params2adapt
        self.init = init
        self.use_extra_class = use_extra_class

    def cosine(self, X, Y):
        return F.normalize(X - self.mu, dim=1) @ F.normalize(Y - self.mu, dim=1).T

    def get_logits(self, prototypes, query_features, bias=True):

        logits = self.cosine(query_features, prototypes)  # [Nq, Ns]
        if bias:
            return self.softmax_temperature * logits - self.biases
        else:
            return self.softmax_temperature * logits

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:

        self.iter_ = 0

        # Metric dic
        num_classes = support_labels.unique().size(0)

        # Initialize weights
        if self.init == "base":
            self.mu = kwargs["train_mean"].squeeze()
        elif self.init == "rand":
            self.mu = 0.1 * torch.randn(1, support_features.size(-1))
        elif self.init == "mean":
            self.mu = torch.cat([support_features, query_features], 0).mean(
                0, keepdim=True
            )

        with torch.no_grad():
            # self.biases = self.get_logits(support_labels, support_features, query_features, bias=False).mean(dim=0)
            self.prototypes = compute_prototypes(support_features, support_labels)
            if self.use_extra_class:
                mu_ood = torch.cat([support_features, query_features], 0).mean(0, keepdim=True)
                # mu_ood = query_features[find_farthest_point(support_features, query_features)].unsqueeze(0)
                self.prototypes = torch.cat([self.prototypes, mu_ood], 0)
                self.biases = torch.zeros(num_classes + 1)
            else:
                self.biases = torch.zeros(num_classes)

        params_list = []
        if "mu" in self.params2adapt:
            self.mu.requires_grad_()
            params_list.append(self.mu)
        if "bias" in self.params2adapt:
            self.biases.requires_grad_()
            params_list.append(self.biases)
        if "prototypes" in self.params2adapt:
            self.prototypes.requires_grad_()
            params_list.append(self.prototypes)

        # Run adaptation
        optimizer = torch.optim.Adam(params_list, lr=self.inference_lr)

        q_cond_ent_values = []
        acc_otsu = []
        aucs = []
        q_ent_values = []
        ce_values = []
        inlier_entropy = []
        precs = []
        recalls = []
        outlier_entropy = []
        inlier_outscore = []
        oulier_outscore = []
        acc_values = []

        for self.iter_ in range(self.inference_steps):
            # proto_labels = torch.arange(num_classes)
            logits_s = self.get_logits(
                self.prototypes, support_features
            )
            logits_q = self.get_logits(
                self.prototypes, query_features
            )

            ce = F.cross_entropy(logits_s, support_labels)
            q_probs = logits_q.softmax(-1)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(-1)

            loss = self.lambda_ce * ce

            em = q_cond_ent.mean(0)
            marginal_y = logits_q.softmax(-1).mean(0)
            div = (marginal_y * torch.log(marginal_y)).sum(0)
            loss += self.lambda_em * em + self.lambda_marg * div

            # logger.warning(marginal_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if self.use_extra_class:
                    outlier_scores = q_probs[:, -1]
                    q_probs = logits_q[:, :-1].softmax(-1)
                    q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(-1)
                else:
                    outlier_scores = q_cond_ent

                q_cond_ent_values.append(q_cond_ent.mean(0).item())
                q_ent_values.append(div.item())
                ce_values.append(ce.item())
                inliers = ~kwargs["outliers"].bool()
                acc_values.append(
                    (q_probs.argmax(-1) == kwargs["query_labels"])[inliers]
                    .float()
                    .mean()
                    .item()
                )
                inlier_entropy.append(q_cond_ent[inliers].mean(0).item())
                outlier_entropy.append(q_cond_ent[~inliers].mean(0).item())
                inlier_outscore.append(outlier_scores[inliers].mean(0).item())
                oulier_outscore.append(outlier_scores[~inliers].mean(0).item())
                aucs.append(self.compute_auc(outlier_scores, **kwargs))
                thresh = threshold_otsu(outlier_scores.numpy())
                believed_inliers = outlier_scores < thresh
                acc_otsu.append((believed_inliers == inliers).float().mean().item())
                precision, recall, thresholds = precision_recall_curve(
                    (~inliers).numpy(), outlier_scores.numpy()
                )
                precs.append(precision[recall > 0.9][-1])
                recalls.append(recall[precision > 0.9][0])

        kwargs['intra_task_metrics']['classifier_losses']['cond_ent'].append(q_cond_ent_values)
        kwargs['intra_task_metrics']['classifier_losses']['marg_ent'].append(q_ent_values)
        kwargs['intra_task_metrics']['classifier_losses']['ce'].append(ce_values)
        kwargs['intra_task_metrics']['main_metrics']['acc'].append(acc_values)
        kwargs['intra_task_metrics']['main_metrics']['rocauc'].append(aucs)
        kwargs['intra_task_metrics']['main_metrics']['acc_otsu'].append(acc_otsu)
        kwargs['intra_task_metrics']['main_metrics']['prec_at_90'].append(precs)
        kwargs['intra_task_metrics']['main_metrics']['rec_at_90'].append(recalls)
        kwargs['intra_task_metrics']['secondary_metrics']['inlier_entropy'].append(inlier_entropy)
        kwargs['intra_task_metrics']['secondary_metrics']['outlier_entropy'].append(outlier_entropy)
        kwargs['intra_task_metrics']['secondary_metrics']['inlier_outscore'].append(inlier_outscore)
        kwargs['intra_task_metrics']['secondary_metrics']['oulier_outscore'].append(oulier_outscore)
        if self.use_extra_class:
            return logits_s[:, :-1].softmax(-1).detach(), logits_q[:, :-1].softmax(-1).detach(), outlier_scores.detach()
        else:
            return logits_s.softmax(-1).detach(), logits_q.softmax(-1).detach(), outlier_scores.detach()


def find_farthest_point(support_features, query_features):
    distances = torch.cdist(query_features, support_features).mean(-1)
    return distances.argmax()