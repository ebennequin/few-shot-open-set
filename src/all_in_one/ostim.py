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
from copy import deepcopy


class OSTIM(AllInOne):
    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        mu_init: str,
        params2adapt: str,
        lambda_ce: float,
        lambda_marg: float,
        lambda_em: float,
        use_explicit_prototype: bool,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_em = lambda_em
        self.lambda_marg = lambda_marg
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature
        self.params2adapt = params2adapt
        self.mu_init = mu_init
        self.use_explicit_prototype = use_explicit_prototype  # use for ablation, to compare with PROSER

    def cosine(self, X, Y):
        return F.normalize(X - self.mu, dim=1) @ F.normalize(Y - self.mu, dim=1).T

    def clear(self):
        delattr(self, 'prototypes')
        delattr(self, 'mu')

    def get_logits(self, prototypes, query_features):

        logits = self.cosine(query_features, prototypes)  # [Nq, K]
        if not self.use_explicit_prototype:
            logits = torch.cat(
                [logits, -logits.mean(-1, keepdim=True)], dim=1
            )  # [Nq, Ns]

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
        if self.mu_init == "base":
            self.mu = deepcopy(kwargs["train_mean"].squeeze())
        elif self.mu_init == "zeros":
            self.mu = torch.zeros(support_features.size(-1))
        elif self.mu_init == "rand":
            self.mu = 0.1 * torch.randn(1, support_features.size(-1))
        elif self.mu_init == "mean":
            self.mu = torch.cat([support_features, query_features], 0).mean(
                0, keepdim=True
            )
        else:
            raise ValueError(f"Mu init {self.mu_init} not recognized.")

        with torch.no_grad():
            self.prototypes = compute_prototypes(support_features, support_labels)
            if self.use_explicit_prototype:
                self.prototypes = torch.cat([self.prototypes, torch.zeros(1, support_features.size(1))], 0)

        params_list = []
        if "mu" in self.params2adapt:
            self.mu.requires_grad_()
            params_list.append(self.mu)
        if "prototypes" in self.params2adapt:
            self.prototypes.requires_grad_()
            params_list.append(self.prototypes)

        # Run adaptation
        optimizer = torch.optim.Adam(params_list, lr=self.inference_lr)

        q_cond_ent_values = []
        rocaucs = []
        q_ent_values = []
        ce_values = []
        inlier_entropy = []
        outlier_entropy = []
        inlier_outscore = []
        oulier_outscore = []
        acc_values = []

        for self.iter_ in range(self.inference_steps):

            # Compute loss

            logits_s = self.get_logits(self.prototypes, support_features)
            logits_q = self.get_logits(self.prototypes, query_features)

            ce = F.cross_entropy(logits_s, support_labels)
            q_probs = logits_q.softmax(-1)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(-1)

            loss = self.lambda_ce * ce
            em = q_cond_ent.mean(0)
            marginal_y = logits_q.softmax(-1).mean(0)
            div = (marginal_y * torch.log(marginal_y)).sum(0)
            loss += self.lambda_em * em + self.lambda_marg * div

            # Note : we track metrics before the optimization is performed

            with torch.no_grad():
                outlier_scores = q_probs[:, -1]
                closed_q_probs = logits_q[:, :-1].softmax(-1)
                q_cond_ent = -(closed_q_probs * torch.log(closed_q_probs + 1e-12)).sum(-1)

                q_cond_ent_values.append(q_cond_ent.mean(0).item())
                q_ent_values.append(div.item())
                ce_values.append(ce.item())
                inliers = ~kwargs["outliers"].bool()
                acc = (closed_q_probs.argmax(-1) == kwargs["query_labels"])[inliers].float().mean().item()
                acc_values.append(acc)
                inlier_entropy.append(q_cond_ent[inliers].mean(0).item())
                outlier_entropy.append(q_cond_ent[~inliers].mean(0).item())
                inlier_outscore.append(outlier_scores[inliers].mean(0).item())
                oulier_outscore.append(outlier_scores[~inliers].mean(0).item())
                rocaucs.append(self.compute_auc(outlier_scores, **kwargs))
                precision, recall, thresholds = precision_recall_curve(
                    (~inliers).numpy(), outlier_scores.numpy()
                )

            # Optimize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        kwargs["intra_task_metrics"]["classifier_losses"]["cond_ent"].append(
            q_cond_ent_values
        )
        kwargs["intra_task_metrics"]["classifier_losses"]["marg_ent"].append(
            q_ent_values
        )
        kwargs["intra_task_metrics"]["classifier_losses"]["ce"].append(ce_values)
        kwargs["intra_task_metrics"]["main_metrics"]["acc"].append(acc_values)
        kwargs["intra_task_metrics"]["main_metrics"]["rocauc"].append(rocaucs)
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_entropy"].append(
            inlier_entropy
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["outlier_entropy"].append(
            outlier_entropy
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_outscore"].append(
            inlier_outscore
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["oulier_outscore"].append(
            oulier_outscore
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["oulier_outscore"].append(
            oulier_outscore
        )

        with torch.no_grad():
            logits_s = self.get_logits(self.prototypes, support_features)
            logits_q = self.get_logits(self.prototypes, query_features)
            outlier_scores = logits_q.softmax(-1)[:, -1]
        
        # Ensure that nothing persists after
        self.clear()
        return (
            logits_s[:, :-1].softmax(-1).detach(),
            logits_q[:, :-1].softmax(-1).detach(),
            outlier_scores.detach(),
        )