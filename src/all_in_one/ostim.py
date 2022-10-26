from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import precision_recall_curve
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from copy import deepcopy
from sklearn.metrics import auc as auc_fn


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
        self.use_explicit_prototype = (
            use_explicit_prototype  # use for ablation, to compare with PROSER
        )

    def normalize_before_cosine(self, x):
        return F.normalize((x - self.mu) / (self.std + 1e-10), dim=1)

    def cosine(self, X, Y):
        return self.normalize_before_cosine(X) @ self.normalize_before_cosine(Y).T

    def clear(self):
        delattr(self, "prototypes")
        delattr(self, "mu")

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
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        self.iter_ = 0

        # Metric dic
        num_classes = support_labels.unique().size(0)

        # Initialize weights
        # Note that here we perform feature transformation (eg. mean centering) inside the OSTIM method
        # This is not the case for other methods we compare to
        # This is because it was easier to implement the ablation study on the feature transformation this way
        # You can still reproduce comparable results with the transformation outside of the method
        # To do so, set mu_init to zeros in configs/detectors.yaml and make the recipe run_ostim
        # with DET_TRANSFORMS="Pool MeanCentering L2norm"
        # We found that the best set of hyper-parameters was different in this setting. Here are the hyper-parameters
        # that we optimized on miniImageNet's val set for this setting:
        # 1-shot: 'inference_steps': 100,
        #           'inference_lr': 0.0001,
        #           'lambda_em': 0.5,
        # 5-shot: 'inference_steps': 50,
        #           'inference_lr': 0.0001,
        #           'lambda_em': 0.1,
        self.std = torch.ones(support_features.size(-1))
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
        elif self.mu_init == "batch":
            self.mu = torch.cat([support_features, query_features], 0).mean(
                0, keepdim=True
            )
            self.std = torch.cat([support_features, query_features], 0).std(
                dim=0, unbiased=False, keepdim=True
            )
        else:
            raise ValueError(f"Mu init {self.mu_init} not recognized.")

        with torch.no_grad():
            self.prototypes = compute_prototypes(support_features, support_labels)
            if self.use_explicit_prototype:
                self.prototypes = torch.cat(
                    [self.prototypes, torch.zeros(1, support_features.size(1))], 0
                )

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
        auprs = []
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
                q_cond_ent = -(closed_q_probs * torch.log(closed_q_probs + 1e-12)).sum(
                    -1
                )

                q_cond_ent_values.append(q_cond_ent.mean(0).item())
                q_ent_values.append(div.item())
                ce_values.append(ce.item())
                outliers = kwargs["outliers"].bool()
                inliers = ~outliers
                acc = (
                    (closed_q_probs.argmax(-1) == kwargs["query_labels"])[inliers]
                    .float()
                    .mean()
                    .item()
                )
                acc_values.append(acc)
                inlier_entropy.append(q_cond_ent[inliers].mean(0).item())
                outlier_entropy.append(q_cond_ent[~inliers].mean(0).item())
                inlier_outscore.append(outlier_scores[inliers].mean(0).item())
                oulier_outscore.append(outlier_scores[~inliers].mean(0).item())
                precision, recall, thresholds = precision_recall_curve(
                    outliers.numpy(), outlier_scores.numpy()
                )
                aupr = auc_fn(recall, precision)
                auprs.append(aupr)
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
        kwargs["intra_task_metrics"]["main_metrics"]["aupr"].append(auprs)
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
