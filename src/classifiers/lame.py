from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes
from skimage.filters import threshold_otsu
import math


class LAME(FewShotMethod):
    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        lambda_kernel: float,
        lambda_ent: float,
        init: str,
        params2adapt: str,
        knn: str,
    ):
        super().__init__()
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature
        self.lambda_kernel = lambda_kernel
        self.lambda_ent = lambda_ent
        self.params2adapt = params2adapt
        self.knn = knn
        self.init = init

    def cosine(self, X, Y):
        return F.normalize(X - self.mu, dim=1) @ F.normalize(Y - self.mu, dim=1).T

    def get_logits(self, support_labels, support_features, query_features, bias=True):

        cossim = self.cosine(query_features, support_features)  # [Nq, Ns]
        sorted_cossim = cossim.sort(descending=True, dim=-1).values  # [Nq, Ns]
        # sorted_cossim = cossim[:, sorted_indexes]
        logits = []

        # Class logits
        for class_ in range(support_labels.unique().size(0)):
            masked_cossim = cossim[:, support_labels == class_]
            knn = (support_labels == class_).sum()
            class_cossim = masked_cossim.topk(knn, dim=-1).values  # [Nq, Ns]
            logits.append(class_cossim.mean(-1))  # [Nq]

        # Outlier logit
        logits.append(-sorted_cossim[:, :knn].mean(-1))
        logits = torch.stack(logits, dim=1)
        if bias:
            return self.softmax_temperature * logits - self.biases
        else:
            return self.softmax_temperature * logits

    def laplacian_optimization(self, unary, kernel, max_steps=10):

        # E_list = []
        # oldE = float('inf')
        Z = unary  # [N, K]
        for i in range(max_steps):
            Z = unary ** (1 / self.lambda_ent) * torch.exp(
                1 / self.lambda_ent * self.lambda_kernel * (kernel @ Z)
            )
            Z /= Z.sum(-1, keepdim=True)
        return Z

    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        if kwargs["use_transductively"] is not None:
            unlabelled_data = query_features[kwargs["use_transductively"]]
        else:
            unlabelled_data = query_features

        # Metric dic
        num_classes = support_labels.unique().size(0)

        # Initialize weights
        if self.init == "base":
            self.mu = kwargs["train_mean"].squeeze()
        elif self.init == "rand":
            self.mu = 0.1 * torch.randn(1, support_features.size(-1))
        elif self.init == "mean":
            self.mu = torch.cat([support_features, unlabelled_data], 0).mean(
                0, keepdim=True
            )

        with torch.no_grad():
            self.biases = self.get_logits(
                support_labels, support_features, unlabelled_data, bias=False
            ).mean(dim=0)
            # self.biases = torch.zeros(num_classes + 1)

        params_list = []
        if "mu" in self.params2adapt:
            self.mu.requires_grad_()
            params_list.append(self.mu)
        if "bias" in self.params2adapt:
            self.biases.requires_grad_()
            params_list.append(self.biases)

        # Run adaptation
        optimizer = torch.optim.Adam(params_list, lr=self.inference_lr)

        q_cond_ent_values = []
        acc_otsu = []
        aucs = []
        q_ent_values = []
        ce_values = []
        inlier_entropy = []
        outlier_entropy = []
        inlier_outscore = []
        oulier_outscore = []
        acc_values = []

        support_onehot = torch.cat(
            [F.one_hot(support_labels), torch.zeros(support_labels.size(0), 1)], dim=1
        )

        for i in range(self.inference_steps):

            logits_q = self.get_logits(
                support_labels, support_features, unlabelled_data
            )
            probs_q = logits_q.softmax(-1)

            # === Perform Z-update ===
            with torch.no_grad():
                unary = torch.cat([support_onehot, probs_q], 0)
                all_features = torch.cat([support_features, query_features], 0)
                N = all_features.size(0)
                cossim = self.cosine(all_features, all_features)
                # W = cossim
                knn_index = cossim.topk(self.knn + 1, -1).indices[:, 1:]  # [N, knn]
                W = torch.zeros(N, N)
                W.scatter_(dim=-1, index=knn_index, value=1.0)
                Z = self.laplacian_optimization(unary, W)
                Z = Z[-query_features.size(0) :]

            # === Perform mu and bias updates ===

            loss = -(Z * logits_q.log_softmax(-1)).sum(-1).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                outlier_scores = Z[:, -1]
                q_probs = logits_q[:, :-1].softmax(-1)
                q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(-1)
                q_cond_ent_values.append(q_cond_ent.mean(0).item())
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

        kwargs["intra_task_metrics"]["classifier_losses"]["cond_ent"].append(
            q_cond_ent_values
        )
        kwargs["intra_task_metrics"]["classifier_losses"]["marg_ent"].append(
            q_ent_values
        )
        kwargs["intra_task_metrics"]["classifier_losses"]["ce"].append(ce_values)
        kwargs["intra_task_metrics"]["main_metrics"]["acc"].append(acc_values)
        kwargs["intra_task_metrics"]["main_metrics"]["rocauc"].append(aucs)
        kwargs["intra_task_metrics"]["main_metrics"]["acc_otsu"].append(acc_otsu)
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

        with torch.no_grad():
            probas_s = self.get_logits(
                support_labels, support_features, support_features
            )[:, :-1].softmax(-1)
            probas_q = self.get_logits(
                support_labels, support_features, query_features
            )[:, :-1].softmax(-1)

        return probas_s, probas_q

    def entropy_energy(self, Z, unary, pairwise):
        """
        pairwise: [N, N]
        Z: [N, K]
        """
        E = (
            -(Z * unary).sum()
            - self.lambda_kernel * (pairwise * (Z @ Z.T)).sum()
            + self.lambda_ent * (Z * torch.log(Z.clip(1e-20))).sum()
        )
        return E
