from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes
from skimage.filters import threshold_otsu
import math


class OOD_TIM(FewShotMethod):
    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        lambda_: float,
        init: str,
        params2adapt: str,
        knn: str,
        lambda_ce: float,
        lambda_marg: float,
        lambda_em: float,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_em = lambda_em
        self.lambda_marg = lambda_marg
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature
        self.lambda_ = lambda_
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
            knn = min(self.knn, masked_cossim.size(-1))
            class_cossim = masked_cossim.topk(knn, dim=-1).values  # [Nq, Ns]
            logits.append(class_cossim.mean(-1))  # [Nq]

        # Outlier logit
        logits.append(-sorted_cossim[:, : self.knn].mean(-1))
        logits = torch.stack(logits, dim=1)
        if bias:
            return self.softmax_temperature * logits - self.biases
        else:
            return self.softmax_temperature * logits

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
        support_labels_one_hot = F.one_hot(support_labels, num_classes)

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
        acc_values = []

        for i in range(self.inference_steps):

            logits_s = self.get_logits(
                support_labels, support_features, support_features
            )
            # logger.warning(logits_s)
            logits_q = self.get_logits(
                support_labels, support_features, unlabelled_data
            )

            ce = F.cross_entropy(logits_s, support_labels)
            q_probs = logits_q.softmax(-1)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(-1)

            loss = self.lambda_ce * ce

            em = q_cond_ent.mean(0)
            marginal_y = q_probs.mean(0)
            div = (marginal_y * torch.log(marginal_y)).sum(0)
            loss += self.lambda_em * em + self.lambda_marg * div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                outlier_scores = q_probs[:, -1]
                q_cond_ent_values.append(q_cond_ent.mean(0).item())
                q_ent_values.append(div.item())
                ce_values.append(ce.item())
                inliers = ~kwargs["outliers"].bool()
                acc_values.append(
                    (q_probs[:, :-1].argmax(-1) == kwargs["query_labels"])[inliers]
                    .float()
                    .mean()
                    .item()
                )
                inlier_entropy.append(q_cond_ent[inliers].mean(0).item())
                outlier_entropy.append(q_cond_ent[~inliers].mean(0).item())
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

        with torch.no_grad():
            probas_s = self.get_logits(
                support_labels, support_features, support_features
            )[:, :-1].softmax(-1)
            probas_q = self.get_logits(
                support_labels, support_features, query_features
            )[:, :-1].softmax(-1)

        return probas_s, probas_q
