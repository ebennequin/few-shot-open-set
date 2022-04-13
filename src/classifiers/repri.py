from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes
from skimage.filters import threshold_otsu
from loguru import logger


class RePRI(FewShotMethod):
    def __init__(
        self,
        softmax_temperature: float,
        inference_steps: int,
        inference_lr: float,
        init: str,
        ensemble_size: int,
        loss_weights: List[float] = None,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.init = init
        self.softmax_temperature = softmax_temperature
        self.ensemble_size = ensemble_size

    def get_logits_from_cosine_distances_to_prototypes(self, samples):
        return (
            self.softmax_temperature
            * F.normalize(samples, dim=-1)  # [ens, N, d]
            @ F.normalize(self.prototypes, dim=-1).permute(0, 2, 1)  # [ens, K, d]
        )

    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        support_features, query_features = (
            support_features.cuda(),
            query_features.cuda(),
        )
        inliers = ~kwargs["outliers"].bool().cuda()

        if kwargs["use_transductively"] is not None:
            unlabelled_data = query_features[kwargs["use_transductively"]]
        else:
            unlabelled_data = query_features

        raw_feat_s = support_features.clone().cuda()
        raw_feat_q = unlabelled_data.clone().cuda()

        # Metric dic
        num_classes = support_labels.unique().size(0)
        support_labels_one_hot = F.one_hot(support_labels, num_classes).cuda()
        support_labels, query_labels = (
            support_labels.cuda(),
            kwargs["query_labels"].cuda(),
        )

        # Initialize weights
        if self.init == "base":
            mu = kwargs["train_mean"].squeeze().cuda()
        elif self.init == "rand":
            mu = (
                torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
                + 0.1 * torch.randn(self.ensemble_size, 1, raw_feat_s.size(-1)).cuda()
            )
        elif self.init == "mean":
            mu = torch.cat([raw_feat_s, raw_feat_q], 0).mean(0, keepdim=True)
        mu.requires_grad_()

        centered_feats_s = raw_feat_s - mu

        # Run adaptation
        # self.prototypes.requires_grad_()
        optimizer = torch.optim.SGD([mu], lr=self.inference_lr)

        q_cond_ent_values = []
        q_ent_values = []
        ce_values = []
        inlier_entropy = []
        outlier_entropy = []
        acc_values = []
        aucs = []
        acc_otsu = []

        for i in range(self.inference_steps):

            # Center data
            centered_feats_s = raw_feat_s.unsqueeze(0) - mu  # [ens, N,  d]
            centered_feats_q = raw_feat_q.unsqueeze(0) - mu  # [ens, N, d]

            # Recompute prototypes
            prots = []
            for j in range(self.ensemble_size):
                prots.append(compute_prototypes(centered_feats_s[j], support_labels))
            self.prototypes = torch.stack(prots, 0)  # [ens, K, d]

            # Compute loss
            logits_s = self.get_logits_from_cosine_distances_to_prototypes(
                centered_feats_s
            )  # [ens, N, K]
            logits_q = self.get_logits_from_cosine_distances_to_prototypes(
                centered_feats_q
            )  # [ens, N, K]

            ce = -(support_labels_one_hot * logits_s.log_softmax(-1)).sum(-1).mean(1)
            q_probs = logits_q.softmax(-1)  # [ens, N, K]
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(-1)  # [ens, N]
            marginal_y = q_probs.mean(1)  # [ens, K]
            if i == 0:
                pi = marginal_y.detach().clone()
            div = (pi - marginal_y).abs().sum(-1)  # [ens]

            loss = (
                self.loss_weights[0] * ce
                + self.loss_weights[1] * div
                + self.loss_weights[2] * q_cond_ent.mean(-1)
            ).sum(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                q_cond_ent_values.append(q_cond_ent.mean().item())
                q_ent_values.append(div.mean(0).item())
                ce_values.append(ce.mean(0).item())
                acc_values.append(
                    (q_probs.mean(0).argmax(-1) == query_labels)[inliers]
                    .float()
                    .mean()
                    .item()
                )
                inlier_entropy.append(q_cond_ent[:, inliers].mean().item())
                outlier_entropy.append(q_cond_ent[:, ~inliers].mean().item())
                aucs.append(self.compute_auc(q_cond_ent.mean(0), **kwargs))
                thresh = threshold_otsu(q_cond_ent.mean(0).cpu().numpy())
                believed_inliers = q_cond_ent.mean(0) < thresh
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
            probas_s = (
                self.get_logits_from_cosine_distances_to_prototypes(
                    support_features - mu
                )
                .softmax(-1)
                .mean(0)
                .cpu()
            )
            probas_q = (
                self.get_logits_from_cosine_distances_to_prototypes(query_features - mu)
                .softmax(-1)
                .mean(0)
                .cpu()
            )

        return probas_s, probas_q
