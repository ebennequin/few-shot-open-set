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
        loss_weights: List[float] = None,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.softmax_temperature = softmax_temperature
        self.lambda_ = lambda_
        self.params2adapt = params2adapt
        self.init = init

    def get_logits_from_cosine_distances_to_prototypes(self, samples):
        return (
            self.softmax_temperature
            * F.normalize(samples - self.mu, dim=1)
            @ F.normalize(self.prototypes - self.mu, dim=1).T
        )

    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        if kwargs['use_transductively'] is not None:
            unlabelled_data = query_features[kwargs['use_transductively']]
        else:
            unlabelled_data = query_features

        # Metric dic
        num_classes = support_labels.unique().size(0)
        support_labels_one_hot = F.one_hot(support_labels, num_classes)

        # Initialize weights
        if self.init == 'base':
            self.mu = kwargs['train_mean'].squeeze()
        elif self.init == 'rand':
            self.mu = 0.1 * torch.randn(1, support_features.size(-1))
        elif self.init == 'mean':
            self.mu = torch.cat([support_features, unlabelled_data], 0).mean(0, keepdim=True)

        self.prototypes = compute_prototypes(support_features - self.mu, support_labels)

        params_list = []
        if 'mu' in self.params2adapt:
            self.mu.requires_grad_()
            params_list.append(self.mu)
        if 'proto' in self.params2adapt:
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
        outlier_entropy = []
        acc_values = []

        for i in range(self.inference_steps):

            logits_s = self.get_logits_from_cosine_distances_to_prototypes(
                support_features
            )
            logits_q = self.get_logits_from_cosine_distances_to_prototypes(
                unlabelled_data
            )

            # otsu_thresh = threshold_otsu(q_cond_ent.detach().numpy())
            # pseudo_outliers = (q_cond_ent > otsu_thresh).float()

            ce = -(support_labels_one_hot * logits_s.log_softmax(1)).sum(1).mean(0)
            q_probs = logits_q.softmax(1)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(1) - math.log(num_classes) / 2
            marginal_y = q_probs.mean(0)
            q_ent = -(marginal_y * torch.log(marginal_y)).sum(0)

            outlier_scores = q_cond_ent
            # outlier_scores = (2 * self.lambda_ * q_cond_ent).sigmoid().detach()
            # logger.warning(outlier_scores.mean())
            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - \
                # self.loss_weights[2] * (((1 - outlier_scores) / (1 - outlier_scores).sum()) * q_cond_ent).sum(0)
                self.loss_weights[2] * q_cond_ent.mean(0)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                q_probs = self.get_logits_from_cosine_distances_to_prototypes(unlabelled_data)
                q_cond_ent_values.append(q_cond_ent.mean(0).item())
                q_ent_values.append(q_ent.item())
                ce_values.append(ce.item())
                inliers = ~ kwargs['outliers'].bool()
                acc_values.append((q_probs.argmax(-1) == kwargs['query_labels'])[inliers].float().mean().item())
                inlier_entropy.append(q_cond_ent[inliers].mean(0).item())
                outlier_entropy.append(q_cond_ent[~inliers].mean(0).item())
                aucs.append(self.compute_auc(outlier_scores, **kwargs))
                thresh = threshold_otsu(outlier_scores.numpy())
                believed_inliers = (outlier_scores < thresh)
                acc_otsu.append((believed_inliers == inliers).float().mean().item())

        kwargs['intra_task_metrics']['classifier_losses']['cond_ent'].append(q_cond_ent_values)
        kwargs['intra_task_metrics']['classifier_losses']['marg_ent'].append(q_ent_values)
        kwargs['intra_task_metrics']['classifier_losses']['ce'].append(ce_values)
        kwargs['intra_task_metrics']['main_metrics']['acc'].append(acc_values)
        kwargs['intra_task_metrics']['main_metrics']['rocauc'].append(aucs)
        kwargs['intra_task_metrics']['main_metrics']['acc_otsu'].append(acc_otsu)
        kwargs['intra_task_metrics']['secondary_metrics']['inlier_entropy'].append(inlier_entropy)
        kwargs['intra_task_metrics']['secondary_metrics']['outlier_entropy'].append(outlier_entropy)

        with torch.no_grad():
            probas_s = self.get_logits_from_cosine_distances_to_prototypes(support_features).softmax(-1)
            probas_q = self.get_logits_from_cosine_distances_to_prototypes(query_features).softmax(-1)

        return probas_s, probas_q
