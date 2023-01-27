from typing import Tuple, List
import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes


class MAP(FewShotMethod):
    def __init__(self, alpha: float, inference_steps: int, lam: float):
        super().__init__()
        self.alpha = alpha
        self.inference_steps = inference_steps
        self.lam = lam

    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ):
        support_features, query_features = (
            support_features.cuda(),
            query_features.cuda(),
        )
        support_labels, query_labels = (
            support_labels.cuda(),
            kwargs["query_labels"].cuda(),
        )
        inliers = ~kwargs["outliers"].bool().cuda()

        self.prototypes = compute_prototypes(support_features, support_labels)
        num_classes = support_labels.unique().size(0)
        probs_s = F.one_hot(support_labels, num_classes)
        all_features = torch.cat([support_features, query_features], 0)
        acc_values = []
        for epoch in range(self.inference_steps):
            probs_q = self.get_probas(query_features)
            all_probs = torch.cat([probs_s, probs_q], dim=0)

            # update centroids
            self.update_prototypes(all_features, all_probs)

            acc_values.append(
                (self.get_probas(query_features).argmax(-1) == query_labels)[inliers]
                .float()
                .mean()
                .item()
            )

        kwargs["intra_task_metrics"]["classifier_metrics"]["acc"].append(acc_values)

        # get final accuracy and return it
        return (
            self.get_probas(support_features).cpu(),
            self.get_probas(query_features).cpu(),
        )

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        """
        M: [N, K]
        """

        r = r.cuda()
        c = c.cuda()
        n, m = M.shape
        P = torch.exp(-self.lam * M)
        P /= P.sum(dim=(0, 1), keepdim=True)

        u = torch.zeros(n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(1))) > epsilon:
            u = P.sum(1)
            P *= (r / u + 1e-10).view(-1, 1)
            P *= (c / P.sum(0) + 1e-10).view(1, -1)
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def get_probas(self, query_features):
        """
        query_features: [Nq, d]
        """

        dist = torch.cdist(query_features, self.prototypes) ** 2  # [Nq, K]

        n_usamples = query_features.size(0)
        n_ways = dist.size(1)

        r = torch.ones(n_usamples)
        c = torch.ones(n_ways) * (n_usamples // n_ways)

        probas_q, _ = self.compute_optimal_transport(dist, r, c, epsilon=1e-6)
        return probas_q

    def update_prototypes(self, features, probas):
        """
        features: [N, d]
        probas: [N, K]

        mus : [K, d]
        """
        new_prototypes = (probas.t() @ features) / probas.sum(0).unsqueeze(1)
        delta = new_prototypes - self.prototypes
        self.prototypes += self.alpha * delta
