"""
Functions used to compute outlier scores from classification predictions.
"""
from collections import defaultdict
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np


def get_pseudo_renyi_entropy(soft_predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute the pseudo-Renyi entropy of the prediction for each query, i.e. the sum of the
        squares of each class' classification score.
    Args:
        soft_predictions: predictions before softmax, shape (n_query*n_way, feature_dimension)
    Returns:
        1-dim tensor of length (n_query*n_way) giving the prediction entropy for each query
    """
    return torch.pow(soft_predictions, 2).sum(dim=1).detach().cpu()


def get_shannon_entropy(soft_predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute the Shannon entropy of the prediction for each query.
    Args:
        soft_predictions: predicted probabilities, shape (n_query*n_way, feature_dimension)
    Returns:
        1-dim tensor of length (n_query*n_way) giving the prediction entropy for each query
    """
    return (soft_predictions * torch.log(soft_predictions)).sum(dim=1).detach().cpu()


def compute_outlier_scores_with_renyi_divergence(
    soft_predictions: torch.Tensor,
    soft_support_predictions: torch.Tensor,
    alpha: int = 2,
    method: str = "topk",
    k: int = 3,
) -> torch.Tensor:
    """
    Compute all Renyi divergences from query instances to support instances, and assign an outlier
    score to each query from its divergence with it's "neighbouring" supports w.r.t. to this
    divergence.
    Args:
        soft_predictions: predictions, shape (n_query*n_way, feature_dimension)
        soft_support_predictions: predictions for support examples, shape (n_shot*n_way, feature_dimension)
        alpha: parameter alpha for Renyi divergence
        method: min or topk. min returns for each query its divergence with the "nearest" support
            example. topk returns for each query with divergence with the k-th "nearest" support
            example.
        k: only used if method=topk. Defines the k.

    Returns:
        1-dim tensor of length (n_query*n_way) containing the outlier score of each query
    """
    pairwise_divergences = (1 / (alpha - 1)) * torch.log(
        torch.pow(soft_predictions, alpha).matmul(
            torch.pow(soft_support_predictions, 1 - alpha).T
        )
    )

    if method == "min":
        return 1 - pairwise_divergences.min(dim=1)[0]
    elif method == "topk":
        return 1 - pairwise_divergences.topk(k, dim=1, largest=False)[0].mean(-1)
    else:
        raise ValueError("Don't know this method.")


def detect_outliers(outlier_detector, data_loader, n_way, n_query):
    accs = []
    metrics = defaultdict(list)
    for support_features, support_labels, query_features, query_labels, _ in tqdm(data_loader):
        outliers = torch.cat([torch.zeros(n_way * n_query), torch.ones(n_way * n_query)])
        outlier_scores, predictions = outlier_detector(
                        support_features, support_labels, query_features, query_labels
                    )
        accs.append((predictions[:n_way * n_query] == query_labels[:n_way * n_query]).float().mean())
        predictions = predictions[:n_way * n_query]
        query_labels = query_labels[:n_way * n_query]

        for metric_name in ['outliers', 'outlier_scores', 'predictions', 'query_labels']:
            metrics[metric_name].append(eval(metric_name))
        # outlier_detection_df_list.append(
        #     pd.DataFrame(
        #         {
        #             "outlier": outliers,
        #             "outlier_score": out_score,
        #             "predictions": predictions,
        #             "labels": query_labels,
        #         }
        #     )
        # )
    for metric_name in metrics:
        metrics[metric_name] = torch.stack(metrics[metric_name], 0)

    return metrics, np.mean(accs)
