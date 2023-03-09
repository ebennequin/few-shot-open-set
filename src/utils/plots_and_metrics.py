"""
Utils for metric computation and plots.
"""
from statistics import mean
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import argparse
from loguru import logger


def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()


def plot_roc(metrics: dict, title: str) -> float:
    """
    Plot the ROC curve from outlier prediction scores and ground truth, and returns
    Args:
        outliers_df: contains a column "outlier" of booleans, and a column "outlier_score" of floats
        title: title of the plot
    Returns:
        the area under the ROC curve.
    """
    aucs = []
    outliers = metrics["outliers"]
    outlier_scores = metrics["outlier_scores"]
    assert outliers.size() == outlier_scores.size()
    for i in range(len(outliers)):
        gt, scores = outliers[i], outlier_scores[i]
        fp_rate, tp_rate, thresholds = roc_curve(gt, scores)
        aucs.append(auc(fp_rate, tp_rate))

    return np.mean(aucs)


def plot_twin_hist(outliers_df: pd.DataFrame, title: str, plot: bool):
    """
    Plot a bi-color histogram showing the predicted outlier score for ground truth outliers and
    ground truth inliers.
    Args:
        outliers_df: contains a column "outlier" of booleans, and a column "outlier_score" of floats
        title: title of the plot
    """
    if plot:
        fig = plt.figure(figsize=(10, 4))
        sns.histplot(data=outliers_df, x="outlier_score", hue="outlier")
        plt.title(title)
        st.pyplot(fig, clear_figure=True)


def check_if_record_exists(args, path: str):
    # Params part of the new record
    group_by_args = args.general_hparams + args.simu_hparams
    new_entry = {}
    for param in group_by_args:
        value = getattr(args, param)
        if isinstance(value, list):
            value = "-".join(value)
        else:
            value = str(value)
        new_entry[param] = value
    try:
        res = pd.read_csv(path)
    except FileNotFoundError:
        res = pd.DataFrame({})

    records = res.to_dict("records")
    for existing_entry in records:
        if any([param not in existing_entry for param in group_by_args]):
            continue
        matches = [
            str(existing_entry[param]) == new_entry[param] for param in group_by_args
        ]
        match = sum(matches) == len(matches)
        if match:
            return True
    return False


def update_csv(args: argparse.Namespace, metrics: dict, path: str):
    # Load records
    try:
        res = pd.read_csv(path)
    except FileNotFoundError:
        res = pd.DataFrame({})
    records = res.to_dict("records")

    # Metrics part of the new record
    fill_entry = metrics

    # Params part of the new record
    group_by_args = args.general_hparams + args.simu_hparams
    new_entry = {}
    for param in group_by_args:
        value = getattr(args, param)
        if isinstance(value, list):
            value = "-".join(value)
        else:
            value = str(value)
        new_entry[param] = value

    # Check whether the row exists already, if yes, simply update the metrics
    match = False
    for existing_entry in records:
        if any([param not in existing_entry for param in group_by_args]):
            continue
        matches = [
            str(existing_entry[param]) == new_entry[param] for param in group_by_args
        ]
        match = sum(matches) == len(matches)
        if match:
            if not args.override:
                print("Matching entry found. Not overriding.")
                return
            elif args.override:
                print("Overriding existing results.")
            existing_entry.update(fill_entry)
            break

    # If entry does not exist, just create it
    if not match:
        new_entry.update(fill_entry)
        records.append(new_entry)

    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


def confidence_interval(standard_deviation, n_samples):
    """
    Computes statistical 95% confidence interval of the results from standard deviation and number of samples
    Args:
        standard_deviation (float): standard deviation of the results
        n_samples (int): number of samples
    Returns:
        float: confidence interval
    """
    return 1.96 * standard_deviation / np.sqrt(n_samples)


def clustering_variances_ratio(features) -> Tuple[float, float, float]:
    sigma_within = (
        np.mean([np.linalg.norm(v.std(axis=0)) for k, v in features.items()]) ** 2
    )

    sigma_between = (
        np.linalg.norm(
            np.stack([v.mean(axis=0) for v in features.values()]).std(axis=0)
        )
        ** 2
    )

    return sigma_within / sigma_between, sigma_within, sigma_between


def compute_mif_with_auroc(features):
    """
    Computes the MIF of the features using the area under the ROC curve.
    This should give the same results as compute_mif_explicitely() but it is a tiny bit faster.
    """
    aurocs = []
    for label in features.keys():
        ground_truth = []
        predictions = []
        centroid = features[label].mean(axis=0)
        for second_label, v in features.items():
            ground_truth += len(v) * [0 if label == second_label else 1]
            distances = np.linalg.norm(v - centroid, axis=1)
            predictions += distances.tolist()
        aurocs.append(sklearn.metrics.roc_auc_score(ground_truth, predictions))

    return 1 - np.mean(aurocs)


def compute_mif_explicitly(features):
    """
    Computes the MIF of the features using the explicit definition.
    This should give the same results as compute_mif_with_auroc() but it is a tiny bit slower.
    """
    mean_imposture_factors = []
    for label, label_features in features.items():
        centroid = label_features.mean(axis=0)
        distances_to_centroid = np.sort(
            np.linalg.norm(label_features - centroid, axis=1)
        )
        class_imposture_factors = []
        for second_label, v in features.items():
            if second_label == label:
                continue
            distances = np.linalg.norm(v - centroid, axis=1)
            imposture_factors = 1 - np.searchsorted(
                distances_to_centroid, distances
            ) / len(distances_to_centroid)
            class_imposture_factors.append(imposture_factors)
        mean_imposture_factors.append(np.mean(np.concatenate(class_imposture_factors)))

    return np.mean(mean_imposture_factors)
