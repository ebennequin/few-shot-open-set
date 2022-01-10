"""
Utils for metric computation and plots.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import argparse

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


def plot_roc(outliers_df: pd.DataFrame, title: str) -> float:
    """
    Plot the ROC curve from outlier prediction scores and ground truth, and returns
    Args:
        outliers_df: contains a column "outlier" of booleans, and a column "outlier_score" of floats
        title: title of the plot
    Returns:
        the area under the ROC curve.
    """
    fp_rate, tp_rate, _ = roc_curve(outliers_df.outlier, -outliers_df.outlier_score)

    # plt.plot(fp_rate, tp_rate)
    # plt.xlim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    # plt.title(title)
    # plt.show()

    return auc(fp_rate, tp_rate)


def plot_twin_hist(outliers_df: pd.DataFrame, title: str):
    """
    Plot a bi-color histogram showing the predicted outlier score for ground truth outliers and
    ground truth inliers.
    Args:
        outliers_df: contains a column "outlier" of booleans, and a column "outlier_score" of floats
        title: title of the plot
    """
    sns.histplot(data=outliers_df, x="outlier_score", hue="outlier")
    plt.title(title)
    plt.show()


def show_all_metrics_and_plots(outliers_df: pd.DataFrame, title: str, objective=0.9):
    """
    Print all metrics and plot all plots for a given set of outlier predictions.
    Args:
        outliers_df: contains a column "outlier" of booleans, and a column "outlier_score" of floats
        title: title of the plots
        objective: two of the metrics are the maximum precision (resp. recall) possible for a fixed
            recall (resp. precision) threshold
    """
    roc_auc = plot_roc(outliers_df, title=title)
    acc = (outliers_df.predictions == outliers_df.labels)[~outliers_df.outlier].mean()
    print(f"ROC AUC: {roc_auc}")
    print(f"Accuracy: {acc}")

    precisions, recalls, _ = precision_recall_curve(
        outliers_df.outlier, -outliers_df.outlier_score
    )
    precision_at_recall_objective = precisions[
        next(i for i, value in enumerate(recalls) if value < objective)
    ]
    recall_at_precision_objective = recalls[
        next(i for i, value in enumerate(precisions) if value > objective)
    ]
    print(f"Precision for recall={objective}: {precision_at_recall_objective}")
    print(f"Recall for precision={objective}: {recall_at_precision_objective}")

    # plot_twin_hist(outliers_df, title=title)

    return roc_auc, precision_at_recall_objective, recall_at_precision_objective


def update_csv(args: argparse.Namespace,
               metrics: dict,
               path: str):

    # The metrics we need to fill in the row
    fill_entry = metrics
    # fill_entry['completed'] = True
    try:
        res = pd.read_csv(path)
    except FileNotFoundError:
        res = pd.DataFrame({})

    group_by_args = args.general_hparams + args.simu_hparams
    records = res.to_dict('records')

    # Check whether the row exists already, if yes, simply update the metrics
    match = False
    for entry in records:
        if any([param not in entry for param in group_by_args]):
            continue
        matches = [str(entry[param]) == str(getattr(args, param)) for param in group_by_args]
        match = (sum(matches) == len(matches))
        if match:
            if not args.override:
                print("Matching entry found. Not overriding.")
                return
            elif args.override:
                print("Overriding existing results.")
            entry.update(fill_entry)

    # If entry does not exist, just create it
    if not match:
        new_entry = {param: getattr(args, param)
                     for param in group_by_args}
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
