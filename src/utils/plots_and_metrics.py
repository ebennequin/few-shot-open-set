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
    outliers = metrics['outliers']
    outlier_scores = metrics['outlier_scores']
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


def show_all_metrics_and_plots(args, metrics: dict, title: str, objective=0.9):
    """
    Print all metrics and plot all plots for a given set of outlier predictions.
    Args:
        outliers_df: contains a column "outlier" of booleans, and a column "outlier_score" of floats
        title: title of the plots
        objective: two of the metrics are the maximum precision (resp. recall) possible for a fixed
            recall (resp. precision) threshold
    """
    roc_auc = plot_roc(metrics, title=title)
    acc = (metrics['query_labels'] == metrics['predictions']).float().mean(-1).mean(-1)
    print(f"ROC AUC: {roc_auc}")
    print(f"Accuracy: {acc}")

    # precisions, recalls, _ = precision_recall_curve(
    #     outliers_df.outlier, -outliers_df.outlier_score
    # )
    # precision_at_recall_objective = precisions[
    #     next(i for i, value in enumerate(recalls) if value < objective)
    # ]
    # recall_at_precision_objective = recalls[
    #     next(i for i, value in enumerate(precisions) if value > objective)
    # ]
    # print(f"Precision for recall={objective}: {precision_at_recall_objective}")
    # print(f"Recall for precision={objective}: {recall_at_precision_objective}")

    # plot_twin_hist(outliers_df, title=title, plot=args.streamlit)

    return roc_auc, acc


def update_csv(args: argparse.Namespace,
               metrics: dict,
               path: str):

    # Load records
    try:
        res = pd.read_csv(path)
    except FileNotFoundError:
        res = pd.DataFrame({})
    records = res.to_dict('records')

    # Metrics part of the new record
    fill_entry = metrics

    # Params part of the new record
    group_by_args = args.general_hparams + args.simu_hparams
    new_entry = {}
    for param in group_by_args:
        value = getattr(args, param)
        if isinstance(value, list):
            value = '-'.join(value)
        else:
            value = str(value)
        new_entry[param] = value

    # Check whether the row exists already, if yes, simply update the metrics
    match = False
    for existing_entry in records:
        if any([param not in existing_entry for param in group_by_args]):
            continue
        matches = [str(existing_entry[param]) == new_entry[param] for param in group_by_args]
        match = (sum(matches) == len(matches))
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
