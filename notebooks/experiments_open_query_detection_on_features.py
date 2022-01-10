"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
import logging
from src.outlier_detection_methods import RenyiEntropyOutlierDetector, KNNOutlierDetector, IterativeKNNOutlierDetector
from src.utils.utils import (
    set_random_seed,
)
import pandas as pd
import numpy as np
from pathlib import Path
from src.few_shot_methods import ALL_FEW_SHOT_CLASSIFIERS
from src.utils.outlier_detectors import (
    detect_outliers,
)
from src.utils.plots_and_metrics import show_all_metrics_and_plots, update_csv
from src.utils.data_fetchers import get_features_data_loader, get_test_features


def parse_args() -> argparse.Namespace:

    # Data
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mini_imagenet")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=6)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--training", type=str, default="classic")

    # Detector
    parser.add_argument("--outlier_detector", type=str, default="knn")
    parser.add_argument("--nn", type=int, default=3)

    # Method
    parser.add_argument("--inference_method", type=str, default="SimpleShot")
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument(
        "--inference_lr",
        type=float,
        default=1e-3,
        help="Learning rate used for methods that perform \
                        gradient-based inference.",
    )

    parser.add_argument(
        "--prepool_transforms",
        type=str,
        nargs='+',
        default='trivial',
        help="What type of transformation to apply before spatial pooling.",
    )
    parser.add_argument(
        "--postpool_transforms",
        nargs='+',
        type=str,
        default="l2_norm",
        help="What type of transformation to apply after spatial pooling.",
    )
    parser.add_argument(
        "--inference_steps",
        type=float,
        default=10,
        help="Steps used for gradient-based inferences.",
    )

    # Logging / Saving results
    parser.add_argument(
        "--exp_name",
    type=str,
    default='default',
    help="Name the experiment for easy grouping.")
    parser.add_argument(
    "--general_hparams",
    type=str,
    nargs='+',
    default=['backbone', 'outlier_detector', 'inference_method', 'n_way', 'n_shot', 'prepool_transforms', 'postpool_transforms'],
    help="Important params that will appear in .csv result file.",
    )
    parser.add_argument(
    "--simu_hparams",
    type=str,
    nargs='*',
    default=[],
    help="Important params that will appear in .csv result file.",
    )
    parser.add_argument(
        "--override",
    action='store_true',
    help='Whether to override results already present in the .csv out file.')

    args = parser.parse_args()
    return args


def main(args):
    set_random_seed(args.random_seed)

    features, _, average_train_features = get_test_features(
        args.backbone, args.dataset, args.training
    )

    data_loader = get_features_data_loader(
        features,
        args.n_way,
        args.n_shot,
        args.n_query,
        args.n_tasks,
        args.n_workers,
    )

    few_shot_classifier = [
        class_
        for class_ in ALL_FEW_SHOT_CLASSIFIERS
        if class_.__name__ == args.inference_method
    ][0].from_cli_args(args, average_train_features)

    if args.outlier_detector == 'renyi':
        outlier_detector = RenyiEntropyOutlierDetector(
            few_shot_classifier=few_shot_classifier
        )
    elif args.outlier_detector == 'knn':
        outlier_detector = KNNOutlierDetector(
            few_shot_classifier=few_shot_classifier, n_neighbors=args.nn
        )
    elif args.outlier_detector == 'iterative_knn':
        outlier_detector = IterativeKNNOutlierDetector(
            few_shot_classifier=few_shot_classifier, n_neighbors=args.nn
        )

    outliers_df, acc = detect_outliers(
        outlier_detector, data_loader, args.n_way, args.n_query
    )

    # Saving results 

    roc_auc, prec_at_90, rec_at_90 = show_all_metrics_and_plots(outliers_df, title=outlier_detector.__class__.__name__)
    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    metrics = {'acc': np.round(acc, 4), 'roc_auc': np.round(roc_auc, 4), 'prec@0.9': np.round(prec_at_90, 4), 'rec@90': np.round(rec_at_90, 4)}
    update_csv(args, metrics, path= res_root / 'out.csv')



if __name__ == "__main__":
    args = parse_args()
    main(args)
