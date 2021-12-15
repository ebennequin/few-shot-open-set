"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse

from src.outlier_detection_methods import RenyiEntropyOutlierDetector
from src.utils.utils import (
    set_random_seed,
)
from src.few_shot_methods import ALL_FEW_SHOT_CLASSIFIERS
from src.utils.outlier_detectors import (
    detect_outliers,
)
from src.utils.plots_and_metrics import show_all_metrics_and_plots
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
    parser.add_argument("--n_workers", type=int, default=12)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--training", type=str, default="classic")

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
        "--inference_steps",
        type=float,
        default=10,
        help="Steps used for gradient-based inferences.",
    )

    args = parser.parse_args()
    return args


def main(args):
    set_random_seed(args.random_seed)

    features, average_train_features = get_test_features(
        args.backbone, args.dataset, args.training
    )

    data_loader = get_features_data_loader(
        features,
        average_train_features,
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
    ][0].from_cli_args(args)

    outlier_detector = RenyiEntropyOutlierDetector(
        few_shot_classifier=few_shot_classifier
    )

    outliers_df = detect_outliers(
        outlier_detector, data_loader, args.n_way, args.n_query
    )

    show_all_metrics_and_plots(outliers_df, title=outlier_detector.__class__.__name__)


if __name__ == "__main__":
    args = parse_args()
    main(args)
