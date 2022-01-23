"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
import logging
from collections import defaultdict
from src.outlier_detection_methods import FewShotDetector, NaiveAggregator
from src.utils.utils import (
    set_random_seed,
)
from typing import Dict
from types import SimpleNamespace
import pandas as pd
import yaml
import numpy as np
import itertools
from pathlib import Path
import pyod
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
    parser.add_argument("--pool", type=bool, default=True)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--training", type=str, default="classic")

    # Detector
    parser.add_argument("--outlier_detectors", type=str, default="knn_3")
    parser.add_argument("--detector_config_file", type=str, default="configs/detectors.yaml")

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
        default=['backbone', 'dataset', 'outlier_detectors', 'inference_method', 'n_way', 'n_shot', 'prepool_transforms', 'postpool_transforms'],
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

    # Tuning
    parser.add_argument(
            "--combination_size",
            type=int,
            default=2)
    parser.add_argument(
            "--mode",
            type=str,
            default='benchmark')
    parser.add_argument(
            "--debug",
            action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.n_tasks = 5
    with open(args.detector_config_file) as f:
        parsed_yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    merge_from_dict(args, parsed_yaml_dict)
    return args


all_detectors = {'knn': pyod.models.knn.KNN,
                 'abod': pyod.models.abod.ABOD,
                 'pca': pyod.models.pca.PCA,
                 'rod': pyod.models.rod.ROD,
                 'sod': pyod.models.sod.SOD,
                 'ocsvm': pyod.models.ocsvm.OCSVM,
                 'iforest': pyod.models.iforest.IForest,
                 'feature_bagging': pyod.models.feature_bagging.FeatureBagging,
                 'sos': pyod.models.sos.SOS,
                 'lof': pyod.models.lof.LOF,
                 'ecod': pyod.models.ecod.ECOD,
                 'copod': pyod.models.copod.COPOD,
                 'cof': pyod.models.cof.COF,
                 'ae': pyod.models.auto_encoder.AutoEncoder}


def merge_from_dict(args, dict_: Dict):
    for key, value in dict_.items():
        if isinstance(value, dict):
            setattr(args, key, SimpleNamespace())
            merge_from_dict(getattr(args, key), value)
        else:
            setattr(args, key, value)


def main(args):
    set_random_seed(args.random_seed)

    features, _, average_train_features = get_test_features(
        args.backbone, args.dataset, args.training
    )

    few_shot_classifier = [
        class_
        for class_ in ALL_FEW_SHOT_CLASSIFIERS
        if class_.__name__ == args.inference_method
    ][0].from_cli_args(args, average_train_features)

    current_detectors = args.outlier_detectors.split('-')

    if args.mode == 'benchmark':

        data_loader = get_features_data_loader(
                            features,
                            args.n_way,
                            args.n_shot,
                            args.n_query,
                            args.n_tasks,
                            args.n_workers,
        )

        # Obtaining the detectors
        current_detectors = [all_detectors[x](**eval(f'args.{x}.default')) for x in current_detectors]

        # final_detector = SUOD(base_estimators=current_detectors,
        #                       n_jobs=2,
        #                       combination='average',
        #                       verbose=False)
        final_detector = NaiveAggregator(current_detectors)

        fewshot_detector = FewShotDetector(few_shot_classifier, final_detector)

        outliers_df, acc = detect_outliers(
            fewshot_detector, data_loader, args.n_way, args.n_query
        )

        # Saving results
        save_results(args, outliers_df)

    elif args.mode == 'tune':

        # For each detector type, create all relevant detectors
        detectors_to_try = defaultdict(list)
        for x in current_detectors:
            detector_args = vars(eval(f'args.{x}.current_params'))  # take default args
            params2tune = eval(f'args.{x}.tuning.hparams2tune')
            values2tune = eval(f'args.{x}.tuning.hparam_values')
            values_combinations = itertools.product(*values2tune)
            for some_combin in values_combinations:
                # Override default args
                for k, v in zip(params2tune, some_combin):
                    detector_args[k] = v
                detectors_to_try[x].append(all_detectors[x](**detector_args))

        # print(detectors_to_try)

        # For each detector type, create all relevant detectors
        for x in detectors_to_try:
            n_combinations = itertools.combinations(detectors_to_try[x], args.combination_size)
            detectors_to_try[x] = [list(x) for x in n_combinations]  # ([d1, d2], [d3, d4], ...)

        # Finally, form the product across detector types
        sequences_to_try = itertools.product(*list(detectors_to_try.values()))

        for detector_sequence in sequences_to_try:

            set_random_seed(args.random_seed)

            data_loader = get_features_data_loader(
                features,
                args.n_way,
                args.n_shot,
                args.n_query,
                args.n_tasks,
                args.n_workers,
            )

            d_sequence = []
            for d in detector_sequence:
                d_sequence += d
            args.current_sequence = [str(x) for x in d_sequence]

            # final_detector = pyod.models.suod.SUOD(base_estimators=d_sequence,
            #                                        n_jobs=1, combination='average',
            #                                        verbose=False)
            final_detector = NaiveAggregator(d_sequence)
            fewshot_detector = FewShotDetector(few_shot_classifier, final_detector)

            outliers_df, acc = detect_outliers(
                fewshot_detector, data_loader, args.n_way, args.n_query
            )

            # Saving results
            save_results(args, outliers_df)


def save_results(args, outliers_df):
    roc_auc, acc = show_all_metrics_and_plots(args, outliers_df, title='')
    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    metrics = {'acc': np.round(acc, 4), 'roc_auc': np.round(roc_auc, 4)}
    update_csv(args, metrics, path=res_root / 'out.csv')


if __name__ == "__main__":
    args = parse_args()
    main(args)
