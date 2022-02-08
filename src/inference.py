"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
from loguru import logger
from collections import defaultdict
from src.outlier_detection_methods import FewShotDetector, NaiveAggregator, local_knn
from src.utils.utils import (
    set_random_seed, load_model, merge_from_dict
)
from typing import Dict
from types import SimpleNamespace
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
from src.utils.data_fetchers import get_task_loader, get_train_features
import torch


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
    parser.add_argument("--pool", action='store_true')
    parser.add_argument("--image_size", type=int, default=84)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--training", type=str, default="feat")
    parser.add_argument("--layers", type=str, default="4_2")

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

    parser.add_argument("--augmentations", type=str, default="trivial")
    parser.add_argument(
        "--prepool_transforms",
        type=str,
        nargs='+',
        default=['trivial'],
        help="What type of transformation to apply before spatial pooling.",
    )
    parser.add_argument(
        "--postpool_transforms",
        nargs='+',
        type=str,
        default=['l2_norm'],
        help="What type of transformation to apply after spatial pooling.",
    )
    parser.add_argument(
        "--aggreg",
        type=str,
        default='concat',
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
                 # 'local_knn': local_knn,
                 # 'abod': pyod.models.abod.ABOD,
                 # 'pca': pyod.models.pca.PCA,
                 # 'rod': pyod.models.rod.ROD,
                 # 'sod': pyod.models.sod.SOD,
                 # 'ocsvm': pyod.models.ocsvm.OCSVM,
                 # 'iforest': pyod.models.iforest.IForest,
                 # 'feature_bagging': pyod.models.feature_bagging.FeatureBagging,
                 # 'sos': pyod.models.sos.SOS,
                 # 'lof': pyod.models.lof.LOF,
                 # 'ecod': pyod.models.ecod.ECOD,
                 # 'copod': pyod.models.copod.COPOD,
                 # 'cof': pyod.models.cof.COF,
                 # 'ae': pyod.models.auto_encoder.AutoEncoder
                 }


def main(args):
    set_random_seed(args.random_seed)

    logger.info("Building model...")
    weights = Path('data') / 'models' / f'{args.backbone}_{args.dataset}_{args.training}.pth'
    feature_extractor = load_model(backbone=args.backbone,
                                   weights=weights,
                                   dataset_name=args.dataset,
                                   device='cuda')

    # logger.info("Loading mean/std from base set ...")
    average_train_features = {}
    std_train_features = {}
    layers = args.layers.split('-')
    for layer in layers:
        average_train_features[layer], std_train_features[layer] = get_train_features(
            args.backbone, args.dataset, args.training, layer
        )

    logger.info("Creating few-shot classifier ...")
    current_detectors = args.outlier_detectors.split('-')
    few_shot_classifier = [
        class_
        for class_ in ALL_FEW_SHOT_CLASSIFIERS
        if class_.__name__ == args.inference_method
    ][0].from_cli_args(args, average_train_features, std_train_features)

    if args.mode == 'tune':

        logger.info("Tuning mode activated !")

        logger.info("Creating all detector combinations ...")

        # For each detector type, create all relevant detectors
        detectors_to_try = defaultdict(list)
        for x in current_detectors:
            detector_args = vars(eval(f'args.{x}.current_params'))  # take default args
            params2tune = eval(f'args.{x}.tuning.hparams2tune')
            values2tune = eval(f'args.{x}.tuning.hparam_values')[args.n_shot]
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

            logger.info(f"Trying detector {detector_sequence} ...")

            set_random_seed(args.random_seed)

            data_loader = get_task_loader(args,
                                          args.dataset,
                                          args.n_way,
                                          args.n_shot,
                                          args.n_query,
                                          args.n_tasks,
                                          split="test",
                                          n_workers=6)
            d_sequence = []
            for d in detector_sequence:
                d_sequence += d
            args.current_sequence = [str(x) for x in d_sequence]

            final_detector = NaiveAggregator(d_sequence)
            fewshot_detector = FewShotDetector(few_shot_classifier=few_shot_classifier,
                                               detector=final_detector,
                                               model=feature_extractor,
                                               layers=args.layers.split('-'),
                                               augmentations=args.augmentations.split('-'),
                                               on_features=False)

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
