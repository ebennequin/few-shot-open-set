"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
from collections import defaultdict
from src.utils.utils import set_random_seed, merge_from_dict, get_modules_to_try
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as auc_fn

from loguru import logger
import os
import json
import inspect
import yaml
import numpy as np
import itertools
from typing import Tuple, List, Dict, Any
import seaborn as sns
from pathlib import Path
from src.utils.utils import load_model

from src.classifiers import __dict__ as CLASSIFIERS
from src.classifiers import FewShotMethod

from src.detectors.feature import __all__ as FEATURE_DETECTORS
from src.detectors.feature import FeatureDetector

from src.detectors.proba import __dict__ as PROBA_DETECTORS
from src.detectors.proba import ProbaDetector
from src.all_in_one import __dict__ as ALL_IN_ONE_METHODS
from src.robust_ssl import __dict__ as SSL_METHODS

from src.models import __dict__ as BACKBONES
from src.transforms import __dict__ as TRANSFORMS
from src.transforms import FeatureTransform
from src.utils.plots_and_metrics import update_csv, check_if_record_exists
from src.utils.data_fetchers import get_task_loader, get_test_features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from copy import deepcopy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:

    # Data
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--tgt_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_id_query", type=int, default=15)
    parser.add_argument("--n_ood_query", type=int, default=15)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--balanced_tasks", type=str2bool, default="True")
    parser.add_argument("--alpha", type=float, default=1.0)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--model_source", type=str, default="feat")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--layers", type=int)

    # Detector
    parser.add_argument("--feature_detector", type=str)
    parser.add_argument("--proba_detector", type=str)
    parser.add_argument(
        "--detector_config_file", type=str, default="configs/detectors.yaml"
    )

    # Transform
    parser.add_argument(
        "--detector_transforms",
        nargs="+",
        type=str,
        default=["l2_norm"],
        help="What type of transformation to apply after spatial pooling.",
    )
    parser.add_argument(
        "--classifier_transforms",
        nargs="+",
        type=str,
        default=["l2_norm"],
        help="What type of transformation to apply after spatial pooling.",
    )
    parser.add_argument(
        "--transforms_config_file", type=str, default="configs/transforms.yaml"
    )

    # Classifier
    parser.add_argument("--classifier", type=str, default="SimpleShot")
    parser.add_argument("--use_filtering", type=str2bool, default=False)
    parser.add_argument("--threshold", type=str, default="otsu")
    parser.add_argument(
        "--classifiers_config_file", type=str, default="configs/classifiers.yaml"
    )

    # Logging / Saving results

    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="Name the experiment for easy grouping.",
    )
    parser.add_argument("--visu_episode", type=str2bool, default="default")
    parser.add_argument("--save_predictions", type=str2bool, default="default")
    parser.add_argument(
        "--general_hparams",
        type=str,
        nargs="+",
        default=[
            "backbone",
            "src_dataset",
            "tgt_dataset",
            "split",
            "feature_detector",
            "proba_detector",
            "classifier",
            "n_way",
            "n_shot",
            "n_id_query",
            "n_ood_query"
        ],
        help="Important params that will appear in .csv result file.",
    )
    parser.add_argument(
        "--simu_hparams",
        type=str,
        nargs="*",
        default=[],
        help="Important params that will appear in .csv result file.",
    )
    parser.add_argument(
        "--override",
        type=str2bool,
        help="Whether to override results already present in the .csv out file.",
    )

    # Tuning
    parser.add_argument("--tune", nargs="+", default=[])
    parser.add_argument("--debug", type=str2bool)

    args = parser.parse_args()

    if args.debug:
        args.n_tasks = 5

    # Merge external config files
    for file in [
        args.detector_config_file,
        args.transforms_config_file,
        args.classifiers_config_file,
    ]:
        with open(file) as f:
            parsed_yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        merge_from_dict(args, parsed_yaml_dict)
    return args


def dump_config(args):
    path = Path(args.res_dir) / 'config.json'
    logger.info(f"Dropping config file at {path}")
    with open(path, "w") as f:
        arg_dict = vars(args)
        json.dump(arg_dict, f)


def main(args):

    args.res_root = os.path.join("results", args.exp_name)
    os.makedirs(args.res_root, exist_ok=True)
    args.layers = BACKBONES[args.backbone]().all_layers[-args.layers :]

    # ================ Prepare Transforms + Detector + Classifier ===================

    # ==== Prepare transforms ====

    classifier_transforms: List[FeatureTransform] = []
    for x in args.classifier_transforms:
        classifier_transforms.append(
            get_modules_to_try(args, "transforms", x, TRANSFORMS, False)[0]
        )
    classifier_transforms: FeatureTransform = TRANSFORMS["SequentialTransform"](
        classifier_transforms
    )

    detector_transforms: List[FeatureTransform] = []
    for x in args.detector_transforms:
        detector_transforms.append(
            get_modules_to_try(args, "transforms", x, TRANSFORMS, False)[0]
        )
    detector_transforms: FeatureTransform = TRANSFORMS["SequentialTransform"](
        detector_transforms
    )

    if args.feature_detector in ALL_IN_ONE_METHODS:

        # ==== Prepare all in one detector ====

        feature_detectors: List[FeatureDetector] = get_modules_to_try(
            args,
            "feature_detectors",
            args.feature_detector,
            ALL_IN_ONE_METHODS,
            "feature_detector" in args.tune,
        )
        proba_detectors = classifiers = [None]

    elif args.feature_detector in SSL_METHODS:

        feature_detectors = get_modules_to_try(
            args,
            "feature_detectors",
            args.feature_detector,
            SSL_METHODS,
            "feature_detector" in args.tune,
        )
        proba_detectors = classifiers = [None]

    else:
        # ==== Prepare few-shot classifier ====
        if args.classifier == 'none':
            classifiers = [None]
        else:
            classifiers: List[FewShotMethod] = get_modules_to_try(
                args, "classifiers", args.classifier, CLASSIFIERS, "classifier" in args.tune
            )

        # ==== Prepare feature detector ====

        if args.feature_detector == "none":
            feature_detectors = [None]
        else:
            feature_detectors = get_modules_to_try(
                args,
                "feature_detectors",
                args.feature_detector,
                FEATURE_DETECTORS,
                "feature_detector" in args.tune,
            )
        # ==== Prepare proba detector ====
        if args.proba_detector == "none":
            proba_detectors = [None]
        else:
            proba_detectors: List[ProbaDetector] = get_modules_to_try(
                args,
                "proba_detectors",
                args.proba_detector,
                PROBA_DETECTORS,
                "proba_detector" in args.tune,
            )

    # ================ Prepare data ===================

    if (
        args.feature_detector in SSL_METHODS
    ):  # SSL methods require raw images and the actual model, not the features
        logger.info("Using model instead of saved features.")
        if args.model_source == "url":
            weights = None
        else:
            weights = (
                Path(args.data_dir)
                / "models"
                / args.training
                / f"{args.backbone}_{args.src_dataset}_{args.model_source}.pth"
            )
        feature_extractor = load_model(
            args, args.backbone, weights, args.src_dataset, args.device
        )
        feature_dic = train_mean = train_std = None
    else:  # Currently, all other methods work directly on features
        feature_extractor = None
        feature_dic = defaultdict(dict)
        train_mean = {}
        train_std = {}
        for i, layer in enumerate(args.layers):
            features, _, train_mean[i], train_std[i], _, _ = get_test_features(
                args.data_dir,
                args.backbone,
                args.src_dataset,
                args.tgt_dataset,
                args.training,
                args.model_source,
                layer,
                args.split,
            )
            for class_ in features:
                feature_dic[class_.item()][layer] = features[class_]

    data_loader = get_task_loader(
        args,
        args.split,
        args.tgt_dataset,
        args.n_way,
        args.n_shot,
        args.n_id_query,
        args.n_ood_query,
        args.n_tasks,
        args.n_workers,
        feature_dic,
    )  # If feature_dic is None, this loader will return raw PIL images !

    for feature_d, proba_d, classifier in itertools.product(
        feature_detectors, proba_detectors, classifiers
    ):

        # ==> Each run is an experiment with some set of hyper-parameters

        logger.info(f"Classifier transforms : {classifier_transforms}")
        logger.info(f"Detector transforms : {detector_transforms}")
        logger.info(f"Feature detector:  {feature_d}")
        logger.info(f"Proba detector: {proba_d}")
        logger.info(f"Classifier {classifier}")
        args.feature_detector = str(feature_d)
        args.proba_detector = str(proba_d)
        args.classifier = str(classifier)


        sub_exp = (str(args.classifier) + str(args.feature_detector)).replace("None", "")
        args.res_dir = os.path.join(args.res_root, sub_exp)
        os.makedirs(args.res_dir, exist_ok=True)

        dump_config(args)

        set_random_seed(args.random_seed)

        if not check_if_record_exists(args, Path(args.res_root) / "out.csv") or args.override:
            metrics = detect_outliers(
                args=args,
                layers=args.layers,
                feature_extractor=feature_extractor,
                detector_transforms=detector_transforms,
                classifier_transforms=classifier_transforms,
                train_mean=train_mean,
                train_std=train_std,
                classifier=classifier,
                feature_detector=feature_d,
                proba_detector=proba_d,
                data_loader=data_loader,
            )
            save_results(args, metrics, args.res_root)
        else:
            logger.warning("Experiment already done, and overriding not activated. Moving to the next.")


def save_results(args, metrics, res_root):
    for metric_name in metrics:
        logger.info(f"{metric_name}: {np.round(100 * metrics[metric_name], 2)}")
    update_csv(args, metrics, path=Path(res_root) / "out.csv")


def detect_outliers(
    args,
    layers,
    classifier_transforms,
    detector_transforms,
    classifier,
    feature_detector,
    proba_detector,
    data_loader,
    train_mean,
    train_std,
    feature_extractor=None,
):

    tensors2save: Dict[str, List[torch.Tensor]] = defaultdict(list)
    metrics: Dict[str, List[float]] = defaultdict(list)
    intra_task_metrics = defaultdict(
        lambda: defaultdict(list)
    )  # used to monitor the average evolution of some metric \
    # within a task (during inference)
    figures: Dict[str, Any] = {}  # not really used anymore

    if feature_extractor is not None:
        initial_state_dict = deepcopy(feature_extractor.state_dict())

    for task_id, (support, support_labels, query, query_labels, outliers) in enumerate(
        tqdm(data_loader)
    ):

        support_labels, query_labels = support_labels.long(), query_labels.long()

        # ====== Extract features and transform them ======
        if feature_extractor is None:
            transformed_features = defaultdict(dict)
            support_features = support
            query_features = query

            # === Transforming features ===
            for layer in support_features:
                (
                    transformed_features["cls_sup"][layer],
                    transformed_features["cls_query"][layer],
                ) = classifier_transforms(
                    raw_feat_s=support_features[layer],
                    raw_feat_q=query_features[layer],
                    train_mean=deepcopy(train_mean[layer]),
                    train_std=deepcopy(train_std[layer]),
                    support_labels=support_labels,
                    query_labels=query_labels,
                    outliers=outliers,
                    intra_task_metrics=intra_task_metrics,
                    figures=figures,
                )
                (
                    transformed_features["det_sup"][layer],
                    transformed_features["det_query"][layer],
                ) = detector_transforms(
                    raw_feat_s=support_features[layer],
                    raw_feat_q=query_features[layer],
                    train_mean=deepcopy(train_mean[layer]),
                    train_std=deepcopy(train_std[layer]),
                    support_labels=support_labels,
                    query_labels=query_labels,
                    outliers=outliers,
                    intra_task_metrics=intra_task_metrics,
                    figures=figures,
                )
        else:
            feature_extractor.load_state_dict(initial_state_dict)
            # assert isinstance(feature_detector, AllInOne), "Currently on AllInOne \
            #     detectors also handle feature extraction"

        # ====== Classification + OOD detection ======

        outlier_scores = []
        soft_preds_q = []

        if feature_extractor is None:  # For methods that work on features directly
            for layer in support_features:
                if feature_detector is None:
                    assert proba_detector is not None
                    probas_s, probas_q = classifier(
                        support_features=transformed_features["cls_sup"][layer],
                        query_features=transformed_features["cls_query"][layer],
                        train_mean=deepcopy(train_mean[layer]),
                        support_labels=support_labels,
                        intra_task_metrics=intra_task_metrics,
                        use_transductively=None,
                        query_labels=query_labels,
                        outliers=outliers,
                    )
                    outlier_scores.append(
                        proba_detector(support_probas=probas_s, query_probas=probas_q)
                    )
                else:
                    output = feature_detector(
                        support_features=transformed_features["det_sup"][layer],
                        query_features=transformed_features["det_query"][layer],
                        train_mean=deepcopy(train_mean[layer]),
                        support_labels=support_labels,
                        query_labels=query_labels,
                        outliers=outliers,
                        intra_task_metrics=intra_task_metrics,
                        figures=figures,
                    )
                    if len(output) == 3:
                        probas_s, probas_q, scores = output
                    else:
                        scores = output
                        if args.use_filtering:  # Then we filter out before giving
                            assert feature_detector is not None
                            if args.threshold == "otsu":
                                thresh = threshold_otsu(scores.numpy())
                            else:
                                thresh = float(args.threshold)
                            believed_inliers = scores < thresh
                            metrics["thresholding_accuracy"].append(
                                (believed_inliers == ~outliers.bool())
                                .float()
                                .mean()
                                .item()
                            )
                        else:
                            believed_inliers = None

                        probas_s, probas_q = classifier(
                            support_features=transformed_features["cls_sup"][layer],
                            query_features=transformed_features["cls_query"][layer],
                            train_mean=train_mean[layer],
                            support_labels=support_labels,
                            intra_task_metrics=intra_task_metrics,
                            query_labels=query_labels,
                            use_transductively=believed_inliers,
                            outliers=outliers,
                        )
                    outlier_scores.append(scores)
                soft_preds_q.append(probas_q)
        else:  # For SSL methods
            output = feature_detector(
                support_images=support,
                query_images=query,
                feature_extractor=feature_extractor,
                support_labels=support_labels,
                query_labels=query_labels,
                outliers=outliers,
                intra_task_metrics=intra_task_metrics,
            )
            probas_s, probas_q, scores = output
            soft_preds_q.append(probas_q)
            outlier_scores.append(scores)
            feature_detector.clear()

        # ====== Aggregate scores and probas across layers ======

        outlier_scores = torch.stack(outlier_scores, 0).mean(0)
        all_probs = torch.stack(soft_preds_q, 0).mean(0)
        predictions = all_probs.argmax(-1)

        # ====== Store predictions in case it needs saving ====

        if args.save_predictions:
            for array_name in ['probas_q', 'outliers', 'query_labels', 'outlier_scores']:
                tensors2save[array_name].append(eval(array_name))

        # ====== Tracking metrics ======

        acc = (
            (predictions[outliers == 0] == query_labels[outliers == 0])
            .float()
            .mean()
            .item()
        )
        metrics["acc"].append(acc)
        if args.n_ood_query:
            fp_rate, tp_rate, _ = roc_curve(outliers.numpy(), outlier_scores.numpy())
            precision, recall, thresholds = precision_recall_curve(
                outliers.numpy(), outlier_scores.numpy()
            )
            aupr = auc_fn(recall, precision)
            precision_at_90 = precision[recall >= 0.9][-1]
            recall_at_90 = recall[precision >= 0.9][0]
            metrics["rocauc"].append(auc_fn(fp_rate, tp_rate))
            metrics["prec_at_90"].append(precision_at_90)
            metrics["rec_at_90"].append(recall_at_90)
            metrics["aupr"].append(aupr)
            metrics["rec_at_90"].append(recall_at_90)
            metrics["outlier_ratio"].append(outliers.sum().item() / outliers.size(0))

    # ====== Computing mean and std of metrics across tasks ======

    final_metrics = {}
    for metric_name in metrics:
        mean, std = compute_confidence_interval(
            np.array(metrics[metric_name]), ignore_value=255
        )
        final_metrics[f"mean_{metric_name}"] = np.round(mean, 4)
        final_metrics[f"std_{metric_name}"] = np.round(std, 4)

    # ====== Quick intra-task metrics ======

    for title in intra_task_metrics.keys():
        fig = plt.Figure((10, 10), dpi=200)
        for legend, values in intra_task_metrics[title].items():
            array = np.array(values)
            assert len(array.shape) == 2
            m, pm = compute_confidence_interval(array, ignore_value=255)
            ax = plt.gca()
            if array.shape[0] == 1:
                ax.scatter([0], m, c="r")
            else:
                x = np.arange(len(m))
                ax.plot(m, label=legend)
                ax.fill_between(x, m - pm, m + pm, alpha=0.5)
        plt.legend()
        plt.savefig(Path(args.res_dir) / f"{title}.png")
        plt.clf()

    # ====== Save predictions and gts in case ====

    if args.save_predictions:
        for array_name, tensor_list in tensors2save.items():
            tensor = torch.stack(tensor_list, 0)
            torch.save(tensor, Path(args.res_dir) / f'{array_name}.pt')
            logger.info(f"Saved {Path(args.res_dir) / array_name}.pt'")

    return final_metrics


def compute_confidence_interval(
    data: np.ndarray,
    axis=0,
    ignore_value=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    assert len(data)
    if ignore_value is None:
        valid = np.ones_like(data)
    else:
        valid = data != ignore_value
    m = np.sum(data * valid, axis=axis, keepdims=True) / valid.sum(
        axis=axis, keepdims=True
    )
    # np.mean(data, axis=axis)
    std = np.sqrt(((data - m) ** 2 * valid).sum(axis=axis) / valid.sum(axis=axis))
    # std = np.std(data, axis=axis)

    pm = 1.96 * (std / np.sqrt(valid.sum(axis=axis)))

    m = np.squeeze(m).astype(np.float64)
    pm = pm.astype(np.float64)

    return m, pm


if __name__ == "__main__":
    args = parse_args()
    main(args)
