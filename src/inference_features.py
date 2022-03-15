"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
from collections import defaultdict
from src.utils.utils import (
    set_random_seed,
    merge_from_dict,
    get_modules_to_try
)
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

from src.classifiers import __dict__ as CLASSIFIERS
from src.detectors.feature import __dict__ as FEATURE_DETECTORS
from src.detectors.proba import __dict__ as PROBA_DETECTORS
from src.all_in_one import __dict__ as ALL_IN_ONE_METHODS
from src.all_in_one import AllInOne

from src.models import __dict__ as BACKBONES
from src.transforms import __dict__ as TRANSFORMS
from src.utils.plots_and_metrics import update_csv
from src.utils.data_fetchers import get_task_loader, get_test_features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> argparse.Namespace:

    # Data
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--tgt_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_id_query", type=int, default=10)
    parser.add_argument("--n_ood_query", type=int, default=10)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default='cuda')
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
    parser.add_argument("--detector_config_file", type=str, default="configs/detectors.yaml")

    # Transform
    parser.add_argument("--feature_transforms", nargs='+', type=str, default=['l2_norm'],
                        help="What type of transformation to apply after spatial pooling.")
    parser.add_argument("--transforms_config_file", type=str, default="configs/transforms.yaml")

    # Classifier
    parser.add_argument("--classifier", type=str, default="SimpleShot")
    parser.add_argument("--use_filtering", type=str2bool, default=False)
    parser.add_argument("--classifiers_config_file", type=str, default="configs/classifiers.yaml")

    # Logging / Saving results

    parser.add_argument("--exp_name", type=str, default='default',
                        help="Name the experiment for easy grouping.")
    parser.add_argument("--general_hparams", type=str, nargs='+',
                        default=['backbone', 'src_dataset', 'tgt_dataset', 'balanced_tasks', 'feature_detectors',
                                 'classifier', 'n_way', 'n_shot'],
                        help="Important params that will appear in .csv result file.",)
    parser.add_argument("--simu_hparams", type=str, nargs='*', default=[],
                        help="Important params that will appear in .csv result file.")
    parser.add_argument("--override", type=str2bool, help='Whether to override results already present in the .csv out file.')

    # Tuning
    parser.add_argument("--tune", nargs='+', default=[])
    parser.add_argument("--debug", type=str2bool)

    args = parser.parse_args()

    if args.debug:
        args.n_tasks = 5

    # Merge external config files
    for file in [args.detector_config_file, args.transforms_config_file, args.classifiers_config_file]:
        with open(file) as f:
            parsed_yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        merge_from_dict(args, parsed_yaml_dict)
    return args


def main(args):

    save_dir = Path(os.path.join('results', args.exp_name))
    save_dir.mkdir(exist_ok=True, parents=True)
    args.layers = BACKBONES[args.backbone]().all_layers[-args.layers:]

    # logger.info(f"Dropping config file at {save_dir / 'config.json'}")
    # with open(save_dir / 'config.json', 'w') as f:
    #     json.dump(vars(args), f)

    # ================ Prepare data ===================

    feature_dic = defaultdict(dict)
    train_mean = {}
    train_std = {}
    for i, layer in enumerate(args.layers):
        features, _, train_mean[i], train_std[i] = get_test_features(
            args.data_dir, args.backbone, args.src_dataset, args.tgt_dataset, args.training, args.model_source, layer
        )
        for class_ in features:
            feature_dic[class_.item()][layer] = features[class_]
    data_loader = get_task_loader(args, "test", args.tgt_dataset, args.n_way, args.n_shot,
                                  args.n_id_query, args.n_ood_query, args.n_tasks, args.n_workers, feature_dic)

    # ================ Prepare Transforms + Detector + Classifier ===================

    transform_names = args.feature_transforms

    # ==== Prepare transforms ====

    transforms = []
    for x in transform_names:
        transforms.append(
                    get_modules_to_try(args, 'transforms', x,
                                       TRANSFORMS, False)[0]
                   )
    transforms = TRANSFORMS['SequentialTransform'](transforms)

    if args.feature_detector in ALL_IN_ONE_METHODS:

        # ==== Prepare all in one detector ====

        feature_detectors = get_modules_to_try(args, 'feature_detectors', args.feature_detector,
                                               ALL_IN_ONE_METHODS, 'all_in_one' in args.tune)
        proba_detectors = classifier = [None]

    else:
        # ==== Prepare few-shot classifier ====

        classifiers = get_modules_to_try(args, 'classifiers', args.classifier,
                                         CLASSIFIERS, 'classifier' in args.tune)

        # ==== Prepare feature detector ====

        feature_detectors = get_modules_to_try(args, 'feature_detectors', args.feature_detector,
                                               FEATURE_DETECTORS, 'feature_detector' in args.tune)

        # ==== Prepare proba detector ====
        proba_detectors = get_modules_to_try(args, 'proba_detectors', args.proba_detector,
                                             PROBA_DETECTORS, 'proba_detector' in args.tune)

    for feature_d, proba_d, classifier in itertools.product(
                feature_detectors, proba_detectors, classifiers):
        logger.info(f"Transforms : {transforms}")
        logger.info(f"Feature detector:  {feature_d}")
        logger.info(f"Proba detector: {proba_d}")
        logger.info(f"Classifier {classifier}")
        args.current_sequence = str((feature_d, proba_d, classifier))

        set_random_seed(args.random_seed)

        metrics = detect_outliers(args=args,
                                  layers=args.layers,
                                  transforms=transforms,
                                  train_mean=train_mean,
                                  train_std=train_std,
                                  classifier=classifier,
                                  feature_detector=feature_d,
                                  proba_detector=proba_d,
                                  data_loader=data_loader,
                                  on_features=True)
        # Saving results
        save_results(args, metrics)


def save_results(args, metrics):
    for metric_name in metrics:
        logger.info(f"{metric_name}: {np.round(100 * metrics[metric_name], 2)}")

    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    update_csv(args, metrics, path=res_root / 'out.csv')


def tsne_plot(figures, feat_s, feat_q, support_labels, query_labels, title):

    fig, axis = plt.subplots(nrows=1, ncols=len(feat_s), figsize=(10, 10), squeeze=True, dpi=200)
    if len(feat_s) == 1:
        axis = [axis]

    for layer, ax in zip(feat_s, axis):
        all_feats = torch.cat([feat_s[layer], feat_q[layer]], 0)
        embedded_feats = TSNE(n_components=2, init='pca', perplexity=5, learning_rate=50).fit_transform(all_feats.squeeze().numpy())
        all_labels = torch.cat([support_labels, query_labels], 0).numpy()
        ax.scatter(embedded_feats[:, 0], embedded_feats[:, 1], c=all_labels)
    figures[title] = fig


def detect_outliers(args, layers, transforms, classifier, feature_detector, proba_detector,
                    data_loader, train_mean, train_std, on_features: bool, model=None):

    metrics = defaultdict(list)
    intra_task_metrics = defaultdict(lambda: defaultdict(list))
    figures: Dict[str, Any] = {}
    for task_id, (support, support_labels, query, query_labels, outliers) in enumerate(tqdm(data_loader)):

        support_labels, query_labels = support_labels.long(), query_labels.long()

        # ====== Extract features ======
        if on_features:
            support_features = support
            query_features = query
        else:
            all_images = torch.cat([support, query], 0)
            with torch.no_grad():
                all_images = all_images.cuda()
                all_features = model(all_images, layers)  # []
            support_features = {k: v[:support.size(0)].cpu() for k, v in all_features.items()}
            query_features = {k: v[support.size(0):].cpu() for k, v in all_features.items()}

        # ====== Transforming features ======
        for layer in support_features:
            support_features[layer], query_features[layer] = transforms(
                  raw_feat_s=support_features[layer],
                  raw_feat_q=query_features[layer],
                  train_mean=train_mean[layer],
                  train_std=train_std[layer],
                  support_labels=support_labels,
                  query_labels=query_labels,
                  outliers=outliers,
                  intra_task_metrics=intra_task_metrics,
                  figures=figures
            )

        # ====== Classification + OOD detection ======

        outlier_scores = defaultdict(list)
        soft_preds_q = []

        for layer in support_features:
            output = feature_detector(
                support_features=support_features[layer],
                query_features=query_features[layer],
                train_mean=train_mean[layer],
                train_std=train_std[layer],
                support_labels=support_labels,
                query_labels=query_labels,
                outliers=outliers,
                intra_task_metrics=intra_task_metrics,
                figures=figures
            )
            if isinstance(feature_detector, AllInOne):
                probas_s, probas_q, scores = output
            else:
                scores = output
                if args.use_filtering:
                    use_transductively = (feature_detector.standardize(scores) < 0.15)
                else:
                    use_transductively = None
                probas_s, probas_q = classifier(support_features=support_features[layer],
                                                query_features=query_features[layer],
                                                support_labels=support_labels,
                                                intra_task_metrics=intra_task_metrics,
                                                query_labels=query_labels,
                                                outlier_scores=scores,
                                                use_transductively=use_transductively,
                                                outliers=outliers)
                outlier_scores['probas'].append(
                    proba_detector(support_probas=probas_s,
                                   query_probas=probas_q)
                )

            outlier_scores['features'].append(scores)
            soft_preds_q.append(probas_q)
        
        # ====== Aggregate scores and probas across layers ======

        for score_type, scores in outlier_scores.items():
            outlier_scores[score_type] = torch.stack(scores, 0).mean(0)
        all_probs = torch.stack(soft_preds_q, 0).mean(0)
        predictions = all_probs.argmax(-1)

        # ====== Tracking metrics ======
        acc = (predictions[outliers == 0] == query_labels[outliers == 0]).float().mean().item()
        metrics['acc'].append(acc)
        if args.n_ood_query:
            for score_type, scores in outlier_scores.items():
                fp_rate, tp_rate, thresholds = roc_curve(outliers.numpy(), scores.numpy())
                precision, recall, _ = precision_recall_curve(outliers.numpy(), scores.numpy())
                metrics[f"{score_type}_rocauc"].append(auc_fn(fp_rate, tp_rate))
            metrics['outlier_ratio'].append(outliers.sum().item() / outliers.size(0))

    final_metrics = {}
    for metric_name in metrics:
        mean, std = compute_confidence_interval(np.array(metrics[metric_name]), ignore_value=255)
        final_metrics[f"mean_{metric_name}"] = np.round(mean, 4)
        final_metrics[f"std_{metric_name}"] = np.round(std, 4)

    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    for title in intra_task_metrics.keys():
        fig = plt.Figure((10, 10), dpi=200)
        for legend, values in intra_task_metrics[title] .items():
            array = np.array(values)
            assert len(array.shape) == 2
            m, pm = compute_confidence_interval(array, ignore_value=255)
            ax = plt.gca()
            if array.shape[0] == 1:
                ax.scatter([0], m, c='r')
            else:
                x = np.arange(len(m))
                ax.plot(m, label=legend)
                ax.fill_between(x, m - pm, m + pm, alpha=0.5)
        plt.legend()
        plt.savefig(res_root / f'{title}.png')
        plt.clf()
    # bins = np.linspace(0, 1, 10)
    # inds = np.digitize(metrics['outlier_ratio'], bins) - 1
    # binned_aucs = []
    # bin_importance = []
    # for score_type, scores in metrics.items()
    # for i in range(len(bins)):
    #     if sum(inds == i):
    #         binned_aucs.append(np.array()[np.where(inds == i)].mean())
    #     else:
    #         binned_aucs.append(0.)
    #     bin_importance.append((sum(inds == i) + 1e-10) / inds.shape[0])

    # bars = plt.bar(bins, binned_aucs, width=0.1, align='edge', label="{} (ROCAUC={:.2f})".format(
    #      str(transforms) + str(feature_detector), 100 * final_metrics['mean_auc']))
    # plt.legend()
    # plt.xlabel(r'Outlier/Inlier ratio in query set')
    # plt.ylabel(r'Average AUROC')
    # for i, b in enumerate(bars):
    #     b.set_color(plt.cm.Blues(bin_importance[i]))
    # plt.savefig(res_root / f'{str(transforms)}+{str(feature_detector)}_binned_auc.png')
    # plt.clf()

    # Save figures
    for title, fig in figures.items():
        fig.savefig(res_root / f'{title}.png')
    return final_metrics


def compute_confidence_interval(data: np.ndarray, axis=0, ignore_value=None,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    assert len(data)
    if ignore_value is None:
        valid = np.ones_like(data)
    else:
        valid = (data != ignore_value)
    m = np.sum(data * valid, axis=axis, keepdims=True) / valid.sum(axis=axis, keepdims=True)
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
