"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

import argparse
from collections import defaultdict
from src.utils.utils import (
    set_random_seed,
    merge_from_dict
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
from typing import Tuple, List, Dict
from pathlib import Path
from src.few_shot_methods import ALL_FEW_SHOT_CLASSIFIERS
from src.detectors import ALL_DETECTORS
from src.utils.plots_and_metrics import update_csv
from src.utils.data_fetchers import get_task_loader, get_test_features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.models import __dict__ as BACKBONES
from src.transforms import __dict__ as TRANSFORMS

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
    parser.add_argument("--n_query", type=int, default=10)
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
    parser.add_argument("--outlier_detectors", type=str)
    parser.add_argument("--detector_config_file", type=str, default="configs/detectors.yaml")

    # Transform
    parser.add_argument("--feature_transforms", nargs='+', type=str, default=['l2_norm'],
                        help="What type of transformation to apply after spatial pooling.")
    parser.add_argument("--transforms_config_file", type=str, default="configs/transforms.yaml")

    # Method
    parser.add_argument("--inference_method", type=str, default="SimpleShot")
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument("--inference_lr", type=float, default=1e-3,
                        help="Learning rate used for methods that perform \
                        gradient-based inference.")
    parser.add_argument("--inference_steps", type=float, default=10,
                        help="Steps used for gradient-based inferences.")

    # Logging / Saving results

    parser.add_argument("--exp_name", type=str, default='default',
                        help="Name the experiment for easy grouping.")
    parser.add_argument("--general_hparams", type=str, nargs='+',
                        default=['backbone', 'src_dataset', 'tgt_dataset', 'balanced_tasks', 'outlier_detectors',
                                 'inference_method', 'n_way', 'n_shot', 'transforms'],
                        help="Important params that will appear in .csv result file.",)
    parser.add_argument("--simu_hparams", type=str, nargs='*', default=[],
                        help="Important params that will appear in .csv result file.")
    parser.add_argument("--override", type=str2bool, help='Whether to override results already present in the .csv out file.')

    # Tuning
    parser.add_argument("--combination_size", type=int, default=2)
    parser.add_argument("--mode", type=str, default='benchmark')
    parser.add_argument("--debug", type=str2bool)

    args = parser.parse_args()

    if args.debug:
        args.n_tasks = 5

    # Merge external config files
    for file in [args.detector_config_file, args.transforms_config_file]:
        with open(file) as f:
            parsed_yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        merge_from_dict(args, parsed_yaml_dict)
    return args


def main(args):
    set_random_seed(args.random_seed)
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
                                  args.n_query, args.n_tasks, args.n_workers, feature_dic)

    # ================ Prepare Few-Shot Classifier ===================

    few_shot_classifier = [
        class_
        for class_ in ALL_FEW_SHOT_CLASSIFIERS
        if class_.__name__ == args.inference_method
    ][0].from_cli_args(args)

    detector_names = args.outlier_detectors.split('-')
    transform_names = args.feature_transforms

    if args.mode == 'benchmark':

        # ================ Prepare detector ===================

        detectors = []
        transforms = []
        for x in detector_names:
            detector_args = eval(f'args.detectors.{x}.current_params')[args.n_shot]  # take default args
            if "args" in inspect.getfullargspec(ALL_DETECTORS[x].__init__).args:
                detector_args['args'] = args
            detectors.append(ALL_DETECTORS[x](**detector_args))
        outlier_detector = ALL_DETECTORS['aggregator'](detectors)
        for x in transform_names:
            if x in vars(args.transforms):
                transform_args = eval(f'args.transforms.{x}.current_params')[args.n_shot]  # take default args
                transforms.append(TRANSFORMS[x](**transform_args))
            else:
                transforms.append(TRANSFORMS[x]())
        transforms = TRANSFORMS['SequentialTransform'](transforms)

        logger.info(outlier_detector.detectors)
        logger.info(transforms.transform_list)

        args.current_sequence = str(outlier_detector.detectors) + str(transforms.transform_list)

        metrics = detect_outliers(layers=args.layers,
                                  transforms=transforms,
                                  train_mean=train_mean,
                                  train_std=train_std,
                                  few_shot_classifier=few_shot_classifier,
                                  detector=outlier_detector,
                                  data_loader=data_loader,
                                  n_way=args.n_way,
                                  n_query=args.n_query,
                                  on_features=True)
        # Saving results
        save_results(args, metrics)

    elif args.mode == 'tune':

        # For each detector type, create all relevant detectors

        detectors_to_try = get_all_detectors(detector_names)
        all_transforms = get_all_transforms(transform_names)

        for aggreg_detector in detectors_to_try:

            for aggreg_transform in all_transforms:

                set_random_seed(args.random_seed)
                logger.info(aggreg_detector.detectors)
                logger.info(aggreg_transform.transform_list)

                args.current_sequence = str(aggreg_detector.detectors) + str(aggreg_transform.transform_list)

                metrics = detect_outliers(layers=args.layers,
                                          transforms=aggreg_transform,
                                          train_mean=train_mean,
                                          train_std=train_std,
                                          few_shot_classifier=few_shot_classifier,
                                          detector=aggreg_detector,
                                          data_loader=data_loader,
                                          n_way=args.n_way,
                                          n_query=args.n_query,
                                          on_features=True)

                # Saving results
                save_results(args, metrics)


def get_all_transforms(transform_names: List[str]):

    """
    ['a', 'b', 'c']
    """
    modules_to_try = defaultdict(list)  # {'a': [a1, a2, a3],  'b': [b1, b2, b3], 'b': [c1]}
    for x in transform_names:
        if x in vars(args.transforms):
            module_args = eval(f'args.transforms.{x}.current_params')[args.n_shot]  # take default args
            params2tune = eval(f'args.transforms.{x}.tuning.hparams2tune')
            values2tune = eval(f'args.transforms.{x}.tuning.hparam_values')[args.n_shot]
            values_combinations = itertools.product(*values2tune)
            for some_combin in values_combinations:
                # Override default args
                for k, v in zip(params2tune, some_combin):
                    module_args[k] = v
                if "args" in inspect.getfullargspec(TRANSFORMS[x].__init__).args:
                    module_args['args'] = args
                modules_to_try[x].append(TRANSFORMS[x](**module_args))
        else:  # some methods just don't have any argument
            modules_to_try[x].append(TRANSFORMS[x]())

    transforms_products = itertools.product(*modules_to_try.values())  # [(a1, b1, c1), (a2, b1, c1), ....]
    all_transforms = [TRANSFORMS['SequentialTransform'](x) for x in transforms_products]
    return all_transforms


def get_all_detectors(detector_names: List[str]):
    """
    ['a', 'b']
    """
    detectors_to_try = defaultdict(list)  # {'a': [a1, a2, a3, a4], 'b': [b1, b2]}
    for x in detector_names:
        detector_args = eval(f'args.detectors.{x}.current_params')[args.n_shot]  # take default args
        params2tune = eval(f'args.detectors.{x}.tuning.hparams2tune')
        values2tune = eval(f'args.detectors.{x}.tuning.hparam_values')[args.n_shot]
        values_combinations = itertools.product(*values2tune)
        for some_combin in values_combinations:
            # Override default args
            for k, v in zip(params2tune, some_combin):
                detector_args[k] = v
            if "args" in inspect.getfullargspec(ALL_DETECTORS[x].__init__).args:
                detector_args['args'] = args
            detectors_to_try[x].append(ALL_DETECTORS[x](**detector_args))

    # For each detector type, create all relevant detectors
    for x in detectors_to_try:
        n_combinations = itertools.combinations(detectors_to_try[x], args.combination_size)
        detectors_to_try[x] = [list(x) for x in n_combinations]  # {'a': [[a1, a2], [a2, a3], ..], 'b': [[b1, b2]], ...}

    # Finally, form the product across detector types
    sequences_to_try = list(itertools.product(*list(detectors_to_try.values())))  # [[[a1, a2], [b2, b3]], ...]
    sequences_to_try = [list(itertools.chain(*x)) for x in sequences_to_try]  # [[a1, a2, b2, b3], ...]
    sequences_to_try = [ALL_DETECTORS['aggregator'](x) for x in sequences_to_try]
    return sequences_to_try


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


def detect_outliers(layers, transforms, few_shot_classifier,
                    detector, data_loader, n_way, n_query, train_mean, train_std,
                    on_features: bool, model=None):

    metrics = defaultdict(list)
    intra_task_metrics = defaultdict(list)
    figures = {}
    for task_id, (support, support_labels, query, query_labels, outliers) in enumerate(tqdm(data_loader)):

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
                      figures=figures)

        # ====== OOD detection ======
        outlier_scores = []
        for layer in support_features:
            detector.fit(support_features[layer], support_labels)
            raw_scores = torch.from_numpy(detector.decision_function(support_features[layer], query_features[layer]))
            outlier_scores.append(raw_scores)
            # outlier_scores.append((raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min()))  # [?,]
        outlier_scores = torch.stack(outlier_scores, 0).mean(0)

        # ====== Few-shot classifier ======
        all_probs = []
        for layer in support_features:
            all_probs.append(few_shot_classifier(support_features[layer], query_features[layer], support_labels)[1])
        all_probs = torch.stack(all_probs, 0).mean(0)
        predictions = all_probs.argmax(-1)

        # ====== Tracking metrics ======
        acc = (predictions[outliers == 0] == query_labels[outliers == 0]).float().mean().item()
        fp_rate, tp_rate, thresholds = roc_curve(outliers.numpy(), outlier_scores.numpy())
        precision, recall, _ = precision_recall_curve(outliers.numpy(), outlier_scores.numpy())
        auc = auc_fn(fp_rate, tp_rate)

        for metric_name in ['auc', 'acc']:
            metrics[metric_name].append(eval(metric_name))
        for t in transforms.transform_list:
            if hasattr(t, 'final_auc'):
                metrics['transform_auc'].append(t.final_auc)
        # logger.warning(metrics)

    for metric_name in metrics:
        metrics[metric_name] = np.round(torch.Tensor(metrics[metric_name]).mean().item(), 4)

    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    for metric_name, values in intra_task_metrics.items():
        array = np.array(values)
        assert len(array.shape) == 2
        m, pm = compute_confidence_interval(array, ignore_value=255)
        fig = plt.Figure((10, 10), dpi=200)
        ax = plt.gca()
        if array.shape[0] == 1:
            ax.scatter([0], m, c='r')
        else:
            x = np.arange(len(m))
            ax.plot(m)
            ax.fill_between(x, m - pm, m + pm, alpha=0.5)
        plt.savefig(res_root / f'{metric_name}.png')
        plt.clf()


    # Save figures
    for title, fig in figures.items():
        fig.savefig(res_root / f'{title}.png')
    return metrics


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
