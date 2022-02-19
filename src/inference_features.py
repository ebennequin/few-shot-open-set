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
from pathlib import Path
from src.few_shot_methods import ALL_FEW_SHOT_CLASSIFIERS
from src.detectors import ALL_DETECTORS
from src.utils.plots_and_metrics import show_all_metrics_and_plots, update_csv
from src.utils.data_fetchers import get_task_loader, get_test_features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.constants import BACKBONES


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
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--pool", action='store_true')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--balanced_tasks", type=str2bool, default="True")
    parser.add_argument("--alpha", type=float, default=1.0)

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--model_source", type=str, default="feat")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--layers", type=str, nargs='+')

    # Detector
    parser.add_argument("--outlier_detectors", type=str)
    parser.add_argument("--detector_config_file", type=str, default="configs/detectors.yaml")

    # Method
    parser.add_argument("--inference_method", type=str, default="SimpleShot")
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument("--inference_lr", type=float, default=1e-3,
                        help="Learning rate used for methods that perform \
                        gradient-based inference.")

    parser.add_argument("--prepool_transforms", type=str, nargs='+',
                        default=['trivial'], help="What type of transformation to apply before spatial pooling.")
    parser.add_argument("--postpool_transforms", nargs='+', type=str, default=['l2_norm'],
                        help="What type of transformation to apply after spatial pooling.")
    parser.add_argument("--aggreg", type=str, default='concat',
                        help="What type of transformation to apply after spatial pooling.")
    parser.add_argument("--inference_steps", type=float, default=10,
                        help="Steps used for gradient-based inferences.")

    # Logging / Saving results

    parser.add_argument("--exp_name", type=str, default='default',
                        help="Name the experiment for easy grouping.")
    parser.add_argument("--general_hparams", type=str, nargs='+',
                        default=['backbone', 'src_dataset', 'tgt_dataset', 'balanced_tasks', 'outlier_detectors',
                                 'inference_method', 'n_way', 'n_shot', 'prepool_transforms', 'postpool_transforms'],
                        help="Important params that will appear in .csv result file.",)
    parser.add_argument("--simu_hparams", type=str, nargs='*', default=[],
                        help="Important params that will appear in .csv result file.")
    parser.add_argument("--override", type=str2bool, help='Whether to override results already present in the .csv out file.')

    # Tuning
    parser.add_argument("--combination_size", type=int, default=2)
    parser.add_argument("--mode", type=str, default='benchmark')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.n_tasks = 5
    with open(args.detector_config_file) as f:
        parsed_yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    merge_from_dict(args, parsed_yaml_dict)
    return args


def main(args):
    set_random_seed(args.random_seed)
    save_dir = Path(os.path.join('results', args.exp_name))
    save_dir.mkdir(exist_ok=True, parents=True)
    args.layers = [BACKBONES[args.backbone]().last_layer_name] if args.layers == ['last'] else args.layers

    # logger.info(f"Dropping config file at {save_dir / 'config.json'}")
    # with open(save_dir / 'config.json', 'w') as f:
    #     json.dump(vars(args), f)

    # ================ Prepare data ===================

    feature_dic = defaultdict(dict)
    average_train_features = {}
    std_train_features = {}
    for i, layer in enumerate(args.layers):
        features, _, average_train_features[i], std_train_features[i] = get_test_features(
            args.backbone, args.src_dataset, args.tgt_dataset, args.training, args.model_source, layer
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
    ][0].from_cli_args(args, average_train_features, std_train_features)

    current_detectors = args.outlier_detectors.split('-')

    if args.mode == 'benchmark':

        # ================ Prepare detector ===================

        detectors = []
        for x in current_detectors:
            detector_args = eval(f'args.{x}.current_params')[args.n_shot]  # take default args
            if "args" in inspect.getfullargspec(ALL_DETECTORS[x].__init__).args:
                detector_args['args'] = args
            detectors.append(ALL_DETECTORS[x](**detector_args))
        args.current_sequence = [str(x) for x in detectors]
        outlier_detector = ALL_DETECTORS['aggregator'](detectors)
        metrics = detect_outliers(layers=args.layers,
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
        detectors_to_try = defaultdict(list)
        for x in current_detectors:
            detector_args = eval(f'args.{x}.current_params')[args.n_shot]  # take default args
            params2tune = eval(f'args.{x}.tuning.hparams2tune')
            values2tune = eval(f'args.{x}.tuning.hparam_values')[args.n_shot]
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
            detectors_to_try[x] = [list(x) for x in n_combinations]  # ([d1, d2], [d3, d4], ...)

        # Finally, form the product across detector types
        sequences_to_try = itertools.product(*list(detectors_to_try.values()))

        for detector_sequence in sequences_to_try:

            logger.info(detector_sequence)

            set_random_seed(args.random_seed)

            d_sequence = []
            for d in detector_sequence:
                d_sequence += d
            args.current_sequence = [str(x) for x in d_sequence]
            outlier_detectors = ALL_DETECTORS['aggregator'](d_sequence)

            metrics = detect_outliers(layers=args.layers,
                                      few_shot_classifier=few_shot_classifier,
                                      detector=outlier_detectors,
                                      data_loader=data_loader,
                                      n_way=args.n_way,
                                      n_query=args.n_query,
                                      on_features=True)

            # Saving results
            save_results(args, metrics)


def save_results(args, metrics):
    roc_auc, acc = show_all_metrics_and_plots(args, metrics, title='')
    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    metrics = {'acc': np.round(acc, 4), 'roc_auc': np.round(roc_auc, 4)}
    update_csv(args, metrics, path=res_root / 'out.csv')


def tsne_plot(figures, feat_s, feat_q, support_labels, query_labels, title):

    fig, axis = plt.subplots(nrows=1, ncols=len(feat_s), figsize=(10, 10), squeeze=True, dpi=200)
    if len(feat_s) == 1:
        axis = [axis]

    for layer, ax in zip(feat_s, axis):
        all_feats = torch.cat([feat_s[layer], feat_q[layer]], 0)
        # logger.warning(all_feats.squeeze().numpy()[0])
        embedded_feats = TSNE(n_components=2, init='pca', perplexity=5, learning_rate=50).fit_transform(all_feats.squeeze().numpy())
        all_labels = torch.cat([support_labels, query_labels], 0).numpy()
        ax.scatter(embedded_feats[:, 0], embedded_feats[:, 1], c=all_labels)
    figures[title] = fig


def detect_outliers(layers, few_shot_classifier, detector, data_loader, n_way, n_query, on_features: bool, model=None):

    metrics = defaultdict(list)
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
        if task_id == 0:
            tsne_plot(figures, support_features, query_features, support_labels, query_labels, 'pre_norm')
        support_features, query_features = few_shot_classifier.transform_features(support_features.copy(),
                                                                                  query_features.copy(),
                                                                                  support_labels,
                                                                                  query_labels,
                                                                                  outliers,
                                                                                  figures)
        if task_id == 0:
            tsne_plot(figures, support_features, query_features, support_labels, query_labels, 'post_norm')

        # ====== OOD detection ======
        outlier_scores = []
        for layer in support_features:
            detector.fit(support_features[layer], support_labels)
            raw_scores = torch.from_numpy(detector.decision_function(support_features[layer], query_features[layer]))
            outlier_scores.append((raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min()))  # [?,]
        outlier_scores = torch.stack(outlier_scores, 0).mean(0)

        # ====== Few-shot classifier ======
        all_probs = []
        for layer in support_features:
            all_probs.append(few_shot_classifier(support_features[layer], query_features[layer], support_labels)[1])
        all_probs = torch.stack(all_probs, 0).mean(0)
        predictions = all_probs.argmax(-1)

        # ====== Tracking metrics ======
        acc = (predictions[outliers == 0] == query_labels[outliers == 0]).float().mean()
        fp_rate, tp_rate, thresholds = roc_curve(outliers.numpy(), outlier_scores.numpy())
        precision, recall, _ = precision_recall_curve(outliers.numpy(), outlier_scores.numpy())
        auc = auc_fn(fp_rate, tp_rate)

        for metric_name in ['auc', 'acc']:
            metrics[metric_name].append(eval(metric_name))

    for metric_name in metrics:
        metrics[metric_name] = torch.Tensor(metrics[metric_name])

    # Save figures
    res_root = Path('results') / args.exp_name
    res_root.mkdir(exist_ok=True, parents=True)
    for title, fig in figures.items():
        fig.savefig(res_root / f'{title}.png')

    return metrics


if __name__ == "__main__":
    args = parse_args()
    main(args)
