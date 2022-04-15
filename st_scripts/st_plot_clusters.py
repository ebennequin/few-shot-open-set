import json
import pickle
from statistics import mean
from typing import Dict, Tuple, List

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.manifold import TSNE
import streamlit as st
from torch import tensor

from src.utils.data_fetchers import get_test_features

IMAGENET_WORDS_PATH = Path("data/mini_imagenet/specs/words.txt")
DATA_ROOT = Path("data")
DATASETS_LIST = [
    "mini_imagenet",
    "imagenet",
    "tiered_imagenet",
    "cub",
    "aircraft",
    "imagenet_val",
    "mini_imagenet_bis",
]
MODELS_LIST = [
    "resnet12",
    "wrn2810",
    "deit_tiny_patch16_224",
    "ssl_resnext101_32x16d",
    "vit_base_patch16_224_in21k",
]


def get_class_names(dataset, key):
    selected_specs_file = st.selectbox(
        "Specs",
        [
            path
            for path in (Path("data") / dataset / "specs").glob("*")
            if path.suffix in {".json", ".csv"}
        ],
        format_func=lambda path: str(path)[len("data/") :],
        key=key,
    )
    if selected_specs_file.suffix == ".json":
        with open(selected_specs_file, "r") as file:
            return json.load(file)["class_names"]
    elif selected_specs_file.suffix == ".csv":
        synset_codes = pd.read_csv(selected_specs_file).class_name.unique()
        words = {}
        with open(IMAGENET_WORDS_PATH, "r") as file:
            for line in file:
                synset, word = line.rstrip().split("\t")
                words[synset] = word.split(",")[0]
        return [words[synset] for synset in synset_codes]
    else:
        raise ValueError


def select_classes(class_names, key):
    container = st.container()
    select_all = st.checkbox("All classes", value=True)

    if select_all:
        selected_options = container.multiselect(
            "Classes", class_names, default=class_names, key=key
        )
    else:
        selected_options = container.multiselect("Classes", class_names, key=key)

    return selected_options


def map_label(features: pd.DataFrame, class_names: List[str]):
    features.label = features.label.apply(lambda label: class_names[label])
    return features


def normalize(features: Dict[int, ndarray]) -> Dict[int, ndarray]:
    return {k: sklearn.preprocessing.normalize(v, axis=1) for k, v in features.items()}


def compute_2d_features(features: Dict[int, ndarray]) -> pd.DataFrame:
    reduced_features = TSNE(n_components=2, init="pca").fit_transform(
        np.concatenate(list(normalize(features).values()))
    )

    return pd.concat(
        [pd.DataFrame({"label": len(v) * [int(k)]}) for k, v in features.items()]
    ).assign(x=reduced_features[:, 0], y=reduced_features[:, 1])


def compute_or_retrieve_2d_features(features_path, features=None):
    # TODO : if features change or the method changes, this will not recompute 2d features
    reduced_features_file_name = features_path.with_name(f"{features_path.stem}_2d.csv")
    if reduced_features_file_name.is_file():
        reduced_features = pd.read_csv(reduced_features_file_name)
    else:
        st.write("Computing TSNE...")
        if features is None:
            with open(features_path, "rb") as stream:
                features = pickle.load(stream)
        reduced_features = compute_2d_features(features)
        reduced_features.to_csv(reduced_features_file_name, index=False)
    return reduced_features


def plot_2d_features(features, classes_to_plot):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for label, group in features.loc[lambda df: df.label.isin(classes_to_plot)].groupby(
        "label"
    ):
        ax.scatter(
            group.x,
            group.y,
            s=0.5,
            marker="o",
            label=label,
        )
    if len(classes_to_plot) < 10:
        plt.legend(loc="best", ncol=2, fontsize="xx-small")
    st.write(fig)


def compute_statistics(features, use_normalize=True):
    if use_normalize:
        features = normalize(features)

    sigma_within = np.mean([np.linalg.norm(v.std(axis=0)) for k, v in features.items()])

    sigma_between = np.linalg.norm(
        np.stack([v.mean(axis=0) for v in features.values()]).std(axis=0)
    )

    return sigma_within, sigma_between


def compute_mean_auroc(features, use_normalize=True):
    if use_normalize:
        features = normalize(features)

    # features = {k: v for k,v in features.items() if k<10}
    # st.write("coucou")

    aurocs = []
    for label in features.keys():
        ground_truth = []
        predictions = []
        centroid = features[label].mean(axis=0)
        for second_label, v in features.items():
            ground_truth += len(v) * [0 if label == second_label else 1]
            distances = np.linalg.norm(v - centroid, axis=1)
            predictions += distances.tolist()
        auroc = sklearn.metrics.roc_auc_score(ground_truth, predictions)
        aurocs.append(auroc)

    return mean(aurocs)


def print_clustering_statistics_for_all_features(features_paths_list):
    all_stats = []
    for feature_path in features_paths_list:
        statistics = compute_statistics(feature_path)
        all_stats.append(
            {
                "backbone": feature_path.stem,
                "split": feature_path.parent.name,
                "sigma_within": statistics[0],
                "sigma_between": statistics[1],
            }
        )

    stats_df = pd.DataFrame(all_stats).assign(
        ratio=lambda df: df.sigma_within / df.sigma_between
    )

    st.title("Clustering statistics")
    st.write(stats_df)


def plot_clusters(key):
    selectors_col, plot_col = st.columns([2, 3])
    with selectors_col:
        st.title("Select stuff")
        source_dataset = st.selectbox(
            "Source dataset",
            DATASETS_LIST,
            key=key,
        )
        target_dataset = st.selectbox(
            "Target dataset",
            DATASETS_LIST,
            key=key,
        )
        backbone = st.selectbox(
            "Model",
            MODELS_LIST,
            key=key,
        )
        model_source = st.selectbox(
            "Model source",
            ["url", "feat"],
            key=key,
        )
        layer = st.selectbox(
            "Layer",
            ["4_4", "last", "4_3", "last_cls"],
            key=key,
        )
        split = st.selectbox(
            "Split",
            ["train", "val", "test"],
            key=key,
        )
        try:
            pickle_basename = (
                f"{backbone}_{source_dataset}_{model_source}_{layer}.pickle"
            )
            test_features_path = (
                DATA_ROOT
                / "features"
                / source_dataset
                / target_dataset
                / split
                / "standard"
                / pickle_basename
            )

            with open(test_features_path, "rb") as stream:
                test_features = pickle.load(stream)
                # WRN returns a weird shape for the features (n_instances, n_channels, 1, 1)
                test_features = {
                    k: v.reshape(v.shape[0], -1) for k, v in test_features.items()
                }
        except FileNotFoundError:
            st.write("No features for this combination")
            return

        mean_auroc = compute_mean_auroc(features=test_features)
        st.write(mean_auroc)
        use_class_names = st.checkbox(
            "Use class names",
            key=key,
        )
        if use_class_names:
            class_names = get_class_names(target_dataset, key)
            selected_classes = select_classes(class_names, key)

        sigma_within, sig_between = compute_statistics(test_features)
        st.write(
            f"Test set stats: sigma_within={sigma_within}, sigma_between={sig_between}, ratio={sigma_within / sig_between}"
        )

        # sigma_within, sig_between = compute_statistics(train_features)
        # st.write(f"Train set stats: sigma_within={sigma_within}, sigma_between={sig_between}, ratio={sigma_within / sig_between}")

    with plot_col:
        reduced_features = compute_or_retrieve_2d_features(
            test_features_path, test_features
        )
        st.title("Look at all those clusters")
        if use_class_names:
            reduced_features = map_label(reduced_features, class_names)
        else:
            selected_classes = reduced_features.label.unique()
        plot_2d_features(reduced_features, selected_classes)


st.set_page_config(page_title="Look at clusters", layout="wide")
plot_clusters(1)
