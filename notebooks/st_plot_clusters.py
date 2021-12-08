import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import sklearn
from matplotlib import pyplot as plt
import streamlit as st
from numpy import ndarray
from sklearn.manifold import TSNE

st.set_page_config(page_title="Look at clusters", layout="wide")


FEATURES_ROOT = Path("data/features")


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
        with open("data/mini_imagenet/specs/words.txt", "r") as file:
            for line in file:
                synset, word = line.rstrip().split("\t")
                words[synset] = word.split(",")[0]
        return [words[synset] for synset in synset_codes]
    else:
        raise ValueError


def select_classes(class_names, key):
    container = st.container()
    select_all = st.checkbox("All classes")

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
        [pd.DataFrame({"label": len(v) * [k]}) for k, v in features.items()]
    ).assign(x=reduced_features[:, 0], y=reduced_features[:, 1])


def compute_or_retrieve_2d_features(features_path):
    # TODO : if features change or the method changes, this will not recompute 2d features
    reduced_features_file_name = features_path.with_name(f"{features_path.stem}_2d.csv")
    if reduced_features_file_name.is_file():
        reduced_features = pd.read_csv(reduced_features_file_name)
    else:
        st.write("Computing TSNE...")
        with open(features_path, "rb") as stream:
            selected_features = pickle.load(stream)
        reduced_features = compute_2d_features(selected_features)
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
    plt.legend(loc="best", ncol=2, fontsize="xx-small")
    st.write(fig)


def compute_statistics(features_path):
    with open(features_path, "rb") as stream:
        features = normalize(pickle.load(stream))

    sigma_within = np.mean([np.linalg.norm(v.std(axis=0)) for k, v in features.items()])

    sigma_between = np.linalg.norm(
        np.stack([v.mean(axis=0) for v in features.values()]).std(axis=0)
    )

    return sigma_within, sigma_between


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
        selected_dataset = st.selectbox(
            "Dataset",
            list(FEATURES_ROOT.glob("*")),
            format_func=lambda path: path.name,
            key=key,
        )
        feature_paths_for_selected_dataset = list(selected_dataset.glob("**/*.pickle"))
        selected_features_path = st.selectbox(
            "Features",
            feature_paths_for_selected_dataset,
            # format_func=lambda path: f"{path.parent.name}/{path.name}",
            key=key,
        )

        class_names = get_class_names(selected_dataset.name, key)
        selected_classes = select_classes(class_names, key)

        reduced_features = compute_or_retrieve_2d_features(selected_features_path)
        reduced_features = map_label(reduced_features, class_names)

        print_clustering_statistics_for_all_features(feature_paths_for_selected_dataset)

    with plot_col:
        st.title("Look at all those clusters")
        plot_2d_features(reduced_features, selected_classes)


plot_clusters(1)
