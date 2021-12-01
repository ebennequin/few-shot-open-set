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
st.title("Clusters")


FEATURES_PATHS = list(Path("data/features").glob("**/*.pickle"))


def get_class_names(specs_file):
    if specs_file.suffix == ".json":
        with open(specs_file, "r") as file:
            return json.load(file)["class_names"]
    else:
        raise ValueError


def map_label(features, class_names):
    return {class_names[k]: v for k, v in features.items()}


def normalize(features: Dict[int, ndarray]) -> Dict[int, ndarray]:
    return {k: sklearn.preprocessing.normalize(v, axis=0) for k, v in features.items()}


def pack_features(features: Dict[int, ndarray]) -> Tuple[ndarray, Dict[int, List[int]]]:
    return (
        np.concatenate(list(features.values())),
        {k: list(range(len(v))) for k, v in features},
    )


@st.cache()
def compute_2d_features(features: Dict[int, ndarray]) -> pd.DataFrame:
    reduced_features = TSNE(n_components=2, init="random").fit_transform(
        np.concatenate(list(features.values()))
    )

    return pd.concat(
        [pd.DataFrame({"label": len(v) * [k]}) for k, v in features.items()]
    ).assign(x=reduced_features[:, 0], y=reduced_features[:, 1])


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


def plot_clusters(key):
    selected_features_path = st.selectbox(
        "Features",
        FEATURES_PATHS,
        format_func=lambda path: str(path)[len("data/features/") :],
        key=key,
    )
    selected_specs_file = st.selectbox(
        "Specs",
        list(Path("data/cifar100/specs").glob("*.json"))
        + list(Path("data/mini_imagenet/specs").glob("*.csv")),
        format_func=lambda path: str(path)[len("data/") :],
        key=key,
    )

    with open(selected_features_path, "rb") as stream:
        selected_features = pickle.load(stream)
    classes = get_class_names(selected_specs_file)

    selected_classes = st.multiselect("Classes", classes, key=key)

    reduced_features = compute_2d_features(
        normalize(map_label(selected_features, get_class_names(selected_specs_file)))
    )

    plot_2d_features(reduced_features, selected_classes)


col1, col2 = st.columns(2)

with col1:
    plot_clusters(1)

with col2:
    plot_clusters(2)
