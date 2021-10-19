from typing import List

import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st

from st_displayers import display_fn
from dvc_getters import (
    read_params,
    read_metrics,
    get_params,
    get_all_exps,
    read_csv,
    get_image,
)
from st_constants import METRICS_DIR
from st_utils import condense_results, st_params_selector, st_commits_selector


def st_dig():

    all_dvc_exps = get_all_exps()
    all_params = get_params(all_dvc_exps.index.to_list())
    st.title("Dig an experiment")

    selected_params = st_params_selector(all_params)

    selected_commits = st_commits_selector(all_dvc_exps)

    column_left, column_right = st.columns(2)
    if len(selected_commits) > 0:
        with column_left:
            accuracies_left = dig_one(
                1, all_dvc_exps, selected_commits, selected_params, all_params
            )
        with column_right:
            accuracies_right = dig_one(
                2, all_dvc_exps, selected_commits, selected_params, all_params
            )

        st.title("Compare performances")

        accuracies_compared = pd.DataFrame(
            {"right_acc": accuracies_left, "left_acc": accuracies_right}
        )

        class_column, task_column = st.columns(2)
        with class_column:
            st.header("On classes")
            st.write(
                accuracies_compared.groupby("true_label")
                .mean()
                .assign(diff=lambda df: abs(df.left_acc - df.right_acc))
                .sort_values("diff", ascending=False)
            )

        with task_column:
            st.header("On tasks")
            st.write(
                accuracies_compared.groupby("task_id")
                .mean()
                .assign(diff=lambda df: abs(df.left_acc - df.right_acc))
                .sort_values("diff", ascending=False)
            )


def dig_one(
    key: int,
    all_dvc_exps: pd.DataFrame,
    selected_commits: List[str],
    selected_params: List[str],
    all_params: pd.DataFrame,
) -> pd.Series:
    # TODO: ce serait cool de pouvoir cacher ça, j'ai l'impression que ça reloade quand je clique sur un expander
    selected_exp = st.selectbox(
        label="Select an experiment",
        options=all_dvc_exps.loc[
            lambda df: df.parent_hash.isin(selected_commits)
        ].index.to_list(),
        format_func=lambda x: display_fn(x, all_dvc_exps, all_params, selected_params),
        key=-key,
    )

    st.expander("Metrics").write(read_metrics(selected_exp))
    st.expander("Params").write(read_params(selected_exp))

    with st.expander("Heatmaps"):
        st.image(
            get_image(
                METRICS_DIR / "training_classes_biconfusion.png",
                selected_exp,
            ),
            caption="Training classes biconfusion",
        )
        st.image(
            get_image(
                METRICS_DIR / "training_classes_sampled_together.png",
                selected_exp,
            ),
            caption="Training classes cosampling",
        )

    with st.expander(
        "Other plots"
    ):  # TODO: ces plots sont trop larges, comment les reduire ?
        fig, ax = plt.subplots()
        read_csv(METRICS_DIR / "task_performances.csv", selected_exp).plot.scatter(
            x="variance",
            y="accuracy",
            ax=ax,
            title="Accuracy depending on intra-task distance on test set",
        )
        st.pyplot(fig)

        fig, ax = plt.subplots()
        read_csv(
            METRICS_DIR / "intra_training_task_distances.csv", selected_exp
        ).assign(
            smooth=lambda df: df.median_distance.rolling(500).mean()
        ).smooth.plot.line(
            ax=ax, title="Evolution of intra-task distances during training (smooth)"
        )
        st.pyplot(fig)

    results = read_csv(METRICS_DIR / "raw_results.csv", selected_exp)

    return condense_results(results)
