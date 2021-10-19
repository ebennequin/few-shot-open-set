import pandas as pd
import json

from hashlib import sha1
from typing import List

import streamlit as st

from st_constants import DEFAULT_DISPLAYED_PARAMS
from git_helpers import get_commit_message


def aggregate_over_seeds(
    metrics_df: pd.DataFrame, params: List[str], seed_column_name: str = "train.seed"
) -> pd.DataFrame:
    return (
        metrics_df.groupby([param for param in params if param != seed_column_name])
        .aggregate(
            {
                **{
                    metric: "mean"
                    for metric in metrics_df.columns
                    if metric not in params
                },
                seed_column_name: "count",
            }
        )
        .rename(columns={seed_column_name: "n_seeds"})
    )


@st.cache
def condense_results(results: pd.DataFrame) -> pd.Series:
    return (
        results.sort_values("score", ascending=False)
        .drop_duplicates(["task_id", "image_id"])
        .sort_values(["task_id", "image_id"])
        .reset_index(drop=True)
        .assign(accuracy=lambda df: df.true_label == df.predicted_label)
        .groupby(["task_id", "true_label"])
        .accuracy.mean()
    )


def get_hash_from_list(list_to_hash: List) -> str:
    return sha1(json.dumps(sorted(list_to_hash)).encode()).hexdigest()


def st_params_selector(all_params: pd.DataFrame) -> List[str]:
    return st.multiselect(
        label="Select displayed params",
        options=all_params.columns.to_list(),
        default=all_params.filter(regex=DEFAULT_DISPLAYED_PARAMS).columns.to_list(),
    )


def st_commits_selector(all_dvc_exps: pd.DataFrame) -> List[str]:
    selected_commits = st.multiselect(
        "Filter pickable experiments by commit",
        options=all_dvc_exps.parent_hash.unique(),
        format_func=lambda x: x[:7]
        + f": {get_commit_message(x)} ({all_dvc_exps.parent_hash.value_counts().loc[x]} experiments)",
    )
    return selected_commits
