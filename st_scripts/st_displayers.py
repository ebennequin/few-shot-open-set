from typing import List

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt


def plot_all_bars(metrics_df: pd.DataFrame):
    bar_plots_columns = st.columns(5)

    with bar_plots_columns[0]:
        bar_plot(metrics_df.accuracy, title="TOP 1 overall accuracy")

    for i, quartile in enumerate(["first", "second", "third", "fourth"]):
        with bar_plots_columns[i + 1]:
            bar_plot(
                metrics_df[f"{quartile}_quartile_acc"],
                title=f"TOP 1 accuracy on {quartile} quartile",
            )


def bar_plot(accuracy: pd.Series, title: str):
    bottom = 0.9 * accuracy.min()

    fig, ax = plt.subplots()
    (accuracy - bottom).plot.bar(
        ax=ax,
        title=title,
        fontsize=15,
        alpha=0.5,
        grid=True,
        bottom=bottom,
    )
    # plt.legend(loc="lower left")
    st.pyplot(fig)


def display_fn(
    x: str, exps_df: pd.DataFrame, all_params: pd.DataFrame, selected_params: List[str]
) -> str:
    to_display = f"{exps_df.parent_hash[x][:7]} - {x}"
    for param in selected_params:
        to_display += f" - {param} {all_params[param][x]}"
    return to_display
