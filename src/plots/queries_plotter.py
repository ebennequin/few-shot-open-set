#%%
from typing import List, Optional

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import typer

from matplotlib import pyplot as plt

from src.plots.csv_plotter import pretty

DEFAULT_METHODS_FOR_QUERY_PLOTS = [
    "$k$-NN",
    "LapShot",
    "BDCSPN",
    r"\textsc{PT-MAP}",
    "\\textsc{TIM}",
    "\\textsc{OSLO}",
]


def make_output_dir(exp_name: str):
    output_dir = Path("plots") / exp_name
    output_dir.mkdir(exist_ok=True)
    return output_dir


def curate_results(
    results: pd.DataFrame,
    metrics: List[str],
    methods: List[str],
    keep_columns: List[str],
):
    metrics_std = [metric.replace("mean", "std") for metric in metrics]
    results = (
        results.assign(
            method_name=lambda df: (
                df.feature_detector.where(df.feature_detector != "None", df.classifier)
            )
            .str.split("(", 1, expand=True)[0]
            .map(pretty),
        )[[*keep_columns, *metrics, *metrics_std]]
        .loc[lambda df: df.method_name.isin(methods)]
        .sort_values(
            ["method_name"],
            key=lambda col: col.map(
                {method: rank for rank, method in enumerate(methods)}
            ),
        )
    )

    for metric in metrics + metrics_std:
        results = results.assign(**{metric: lambda df: 100 * df[metric]})

    return results


def create_canvas(metrics, shots):
    axes = {n_shot: {} for n_shot in shots}
    fig, (
        (
            axes[shots[0]][metrics[0]],
            axes[shots[1]][metrics[0]],
        ),
        (
            axes[shots[0]][metrics[1]],
            axes[shots[1]][metrics[1]],
        ),
    ) = plt.subplots(figsize=(14, 4), nrows=2, ncols=2)

    return fig, axes


def main(
    exp_name: str,
    metrics: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    plot_std: bool = True,
):
    metrics = metrics if metrics else ["mean_acc", "mean_rocauc"]
    shots = [1, 5]
    selected_methods = methods if methods else DEFAULT_METHODS_FOR_QUERY_PLOTS
    output_dir = make_output_dir(exp_name)

    assert len(metrics) == 2, "Can only handle two metrics for this plot."
    assert len(shots) == 2, "Can only handle two different n_shot for this plot."

    #  ===== Recover all csv result files =====
    p = Path("results") / exp_name
    csv_files = list(
        p.glob("**/*.csv")
    )  # every csv file represents the  summary of an experiment
    assert len(csv_files)

    #  ===== Get all results in one dataframe =====
    all_results = curate_results(
        pd.concat([pd.read_csv(file) for file in csv_files]).assign(
            n_query=lambda df: df.n_id_query,
        ),
        metrics=metrics,
        methods=selected_methods,
        keep_columns=["src_dataset", "n_shot", "method_name", "n_query"],
    )

    # ===== Parameterize plotlib =====
    sequential_colors = sns.color_palette("RdPu", 4)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )

    ylim = {
        "mini_imagenet": (50.0, 85.0),
        "tiered_imagenet": (55.0, 89.0),
    }

    # ===== Plot =====
    for dataset in all_results.src_dataset.unique():

        fig, axes = create_canvas(metrics, shots)

        for shot_id, n_shot in enumerate(shots):

            for metric_id, metric in enumerate(metrics):
                results = all_results.loc[lambda df: df.src_dataset == dataset].loc[
                    lambda df: df.n_shot == n_shot
                ]
                sns.barplot(
                    ax=axes[n_shot][metric],
                    data=results,
                    x="method_name",
                    y=metric,
                    hue="n_query",
                    palette=sequential_colors,
                    alpha=0.6,
                )

                if plot_std:
                    num_hues = len(results.n_query.unique())
                    for (hue, df_hue), dogde_dist in zip(
                        results.groupby("n_query"),
                        np.linspace(-0.4, 0.4, 2 * num_hues + 1)[1::2],
                    ):
                        bars = axes[n_shot][metric].errorbar(
                            data=df_hue,
                            x="method_name",
                            y=metric,
                            yerr=metric.replace("mean", "std"),
                        )
                        xys = bars.lines[0].get_xydata()
                        bars.remove()
                        axes[n_shot][metric].errorbar(
                            data=df_hue,
                            x=xys[:, 0] + dogde_dist,
                            y=metric,
                            yerr=metric.replace("mean", "std"),
                            capsize=1.0,
                            ls="",
                            lw=0.5,
                            color="#91017a",
                        )

                axes[n_shot][metric].set_ylim(*ylim[dataset])
                axes[n_shot][metric].set(xlabel="")

                if shot_id == 0:
                    axes[n_shot][metric].yaxis.tick_right()
                    axes[n_shot][metric].yaxis.set_label_position("right")
                    axes[n_shot][metric].spines.left.set_visible(False)
                    axes[n_shot][metric].invert_xaxis()
                    axes[n_shot][metric].set(ylabel="")
                else:
                    axes[n_shot][metric].spines.right.set_visible(False)
                    axes[n_shot][metric].set_ylabel(
                        rf"\textbf{{{pretty[metric]}}}",
                        rotation=0,
                        fontsize=12,
                    )
                    if metric_id == 0:
                        axes[n_shot][metric].yaxis.set_label_coords(-0.11, 0.89)
                    else:
                        axes[n_shot][metric].yaxis.set_label_coords(-0.11, 0.0)
                if metric_id == 0:
                    if shot_id == 0:
                        axes[n_shot][metric].set_title(
                            rf"\textbf{{{n_shot}-shot}}",
                            x=-0.025,
                            y=-0.33,
                            rotation=90,
                            fontsize=13,
                        )
                    else:
                        axes[n_shot][metric].set_title(
                            rf"\textbf{{{n_shot}-shot}}",
                            x=1.025,
                            y=-0.33,
                            rotation=-90,
                            fontsize=13,
                        )

                    axes[n_shot][metric].spines.top.set_visible(False)
                    axes[n_shot][metric].tick_params(axis="x", pad=9, bottom=False)
                else:
                    axes[n_shot][metric].spines.bottom.set_visible(False)
                    axes[n_shot][metric].invert_yaxis()
                    axes[n_shot][metric].set(xticks=[])

                axes[n_shot][metric].legend().remove()

        handles, labels = axes[shots[1]][metrics[1]].get_legend_handles_labels()
        fig.legend(
            handles[:4],
            labels[:4],
            loc=(0.481, 0.38),
            title=r"$N_Q$",
            frameon=False,
            handlelength=1.0,
            labelspacing=0.8,
        )
        plt.subplots_adjust(wspace=0.1, hspace=1.0)
        plt.tight_layout()

        plt.savefig(output_dir / f"{dataset}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(main)
