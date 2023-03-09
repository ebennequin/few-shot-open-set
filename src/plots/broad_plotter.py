from textwrap import fill
from typing import List, Optional

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import typer

from matplotlib import pyplot as plt

from src.plots.bar_plotter import barplot_colors
from src.plots.csv_plotter import pretty
from src.plots.queries_plotter import (
    DEFAULT_METHODS_FOR_QUERY_PLOTS,
    make_output_dir,
    curate_results,
    create_canvas,
)


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

    #  ===== Get all results in one dataframe =====
    p = Path("results") / exp_name
    metrics_std = [metric.replace("mean", "std") for metric in metrics]
    all_results = curate_results(
        pd.concat(
            [
                pd.read_csv(
                    p
                    / str(is_broad).lower()
                    / "mini_imagenet-->mini_imagenet(test)/resnet12/feat"
                    / str(shot)
                    / "out.csv"
                )
                .assign(broad=is_broad)
                .iloc[:6]
                for is_broad in [True, False]
                for shot in [1, 5]
            ]
        ),
        metrics=metrics,
        methods=selected_methods,
        keep_columns=["src_dataset", "n_shot", "method_name", "broad"],
    )

    # ===== Parameterize plotlib =====
    sequential_colors = barplot_colors

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )

    ylim = {
        1: {
            "mean_acc": (60, 78),
            "mean_rocauc": (50, 78),
        },
        5: {
            "mean_acc": (70, 88),
            "mean_rocauc": (50, 88),
        },
    }

    # ===== Plot =====
    dataset = "mini_imagenet"

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
                hue="broad",
                palette=sequential_colors,
                alpha=0.6,
            )

            if plot_std:
                num_hues = len(results.broad.unique())
                for (hue, df_hue), dogde_dist in zip(
                    results.groupby("broad"),
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
                        capsize=1.5,
                        ls="",
                        lw=0.5,
                        color="#91017a",
                    )

            axes[n_shot][metric].set_ylim(*ylim[n_shot][metric])
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
                    axes[n_shot][metric].yaxis.set_label_coords(-0.152, 0.89)
                else:
                    axes[n_shot][metric].yaxis.set_label_coords(-0.152, 0.0)
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
        handles[:2],
        [
            fill(label, 11)
            for label in ["...from 5 classes", "...from all remaining classes"]
        ],
        loc=(0.465, 0.32),
        title=fill("Open-set queries sampled...", 12),
        frameon=False,
        handlelength=1.0,
        labelspacing=0.8,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.265)
    plt.show()
    plt.savefig(output_dir / f"{dataset}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(main)
