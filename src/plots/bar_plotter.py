from pathlib import Path
import numpy as np
from itertools import cycle
from collections import defaultdict
from functools import partial
from loguru import logger
from typing import Any, List, Tuple
from .plotter import Plotter
import pandas as pd
import argparse
from .csv_plotter import CSVPlotter, parse_args, pretty, pretty_training, pretty_arch
import matplotlib.pyplot as plt
import os


CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"


barplot_colors = [
    CB91_Pink,
    CB91_Blue,
    CB91_Green,
    CB91_Amber,
    CB91_Purple,
    CB91_Violet,
    "r",
    "m",
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class BarPlotter(CSVPlotter):
    """
    An abstract plotter.
    """

    def plot(self, **kwargs):
        """
        metric_dic[metric_name][method] = {
                                    'x': ndarray [n_points_found],
                                    'y': ndarray [n_points_found],
                                    'pm': Optional[ndarray],
                                   }
        """
        fig, axes = plt.subplots(figsize=(10, 6), ncols=2)

        metric_names = list(self.metric_dic.keys())
        assert (
            len(metric_names) == 2
        ), "Mirror BarPlotter only supports 2 metrics (one on each side)"

        for metric_index, (metric_name, metric_dic) in enumerate(
            self.metric_dic.items()
        ):

            ax = axes[metric_index]

            methods = list(metric_dic.keys())

            # ==== Recover all architectures ======
            labels = metric_dic[methods[0]]["x"]

            # ==== Suu-group by architecture ======
            sorted_items = sorted(metric_dic.items(), key=lambda x: np.mean(x[1]["y"]))
            assert len(metric_dic) == 2, "Currently only support 2 methods at a time."
            bottoms = defaultdict(float)
            for method_index, (method, method_dic) in enumerate(sorted_items):

                grouped_items = defaultdict(list)
                for arch, result in zip(method_dic["x"], method_dic["y"]):
                    grouped_items[pretty_arch[arch]].append(
                        (pretty_training[arch], result)
                    )

                current_height = 0
                yticks = []
                yticks_labels = []
                for arch in grouped_items:
                    for training_index, (training, value) in enumerate(
                        grouped_items[arch]
                    ):
                        value = value - bottoms[training]
                        ax.barh(
                            [current_height],
                            [value],
                            edgecolor="white",
                            color=barplot_colors[methods.index(method)],
                            height=0.01,
                            left=[bottoms[training]],
                            label=r"Strong baseline" if method_index == 0 else method,
                        )
                        bottoms[training] += value
                        if method_index == 1:
                            ax.text(
                                bottoms[training] + 0.01,
                                current_height - 0.0001,
                                rf"$\mathbf{{+{np.round(100 * value, 1)}}}$",
                                color=barplot_colors[methods.index(method)],
                                va="center",
                                ha="right" if metric_index == 0 else "left",
                                fontsize=12,
                            )
                        if (metric_index == 1) and (
                            training_index == len(grouped_items[arch]) - 1
                        ):
                            ax.text(
                                0.24,
                                current_height + 0.02,
                                arch,
                                va="center",
                                ha="center",
                                fontsize=11,
                            )
                        yticks_labels.append(rf"{training}")
                        yticks.append(current_height)
                        current_height += 0.02
                    current_height += 0.03

            current_height -= 0.03
            ax.set_xticks(np.arange(4, 9) / 10)
            ax.set_xlim(0.4, 0.85)
            ax.set_xticklabels([rf"${10 * x}$" for x in range(4, 9)])
            ax.set_title(rf"\textbf{{{pretty[metric_name]}}}", fontsize=12)

            if metric_index == 0:
                ax.set(yticks=yticks, yticklabels=["" for x in labels])
                ax.yaxis.tick_right()
            else:
                ax.set(yticks=yticks)
                ax.set_yticklabels(
                    yticks_labels,
                    ha="center",
                    va="center",
                    position=(-0.32, 0),
                    fontsize=12,
                )
                # ax.yaxis.tick_left()
            plt.subplots_adjust(wspace=0.7)
            ax.set_ylim(-0.03, current_height)

            # Hide the right and top spines
            if metric_index == 1:
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

                # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position("left")
                ax.xaxis.set_ticks_position("bottom")
            else:
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)

                # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position("right")
                ax.xaxis.set_ticks_position("bottom")

            ax.xaxis.set_tick_params(labelsize=12)

            if metric_index == 1:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                fig.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="center",
                    bbox_to_anchor=[0.53, 0.97],
                    fontsize=12,  # bottom-right
                    ncol=2,
                    frameon=False,  # don't put a frame)
                )
            if metric_index == 0:
                # If you have positive numbers and want to invert the x-axis of the left plot
                ax.invert_xaxis()

                # # To show data from highest to lowest
                # ax.invert_yaxis()
        # plt.subplots_adjust(wspace=None)
        os.makedirs(self.out_dir, exist_ok=True)
        # fig.tight_layout()
        fig.savefig(self.out_dir / f"barplot.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args()
    plt.style.use("classic")
    if args.latex:
        logger.info("Activating latex")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
            }
        )
    plotter = BarPlotter()
    plotter.fit(**vars(args))
    plotter.plot()
