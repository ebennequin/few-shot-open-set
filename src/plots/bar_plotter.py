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
from .csv_plotter import CSVPlotter, parse_args, pretty
import matplotlib.pyplot as plt
import os
from src.inference import str2bool



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
        fig, axes = plt.subplots(figsize=(10, 7), ncols=2)

        metric_names = list(self.metric_dic.keys())
        assert len(metric_names) == 2, 'Mirror BarPlotter only supports 2 metrics (one on each side)'

        for i, (metric_name, metric_dic) in enumerate(self.metric_dic.items()):

            ax = axes[i]

            methods = list(metric_dic.keys())
            labels = metric_dic[methods[0]]["x"]
            bottoms = defaultdict(float)
            sorted_items = sorted(metric_dic.items(), key=lambda x: np.mean(x[1]["y"]))
            assert len(metric_dic) == 2, "Currently only support 2 methods at a time."
            for index, (method, method_dic) in enumerate(sorted_items):
                sorted_tuples = sorted(
                    zip(method_dic["x"], method_dic["y"]), key=lambda x: x[0]
                )
                for j, (label, value) in enumerate(sorted_tuples):
                    value = value - bottoms[label]
                    ax.barh(
                        [j],
                        [value],
                        edgecolor="white",
                        color=barplot_colors[methods.index(method)],
                        height=0.15,
                        left=[bottoms[label]],
                        label=r"Strong baseline" if index == 0 else method,
                    )
                    bottoms[label] += value
                    if index == 1:
                        ax.text(bottoms[label] + 0.01, j, rf"$\mathbf{{+{np.round(100 * value, 1)}}}$",
                                color=barplot_colors[methods.index(method)], va='center', ha='right' if i == 0 else 'left', fontsize=15)
            ax.set_xticks(np.arange(4, 9) / 10)
            ax.set_xlim(0.4, 0.85)
            ax.set_xticklabels([rf"${10 * x}$" for x in range(4, 9)])
            ax.set_title(rf"\textbf{{{pretty[metric_name]}}}", fontsize=15)

            if i == 0:
                ax.set(yticks=range(len(labels)), yticklabels=["" for x in labels])
                ax.yaxis.tick_right()
            else:
                ax.set(yticks=range(len(labels)))
                ax.set_yticklabels([pretty[x] for x in labels],
                                   ha='center', va='center', position=(-0.32, 0), fontsize=12)
                # ax.yaxis.tick_left()
            plt.subplots_adjust(wspace=0.7)
            ax.set_ylim(-0.5, len(labels) - 0.5)

            # Hide the right and top spines
            if i == 1:
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

            ax.xaxis.set_tick_params(labelsize=15)

            if i == 1:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                fig.legend(by_label.values(), by_label.keys(),
                           loc="center",
                           bbox_to_anchor=[0.5, 1.01],  # bottom-right
                           ncol=2,
                           frameon=False,  # don't put a frame)
                           )
            if i == 0:
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
        # for Palatino and other serif fonts use:
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Palatino"],
        # })
    plotter = BarPlotter()
    plotter.fit(**vars(args))
    plotter.plot()
