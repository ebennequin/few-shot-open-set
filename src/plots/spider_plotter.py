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


class SpiderPlotter(CSVPlotter):
    """
    An abstract plotter.
    """
    def plot(self, **kwargs):
        '''
        metric_dic[metric_name][method] = {
                                    'x': ndarray [n_points_found],
                                    'y': ndarray [n_points_found],
                                    'pm': Optional[ndarray],
                                   }
        '''
        assert len(self.metric_dic)

        fig, axes = plt.subplots(nrows=1, ncols=len(self.metric_dic),
                                 subplot_kw=dict(projection="polar"),
                                 figsize=(40, 13), squeeze=True)

        for i, metric_name in enumerate(self.metric_dic):
            ax = axes[i]

            BG_WHITE = "#fbf9f4"
            BLUE = "#2a475e"
            GREY70 = "#b3b3b3"
            GREY_LIGHT = "#f2efe8"

            min_val = min([min(arr['y']) for arr in self.metric_dic[metric_name].values()])
            max_val = max([max(arr['y']) for arr in self.metric_dic[metric_name].values()])
            max_avg_perf = max([np.mean(arr['y']) for arr in self.metric_dic[metric_name].values()])
            l = np.linspace(min_val - 0.02, max_val + 0.02, 5)
            angle = 2.2

            first_method = list(self.metric_dic[metric_name].keys())[0]
            x_names = self.metric_dic[metric_name][first_method]['x']
            VARIABLES = x_names
            VARIABLES_N = len(VARIABLES)

            # The angles at which the values of the numeric variables are placed
            ANGLES = [n / VARIABLES_N * 2 * np.pi for n in range(VARIABLES_N)]
            ANGLES += ANGLES[:1]

            # Padding used to customize the location of the tick labels
            # X_VERTICAL_TICK_PADDING = 5
            # X_HORIZONTAL_TICK_PADDING = 50

            # Angle values going from 0 to 2*pi
            HANGLES = np.linspace(0, 2 * np.pi)

            # Used for the equivalent of horizontal lines in cartesian coordinates plots
            # The last one is also used to add a fill which acts a background color.
            H = [np.ones(len(HANGLES)) * li for li in l]

            # fig.patch.set_facecolor(BG_WHITE)
            # ax.set_facecolor(BG_WHITE)

            # Rotate the "" 0 degrees on top.
            # There it where the first variable, avg_bill_length, will go.
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Setting lower limit to negative value reduces overlap
            # for values that are 0 (the minimums)
            ax.set_ylim(l[0] - 0.02, l[-1])

            # Set values for the angular axis (x)
            ax.set_xticks(ANGLES[:-1])
            ax.set_xticklabels(VARIABLES, size=17, y=-0.1)

            # Remove lines for radial axis (y)
            ax.set_yticks([])
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            # Remove spines
            ax.spines["start"].set_color("none")
            ax.spines["polar"].set_color("none")

            # Add custom lines for radial axis (y) at 0, 0.5 and 1.
            _ = [ax.plot(HANGLES, h, ls=(0, (6, 6)), c=GREY70) for h in H]

            # Add levels -----------------------------------------------------
            # These labels indicate the values of the radial axis
            PAD = 0.005
            size = 15
            _ = [ax.text(angle, li + PAD, f"{int(li * 100)}", size=size) for li in l]

            # Now fill the area of the circle with radius 1.
            # This create the effect of gray background.
            ax.fill(HANGLES, H[-1], GREY_LIGHT)

            # Fill lines and dots --------------------------------------------
            for idx, (method, method_result) in enumerate(self.metric_dic[metric_name].items()):
                assert method_result['x'] == x_names, (method, method_result['x'], x_names)
                values = method_result['y']
                perf = np.mean(values)
                values += values[:1]  # to close the spider
                is_best = (max_avg_perf == perf)
                logger.warning((max_avg_perf, perf))
                label = f"{method.split('(')[0]} ({np.round(100 * perf, 2)})"
                ax.plot(ANGLES, values, linewidth=3, label=rf"\textbf{{{label}}}" if is_best else label)
                ax.scatter(ANGLES, values, s=130, zorder=10)
                # ax.plot(ANGLES, values, c=method2color[method], linewidth=3, label=,)
                # ax.scatter(ANGLES, values, s=130, c=method2color[method], zorder=10)

            ax.set_title(pretty[metric_name], fontdict={'fontsize': 30}, y=1.2)
            ax.legend(
                loc='center',
                bbox_to_anchor=[1.2, 1.06],       # bottom-right
                ncol=1,
                frameon=False,     # don't put a frame
                prop={'size': 17}
            )

        # ---- Save plots ----
        # plt.subplots_adjust(wspace=-0.1)
        fig.tight_layout()
        os.makedirs(self.out_dir, exist_ok=True)
        fig.savefig(self.out_dir / f'spider.pdf', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    args = parse_args()
    if args.latex:
        logger.info("Activating latex")
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        # for Palatino and other serif fonts use:
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Palatino"],
        # })
    plotter = SpiderPlotter()
    plotter.fit(**vars(args))
    plotter.plot()