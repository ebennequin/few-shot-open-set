import numpy as np
from loguru import logger
from src.plots.csv_plotter import CSVPlotter, parse_args, pretty, spider_colors
import matplotlib.pyplot as plt
import os
from copy import deepcopy


class SpiderPlotter(CSVPlotter):
    """
    An abstract plotter.
    """

    def plot(self, horizontal: bool):
        """
        metric_dic[metric_name][method] = {
                                    'x': ndarray [n_points_found],
                                    'y': ndarray [n_points_found],
                                    'pm': Optional[ndarray],
                                   }
        """
        assert len(self.metric_dic)
        if horizontal:
            nrows = 1
            n_cols = len(self.metric_dic)
            figsize = (10 * len(self.metric_dic), 13)
        else:
            nrows = len(self.metric_dic)
            n_cols = 1
            figsize = (10, 10 * len(self.metric_dic))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=n_cols,
            subplot_kw=dict(projection="polar"),
            figsize=figsize,
            squeeze=True,
        )

        # Form the differences instead of absolute values
        baseline_values = {}
        for i, metric in enumerate(self.metric_dic):
            if i == 0:
                methods = list(self.metric_dic[metric].keys())
                print(
                    f"Methods detected : {methods} \n  Which methods to use ? ('all' or space-separated indices.)"
                )
                selection = input()
                if selection == "all":
                    pass
                else:
                    methods = [methods[int(x)] for x in selection.split(" ")]
            if i == 0:
                print(
                    f"Methods detected : {methods} \n  Which one to use as a baseline ?"
                )
                baseline_method = methods[int(input())]
                methods.remove(baseline_method)
            x_names = deepcopy(self.metric_dic[metric][baseline_method]["x"])
            y_baseline = np.array(self.metric_dic[metric][baseline_method]["y"])
            baseline_values[metric] = y_baseline
            for method in methods:
                assert self.metric_dic[metric][method]["x"] == x_names, (
                    self.metric_dic[metric][method]["x"],
                    x_names,
                )
                self.metric_dic[metric][method]["y"] = (
                    np.array(self.metric_dic[metric][method]["y"]) - y_baseline
                )
            del self.metric_dic[metric][baseline_method]

        for i, metric_name in enumerate(self.metric_dic):
            if nrows*n_cols > 1:
                ax = axes[i]
            else:
                ax = axes

            BG_WHITE = "#fbf9f4"
            BLUE = "#2a475e"
            GREY70 = "#b3b3b3"
            GREY_LIGHT = "#f2efe8"

            min_val = min(
                [min(self.metric_dic[metric_name][method]["y"]) for method in methods]
            )
            max_val = max(
                [max(self.metric_dic[metric_name][method]["y"]) for method in methods]
            )
            max_avg_perf = max(
                [
                    np.mean(self.metric_dic[metric_name][method]["y"])
                    for method in methods
                ]
            )
            yticks = np.linspace(min_val - 0.005, max_val + 0.005, 5)
            cloest_to_0 = np.abs(yticks).argmin()
            # yticks = np.delete(yticks, cloest_to_0)

            PAD = ((max_val - min_val) / 5) * 0.5
            angle = 3.14

            first_method = list(self.metric_dic[metric_name].keys())[0]
            x_names = self.metric_dic[metric_name][first_method]["x"]
            VARIABLES = [
                f"{x} \n ({np.round(100 * y, 1)})"
                for x, y in zip(x_names, baseline_values[metric_name])
            ]
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
            H = [np.ones(len(HANGLES)) * li for li in yticks]

            # fig.patch.set_facecolor(BG_WHITE)
            # ax.set_facecolor(BG_WHITE)

            # Rotate the "" 0 degrees on top.
            # There it where the first variable, avg_bill_length, will go.
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Setting lower limit to negative value reduces overlap
            # for values that are 0 (the minimums)
            ax.set_ylim(yticks[0] - 0.01, yticks[-1])

            # Set values for the angular axis (x)
            ax.set_xticks(ANGLES[:-1])
            ax.set_xticklabels(VARIABLES, size=30, y=-0.35)

            # Remove lines for radial axis (y)
            ax.set_yticks([])
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            # Remove spines
            ax.spines["start"].set_color("none")
            ax.spines["polar"].set_color("none")

            # Add custom lines for radial axis (y) at 0, 0.5 and 1.
            _ = [ax.plot(HANGLES, h, ls=(0, (6, 6)), c=GREY70) for h in H]
            _ = ax.plot(
                HANGLES,
                [0.0] * len(HANGLES),
                ls=(0, (6, 6)),
                c="black",
                label="Strong baseline",
                linewidth=2,
            )

            # Add levels -----------------------------------------------------
            # These labels indicate the values of the radial axis
            size = 18
            _ = [
                ax.text(
                    angle,
                    li + PAD,
                    f"+{np.round(li * 100, 1)}" if li > 0 else np.round(li * 100, 1),
                    size=size,
                )
                for li in yticks
            ]
            ax.text(angle, 0.0 + PAD, 0.0, size=size)

            # Now fill the area of the circle with radius 1.
            # This create the effect of gray background.
            ax.fill(HANGLES, H[-1], GREY_LIGHT)

            # Fill lines and dots --------------------------------------------
            methods.sort()
            for idx, method in enumerate(methods):
                method_result = self.metric_dic[metric_name][method]
                assert method_result["x"] == x_names, (
                    method,
                    method_result["x"],
                    x_names,
                )
                values = method_result["y"]
                perf = np.mean(values)
                values = np.concatenate([values, [values[0]]])  # to close the spider
                is_best = max_avg_perf == perf
                label = f"{method.split('(')[0]} ({np.round(100 * perf, 2)})"
                ax.plot(
                    ANGLES,
                    values,
                    linewidth=4 if "OSTIM" in method else 2,
                    c=spider_colors[idx],
                    label=rf"\textbf{{{label}}}" if is_best else label,
                )
                ax.scatter(ANGLES, values, c=spider_colors[idx], s=130, zorder=10)
                # ax.plot(ANGLES, values, c=method2color[method], linewidth=3, label=,)
                # ax.scatter(ANGLES, values, s=130, c=method2color[method], zorder=10)

            ax.set_title(
                rf"\textbf{{{pretty[metric_name]}}}", fontdict={"fontsize": 30}, y=1.5
            )
            ax.legend(
                loc="center",
                bbox_to_anchor=[1.25, 1.3],  # bottom-right
                ncol=1,
                frameon=False,  # don't put a frame
                prop={"size": 25},
            )

        # ---- Save plots ----
        fig.tight_layout()
        os.makedirs(self.out_dir, exist_ok=True)
        if horizontal:
            fig.savefig(self.out_dir / f"main_spider.pdf", dpi=300, bbox_inches="tight")
        else:
            fig.savefig(
                self.out_dir / f"{self.filters}.pdf", dpi=300, bbox_inches="tight"
            )


if __name__ == "__main__":
    args = parse_args()
    if args.latex:
        logger.info("Activating latex")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
            }
        )
    plotter = SpiderPlotter()
    plotter.fit(**vars(args))
    plotter.plot(horizontal=args.horizontal)
