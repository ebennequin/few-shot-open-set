from pathlib import Path
import numpy as np
from itertools import cycle
from collections import defaultdict
from functools import partial
from loguru import logger
from typing import Any, List, Tuple
from .plotter import Plotter
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from src.inference import str2bool


class my_default_dict(dict):
    def __init__(self, fn):
        self.fn = fn

    def __missing__(self, key):
        return self.fn(key)


pretty = my_default_dict(lambda x: x)

pretty["mini_imagenet"] = r"\textit{Mini}"
pretty["tiered_imagenet"] = r"\textit{Tiered}"
pretty["cub"] = "CUB"
pretty["aircraft"] = "Aircraft"
pretty["fungi"] = "Fungi"
pretty["None"] = ""

pretty["OTTIM"] = "OTTIM"

pretty["mean_acc"] = "Accuracy"
pretty["mean_rocauc"] = "AUROC"
pretty["mean_rec_at_90"] = "Rec@0.9"
pretty["mean_prec_at_90"] = "Prec@0.9"


pretty["ssl_resnext101_32x16d"] = "ResNext101"
pretty["resnet101"] = "ResNet101"
pretty["resnet152"] = "ResNet152"
pretty["resnet50"] = "ResNet50"
pretty["resnet18"] = "ResNet18"

pretty["efficientnet_b0"] = "EfficientNet-B0"
pretty["efficientnet_b1"] = "EfficientNet-B1"
pretty["efficientnet_b2"] = "EfficientNet-B2"
pretty["efficientnet_b3"] = "EfficientNet-B3"
pretty["efficientnet_b4"] = "EfficientNet- B4"
pretty["efficientnet_b5"] = "EfficientNet-B5"
pretty["efficientnet_b6"] = "EfficientNet-B6"
pretty["efficientnet_b7"] = "EfficientNet-B7"
pretty["efficientnet_b8"] = "EfficientNet-B8"
pretty["efficientnet_l2"] = "EfficientNet-L2"

pretty["vit_tiny_patch16_384"] = "ViT-tiny"
pretty["vit_small_patch16_384"] = "ViT-small"
pretty["vit_base_patch16_384"] = "ViT-base"
pretty["vit_large_patch16_384"] = "ViT-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--exp", type=str, help="Name of the experiment")
    parser.add_argument("--action", type=str, default="plot")
    parser.add_argument("--plot_versus", type=str, nargs="+")
    parser.add_argument("--use_pretty", type=str2bool, default=True)
    parser.add_argument("--latex", type=str2bool, default=True)
    parser.add_argument(
        "--filters",
        type=str,
        nargs="+",
        default=[],
        help="Format: n_query=5 n_shot=1 ...",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=["mean_acc", "mean_features_rocauc"]
    )
    parser.add_argument(
        "--groupby",
        type=str,
        nargs="+",
        help="Defines the methods compared. Ex: postpool_transforms",
    )

    args = parser.parse_args()
    return args


class CSVPlotter(Plotter):
    """
    An abstract plotter.
    """

    def fit(self, **kwargs):
        """
        Reads metrics in the folder and fill the dictionnary of metrics.
        At the end of this function, metric_dic should be filled as:

            metric_dic[metric_name][method] = {
                                                'x': ndarray [n_points_found],
                                                'y': ndarray [n_points_found],
                                                'pm': Optional[ndarray],
                                               }
        """
        if kwargs['use_pretty']:
            process_dic = pretty
        else:
            process_dic = my_default_dict(lambda x: x)


        #  ===== Recover all csv result files =====
        p = Path("results") / kwargs["exp"]
        csv_files = list(
            p.glob("**/*.csv")
        )  # every csv file represents the  summary of an experiment
        csv_files.sort()
        assert len(csv_files)

        #  ===== Recover all csv result files =====
        for file in csv_files:
            result_dir = self.nested_default_dict(3, list)
            df = pd.read_csv(file)

            # Perform necesary filtering
            filters = [x.split("=") for x in kwargs["filters"]]
            for key, value in filters:
                expected_type = df[key].dtypes
                cast_value = np.array([value]).astype(expected_type)[0]
                df = df[df[key] == cast_value]

            # Read remaining rows and add it to result_dir
            for index, row in df.iterrows():
                for metric in kwargs["metrics"]:
                    if kwargs['use_pretty']:
                        full_method_name = [
                            process_dic[row[x].split("(")[0]] for x in kwargs["groupby"]
                        ]
                    else:
                        full_method_name = [
                            process_dic[row[x]] for x in kwargs["groupby"]
                        ]
                    full_method_name = list(filter(lambda x: len(x), full_method_name))
                    method_at_row = " + ".join(full_method_name)
                    x_value = "\n".join([process_dic[row[x]] for x in kwargs["plot_versus"]])
                    if metric in row:
                        result_dir[metric][method_at_row][x_value].append(row[metric])

            # Fill the metric_dic
            for metric in result_dir:
                for method, method_dic in result_dir[metric].items():
                    for x_value, values in method_dic.items():
                        if len(method_dic[x_value]) > 1:
                            logger.warning(
                                f"Method {method} contains {len(method_dic[x_value])} \
                                             possible values for {kwargs['plot_versus']}={x_value}. \
                                             Choosing the best among them."
                            )
                        self.metric_dic[metric][method]["x"].append(x_value)
                        self.metric_dic[metric][method]["y"].append(
                            max(method_dic[x_value])
                        )
                    self.metric_dic[metric][method]["xlabel"] = kwargs["plot_versus"]
                    sorted_indexes = np.argsort(self.metric_dic[metric][method]["x"])
                    self.metric_dic[metric][method]["x"] = [
                        self.metric_dic[metric][method]["x"][i] for i in sorted_indexes
                    ]
                    self.metric_dic[metric][method]["y"] = [
                        self.metric_dic[metric][method]["y"][i] for i in sorted_indexes
                    ]

        self.out_dir = Path("plots") / kwargs["exp"] / "-".join(kwargs["filters"])


class CSVPrinter(CSVPlotter):
    def log_best(self, **kwargs):
        assert hasattr(self, "metric_dic")
        all_metrics = self.metric_dic.keys()
        for metric in self.metric_dic:
            all_methods = self.metric_dic[metric].keys()
            for method, res in self.metric_dic[metric].items():
                assert len(res["x"]) == len(res["y"]) == 1, res
        all_items = [(method, [self.metric_dic[metric][method]["y"][0] for metric in all_metrics]) for method in all_methods]
        sorted_methods = list(
            sorted(
                all_items,
                key=lambda item: np.mean(item[1]),
                reverse=True,
            )
        )
        best_method = sorted_methods[0]
        msg = f"Best method {best_method[0]} achieved overall : {best_method[1]}"
        for metric in all_metrics:
            msg += f" and {metric}={self.metric_dic[metric][best_method[0]]['y'][0]}"
        logger.info(
            msg
        )

    def log_latex(self, **kwargs):
        assert hasattr(self, "metric_dic")
        for metric in self.metric_dic:
            all_methods = self.metric_dic[metric].keys()
        for method in all_methods:
            msg = f"{method} &"
            for metric in kwargs['metrics']:
                value = self.metric_dic[metric][method]["y"][0]
                msg += f"{value} & "

            logger.info(
                msg
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
        # for Palatino and other serif fonts use:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Palatino"],
            }
        )

    if args.action == "plot":
        plotter = CSVPlotter()
        plotter.fit(**vars(args))
        plotter.plot()
    elif args.action == "log_best":
        plotter = CSVPrinter()
        plotter.fit(**vars(args))
        plotter.log_best()
    elif args.action == "log_latex":
        plotter = CSVPrinter()
        plotter.fit(**vars(args))
        plotter.log_latex(**vars(args))

