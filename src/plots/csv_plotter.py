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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--action', type=str, default='plot')
    parser.add_argument('--plot_versus', type=str, default='alpha')
    parser.add_argument('--filters', type=str, nargs='+', default=[],
                        help="Format: n_query=5 n_shot=1 ...")
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['mean_acc', 'mean_features_rocauc'])
    parser.add_argument('--groupby', type=str, help="Defines the methods compared. Ex: postpool_transforms")

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
        #  ===== Recover all csv result files =====
        p = Path('results') / args.exp
        csv_files = list(p.glob('**/*.csv'))  # every csv file represents the  summary of an experiment
        csv_files.sort()
        assert len(csv_files)

        #  ===== Recover all csv result files =====
        for file in csv_files:
            result_dir = self.nested_default_dict(3, list)
            df = pd.read_csv(file)

            # Perform necesary filtering
            filters = [x.split('=') for x in args.filters]
            for key, value in filters:
                expected_type = df[key].dtypes
                cast_value = np.array([value]).astype(expected_type)[0]
                df = df[df[key] == cast_value]

            # Read remaining rows and add it to result_dir
            for index, row in df.iterrows():
                for metric in kwargs['metrics']:
                    method_at_row = row[kwargs['groupby']]
                    x_value = row[kwargs['plot_versus']]
                    if metric in row:
                        result_dir[metric][method_at_row][x_value].append(row[metric])

            # Fill the metric_dic
            for metric in result_dir:
                for method, method_dic in result_dir[metric].items():
                    for x_value, values in method_dic.items():
                        if len(method_dic[x_value]) > 1:
                            logger.warning(f"Method {method} contains {len(method_dic[x_value])} \
                                             possible values for {kwargs['plot_versus']}={x_value}. \
                                             Choosing the best among them.")
                        self.metric_dic[metric][method]['x'].append(x_value)
                        self.metric_dic[metric][method]['y'].append(max(method_dic[x_value]))
                    self.metric_dic[metric][method]['xlabel'] = kwargs['plot_versus']
                    sorted_indexes = np.argsort(self.metric_dic[metric][method]['x'])
                    self.metric_dic[metric][method]['x'] = [self.metric_dic[metric][method]['x'][i] for i in sorted_indexes]
                    self.metric_dic[metric][method]['y'] = [self.metric_dic[metric][method]['y'][i] for i in sorted_indexes]

        self.out_dir = Path(kwargs['exp']) / '-'.join(args.filters)


class CSVPrinter(CSVPlotter):

    def log_best(self, **kwargs):
        assert hasattr(self, 'metric_dic')
        assert len(self.metric_dic) == 1, list(self.metric_dic.keys())
        for metric in self.metric_dic:
            for method, res in self.metric_dic[metric].items():
                assert len(res['x']) == 1, res
            sorted_methods = list(sorted(self.metric_dic[metric].items(),
                                         key=lambda res: res[1]['y'][0],
                                         reverse=True)
                                  )
            best_method = sorted_methods[0]
            logger.info(f"Best method {best_method[0]} achieved {best_method[1]['y'][0]} {metric}")


if __name__ == "__main__":
    args = parse_args()
    if args.action == 'plot':
        plotter = CSVPlotter()
        plotter.fit(**vars(args))
        plotter.plot()
    else:
        plotter = CSVPrinter()
        plotter.fit(**vars(args))
        plotter.log_best()
