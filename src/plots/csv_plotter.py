from pathlib import Path
from itertools import cycle
from collections import defaultdict
from functools import partial
from typing import Any, List, Tuple
from .plotter import Plotter
import pandas as pd
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str,
                        help='Folder to search')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--param_plot', type=str, default='alpha')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['acc', 'roc_auc'])

    args = parser.parse_args()
    return args


class CSVPlotter(Plotter):
    """
    An abstract plotter.
    """

    def fit(self, folder: Path, **kwargs):
        """
        Reads metrics in the folder and fill the dictionnary of metrics.
        At the end of this function, metric_dic should be filled as:

            metric_dic[metric_name][method] = {
                                                'x': ndarray [n_iter],
                                                'y': ndarray [n_iter],
                                                'pm': Optional[ndarray],
                                               }
        """
        # Recover all files that match .npy pattern in folder/
        p = Path(folder)
        csv_files = list(p.glob('**/*.csv'))  # every csv file represents the  summary of an experiment
        csv_files.sort()
        assert len(csv_files)
        for file in csv_files:
            experiment_name = file.parent
            if kwargs['exp'] in str(experiment_name):
                self.out_dir = kwargs['exp']
                df = pd.read_csv(file)
                x_values = df[kwargs['param_plot']].values
                sorted_indexes = x_values.argsort()
                for metric in kwargs['metrics']:
                    self.metric_dic[metric][experiment_name]['xlabel'] = kwargs['param_plot']
                    self.metric_dic[metric][experiment_name]['x'] = x_values[sorted_indexes]
                    self.metric_dic[metric][experiment_name]['y'] = df[metric].values[sorted_indexes]


if __name__ == "__main__":
    args = parse_args()
    plotter = CSVPlotter()
    plotter.fit(**vars(args))
    plotter.plot()
