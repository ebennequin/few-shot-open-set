from .aggregator import NaiveAggregator
from .repri import RepriDetector
from .finetune import FinetuneDetector
from .knn import kNNDetector
from.pyod_wrapper import PyodWrapper
from loguru import logger
from functools import partial


def instanciate_wrapper(pyod_detector, **kwargs):
    return PyodWrapper(pyod_detector, **kwargs)


__all__ = {
    'NaiveAggregator': NaiveAggregator,
    'RepriDetector': RepriDetector,
    'FinetuneDetector': FinetuneDetector,
    'kNNDetector': kNNDetector
}

for pyod_detector in ['ABOD', 'IForest', 'LOF', 'MCD', 'PCA']:
    __all__[pyod_detector] = partial(instanciate_wrapper, pyod_detector=pyod_detector)
