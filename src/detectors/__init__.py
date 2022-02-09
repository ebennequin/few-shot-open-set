from .abstract_detector import *
from .aggregator import NaiveAggregator
from .snatcher import SNATCHERF
import pyod

ALL_DETECTORS = {
    'aggregator': NaiveAggregator,
    'knn': pyod.models.knn.KNN,
    'snatcher_f': SNATCHERF
    }