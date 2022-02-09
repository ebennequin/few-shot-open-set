from .abstract_detector import *
from .aggregator import NaiveAggregator
from .snatcher import SNATCHERF
import pyod
from pyod.models.knn import KNN

ALL_DETECTORS = {
    'aggregator': NaiveAggregator,
    'knn': KNN,
    'snatcher_f': SNATCHERF
    }