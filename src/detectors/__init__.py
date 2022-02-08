from .abstract_detector import *
from .aggregator import NaiveAggregator
import pyod

ALL_DETECTORS = {
    'aggregator': NaiveAggregator,
    'knn': pyod.models.knn.KNN
    }