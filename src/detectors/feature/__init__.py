from .abstract import *
from .aggregator import NaiveAggregator
from .snatcher import SNATCHERF
from .alternate_detector import AlternateDetector
from .repri import RepriDetector
from .finetune import FinetuneDetector
from pyod.models.knn import KNN

ALL_DETECTORS = {
    'aggregator': NaiveAggregator,
    'knn': KNN,
    'snatcher_f': SNATCHERF,
    'alternate': AlternateDetector,
    'repri': RepriDetector,
    'finetune': FinetuneDetector,
    }