from .abstract_few_shot_method import AbstractFewShotMethod
from .tim import AbstractTIM, TIM_GD
from .bd_cspn import BDCSPN
from .simpleshot import SimpleShot
from .finetune import Finetune

ALL_FEW_SHOT_CLASSIFIERS = [
    BDCSPN,
    Finetune,
    SimpleShot,
    TIM_GD,
]
