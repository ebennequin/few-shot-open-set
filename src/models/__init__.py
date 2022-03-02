from .resnet import *

from .vit import *

from .resnet import default_cfgs as resnet_config
from .custom_resnet import default_cfgs as custom_resnet_config
from .visiontransformer import default_cfgs as vit_config

BACKBONE_CONFIGS = {}
BACKBONE_CONFIGS.update(resnet_config)
BACKBONE_CONFIGS.update(vit_config)
BACKBONE_CONFIGS.update(custom_resnet_config)