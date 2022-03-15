from .resnet import default_cfgs as resnet_config
from .custom_resnet import default_cfgs as custom_resnet_config
from .visiontransformer import default_cfgs as vit_config
from .efficientnet import default_cfgs as efficientnet_config
from .wide_resnet import default_cfgs as wideres_config
from .misc import *

from .resnet import *
from .wide_resnet import *
from .custom_resnet import *
from .visiontransformer import *
from .efficientnet import *

BACKBONE_CONFIGS = {}
BACKBONE_CONFIGS.update(resnet_config)
BACKBONE_CONFIGS.update(wideres_config)
BACKBONE_CONFIGS.update(vit_config)
BACKBONE_CONFIGS.update(custom_resnet_config)
BACKBONE_CONFIGS.update(efficientnet_config)