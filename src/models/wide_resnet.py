import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import List
from torch import Tensor
import sys
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np


def _cfg(url="", **kwargs):
    return {
        "input_size": (3, 84, 84),
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        **kwargs,
    }


default_cfgs = {
    "wrn2810": _cfg(),
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, feats: List[Tensor]):
        x = feats[-1]
        new_feats = []
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        new_feats.append(out)
        out = self.conv2(F.relu(self.bn2(out)))
        new_feats.append(out)
        out += self.shortcut(x)
        new_feats.append(out)

        return new_feats


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=64, **kwargs):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.all_layers = [f"{i}_{j}" for i in range(1, 4) for j in range(3)] + ["last"]
        self.last_layer_name = "last"

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = int((depth - 4) / 6)
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1
        )
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2
        )
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.fc = nn.Linear(640, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, layers: List[str]):
        """
        layers: List[str]
        """
        all_feats = {}
        x = self.conv1(x)
        layer_feats = [x]
        for block in range(1, 4):
            layer_feats = eval(f"self.layer{block}")(layer_feats)
            pooled_maps = [f.mean((-2, -1)) for f in layer_feats]
            for block_layer, pooled_map in enumerate(pooled_maps):
                layer_name = f"{block}_{block_layer}"
                if layer_name in layers:
                    all_feats[layer_name] = pooled_map
        if "last" in layers:
            last_map = F.relu(self.bn1(layer_feats[-1]))
            all_feats["last"] = last_map.mean((-2, -1), keepdim=True)
        return all_feats


def wrn2810(**kwargs):

    model = Wide_ResNet(28, 10, 0.0, **kwargs)
    return model
