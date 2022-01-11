"""
Custom ResNet from https://github.com/fiveai/on-episodes-fsl/blob/master/src/models/ResNet.py
"""


import torch.nn as nn
import torch
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

__all__ = [
    "resnet10",
    "resnet12",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        widths=None,
        num_classes=1000,
        zero_init_residual=False,
        use_fc=False,
        imagenet_setup=False,
    ):
        super(ResNet, self).__init__()
        if widths is None:
            widths = [64, 128, 256, 512]

        self.inplanes = 64
        self.use_fc = use_fc

        self.conv1 = (
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=1, bias=False)
            if imagenet_setup
            else nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Only used when self.use_fc is True
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = torch.flatten(
            self.avgpool(
                self.layer4(
                    self.layer3(
                        self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))
                    )
                )
            ),
            1,
        )

        if self.use_fc:
            return self.fc(features)

        return features


def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet12(**kwargs):
    """Constructs a ResNet-12 model."""
    model = ResNet(BasicBlock, [1, 1, 2, 1], widths=[64, 160, 320, 640], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet12imagenet(**kwargs):
    """Constructs a ResNet-12 model with a bottleneck at the beginning."""
    model = ResNet(
        BasicBlock,
        [1, 1, 2, 1],
        widths=[64, 160, 320, 640],
        imagenet_setup=True,
        **kwargs
    )
    return model


def resnet18imagenet(**kwargs):
    """Constructs a ResNet-18 model with a bottleneck at the beginning."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], imagenet_setup=True, **kwargs)
    return model
