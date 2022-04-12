import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List
from torch.distributions import Bernoulli
import torch.distributed as dist
import numpy as np

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


def _cfg(url="", **kwargs):
    return {
        "input_size": (3, 84, 84),
        "keep_prob": 1.0,
        "mean": np.array(
            [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        ),
        "std": np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]),
        **kwargs,
    }


default_cfgs = {
    "resnet12": _cfg(),
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (
                    batch_size,
                    channels,
                    height - (self.block_size - 1),
                    width - (self.block_size - 1),
                )
            )
            if torch.cuda.is_available():
                rank = dist.get_rank()
                mask = mask.to(rank)
            block_mask = self._compute_block_mask(mask)
            countM = (
                block_mask.size()[0]
                * block_mask.size()[1]
                * block_mask.size()[2]
                * block_mask.size()[3]
            )
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size)
                .view(-1, 1)
                .expand(self.block_size, self.block_size)
                .reshape(-1),  # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t()
        offsets = torch.cat(
            (torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), 1
        )
        if torch.cuda.is_available():
            rank = dist.get_rank()
            offsets = offsets.to(rank)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(
                mask, (left_padding, right_padding, left_padding, right_padding)
            )
            padded_mask[
                block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]
            ] = 1.0
        else:
            padded_mask = F.pad(
                mask, (left_padding, right_padding, left_padding, right_padding)
            )

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, layer_feats):
        x = layer_feats[-1]
        self.num_batches_tracked += 1

        residual = x

        feats = []

        out = self.conv1(x)
        out = self.bn1(out)
        feats.append(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        feats.append(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        feats.append(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        feats.append(out)

        out = self.relu(out)
        out = self.maxpool(out)

        feats.append(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked),
                    1.0 - self.drop_rate,
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size ** 2
                    * feat_size ** 2
                    / (feat_size - self.block_size + 1) ** 2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(
                    out, p=self.drop_rate, training=self.training, inplace=True
                )

        return feats


# TODO: this is actually hardcoded as a Resnet12
class ResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        keep_prob=1.0,
        avg_pool=False,
        drop_rate=0.1,
        dropblock_size=5,
        num_classes=64,
    ):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.last_layer_name = "4_4"
        self.all_layers = [f"{i}_{j}" for i in range(1, 5) for j in range(5)]
        channels = [64, 160, 320, 640]
        self.layer_dims = [
            channels[i] * block.expansion for i in range(4) for j in range(4)
        ]

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        # if avg_pool:
        #     self.avgpool = nn.AvgPool2d(1, stride=1)
        self.blocks = [getattr(self, f"layer{i}") for i in range(1, 5)]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.fc = nn.Linear(640, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                drop_rate,
                drop_block,
                block_size,
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, layers: List[str]):
        """
        layers: List[str]
        """
        all_feats = {}
        layer_feats = [x]
        for block in range(1, 5):
            layer_feats = eval(f"self.layer{block}")(layer_feats)
            pooled_maps = [f.mean((-2, -1)) for f in layer_feats]
            for block_layer, pooled_map in enumerate(pooled_maps):
                layer_name = f"{block}_{block_layer}"
                if layer_name in layers:
                    all_feats[layer_name] = pooled_map
        return all_feats


def resnet12(**kwargs):
    """Constructs a ResNet-12 model."""
    return ResNet(BasicBlock, **kwargs)
