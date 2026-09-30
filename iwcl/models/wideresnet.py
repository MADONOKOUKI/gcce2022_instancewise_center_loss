"""Wide residual network (Zagoruyko and Komodakis, BMVC 2016) for 32x32 inputs.

Adapted from https://github.com/xternalz/WideResNet-pytorch, as used in the release code
(``archive/release_2023/models/WideResNet.py``). The release model also created two
unused linear heads (``fc2``, ``fc2_sep``); they never received gradients and are omitted.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = in_planes == out_planes
        self.convShortcut = (
            None
            if self.equal_in_out
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            *[
                BasicBlock(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate)
                for i in range(nb_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    """WRN-``depth``-``widen_factor``; the release default is WRN-28-10.

    ``forward`` returns ``(logits, features)`` with the globally pooled penultimate
    features of dimension ``feature_dim = 64 * widen_factor``.
    """

    def __init__(self, depth: int = 28, num_classes: int = 10, widen_factor: int = 10, drop_rate: float = 0.0) -> None:
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError(f"WideResNet depth must be 6n+4, got {depth}")
        n = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], 1, drop_rate)
        self.block2 = NetworkBlock(n, channels[1], channels[2], 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.feature_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.block3(self.block2(self.block1(self.conv1(x))))
        out = self.relu(self.bn1(out))
        features = F.avg_pool2d(out, 8).flatten(1)
        return self.fc(features), features
