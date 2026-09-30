"""ResNeXt-29 for 32x32 inputs (Xie et al., CVPR 2017).

Adapted from https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py, as
used in the release code (``archive/release_2023/models/resnext.py``).

The paper uses ResNeXt-29 8x32d (``base_width=32``, the default here). The release code
built ``CifarResNeXt()``, whose bottleneck width ``cardinality * out_channels //
widen_factor`` gives 8x64d; ``base_width=64`` reproduces that network exactly. The release
also passed ``depth=32``; the number of bottlenecks per stage is ``(depth - 2) // 9 = 3``
for both 29 and 32, so the depth is the same.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    """ResNeXt bottleneck type C."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, cardinality: int, widen_factor: int, base_width: int = 64
    ) -> None:
        super().__init__()
        d = cardinality * base_width * out_channels // (widen_factor * 64)  # base_width=64: release formula
        self.conv_reduce = nn.Conv2d(in_channels, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(d)
        self.conv_conv = nn.Conv2d(d, d, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(d)
        self.conv_expand = nn.Conv2d(d, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "shortcut_conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            )
            self.shortcut.add_module("shortcut_bn", nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_reduce(self.conv_reduce(x)), inplace=True)
        out = F.relu(self.bn(self.conv_conv(out)), inplace=True)
        out = self.bn_expand(self.conv_expand(out))
        return F.relu(self.shortcut(x) + out, inplace=True)


class CifarResNeXt(nn.Module):
    """ResNeXt-29 ``cardinality`` x ``base_width``d (default 8x32d, the paper's model).

    ``forward`` returns ``(logits, features)``; ``feature_dim = 256 * widen_factor`` (1024).
    """

    def __init__(
        self, cardinality: int = 8, depth: int = 29, num_classes: int = 10, widen_factor: int = 4, base_width: int = 32
    ) -> None:
        super().__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.block_depth = (depth - 2) // 9
        self.widen_factor = widen_factor
        self.stages = [64, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor]
        self.feature_dim = self.stages[3]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self._block("stage_1", self.stages[0], self.stages[1], 1)
        self.stage_2 = self._block("stage_2", self.stages[1], self.stages[2], 2)
        self.stage_3 = self._block("stage_3", self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # same initialisation as the reference code
        init.kaiming_normal_(self.classifier.weight)
        for key, value in self.state_dict().items():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(value, mode="fan_out")
                if "bn" in key:
                    value[...] = 1
            elif key.split(".")[-1] == "bias":
                value[...] = 0

    def _block(self, name: str, in_channels: int, out_channels: int, pool_stride: int = 2) -> nn.Sequential:
        block = nn.Sequential()
        for i in range(self.block_depth):
            block.add_module(
                f"{name}_bottleneck_{i}",
                ResNeXtBottleneck(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    pool_stride if i == 0 else 1,
                    self.cardinality,
                    self.widen_factor,
                    self.base_width,
                ),
            )
        return block

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_1(self.conv_1_3x3(x)), inplace=True)
        x = self.stage_3(self.stage_2(self.stage_1(x)))
        features = F.avg_pool2d(x, 8, 1).flatten(1)
        return self.classifier(features), features
