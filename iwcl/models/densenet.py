"""DenseNet-BC for 32x32 inputs (Huang et al., CVPR 2017).

Adapted from https://github.com/bamos/densenet.pytorch, as used in the release code
(``archive/release_2023/models/densenet.py``; DenseNet-BC-100, growth rate 12).

Note: like the release model, ``forward`` returns **log-probabilities**
(``log_softmax`` of the logits) as its first output. Cross-entropy and accuracy are
unaffected (``log_softmax`` is idempotent), but the instance-wise center loss then
compares log-probabilities rather than raw logits for this architecture.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int) -> None:
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat((x, out), 1)


class SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.conv1(F.relu(self.bn1(x)))), 1)


class Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(self.conv1(F.relu(self.bn1(x))), 2)


class DenseNet(nn.Module):
    """``forward`` returns ``(log_probs, features)``; ``feature_dim = 342`` for the
    default DenseNet-BC-100 (k=12)."""

    def __init__(
        self,
        growth_rate: int = 12,
        depth: int = 100,
        reduction: float = 0.5,
        num_classes: int = 10,
        bottleneck: bool = True,
    ) -> None:
        super().__init__()
        n_dense_blocks = (depth - 4) // 3
        if bottleneck:
            n_dense_blocks //= 2

        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate

        self.bn1 = nn.BatchNorm2d(n_channels)
        self.fc = nn.Linear(n_channels, num_classes)
        self.feature_dim = n_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @staticmethod
    def _make_dense(n_channels: int, growth_rate: int, n_dense_blocks: int, bottleneck: bool) -> nn.Sequential:
        layers = []
        for _ in range(int(n_dense_blocks)):
            layers.append(Bottleneck(n_channels, growth_rate) if bottleneck else SingleLayer(n_channels, growth_rate))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.trans1(self.dense1(self.conv1(x)))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        features = F.avg_pool2d(F.relu(self.bn1(out)), 8).flatten(1)
        return F.log_softmax(self.fc(features), dim=1), features
