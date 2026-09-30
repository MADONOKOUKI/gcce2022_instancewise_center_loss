"""Shake-Shake-26 2x32d ("Shake-Shake-Image") for 32x32 inputs (Gastaldi, 2017).

Adapted from https://github.com/hysts/pytorch_shake_shake, as used in the release code
(``archive/release_2023/models/shakeshake.py``). The release model computed the feature
size with a dummy forward pass in ``__init__``; here it is computed analytically
(``4 * base_channels``), which gives the same architecture.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ShakeFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(x1, x2, alpha, beta)
        return x1 * alpha + x2 * (1 - alpha)

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, alpha, beta = ctx.saved_tensors
        grad_x1 = grad_x2 = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)
        return grad_x1, grad_x2, None, None


shake_function = ShakeFunction.apply


def get_alpha_beta(batch_size: int, shake_config: Tuple[bool, bool, bool], device) -> Tuple[torch.Tensor, torch.Tensor]:
    forward_shake, backward_shake, shake_image = shake_config
    if forward_shake and not shake_image:
        alpha = torch.rand(1)
    elif forward_shake and shake_image:
        alpha = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        alpha = torch.FloatTensor([0.5])
    if backward_shake and not shake_image:
        beta = torch.rand(1)
    elif backward_shake and shake_image:
        beta = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        beta = torch.FloatTensor([0.5])
    return alpha.to(device), beta.to(device)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_out")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ResidualPath(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        return self.bn2(self.conv2(x))


class SkipConnection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x, inplace=False)
        y1 = self.conv1(F.avg_pool2d(x, kernel_size=1, stride=self.stride, padding=0))
        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = self.conv2(F.avg_pool2d(y2, kernel_size=1, stride=self.stride, padding=0))
        return self.bn(torch.cat([y1, y2], dim=1))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, shake_config) -> None:
        super().__init__()
        self.shake_config = shake_config
        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module("skip", SkipConnection(in_channels, out_channels, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)
        shake_config = self.shake_config if self.training else (False, False, False)
        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        return self.shortcut(x) + shake_function(x1, x2, alpha, beta)


class ShakeShake(nn.Module):
    """``forward`` returns ``(logits, features)``; ``feature_dim = 4 * base_channels`` (128)."""

    def __init__(self, num_classes: int = 10, base_channels: int = 32, depth: int = 26) -> None:
        super().__init__()
        self.shake_config = (True, True, True)
        n_blocks_per_stage = (depth - 2) // 6
        if n_blocks_per_stage * 6 + 2 != depth:
            raise ValueError(f"Shake-Shake depth must be 6n+2, got {depth}")
        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, n_channels[0], n_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, stride=2)
        self.feature_dim = n_channels[2]
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(
                f"block{index + 1}",
                BasicBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    stride=stride if index == 0 else 1,
                    shake_config=self.shake_config,
                ),
            )
        return stage

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bn(self.conv(x))
        x = self.stage3(self.stage2(self.stage1(x)))
        features = F.adaptive_avg_pool2d(F.relu(x, inplace=True), output_size=1).flatten(1)
        return self.fc(features), features
