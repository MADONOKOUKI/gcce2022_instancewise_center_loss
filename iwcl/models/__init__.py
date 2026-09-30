"""CIFAR-style networks used in the paper's experiments (release code architectures).

Every model returns ``(logits, features)`` from ``forward`` and exposes ``feature_dim``,
the size of the pooled penultimate features (needed by the class-centre baselines; the
release code hard-coded it in ``train.py``).
"""
from __future__ import annotations

import torch.nn as nn

from .densenet import DenseNet
from .resnet import ResNet, ResNet18
from .resnext import CifarResNeXt
from .shakeshake import ShakeShake
from .wideresnet import WideResNet

MODELS = ("resnet18", "resnext", "wideresnet", "densenet", "shakeshake")


def build_model(
    name: str,
    num_classes: int,
    depth: int = 28,
    widen_factor: int = 10,
    resnext_base_width: int = 32,
    base_channels: int = 32,
) -> nn.Module:
    """Build a network. The first three are the models of the paper (Sec. III-A).

    * ``resnet18``: ResNet-18
    * ``resnext``: ResNeXt-29 8x``resnext_base_width``d (default 8x32d as in the paper;
      the release code used 8x64d)
    * ``wideresnet``: WRN-``depth``-``widen_factor`` (default WRN-28-10, dropout 0)
    * ``densenet``: DenseNet-BC-100 (k=12); returns log-probabilities (see its docstring)
    * ``shakeshake``: Shake-Shake-26 2x``base_channels``d (default 2x32d)
    """
    if name == "wideresnet":
        return WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, drop_rate=0.0)
    if name == "resnet18":
        return ResNet18(num_classes=num_classes)
    if name == "resnext":
        return CifarResNeXt(num_classes=num_classes, base_width=resnext_base_width)
    if name == "densenet":
        return DenseNet(growth_rate=12, depth=100, reduction=0.5, num_classes=num_classes, bottleneck=True)
    if name == "shakeshake":
        return ShakeShake(num_classes=num_classes, base_channels=base_channels, depth=26)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


__all__ = ["MODELS", "CifarResNeXt", "DenseNet", "ResNet", "ResNet18", "ShakeShake", "WideResNet", "build_model"]
