"""Factory for the proposed method and the comparison methods (``--method`` of ``train.py``)."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .baselines import AugMixJSDLoss, CenterLoss, ContrastiveCenterLoss, InstanceTripletLoss, MultiViewCrossEntropy
from .losses import InstanceWiseCenterLoss

METHODS = ("proposed", "baseline", "center_loss", "contrastive_center_loss", "triplet_loss", "jsd")
ALIASES = {"augmix": "jsd"}  # name of the JS-divergence method in the release config

# weight lambda of the extra term of each comparison method (paper Sec. III-A: 0.1; the
# JS-divergence uses the AugMix code's value, 12)
DEFAULT_AUX_WEIGHTS = {"center_loss": 0.1, "contrastive_center_loss": 0.1, "triplet_loss": 0.1, "jsd": 12.0}


def build_criterion(
    method: str = "proposed",
    num_classes: Optional[int] = None,
    feat_dim: Optional[int] = None,
    alpha: float = 0.5,
    distance: str = "mse",
    stopgrad: bool = True,
    reduction: str = "mean",
    center_on: str = "logits",
    aux_weight: Optional[float] = None,
    learnable_centers: bool = False,
    triplet_on: str = "logits",
    ignore_index: Optional[int] = -1,
) -> nn.Module:
    """Return the training objective of ``method``.

    ``'proposed'`` is :class:`~iwcl.InstanceWiseCenterLoss` (``alpha`` = lambda_IC,
    ``distance``, ``stopgrad``, ``reduction``, ``center_on``); ``'baseline'`` is the
    cross-entropy on the views ("None" in the paper's tables). ``aux_weight`` overrides the
    weight of a comparison method's extra term; ``num_classes`` and ``feat_dim`` are needed by
    the class-centre losses.
    """
    method = ALIASES.get(method, method)
    if method == "proposed":
        return InstanceWiseCenterLoss(
            alpha=alpha, distance=distance, stopgrad=stopgrad, reduction=reduction, center_on=center_on, ignore_index=ignore_index
        )
    if method == "baseline":
        return MultiViewCrossEntropy(ignore_index=ignore_index)
    weight = DEFAULT_AUX_WEIGHTS.get(method) if aux_weight is None else aux_weight
    if method in ("center_loss", "contrastive_center_loss") and (num_classes is None or feat_dim is None):
        raise ValueError(f"{method} needs num_classes and feat_dim")
    if method == "center_loss":
        return CenterLoss(num_classes, feat_dim, weight=weight, ignore_index=ignore_index)
    if method == "contrastive_center_loss":
        return ContrastiveCenterLoss(
            num_classes, feat_dim, weight=weight, learnable_centers=learnable_centers, ignore_index=ignore_index
        )
    if method == "triplet_loss":
        return InstanceTripletLoss(weight=weight, on=triplet_on, ignore_index=ignore_index)
    if method == "jsd":
        return AugMixJSDLoss(weight=weight, ignore_index=ignore_index)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
