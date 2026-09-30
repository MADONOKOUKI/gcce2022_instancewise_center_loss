"""Instance-wise center loss (IWCL), the regulariser proposed in the paper.

K. Madono, M. Tanaka, M. Onishi, "Instance-wise Center Loss for Efficient Training of
Deep Convolutional Neural Networks", IEEE GCCE 2022, pp. 692-696,
https://ieeexplore.ieee.org/document/10014037

Notation of Sec. II of the paper. Every image ``x_i`` of a mini-batch ``B`` is augmented
``N`` times with random parameters ``eta_n``; ``h(g_n(x_i))`` is the logit vector
(``C`` classes, before the softmax) of the ``n``-th view.

* Instance-wise centre, Eq. (2): ``c_i = 1/N * sum_n h(g_n(x_i))``
* Instance-wise center loss, Eq. (1)/(6):
  ``L_IC = 1/(|B| N) * sum_i sum_n || h(g_n(x_i)) - StopGrad(c_i) ||_2^2``
* Cross-entropy on all views, Eq. (4)/(5): ``L_CE = 1/(|B| N) * sum_i sum_n CE(softmax(h(g_n(x_i))), y_i)``
* Training loss, Eq. (3): ``L = (1 - lambda_IC) * L_CE + lambda_IC * L_IC``

Normalisation. The released code computes ``L_IC`` with ``torch.nn.MSELoss()``, which
also averages over the ``C`` logits, i.e. it uses ``L_IC / C``; the other distances of
Table IV are PyTorch losses with their default reduction as well (the paper cites the
PyTorch default ``delta = 1`` for the Huber loss). ``reduction='mean'`` (default)
reproduces this; ``reduction='sum'`` sums over the logits as written in Eq. (6).

Distances (Table IV of the paper): ``'mse'`` (the L2 loss above, default), ``'l1'``,
``'huber'`` (SmoothL1, delta = 1) and ``'kl'`` = ``KL(softmax(c_i) || softmax(z))``.

Every view is pulled towards the centre of *its own* base image instead of a class
centre, so ``L_IC`` needs no labels: images labelled ``ignore_index`` (``-1`` =
unlabelled) get no cross-entropy but still contribute to ``L_IC``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Views = Union[torch.Tensor, Sequence[torch.Tensor]]

DISTANCES = ("mse", "l1", "huber", "kl")
_ALIASES = {"l2": "mse", "hubor": "huber", "smoothl1": "huber", "smooth_l1": "huber"}  # 'Hubor' in the release config


def as_views(x: Views, num_views: Optional[int] = None) -> torch.Tensor:
    """Return per-view outputs as one tensor of shape ``(N, B, ...)``.

    Accepts a sequence of ``N`` tensors of shape ``(B, ...)``, a tensor of shape
    ``(N, B, D)``, or - when ``num_views`` is given - a flat tensor of shape ``(N*B, D)``
    whose rows are ordered view-major, i.e. ``torch.cat([view_1, ..., view_N])``.
    """
    if isinstance(x, (list, tuple)):
        return torch.stack(list(x), dim=0)
    if not torch.is_tensor(x):
        raise TypeError(f"expected a tensor or a sequence of tensors, got {type(x).__name__}")
    if x.dim() == 2:
        if num_views is None:
            raise ValueError("a 2-D input (N*B, D) needs num_views; or pass a (N, B, D) tensor / list of views")
        if x.size(0) % num_views:
            raise ValueError(f"first dimension {x.size(0)} is not divisible by num_views={num_views}")
        return x.reshape(num_views, x.size(0) // num_views, *x.shape[1:])
    if x.dim() < 2:
        raise ValueError(f"expected at least 2 dimensions, got shape {tuple(x.shape)}")
    return x


def instance_centers(x: Views, num_views: Optional[int] = None) -> torch.Tensor:
    """Instance-wise centres, Eq. (2): ``(N, B, D) -> (B, D)``."""
    return as_views(x, num_views).mean(dim=0)


@torch.no_grad()
def view_spread(x: Views, num_views: Optional[int] = None) -> torch.Tensor:
    """Mean squared distance of the views to their instance centre (a diagnostic);
    zero when all views of every image give the same output."""
    v = as_views(x, num_views)
    return (v - v.mean(dim=0, keepdim=True)).pow(2).mean()


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = -1) -> torch.Tensor:
    """Cross-entropy averaged over the labelled rows (``targets != ignore_index``).

    Identical to ``nn.CrossEntropyLoss()`` when every row is labelled; returns 0 when no
    row is labelled.
    """
    if ignore_index is None:
        return F.cross_entropy(logits, targets)
    if not bool((targets != ignore_index).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def center_distance(outputs: torch.Tensor, center: torch.Tensor, distance: str = "mse", reduction: str = "mean") -> torch.Tensor:
    """Distance between the outputs of one view ``(B, D)`` and the centres ``(B, D)``.

    ``reduction='mean'`` averages over the ``B x D`` entries (PyTorch default, as in the
    released code); ``reduction='sum'`` sums over the ``D`` entries and averages over the
    ``B`` images (Eq. (6) of the paper for ``'mse'``).
    """
    if distance == "kl":
        # nn.KLDivLoss()(log softmax(z), softmax(c)); log_softmax is the stable form of softmax(z).log()
        total = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(center, dim=1), reduction="sum")
    elif distance == "mse":
        total = F.mse_loss(outputs, center, reduction="sum")
    elif distance == "l1":
        total = F.l1_loss(outputs, center, reduction="sum")
    elif distance == "huber":
        total = F.smooth_l1_loss(outputs, center, reduction="sum", beta=1.0)
    else:
        raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")
    return total / (outputs.numel() if reduction == "mean" else outputs.size(0))


class InstanceWiseCenterLoss(nn.Module):
    """Instance-wise center loss, Eq. (3) of the paper:
    ``L = (1 - lambda_IC) * L_CE + lambda_IC * L_IC`` (see the module docstring).

    Args:
        alpha: ``lambda_IC``, the weight of ``L_IC`` (``1 - alpha`` weights the
            cross-entropy). The paper uses 0.7 for ResNet-18 and 0.5 for ResNeXt-29 and
            WideResNet-28-10 (0.1 in the distance ablation, Table IV).
        distance: ``'mse'`` (L2, default), ``'l1'``, ``'huber'`` or ``'kl'`` (Table IV).
        stopgrad: treat the instance centre as a constant (``StopGrad`` in Eq. (6);
            default ``True``). Because the centre is the mean of the views, the gradient
            that flows through it vanishes for ``'mse'`` and ``'kl'``, so there the option
            changes nothing; it only matters for ``'l1'`` and ``'huber'``.
        reduction: ``'mean'`` (default, as the released code: also averaged over the
            ``C`` logits) or ``'sum'`` (summed over the logits, as written in Eq. (6)).
        center_on: ``'logits'`` (default, the paper) or ``'features'`` (pull pooled
            penultimate features instead; ``forward`` then needs ``features``).
        num_views: only needed when the outputs are passed as a flat ``(N*B, D)`` tensor.
        ignore_index: label of unlabelled images; they get no cross-entropy but still
            contribute to ``L_IC``.

    Shapes:
        ``logits``: ``(N, B, C)``, a list of ``N`` tensors ``(B, C)``, or ``(N*B, C)``
        with ``num_views``; ``targets``: ``(B,)``; ``features`` (optional):
        ``(N, B, D)`` like ``logits``. Returns a scalar.

    Example::

        criterion = InstanceWiseCenterLoss(alpha=0.5)                # lambda_IC = 0.5, L2
        views = make_views(images, augment, num_views=2)             # (N, B, 3, H, W)
        logits = torch.stack([model(v) for v in views])               # (N, B, C)
        loss = criterion(logits, labels)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        distance: str = "mse",
        stopgrad: bool = True,
        reduction: str = "mean",
        center_on: str = "logits",
        num_views: Optional[int] = None,
        ignore_index: Optional[int] = -1,
    ) -> None:
        super().__init__()
        distance = _ALIASES.get(distance.lower(), distance.lower())
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        if center_on not in ("logits", "features"):
            raise ValueError(f"center_on must be 'logits' or 'features', got {center_on!r}")
        if distance == "kl" and center_on != "logits":
            raise ValueError("distance='kl' compares class distributions and needs center_on='logits'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha (lambda_IC) must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.distance = distance
        self.stopgrad = bool(stopgrad)
        self.reduction = reduction
        self.center_on = center_on
        self.num_views = num_views
        self.ignore_index = ignore_index

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, distance={self.distance!r}, stopgrad={self.stopgrad}, "
            f"reduction={self.reduction!r}, center_on={self.center_on!r}"
        )

    def forward(self, logits: Views, targets: torch.Tensor, features: Optional[Views] = None) -> torch.Tensor:
        logits = as_views(logits, self.num_views)
        if self.center_on == "features":
            if features is None:
                raise ValueError("center_on='features' needs the per-view features")
            pulled = as_views(features, self.num_views).flatten(2)
        else:
            pulled = logits
        num_views = logits.size(0)
        if targets.size(0) != logits.size(1):
            raise ValueError(f"got {targets.size(0)} targets for a batch of {logits.size(1)} images")

        center = pulled.mean(dim=0)                                                  # Eq. (2)
        if self.stopgrad:
            center = center.detach()
        has_labels = self.ignore_index is None or bool((targets != self.ignore_index).any())

        loss_ce = logits.new_zeros(())
        loss_ic = logits.new_zeros(())
        for n in range(num_views):
            if has_labels:
                loss_ce = loss_ce + masked_cross_entropy(logits[n], targets, self.ignore_index)   # Eq. (4)
            loss_ic = loss_ic + center_distance(pulled[n], center, self.distance, self.reduction)  # Eq. (6)
        return ((1.0 - self.alpha) * loss_ce + self.alpha * loss_ic) / num_views                  # Eq. (3)
