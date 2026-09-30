"""Comparison methods of the paper, as implemented in the release code.

All objectives share the interface of :class:`iwcl.InstanceWiseCenterLoss`::

    loss = criterion(logits, targets, features)   # logits (N, B, C), features (N, B, D)

and, like the release loop (``archive/release_2023/train.py``), sum a per-view loss
over the ``N`` views and divide by ``N``. Constants (weight 0.1, JSD weight 12, margin 1,
...) are those of the release code. Call ``criterion.after_backward()`` between
``loss.backward()`` and ``optimizer.step()`` (a no-op except for the class-centre
losses, whose centre gradients the release code rescaled by ``1 / weight``).

* :class:`MultiViewCrossEntropy` - ``'baseline'``: cross-entropy on every view.
* :class:`CenterLoss` - ``'center_loss'``: Wen et al., ECCV 2016.
* :class:`ContrastiveCenterLoss` - ``'contrastive_center_loss'``: Qi and Su, 2017.
* :class:`InstanceTripletLoss` - ``'triplet_loss'``: triplet margin loss between views.
* :class:`AugMixJSDLoss` - ``'jsd'`` (``'augmix'`` in the release config): Jensen-Shannon
  consistency of AugMix (Hendrycks et al., ICLR 2020), the "JS-divergence" of Table II.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import Views, as_views, masked_cross_entropy


class _MultiViewObjective(nn.Module):
    """Shared plumbing: shape handling and the per-view average."""

    def __init__(self, ignore_index: Optional[int] = -1, num_views: Optional[int] = None) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.num_views = num_views

    def ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(logits, targets, self.ignore_index)

    def labelled(self, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is None:
            return torch.ones_like(targets, dtype=torch.bool)
        return targets != self.ignore_index

    def after_backward(self) -> None:
        """Hook called after ``loss.backward()``; no-op by default."""


class MultiViewCrossEntropy(_MultiViewObjective):
    """``'baseline'``: ``L = 1/N sum_k CE(z^(k), y)``.

    With ``N = 1`` this is standard training; with ``N > 1`` every base image is seen
    in ``N`` augmented versions per step (``N = 2`` in the paper's Table I).
    """

    def forward(self, logits: Views, targets: torch.Tensor, features: Optional[Views] = None) -> torch.Tensor:
        logits = as_views(logits, self.num_views)
        return sum(self.ce(z, targets) for z in logits) / logits.size(0)


def center_loss_term(features: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Center loss of Wen et al. (ECCV 2016), KaiyangZhou/pytorch-center-loss formulation:
    ``1/B sum_i clamp(||f_i - c_{y_i}||^2, 1e-12, 1e12)`` (the masked-out entries are
    clamped to 1e-12 too, exactly as in the reference code)."""
    batch_size, num_classes = features.size(0), centers.size(0)
    distmat = (
        features.pow(2).sum(dim=1, keepdim=True).expand(batch_size, num_classes)
        + centers.pow(2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
    )
    distmat = torch.addmm(distmat, features, centers.t(), beta=1, alpha=-2)
    classes = torch.arange(num_classes, device=features.device)
    mask = labels.unsqueeze(1).expand(batch_size, num_classes).eq(classes.expand(batch_size, num_classes))
    return (distmat * mask.float()).clamp(min=1e-12, max=1e12).sum() / batch_size


class CenterLoss(_MultiViewObjective):
    """``'center_loss'``: ``L = 1/N sum_k [CE(z^(k), y) + weight * center(f^(k), y)]``.

    The class centres are learnable (``torch.randn`` initialisation) and trained by the
    same SGD optimiser as the network; as in the release code their gradient is
    multiplied by ``1 / weight`` in :meth:`after_backward`, so they move with the
    un-weighted center-loss gradient. The penultimate features ``f`` must be passed.
    """

    def __init__(self, num_classes: int, feat_dim: int, weight: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def extra_repr(self) -> str:
        return f"num_classes={self.centers.size(0)}, feat_dim={self.centers.size(1)}, weight={self.weight}"

    def forward(self, logits: Views, targets: torch.Tensor, features: Optional[Views] = None) -> torch.Tensor:
        if features is None:
            raise ValueError("CenterLoss needs the per-view features")
        logits, features = as_views(logits, self.num_views), as_views(features, self.num_views).flatten(2)
        keep = self.labelled(targets)
        loss = logits.new_zeros(())
        for z, f in zip(logits, features):
            reg = center_loss_term(f[keep], targets[keep], self.centers) if bool(keep.any()) else z.sum() * 0.0
            loss = loss + self.ce(z, targets) + self.weight * reg
        return loss / logits.size(0)

    def after_backward(self) -> None:
        if self.centers.grad is not None:
            self.centers.grad.mul_(1.0 / self.weight)


def contrastive_center_term(features: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, lambda_c: float = 1.0) -> torch.Tensor:
    """Contrastive-center loss (Qi and Su, 2017), lyakaap/image-feature-learning-pytorch
    formulation as modified in the release code (extra ``/ 0.1``)::

        lambda_c / (2 B) * sum_i ||f_i - c_{y_i}||^2 / (sum_i sum_{j != y_i} ||f_i - c_j||^2 + 1e-6) / 0.1
    """
    batch_size = features.size(0)
    distance = (features.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)  # (B, num_classes)
    intra = distance.gather(1, labels.unsqueeze(1)).sum()
    inter = distance.sum() - intra
    return (lambda_c / 2.0 / batch_size) * intra / (inter + 1e-6) / 0.1


class ContrastiveCenterLoss(_MultiViewObjective):
    """``'contrastive_center_loss'``: ``L = 1/N sum_k [CE(z^(k), y) + weight * ccl(f^(k), y)]``.

    Release-code behaviour: the centres were created as
    ``nn.Parameter(torch.randn(...)).cuda()``, which returns a plain tensor, so they were
    never registered as parameters and stayed at their random initialisation. This is
    the default here (``learnable_centers=False``, centres stored as a buffer). Pass
    ``learnable_centers=True`` to train them like :class:`CenterLoss` (gradient rescaled
    by ``1 / weight``).
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        weight: float = 0.1,
        lambda_c: float = 1.0,
        learnable_centers: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.lambda_c = float(lambda_c)
        self.learnable_centers = bool(learnable_centers)
        centers = torch.randn(num_classes, feat_dim)
        if self.learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.centers.size(0)}, feat_dim={self.centers.size(1)}, weight={self.weight}, "
            f"lambda_c={self.lambda_c}, learnable_centers={self.learnable_centers}"
        )

    def forward(self, logits: Views, targets: torch.Tensor, features: Optional[Views] = None) -> torch.Tensor:
        if features is None:
            raise ValueError("ContrastiveCenterLoss needs the per-view features")
        logits, features = as_views(logits, self.num_views), as_views(features, self.num_views).flatten(2)
        keep = self.labelled(targets)
        loss = logits.new_zeros(())
        for z, f in zip(logits, features):
            if bool(keep.any()):
                reg = contrastive_center_term(f[keep], targets[keep], self.centers, self.lambda_c)
            else:
                reg = z.sum() * 0.0
            loss = loss + self.ce(z, targets) + self.weight * reg
        return loss / logits.size(0)

    def after_backward(self) -> None:
        if self.learnable_centers and self.centers.grad is not None:
            self.centers.grad.mul_(1.0 / self.weight)


class InstanceTripletLoss(_MultiViewObjective):
    """``'triplet_loss'``: ``L = 1/N sum_k [CE(z^(k), y) + weight * T(a^(k), a^(k+1), roll(a^(k+1), shift))]``.

    ``T`` is ``nn.TripletMarginLoss(margin=1.0, p=2)``: the anchor is a view of image
    ``i``, the positive the next view (cyclically) of the same image and the negative the
    next view of image ``i - shift`` in the batch (the release code built it as
    ``torch.cat([z[B-2:], z[:B-2]])``, i.e. ``shift = 2``). ``a`` are the logits, as in the
    release code (``on='logits'``), or the pooled features (``on='features'``). Needs ``N >= 2``.
    """

    def __init__(self, weight: float = 0.1, margin: float = 1.0, shift: int = 2, on: str = "logits", **kwargs) -> None:
        super().__init__(**kwargs)
        if on not in ("logits", "features"):
            raise ValueError(f"on must be 'logits' or 'features', got {on!r}")
        self.weight = float(weight)
        self.shift = int(shift)
        self.on = on
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)

    def extra_repr(self) -> str:
        return f"weight={self.weight}, shift={self.shift}, on={self.on!r}"

    def forward(self, logits: Views, targets: torch.Tensor, features: Optional[Views] = None) -> torch.Tensor:
        logits = as_views(logits, self.num_views)
        if self.on == "features":
            if features is None:
                raise ValueError("InstanceTripletLoss(on='features') needs the per-view features")
            embed = as_views(features, self.num_views).flatten(2)
        else:
            embed = logits
        num_views = logits.size(0)
        if num_views < 2:
            raise ValueError("InstanceTripletLoss needs at least two views per image")
        loss = logits.new_zeros(())
        for k in range(num_views):
            positive = embed[(k + 1) % num_views]
            negative = torch.roll(positive, shifts=self.shift, dims=0)
            loss = loss + self.ce(logits[k], targets) + self.weight * self.triplet(embed[k], positive, negative)
        return loss / num_views


def augmix_jsd(logits_clean: torch.Tensor, logits_aug1: torch.Tensor, logits_aug2: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence of the three predictive distributions (AugMix):
    ``mean_j KL(p_j || M)`` with the mixture ``M = clamp(mean_j p_j, 1e-7, 1)``.

    Same value as the AugMix code (``F.kl_div(log M, p_j)``), but computed from
    log-probabilities, so the gradient stays finite when a probability underflows to 0
    (the original form then gives NaN).
    """
    log_probs = [F.log_softmax(z, dim=1) for z in (logits_clean, logits_aug1, logits_aug2)]
    # clamp the mixture distribution to avoid exploding KL divergence (as in AugMix)
    log_mixture = torch.clamp(sum(lp.exp() for lp in log_probs) / 3.0, 1e-7, 1).log()
    return sum(F.kl_div(log_mixture, lp, reduction="batchmean", log_target=True) for lp in log_probs) / 3.0


class AugMixJSDLoss(_MultiViewObjective):
    """``'jsd'``: ``L = CE(z^(0), y) + weight * JSD(z^(0), z^(1), z^(2))``, ``weight = 12`` (AugMix code).

    Needs exactly ``N = 3`` views ordered ``[clean, augmented, augmented]``; in the
    release data pipeline the clean view is the un-augmented image (``Resize + ToTensor``)
    and the other two are independent AugMix views.
    """

    def __init__(self, weight: float = 12.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = float(weight)

    def extra_repr(self) -> str:
        return f"weight={self.weight}"

    def forward(self, logits: Views, targets: torch.Tensor, features: Optional[Views] = None) -> torch.Tensor:
        logits = as_views(logits, self.num_views)
        if logits.size(0) != 3:
            raise ValueError(f"AugMixJSDLoss needs 3 views [clean, aug1, aug2], got {logits.size(0)}")
        return self.ce(logits[0], targets) + self.weight * augmix_jsd(logits[0], logits[1], logits[2])
