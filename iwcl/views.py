"""Helpers that turn images into ``N`` augmented views of the same base image."""
from __future__ import annotations

from typing import Callable, List, Optional

import torch


def make_views(images: torch.Tensor, transform: Callable[[torch.Tensor], torch.Tensor], num_views: int = 2) -> torch.Tensor:
    """Turn a batch into ``N`` independently augmented views of every image.

    Args:
        images: a batch ``(B, C, H, W)``.
        transform: a random augmentation applied to one image tensor ``(C, H, W)``,
            e.g. ``torchvision.transforms.RandomCrop(32, padding=4)``. It is called
            separately for every image and every view.
        num_views: ``N``.

    Returns:
        A tensor ``(N, B, C', H', W')``; ``views[k]`` is the ``k``-th view of the batch.
    """
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    return torch.stack([torch.stack([transform(img) for img in images]) for _ in range(num_views)])


def flatten_views(views: torch.Tensor) -> torch.Tensor:
    """``(N, B, ...) -> (N*B, ...)``, view-major (the layout ``InstanceWiseCenterLoss`` expects
    for flat inputs with ``num_views=N``). Note that one forward pass over the flattened
    batch computes BatchNorm statistics over all views together, whereas the release code
    forwarded every view separately."""
    return views.reshape(views.size(0) * views.size(1), *views.shape[2:])


class MultiViewTransform:
    """Dataset transform that returns ``N`` augmented views of one image.

    Use it as the ``transform`` of any torchvision dataset; the default collate function
    then yields a list of ``N`` tensors ``(B, C, H, W)`` per batch.

    Args:
        transform: the random training transform (PIL image -> tensor).
        num_views: ``N``.
        clean_transform: if given, view 0 is ``clean_transform(img)`` (the un-augmented
            image, as needed by the JS-divergence of AugMix) and views ``1..N-1`` are augmented.
    """

    def __init__(self, transform: Callable, num_views: int = 2, clean_transform: Optional[Callable] = None) -> None:
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        self.transform = transform
        self.num_views = num_views
        self.clean_transform = clean_transform

    def __call__(self, img) -> List[torch.Tensor]:
        if self.clean_transform is None:
            return [self.transform(img) for _ in range(self.num_views)]
        return [self.clean_transform(img)] + [self.transform(img) for _ in range(self.num_views - 1)]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_views={self.num_views}, clean_view={self.clean_transform is not None},\n"
            f"  transform={self.transform})"
        )
