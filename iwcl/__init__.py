"""Instance-wise Center Loss (IWCL) - official PyTorch implementation.

K. Madono, M. Tanaka, M. Onishi, "Instance-wise Center Loss for Efficient Training of
Deep Convolutional Neural Networks", IEEE GCCE 2022, https://ieeexplore.ieee.org/document/10014037

Minimal use::

    import torch, iwcl
    criterion = iwcl.InstanceWiseCenterLoss(alpha=0.5)         # lambda_IC = 0.5, L2 distance
    views = iwcl.make_views(images, augment, num_views=2)      # (N, B, 3, H, W)
    logits = torch.stack([model(v) for v in views])            # (N, B, num_classes)
    loss = criterion(logits, labels)
"""
from .baselines import AugMixJSDLoss, CenterLoss, ContrastiveCenterLoss, InstanceTripletLoss, MultiViewCrossEntropy
from .criteria import ALIASES, METHODS, build_criterion
from .losses import InstanceWiseCenterLoss, as_views, instance_centers, view_spread
from .views import MultiViewTransform, flatten_views, make_views

__version__ = "1.0.0"

__all__ = [
    "ALIASES",
    "AugMixJSDLoss",
    "CenterLoss",
    "ContrastiveCenterLoss",
    "InstanceTripletLoss",
    "InstanceWiseCenterLoss",
    "METHODS",
    "MultiViewCrossEntropy",
    "MultiViewTransform",
    "as_views",
    "build_criterion",
    "flatten_views",
    "instance_centers",
    "make_views",
    "view_spread",
]
