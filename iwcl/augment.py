"""Data augmentation pipelines of the release code (``archive/release_2023``).

The five training pipelines (``Solver.load_data`` in ``archive/release_2023/train.py``);
the paper uses standard ("Flip and Cropping"), cutout and autoaug (Table I) and augmix
(Table II)::

    standard : Resize(32), RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor
    cutout   : standard + Cutout(n_holes=1, length=8)
    randaug  : RandAugment(N=1, M=2), then standard
    autoaug  : Resize, RandomCrop, Flip, AutoAugment (CIFAR-10 policy), ToTensor
    augmix   : Resize, RandomCrop, Flip, AugMix(severity=3, width=3, depth=1-3), ToTensor

and the test pipeline is ``Resize(32), ToTensor``. There is no mean/std normalisation.

torchvision's own ``AutoAugment``, ``RandAugment`` and ``AugMix`` use different magnitude
mappings and operation sets than the implementations the release code used, so the
original implementations are ported here instead (credits below, kept from the release):

* ``Cutout``      - https://github.com/uoguelph-mlrg/Cutout (DeVries and Taylor, 2017)
* ``AutoAugment`` - https://github.com/4uiiurz1/pytorch-auto-augment (Cubuk et al., CVPR 2019)
* ``RandAugment`` - https://github.com/ildoonet/pytorch-randaugment (Cubuk et al., 2020);
  the release code imported it with ``pip install git+https://github.com/ildoonet/pytorch-randaugment``
* ``AugMix``      - https://github.com/google-research/augmix (Hendrycks et al., ICLR 2020),
  Copyright 2019 Google LLC, Apache License 2.0

Random numbers come from Python's ``random`` (AutoAugment, RandAugment) and NumPy's global
generator (Cutout, AugMix, RandAugment's CutoutAbs), exactly as in the originals.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as TF

try:  # Pillow >= 9.1
    _AFFINE = Image.Transform.AFFINE
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - old Pillow
    _AFFINE = Image.AFFINE
    _BILINEAR = Image.BILINEAR

AUGMENTATIONS = ("standard", "cutout", "randaug", "autoaug", "augmix")


# --------------------------------------------------------------------------- Cutout
class Cutout:
    """Randomly mask out ``n_holes`` square patches of side ``length`` from a tensor image
    ``(C, H, W)`` (uoguelph-mlrg/Cutout). The release config used ``n_holes=1, length=8``."""

    def __init__(self, n_holes: int = 1, length: int = 8) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        return img * torch.from_numpy(mask).expand_as(img)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_holes={self.n_holes}, length={self.length})"


# --------------------------------------------------------------------------- AutoAugment
def _affine_ndimage(img: Image.Image, matrix: np.ndarray) -> Image.Image:
    """Apply a 3x3 matrix around the image centre with ``scipy.ndimage.affine_transform``
    (cubic spline, zero fill), channel by channel, as in 4uiiurz1/pytorch-auto-augment."""
    from scipy import ndimage

    arr = np.array(img)
    o_x, o_y = float(arr.shape[0]) / 2 + 0.5, float(arr.shape[1]) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    matrix = offset_matrix @ matrix @ reset_matrix
    arr = np.stack(
        [ndimage.affine_transform(arr[:, :, c], matrix[:2, :2], matrix[:2, 2]) for c in range(arr.shape[2])], axis=2
    )
    return Image.fromarray(arr)


def _uniform_bin(low: float, high: float, magnitude: int) -> float:
    bins = np.linspace(low, high, 11)
    return random.uniform(bins[magnitude], bins[magnitude + 1])


def _aa_shear_x(img, m):
    return _affine_ndimage(img, np.array([[1, _uniform_bin(-0.3, 0.3, m), 0], [0, 1, 0], [0, 0, 1]]))


def _aa_shear_y(img, m):
    return _affine_ndimage(img, np.array([[1, 0, 0], [_uniform_bin(-0.3, 0.3, m), 1, 0], [0, 0, 1]]))


def _aa_translate_x(img, m):
    shift = np.array(img).shape[1] * _uniform_bin(-150 / 331, 150 / 331, m)
    return _affine_ndimage(img, np.array([[1, 0, 0], [0, 1, shift], [0, 0, 1]]))


def _aa_translate_y(img, m):
    shift = np.array(img).shape[0] * _uniform_bin(-150 / 331, 150 / 331, m)
    return _affine_ndimage(img, np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1]]))


def _aa_rotate(img, m):
    theta = np.deg2rad(_uniform_bin(-30, 30, m))
    return _affine_ndimage(
        img, np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    )


_AA_OPS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "ShearX": _aa_shear_x,
    "ShearY": _aa_shear_y,
    "TranslateX": _aa_translate_x,
    "TranslateY": _aa_translate_y,
    "Rotate": _aa_rotate,
    "AutoContrast": lambda img, m: ImageOps.autocontrast(img),
    "Invert": lambda img, m: ImageOps.invert(img),
    "Equalize": lambda img, m: ImageOps.equalize(img),
    "Solarize": lambda img, m: ImageOps.solarize(img, _uniform_bin(0, 256, m)),
    "Posterize": lambda img, m: ImageOps.posterize(img, int(round(_uniform_bin(4, 8, m)))),
    "Contrast": lambda img, m: ImageEnhance.Contrast(img).enhance(_uniform_bin(0.1, 1.9, m)),
    "Color": lambda img, m: ImageEnhance.Color(img).enhance(_uniform_bin(0.1, 1.9, m)),
    "Brightness": lambda img, m: ImageEnhance.Brightness(img).enhance(_uniform_bin(0.1, 1.9, m)),
    "Sharpness": lambda img, m: ImageEnhance.Sharpness(img).enhance(_uniform_bin(0.1, 1.9, m)),
}

# CIFAR-10 policy of AutoAugment: (op1, prob1, magnitude1, op2, prob2, magnitude2)
CIFAR10_POLICY: List[Tuple[str, float, int, str, float, int]] = [
    ("Invert", 0.1, 7, "Contrast", 0.2, 6),
    ("Rotate", 0.7, 2, "TranslateX", 0.3, 9),
    ("Sharpness", 0.8, 1, "Sharpness", 0.9, 3),
    ("ShearY", 0.5, 8, "TranslateY", 0.7, 9),
    ("AutoContrast", 0.5, 8, "Equalize", 0.9, 2),
    ("ShearY", 0.2, 7, "Posterize", 0.3, 7),
    ("Color", 0.4, 3, "Brightness", 0.6, 7),
    ("Sharpness", 0.3, 9, "Brightness", 0.7, 9),
    ("Equalize", 0.6, 5, "Equalize", 0.5, 1),
    ("Contrast", 0.6, 7, "Sharpness", 0.6, 5),
    ("Color", 0.7, 7, "TranslateX", 0.5, 8),
    ("Equalize", 0.3, 7, "AutoContrast", 0.4, 8),
    ("TranslateY", 0.4, 3, "Sharpness", 0.2, 6),
    ("Brightness", 0.9, 6, "Color", 0.2, 8),
    ("Solarize", 0.5, 2, "Invert", 0.0, 3),
    ("Equalize", 0.2, 0, "AutoContrast", 0.6, 0),
    ("Equalize", 0.2, 8, "Equalize", 0.6, 4),
    ("Color", 0.9, 9, "Equalize", 0.6, 6),
    ("AutoContrast", 0.8, 4, "Solarize", 0.2, 8),
    ("Brightness", 0.1, 3, "Color", 0.7, 0),
    ("Solarize", 0.4, 5, "AutoContrast", 0.9, 3),
    ("TranslateY", 0.9, 9, "TranslateY", 0.7, 9),
    ("AutoContrast", 0.9, 2, "Solarize", 0.8, 3),
    ("Equalize", 0.8, 8, "Invert", 0.1, 3),
    ("TranslateY", 0.7, 9, "AutoContrast", 0.9, 1),
]


class AutoAugment:
    """AutoAugment with the CIFAR-10 policy (4uiiurz1/pytorch-auto-augment): one of the
    25 sub-policies is drawn uniformly and each of its two operations is applied with its
    probability; magnitudes are sampled uniformly inside their bin. PIL image -> PIL image."""

    def __init__(self, policy=None) -> None:
        self.policy = list(policy or CIFAR10_POLICY)

    def __call__(self, img: Image.Image) -> Image.Image:
        op1, p1, m1, op2, p2, m2 = self.policy[random.randrange(len(self.policy))]
        if random.random() < p1:
            img = _AA_OPS[op1](img, m1)
        if random.random() < p2:
            img = _AA_OPS[op2](img, m2)
        return img

    def __repr__(self) -> str:
        return f"{type(self).__name__}(policy=CIFAR10, {len(self.policy)} sub-policies)"


# --------------------------------------------------------------------------- RandAugment
def _ra_shear_x(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, _AFFINE, (1, v, 0, 0, 1, 0))


def _ra_shear_y(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, _AFFINE, (1, 0, 0, v, 1, 0))


def _ra_translate_x_abs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, _AFFINE, (1, 0, v, 0, 1, 0))


def _ra_translate_y_abs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, _AFFINE, (1, 0, 0, 0, 1, v))


def _ra_rotate(img, v):
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def _ra_solarize_add(img, addition=0, threshold=128):
    arr = np.clip(np.array(img).astype(np.int64) + addition, 0, 255).astype(np.uint8)
    return ImageOps.solarize(Image.fromarray(arr), threshold)


def _ra_posterize(img, v):
    return ImageOps.posterize(img, max(1, int(v)))


def _ra_cutout_abs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)  # sic: numpy's uniform(low=w, high=1.0), as in the original
    y0 = np.random.uniform(h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    img = img.copy()
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), (125, 123, 114))
    return img


# (operation, min value, max value) - ildoonet's augment_list(), 16 operations
RANDAUGMENT_OPS = [
    (lambda img, v: ImageOps.autocontrast(img), 0, 1),
    (lambda img, v: ImageOps.equalize(img), 0, 1),
    (lambda img, v: ImageOps.invert(img), 0, 1),
    (_ra_rotate, 0, 30),
    (_ra_posterize, 0, 4),
    (lambda img, v: ImageOps.solarize(img, v), 0, 256),
    (_ra_solarize_add, 0, 110),
    (lambda img, v: ImageEnhance.Color(img).enhance(v), 0.1, 1.9),
    (lambda img, v: ImageEnhance.Contrast(img).enhance(v), 0.1, 1.9),
    (lambda img, v: ImageEnhance.Brightness(img).enhance(v), 0.1, 1.9),
    (lambda img, v: ImageEnhance.Sharpness(img).enhance(v), 0.1, 1.9),
    (_ra_shear_x, 0.0, 0.3),
    (_ra_shear_y, 0.0, 0.3),
    (_ra_cutout_abs, 0, 40),
    (_ra_translate_x_abs, 0.0, 100),
    (_ra_translate_y_abs, 0.0, 100),
]


class RandAugment:
    """RandAugment (ildoonet/pytorch-randaugment): apply ``n`` operations drawn with
    replacement from 16, each at value ``m / 30 * (max - min) + min``. PIL -> PIL.
    The release config used ``n=1, m=2``."""

    def __init__(self, n: int = 1, m: int = 2) -> None:
        if not 0 <= m <= 30:
            raise ValueError(f"RandAugment magnitude must be in [0, 30], got {m}")
        self.n = n
        self.m = m
        self.augment_list = RANDAUGMENT_OPS

    def __call__(self, img: Image.Image) -> Image.Image:
        for op, minval, maxval in random.choices(self.augment_list, k=self.n):
            img = op(img, (float(self.m) / 30) * float(maxval - minval) + minval)
        return img

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n}, m={self.m})"


# --------------------------------------------------------------------------- AugMix
# Adapted from https://github.com/google-research/augmix (augmentations.py, cifar.py):
# Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0); distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. Changes: image size taken from the image
# instead of a module constant, Pillow enum constants, packaged as a transform class.
def _int_parameter(level: float, maxval: float) -> int:
    return int(level * maxval / 10)


def _float_parameter(level: float, maxval: float) -> float:
    return float(level) * maxval / 10.0


def _sample_level(n: float) -> float:
    return np.random.uniform(low=0.1, high=n)


def _am_posterize(img, level):
    return ImageOps.posterize(img, 4 - _int_parameter(_sample_level(level), 4))


def _am_rotate(img, level):
    degrees = _int_parameter(_sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return img.rotate(degrees, resample=_BILINEAR)


def _am_solarize(img, level):
    return ImageOps.solarize(img, 256 - _int_parameter(_sample_level(level), 256))


def _am_shear_x(img, level):
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return img.transform(img.size, _AFFINE, (1, level, 0, 0, 1, 0), resample=_BILINEAR)


def _am_shear_y(img, level):
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return img.transform(img.size, _AFFINE, (1, 0, 0, level, 1, 0), resample=_BILINEAR)


def _am_translate_x(img, level):
    level = _int_parameter(_sample_level(level), img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return img.transform(img.size, _AFFINE, (1, 0, level, 0, 1, 0), resample=_BILINEAR)


def _am_translate_y(img, level):
    level = _int_parameter(_sample_level(level), img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return img.transform(img.size, _AFFINE, (1, 0, 0, 0, 1, level), resample=_BILINEAR)


def _am_enhance(enhancer):
    def op(img, level):
        return enhancer(img).enhance(_float_parameter(_sample_level(level), 1.8) + 0.1)

    return op


AUGMIX_OPS = [
    lambda img, level: ImageOps.autocontrast(img),
    lambda img, level: ImageOps.equalize(img),
    _am_posterize,
    _am_rotate,
    _am_solarize,
    _am_shear_x,
    _am_shear_y,
    _am_translate_x,
    _am_translate_y,
]
# operations that overlap with the corruptions of CIFAR-10-C (off by default, as in the release)
AUGMIX_OPS_ALL = AUGMIX_OPS + [
    _am_enhance(ImageEnhance.Color),
    _am_enhance(ImageEnhance.Contrast),
    _am_enhance(ImageEnhance.Brightness),
    _am_enhance(ImageEnhance.Sharpness),
]


class AugMix:
    """AugMix (google-research/augmix): mix ``width`` chains of 1-3 random operations
    with Dirichlet(1) weights and blend with the input using Beta(1, 1). PIL -> PIL (the
    mixture is converted back to 8 bit, as in the release code)."""

    def __init__(self, severity: int = 3, width: int = 3, depth: int = -1, all_ops: bool = False) -> None:
        self.severity = severity
        self.width = width
        self.depth = depth
        self.all_ops = all_ops

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = AUGMIX_OPS_ALL if self.all_ops else AUGMIX_OPS
        ws = np.float32(np.random.dirichlet([1] * self.width))
        m = np.float32(np.random.beta(1, 1))
        mix = torch.zeros_like(TF.to_tensor(img))
        for i in range(self.width):
            image_aug = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(ops)
                image_aug = op(image_aug, self.severity)
            mix += ws[i] * TF.to_tensor(image_aug)
        mixed = (1 - m) * TF.to_tensor(img) + m * mix
        return TF.to_pil_image(mixed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(severity={self.severity}, width={self.width}, depth={self.depth}, all_ops={self.all_ops})"


# --------------------------------------------------------------------------- pipelines
def build_train_transform(
    name: str = "cutout",
    image_size: int = 32,
    cutout_length: int = 8,
    cutout_holes: int = 1,
    randaug_n: int = 1,
    randaug_m: int = 2,
) -> T.Compose:
    """Training transform of the release code (PIL image -> tensor in [0, 1])."""
    # every dataset is resized to image_size x image_size (paper Sec. III-A; the release used
    # Resize(32), which is the same for the square CIFAR/SVHN/STL-10 images)
    base = [T.Resize((image_size, image_size)), T.RandomCrop(image_size, padding=4), T.RandomHorizontalFlip()]
    if name == "standard":
        return T.Compose(base + [T.ToTensor()])
    if name == "cutout":
        return T.Compose(base + [T.ToTensor(), Cutout(n_holes=cutout_holes, length=cutout_length)])
    if name == "randaug":
        return T.Compose([RandAugment(randaug_n, randaug_m)] + base + [T.ToTensor()])
    if name == "autoaug":
        return T.Compose(base + [AutoAugment(), T.ToTensor()])
    if name == "augmix":
        return T.Compose(base + [AugMix(), T.ToTensor()])
    raise ValueError(f"unknown augmentation {name!r}; choose from {AUGMENTATIONS}")


def build_test_transform(image_size: int = 32) -> T.Compose:
    """Test (and JS-divergence 'clean view') transform of the release code."""
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])


__all__ = [
    "AUGMENTATIONS",
    "AugMix",
    "AutoAugment",
    "CIFAR10_POLICY",
    "Cutout",
    "RandAugment",
    "build_test_transform",
    "build_train_transform",
]

