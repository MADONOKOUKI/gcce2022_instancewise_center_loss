"""Datasets of the paper's experiments, returning ``N`` augmented views per training image.

The paper uses CIFAR-10, CIFAR-100 and CUB-200-2011, all resized to 32x32 (Sec. III-A);
SVHN and STL-10 were also supported by the release code. The release code shipped modified
copies of the torchvision datasets that returned ten transformed copies of every image;
here the standard torchvision datasets are wrapped with
:class:`~iwcl.views.MultiViewTransform`, which produces exactly the ``N`` views used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF

from .views import MultiViewTransform

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "cub200": 200, "svhn": 10, "stl10": 10, "fake": 10}
DATASETS = tuple(NUM_CLASSES)

CUB_URL = "https://www.vision.caltech.edu/datasets/cub_200_2011/"


class CUB200(Dataset):
    """Caltech-UCSD Birds-200-2011 with its official train/test split (``train_test_split.txt``).

    ``root`` is the extracted ``CUB_200_2011`` directory (the one containing
    ``images.txt``) or its parent. The dataset is not downloaded automatically; get
    ``CUB_200_2011.tgz`` from https://www.vision.caltech.edu/datasets/cub_200_2011/.
    Images are decoded once and resized to ``image_size`` x ``image_size`` (the paper
    resized every dataset to 32x32; the training transforms start with the same resize,
    so this only saves time). ``image_size=None`` keeps the original images.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, image_size: Optional[int] = 32) -> None:
        root = Path(root).expanduser()
        if not (root / "images.txt").exists() and (root / "CUB_200_2011" / "images.txt").exists():
            root = root / "CUB_200_2011"
        if not (root / "images.txt").exists():
            raise FileNotFoundError(
                f"CUB-200-2011 not found in {root}: download CUB_200_2011.tgz from {CUB_URL}, extract it and "
                "pass the CUB_200_2011 directory (or its parent) as --data-root"
            )

        def table(name: str) -> dict:
            with open(root / name) as f:
                return dict(line.split(maxsplit=1) for line in f if line.strip())

        paths, labels, split = table("images.txt"), table("image_class_labels.txt"), table("train_test_split.txt")
        ids = sorted((i for i in paths if split[i].strip() == ("1" if train else "0")), key=int)
        self.root = root
        self.train = train
        self.transform = transform
        self.image_size = image_size
        self.samples: List[Path] = [root / "images" / paths[i].strip() for i in ids]
        self.targets: List[int] = [int(labels[i]) - 1 for i in ids]
        self._images = None
        if image_size is not None:
            with ThreadPoolExecutor(max_workers=8) as pool:
                self._images = list(pool.map(self._load, self.samples))

    def _load(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.image_size is not None:
            img = TF.resize(img, [self.image_size, self.image_size])
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img = self._images[index] if self._images is not None else self._load(self.samples[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]


def _labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise ValueError(f"cannot read the labels of {type(dataset).__name__}")


def class_balanced_subset(dataset: Dataset, per_class: int, seed: int = 0) -> Subset:
    """Keep ``per_class`` random labelled images of every class (seeded)."""
    labels = _labels(dataset)
    rng = np.random.RandomState(seed)
    indices = []
    for c in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == c)
        if len(idx) < per_class:
            raise ValueError(f"class {c} has only {len(idx)} images, fewer than {per_class}")
        indices.extend(rng.choice(idx, per_class, replace=False).tolist())
    return Subset(dataset, sorted(indices))


def build_datasets(
    name: str,
    root: str = "./data",
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    num_views: int = 2,
    clean_view: bool = False,
    download: bool = True,
    subset_per_class: int = 0,
    stl10_split: str = "train",
    fake_size: int = 256,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, int]:
    """Return ``(train_set, test_set, num_classes)``.

    Training items are ``(views, label)`` with ``views`` a list of ``num_views`` tensors
    (view 0 is the un-augmented image when ``clean_view=True``, for the JS-divergence);
    test items are ``(image, label)``.

    Args:
        name: ``cifar10``, ``cifar100``, ``cub200`` (``root`` = the ``CUB_200_2011``
            directory; no automatic download), ``svhn`` (``train``/``test`` splits),
            ``stl10`` or ``fake`` (``torchvision.datasets.FakeData``, 32x32, 10 classes,
            ``fake_size`` training and ``fake_size // 2`` test images; for smoke tests).
        subset_per_class: if > 0, train on this many random images per class (seeded by
            ``seed``; Table III of the paper uses 10, 50 and 100 images per class of CIFAR-10).
        stl10_split: ``'train'`` (5,000 labelled images) or ``'train+unlabeled'``
            (adds 100,000 unlabelled images with label -1, as in the release config;
            they get no cross-entropy but still enter the instance-wise centre term).
    """
    if name not in NUM_CLASSES:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    train_tf = MultiViewTransform(train_transform, num_views, clean_transform=test_transform if clean_view else None)

    if name == "cifar10":
        train = datasets.CIFAR10(root, train=True, transform=train_tf, download=download)
        test = datasets.CIFAR10(root, train=False, transform=test_transform, download=download)
    elif name == "cifar100":
        train = datasets.CIFAR100(root, train=True, transform=train_tf, download=download)
        test = datasets.CIFAR100(root, train=False, transform=test_transform, download=download)
    elif name == "cub200":
        train = CUB200(root, train=True, transform=train_tf)
        test = CUB200(root, train=False, transform=test_transform)
    elif name == "svhn":
        train = datasets.SVHN(root, split="train", transform=train_tf, download=download)
        test = datasets.SVHN(root, split="test", transform=test_transform, download=download)
    elif name == "stl10":
        if stl10_split not in ("train", "train+unlabeled"):
            raise ValueError(f"stl10_split must be 'train' or 'train+unlabeled', got {stl10_split!r}")
        train = datasets.STL10(root, split=stl10_split, transform=train_tf, download=download)
        test = datasets.STL10(root, split="test", transform=test_transform, download=download)
    else:  # fake
        train = datasets.FakeData(fake_size, (3, 32, 32), 10, transform=train_tf, random_offset=0)
        test = datasets.FakeData(max(fake_size // 2, 1), (3, 32, 32), 10, transform=test_transform, random_offset=10**6)

    if subset_per_class > 0:
        if name == "fake":
            raise ValueError("subset_per_class is not supported for the fake dataset")
        train = class_balanced_subset(train, subset_per_class, seed)
    return train, test, NUM_CLASSES[name]
