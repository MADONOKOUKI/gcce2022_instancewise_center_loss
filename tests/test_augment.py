import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from conftest import seed_all
from iwcl.augment import AUGMENTATIONS, RANDAUGMENT_OPS, AugMix, AutoAugment, Cutout, RandAugment, build_test_transform, build_train_transform


def _image(seed=0, size=32):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, (size, size, 3), dtype=np.uint8))


def _release_pipeline(name, cutout, autoaug, augmix):
    """Training transforms as built in archive/release_2023/train.py (Solver.load_data)."""
    base = [T.Resize(32), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if name == "cutout":
        return T.Compose(base + [T.ToTensor(), cutout.Cutout(n_holes=1, length=8)])
    if name == "autoaug":
        return T.Compose(base + [autoaug.AutoAugment(), T.ToTensor()])
    return T.Compose(base + [augmix.AugMix(), T.ToTensor()])


@pytest.mark.parametrize("name", ["cutout", "autoaug", "augmix"])
def test_pipelines_match_release_code(name, load_release):
    cutout = load_release("augmentation/cutout.py")
    autoaug = load_release("augmentation/autoaug.py")
    augmix = load_release("augmentation/augmix.py", argv=["augmix"])
    reference = _release_pipeline(name, cutout, autoaug, augmix)
    ours = build_train_transform(name)
    for seed in range(20):
        img = _image(seed)
        seed_all(seed)
        expected = reference(img)
        seed_all(seed)
        torch.testing.assert_close(ours(img), expected)


def test_ported_operations_match_release_code(load_release):
    autoaug = load_release("augmentation/autoaug.py")
    augmix = load_release("augmentation/augmix.py", argv=["augmix"])
    cutout = load_release("augmentation/cutout.py")
    for seed in range(30):
        img = _image(seed)
        seed_all(seed)
        a = np.array(autoaug.AutoAugment()(img))
        seed_all(seed)
        np.testing.assert_array_equal(np.array(AutoAugment()(img)), a)
        seed_all(seed)
        a = np.array(augmix.AugMix()(img))
        seed_all(seed)
        np.testing.assert_array_equal(np.array(AugMix()(img)), a)
        x = torch.rand(3, 32, 32)
        seed_all(seed)
        a = cutout.Cutout(n_holes=1, length=8)(x)
        seed_all(seed)
        torch.testing.assert_close(Cutout(n_holes=1, length=8)(x), a)


def test_randaugment_operation_list_and_values():
    # ildoonet/pytorch-randaugment: 16 operations; value = m / 30 * (max - min) + min
    assert len(RANDAUGMENT_OPS) == 16
    ranges = [(lo, hi) for _, lo, hi in RANDAUGMENT_OPS]
    assert ranges[4] == (0, 4) and ranges[5] == (0, 256) and ranges[13] == (0, 40) and ranges[14] == (0.0, 100)
    img = _image(1)
    for m in (0, 2, 14, 30):
        for n in (1, 2, 3):
            for seed in range(10):
                seed_all(seed)
                out = RandAugment(n, m)(img)
                assert out.size == img.size and out.mode == "RGB"
    with pytest.raises(ValueError):
        RandAugment(1, 31)


@pytest.mark.parametrize("name", AUGMENTATIONS)
def test_train_transforms_are_deterministic_under_a_seed(name):
    tf = build_train_transform(name)
    img = _image(3)
    seed_all(7)
    a = tf(img)
    seed_all(7)
    b = tf(img)
    assert a.shape == (3, 32, 32) and a.dtype == torch.float32
    assert 0.0 <= a.min().item() and a.max().item() <= 1.0
    torch.testing.assert_close(a, b)


def test_every_image_is_resized_to_32x32():
    # the paper resizes all datasets (also the non-square CUB-200 images) to 32x32
    wide = Image.fromarray(np.random.RandomState(0).randint(0, 256, (60, 90, 3), dtype=np.uint8))
    for img in (_image(0, size=96), wide):
        assert build_test_transform()(img).shape == (3, 32, 32)
        for name in AUGMENTATIONS:
            assert build_train_transform(name)(img).shape == (3, 32, 32)
