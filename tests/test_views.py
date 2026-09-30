import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

import iwcl
from iwcl.data import CUB200, build_datasets, class_balanced_subset


def test_make_views_shape_and_independence():
    torch.manual_seed(0)
    images = torch.rand(5, 3, 16, 16)
    views = iwcl.make_views(images, T.RandomCrop(16, padding=4), num_views=3)
    assert views.shape == (3, 5, 3, 16, 16)
    assert not torch.equal(views[0], views[1])
    assert iwcl.flatten_views(views).shape == (15, 3, 16, 16)
    torch.testing.assert_close(iwcl.flatten_views(views)[5:10], views[1])
    with pytest.raises(ValueError):
        iwcl.make_views(images, T.RandomCrop(16), num_views=0)


def test_multiview_transform_with_and_without_clean_view():
    img = Image.fromarray(np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8))
    aug = T.Compose([T.RandomCrop(32, padding=4), T.ToTensor()])
    clean = T.ToTensor()
    views = iwcl.MultiViewTransform(aug, num_views=3)(img)
    assert len(views) == 3 and all(v.shape == (3, 32, 32) for v in views)
    views = iwcl.MultiViewTransform(aug, num_views=3, clean_transform=clean)(img)
    torch.testing.assert_close(views[0], clean(img))


def test_fake_dataset_yields_k_views_and_subset_is_class_balanced():
    train, test, num_classes = build_datasets(
        "fake", train_transform=T.ToTensor(), test_transform=T.ToTensor(), num_views=2, fake_size=8
    )
    views, label = train[0]
    assert num_classes == 10 and len(train) == 8 and len(test) == 4
    assert len(views) == 2 and views[0].shape == (3, 32, 32) and isinstance(label, int)

    class Toy(torch.utils.data.Dataset):
        targets = [0, 1, 2] * 5

        def __len__(self):
            return 15

    subset = class_balanced_subset(Toy(), per_class=2, seed=0)
    assert sorted(np.array(Toy.targets)[subset.indices].tolist()) == [0, 0, 1, 1, 2, 2]


def test_cub200_reads_the_official_split(tmp_path):
    root = tmp_path / "CUB_200_2011"
    rng = np.random.RandomState(0)
    lines = {"images.txt": [], "image_class_labels.txt": [], "train_test_split.txt": []}
    for i, (cls, is_train) in enumerate([(1, 1), (1, 0), (2, 1), (3, 0), (200, 1)], start=1):
        rel = f"{cls:03d}.bird/img_{i}.jpg"
        (root / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.randint(0, 256, (50 + i, 70, 3), dtype=np.uint8)).save(root / "images" / rel)
        lines["images.txt"].append(f"{i} {rel}")
        lines["image_class_labels.txt"].append(f"{i} {cls}")
        lines["train_test_split.txt"].append(f"{i} {is_train}")
    for name, rows in lines.items():
        (root / name).write_text("\n".join(rows) + "\n")

    train, test = CUB200(tmp_path, train=True), CUB200(root, train=False)   # the parent directory also works
    assert (len(train), len(test)) == (3, 2)
    assert train.targets == [0, 1, 199] and test.targets == [0, 2]
    img, label = train[2]
    assert img.size == (32, 32) and label == 199
    train_set, test_set, num_classes = build_datasets(
        "cub200", root=str(root), train_transform=T.ToTensor(), test_transform=T.ToTensor(), num_views=2
    )
    views, _ = train_set[0]
    assert num_classes == 200 and len(views) == 2 and views[0].shape == (3, 32, 32)
    with pytest.raises(FileNotFoundError):
        CUB200(tmp_path / "missing")
