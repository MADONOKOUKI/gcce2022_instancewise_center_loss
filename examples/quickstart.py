"""Quick start: the instance-wise center loss on a toy problem (CPU only, no downloads).

A tiny CNN is trained twice on synthetic 16x16 images, each step on N=4 augmented views
of every image: once with cross-entropy on the views only, and once with the
instance-wise center loss (lambda_IC=0.5, L2 distance, stop-gradient). The
figure shows the logits of the views of a few held-out images (2-D PCA) and how spread
out the views of an image are relative to the spread between images. This is an
illustration of the idea on synthetic data, not a result of the paper.

    python examples/quickstart.py            # writes assets/quickstart.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import transforms as T  # noqa: E402

import iwcl  # noqa: E402

NUM_CLASSES, SIZE, NUM_VIEWS, BATCH = 4, 16, 4, 32  # NUM_VIEWS = N


def make_data(n: int, generator: torch.Generator):
    """Smooth random 16x16 RGB images: class template + image-specific pattern + noise."""
    templates = F.interpolate(torch.rand(NUM_CLASSES, 3, 4, 4, generator=torch.Generator().manual_seed(0)), size=SIZE, mode="bilinear")
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=generator)
    own = F.interpolate(torch.rand(n, 3, 4, 4, generator=generator), size=SIZE, mode="bilinear")
    noise = 0.1 * torch.randn(n, 3, SIZE, SIZE, generator=generator)
    return (0.5 * templates[labels] + 0.5 * own + noise).clamp(0, 1), labels


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        return self.fc(self.body(x))


# a random augmentation of one image tensor (3, H, W); make_views calls it per image and view
augment = T.Compose([
    T.RandomCrop(SIZE, padding=3, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
    T.RandomErasing(p=0.5, scale=(0.05, 0.2), value=0.5),
])


@torch.no_grad()
def view_logits(model, images, num_views: int = 24, seed: int = 1):
    """Logits of ``num_views`` fixed random views of every image: (views, images, classes)."""
    was_training = model.training
    model.eval()
    with torch.random.fork_rng():  # the same views for every call, without touching the training RNG
        torch.manual_seed(seed)
        z = torch.stack([model(v) for v in iwcl.make_views(images, augment, num_views)])
    model.train(was_training)
    return z


def relative_spread(z: torch.Tensor) -> float:
    """Variance of the views around their instance centre / total variance of the logits."""
    within = (z - z.mean(0, keepdim=True)).pow(2).sum(-1).mean()
    total = (z - z.mean((0, 1), keepdim=True)).pow(2).sum(-1).mean()
    return (within / total).item()


def train(use_iwcl: bool, images, labels, probe, steps: int = 300, seed: int = 0):
    torch.manual_seed(seed)
    model = TinyCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    if use_iwcl:
        criterion = iwcl.InstanceWiseCenterLoss(alpha=0.5)                   # the paper's loss, lambda_IC = 0.5
    else:
        criterion = iwcl.MultiViewCrossEntropy()                               # CE on the same views
    history = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(images), (BATCH,))
        views = iwcl.make_views(images[idx], augment, num_views=NUM_VIEWS)   # (N, B, 3, H, W)
        logits = torch.stack([model(v) for v in views])                      # (N, B, C)
        loss = criterion(logits, labels[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            history.append((step, relative_spread(view_logits(model, probe, num_views=8))))
    return model.eval(), history


def main(out: Path, steps: int = 300) -> dict:
    g = torch.Generator().manual_seed(0)
    train_x, train_y = make_data(512, g)
    test_x, test_y = make_data(512, g)

    results = {}
    for name, use_iwcl in [("cross-entropy on N views", False), ("+ instance-wise center loss", True)]:
        model, history = train(use_iwcl, train_x, train_y, test_x[:64], steps)
        with torch.no_grad():
            acc = (model(test_x).argmax(1) == test_y).float().mean().item() * 100
        z = view_logits(model, test_x[:64])
        results[name] = dict(z=z, acc=acc, spread=relative_spread(z), history=history)
        print(f"{name:28s} test acc {acc:5.1f}%  relative view spread {results[name]['spread']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = plt.cm.tab10.colors
    for ax, (name, r) in zip(axes[:2], results.items()):
        shown = r["z"][:, :8]                                      # 8 held-out images x 24 views
        flat = shown.reshape(-1, shown.size(-1))
        mean = flat.mean(0)
        _, _, v = torch.pca_lowrank(flat - mean, q=2, center=False)
        pts = (shown - mean) @ v[:, :2]                            # (views, 8, 2)
        centers = pts.mean(0)
        for i in range(pts.size(1)):
            c = colors[i % 10]
            for k in range(pts.size(0)):
                ax.plot([pts[k, i, 0], centers[i, 0]], [pts[k, i, 1], centers[i, 1]], color=c, lw=0.5, alpha=0.35)
            ax.scatter(pts[:, i, 0], pts[:, i, 1], s=12, color=c, alpha=0.8)
            ax.scatter(centers[i, 0], centers[i, 1], s=160, marker="*", color=c, edgecolor="black", linewidth=0.8, zorder=3)
        ax.set_title(f"{name}\ntest acc {r['acc']:.1f}%, relative view spread {r['spread']:.3f}", fontsize=10)
        ax.set_xlabel("logit PC 1")
        ax.set_ylabel("logit PC 2")
    for (name, r), color in zip(results.items(), ["#7f7f7f", "#d62728"]):
        s, h = zip(*r["history"])
        axes[2].plot(s, h, marker="o", ms=3, color=color, label=name)
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("within-image / total logit variance")
    axes[2].set_title("relative view spread on 64 held-out images\n(lower = views of an image agree)", fontsize=10)
    axes[2].legend()
    fig.suptitle("Toy example: views (dots) of the same image are pulled to their instance centre (star)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "quickstart.png")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    main(args.out, args.steps)
