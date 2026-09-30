#!/usr/bin/env python
"""Train an image classifier with the instance-wise center loss (IWCL) or a comparison method.

Defaults follow the paper (Sec. III-A): N=2 augmented views, L2 distance with stop-gradient,
lambda_IC = 0.7 for ResNet-18 and 0.5 for ResNeXt-29 8x32d / WideResNet-28-10, SGD for 200
epochs (lr 0.1, divided by 10 at epochs 60/120/180, weight decay 5e-4, momentum 0.9) with
256 images per step. Examples::

    python train.py                                              # CIFAR-100, WideResNet-28-10, Cutout
    python train.py --dataset cifar10 --model resnet18 --augmentation autoaug --method baseline
    python train.py --model resnet18 --method jsd --augmentation augmix    # JS-divergence (N=3)
    python train.py --dataset cub200 --data-root /path/to/CUB_200_2011
    python train.py --config configs/example.yaml --epochs 100   # YAML defaults, CLI overrides
    python train.py --dataset fake --model resnet18 --epochs 1 --batch-size 32 --num-workers 0   # smoke test

``scripts/reproduce_table{1,2,3,4}.sh`` run the experiments of the paper's tables.
Outputs in ``--out-dir``: ``args.json``, ``metrics.csv`` (one row per epoch),
``summary.json`` (best and final test accuracy), ``last.pt`` (resumable with ``--resume``)
and ``best.pt`` (weights of the epoch with the best test accuracy, as the release code saved).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from iwcl.augment import AUGMENTATIONS, build_test_transform, build_train_transform
from iwcl.criteria import ALIASES, METHODS, build_criterion
from iwcl.data import DATASETS, build_datasets
from iwcl.losses import DISTANCES
from iwcl.models import MODELS, build_model

# lambda_IC of the paper (Sec. III-A); models that are not in the paper use 0.5 (release config)
LAMBDA_IC = {"resnet18": 0.7, "resnext": 0.5, "wideresnet": 0.5}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train with the instance-wise center loss (Madono et al., GCCE 2022) or a comparison method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="optional YAML file whose keys (argument names) set the defaults")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="cifar100", choices=DATASETS, help="fake = synthetic smoke-test data")
    g.add_argument("--data-root", default="./data", help="dataset directory; for cub200 the CUB_200_2011 directory or its parent")
    g.add_argument("--download", action=argparse.BooleanOptionalAction, default=True, help="download missing datasets (not cub200)")
    g.add_argument("--augmentation", default="cutout", choices=AUGMENTATIONS, help="standard = flip and cropping")
    g.add_argument("--cutout-length", type=int, default=8, help="Cutout patch side (release config)")
    g.add_argument("--cutout-holes", type=int, default=1, help="Cutout patches per image")
    g.add_argument("--randaug-n", type=int, default=1, help="RandAugment: operations per image")
    g.add_argument("--randaug-m", type=int, default=2, help="RandAugment: magnitude in [0, 30]")
    g.add_argument("--subset-per-class", type=int, default=0, help="train on N random images per class (0 = all; Table III: 10/50/100)")
    g.add_argument("--stl10-split", default="train", choices=["train", "train+unlabeled"], help="unlabelled images only enter the IWCL term")
    g.add_argument("--fake-size", type=int, default=256, help="number of training images of --dataset fake")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="wideresnet", choices=MODELS, help="resnet18, resnext (8x32d) and wideresnet (28-10) are the paper's models")
    g.add_argument("--depth", type=int, default=28, help="WideResNet depth")
    g.add_argument("--widen-factor", type=int, default=10, help="WideResNet widen factor")
    g.add_argument("--resnext-base-width", type=int, default=32, help="ResNeXt-29 8xNd; the release code used 64")

    g = p.add_argument_group("method")
    g.add_argument("--method", default="proposed", choices=METHODS + tuple(ALIASES), help="proposed = instance-wise center loss; baseline = 'None' in the paper")
    g.add_argument("--num-views", type=int, default=2, help="N augmented views per image (jsd: always 3)")
    g.add_argument("--lambda-ic", "--alpha", dest="alpha", type=float, default=None, help="lambda_IC; default 0.7 for resnet18, else 0.5")
    g.add_argument("--distance", default="mse", choices=DISTANCES, help="IWCL distance between a view and its instance centre (mse = L2)")
    g.add_argument("--stopgrad", action=argparse.BooleanOptionalAction, default=True, help="treat the instance centre as a constant")
    g.add_argument("--ic-reduction", default="mean", choices=["mean", "sum"], help="average the IWCL distance over the logits (release code) or sum it (Eq. 6)")
    g.add_argument("--center-on", default="logits", choices=["logits", "features"], help="IWCL pulls logits (paper) or pooled features")
    g.add_argument("--aux-weight", type=float, default=None, help="weight of a comparison method's extra term (0.1; jsd: 12)")
    g.add_argument("--learnable-centers", action="store_true", help="train the contrastive-center-loss centres (release: fixed)")
    g.add_argument("--triplet-on", default="logits", choices=["logits", "features"], help="triplet loss on logits (release) or features")

    g = p.add_argument_group("optimisation (paper Sec. III-A)")
    g.add_argument("--epochs", type=int, default=200, help="training epochs")
    g.add_argument("--batch-size", type=int, default=256, help="images per step, each giving N views (the paper: 256 on 4 GPUs)")
    g.add_argument("--test-batch-size", type=int, default=None, help="default: --batch-size")
    g.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    g.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    g.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    g.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True, help="Nesterov momentum (release config)")
    g.add_argument("--milestones", type=int, nargs="+", default=[60, 120, 180], help="epochs where the lr is multiplied by --gamma")
    g.add_argument("--gamma", type=float, default=0.1, help="lr decay factor")

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, cuda:1, mps or cpu")
    g.add_argument("--data-parallel", action="store_true", help="nn.DataParallel over all visible GPUs (the paper used 4 GPUs)")
    g.add_argument("--num-workers", type=int, default=4, help="data loading processes")
    g.add_argument("--seed", type=int, default=1, help="random seed (also selects the --subset-per-class images)")
    g.add_argument("--out-dir", default=None, help="default: runs/<dataset>-<model>-<augmentation>-<method>...")
    g.add_argument("--resume", default=None, help="path to a last.pt checkpoint")
    g.add_argument("--log-interval", type=int, default=50, help="print progress every N steps (0 = off)")
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        import yaml

        with open(pre.config) as f:
            cfg = {k.replace("-", "_"): v for k, v in (yaml.safe_load(f) or {}).items()}
        cfg = {("alpha" if k == "lambda_ic" else k): v for k, v in cfg.items()}
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            parser.error(f"unknown keys in {pre.config}: {unknown}")
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    args.method = ALIASES.get(args.method, args.method)
    if args.alpha is None:
        args.alpha = LAMBDA_IC.get(args.model, 0.5)
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    if args.method == "jsd" and args.num_views != 3:
        print(f"note: --method jsd uses 3 views [clean, augmented, augmented]; overriding --num-views {args.num_views}")
        args.num_views = 3
    if args.method == "triplet_loss" and args.num_views < 2:
        parser.error("--method triplet_loss needs --num-views >= 2")
    if args.out_dir is None:
        name = f"{args.dataset}-{args.model}-{args.augmentation}-{args.method}"
        if args.method == "proposed":
            name += f"-{args.distance}-l{args.alpha:g}"
        if args.subset_per_class:
            name += f"-sub{args.subset_per_class}"
        args.out_dir = os.path.join("runs", f"{name}-N{args.num_views}-s{args.seed}")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at_epoch(epoch: int, base_lr: float, milestones: List[int], gamma: float) -> float:
    """Step decay of the release code: ``MultiStepLR.step(epoch)`` was called at the start
    of every 1-indexed epoch, so the decay takes effect *at* each milestone epoch."""
    return base_lr * gamma ** sum(epoch >= m for m in milestones)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(model, criterion, loader, optimizer, device, epoch: int, log_interval: int = 0):
    model.train()
    criterion.train()
    loss_sum, correct, seen, steps = 0.0, 0, 0, 0
    after_backward = getattr(criterion, "after_backward", None)
    for step, (views, targets) in enumerate(loader, start=1):
        targets = targets.to(device, non_blocking=True)
        # every view is forwarded separately, as in the release code (per-view BatchNorm statistics)
        outputs = [model(v.to(device, non_blocking=True)) for v in views]
        logits = torch.stack([o[0] for o in outputs])
        features = torch.stack([o[1] for o in outputs])
        loss = criterion(logits, targets, features)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if after_backward is not None:
            after_backward()
        optimizer.step()

        loss_sum += loss.item()
        steps += 1
        labelled = targets != -1  # train accuracy on the last view, as in the release code
        correct += (logits[-1].argmax(dim=1)[labelled] == targets[labelled]).sum().item()
        seen += int(labelled.sum().item())
        if log_interval and step % log_interval == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss_sum / steps:.4f} acc {100.0 * correct / max(seen, 1):.2f}", flush=True)
    return loss_sum / max(steps, 1), 100.0 * correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits, _ = model(images)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def rng_state() -> Dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def main(argv: Optional[List[str]] = None) -> Dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # data
    train_tf = build_train_transform(
        args.augmentation,
        cutout_length=args.cutout_length,
        cutout_holes=args.cutout_holes,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
    )
    test_tf = build_test_transform()
    train_set, test_set, num_classes = build_datasets(
        args.dataset,
        root=args.data_root,
        train_transform=train_tf,
        test_transform=test_tf,
        num_views=args.num_views,
        clean_view=args.method == "jsd",
        download=args.download,
        subset_per_class=args.subset_per_class,
        stl10_split=args.stl10_split,
        fake_size=args.fake_size,
        seed=args.seed,
    )
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, generator=generator
    )
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # model, objective, optimiser
    model = build_model(
        args.model, num_classes, depth=args.depth, widen_factor=args.widen_factor, resnext_base_width=args.resnext_base_width
    ).to(device)
    criterion = build_criterion(
        args.method,
        num_classes=num_classes,
        feat_dim=model.feature_dim,
        alpha=args.alpha,
        distance=args.distance,
        stopgrad=args.stopgrad,
        reduction=args.ic_reduction,
        center_on=args.center_on,
        aux_weight=args.aux_weight,
        learnable_centers=args.learnable_centers,
        triplet_on=args.triplet_on,
    ).to(device)
    params = list(model.parameters()) + list(criterion.parameters())  # class centres share the optimiser
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    start_epoch, best_acc, best_epoch = 1, 0.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_acc, best_epoch = ckpt["epoch"] + 1, ckpt["best_acc"], ckpt["best_epoch"]
        set_rng_state(ckpt["rng"])
        generator.set_state(ckpt["loader_generator"])
        print(f"resumed from {args.resume} (epoch {ckpt['epoch']})")

    n_params = sum(p.numel() for p in unwrap(model).parameters())
    print(f"device {device} | {args.dataset} ({len(train_set)} train / {len(test_set)} test) | {args.model} ({n_params / 1e6:.2f}M params)")
    print(f"method {args.method} | {criterion} | N={args.num_views} | batch {args.batch_size} | epochs {args.epochs} | out {out_dir}")

    metrics_path = out_dir / "metrics.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "best_test_acc", "seconds"]
    if not (args.resume and metrics_path.exists()):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    t_start = time.time()
    test_acc = float("nan")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = lr_at_epoch(epoch, args.lr, args.milestones, args.gamma)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.log_interval)
        test_loss, test_acc = evaluate(model, test_loader, device)
        if test_acc >= best_acc:  # the release code saved the model on ties as well
            best_acc, best_epoch = test_acc, epoch
            torch.save({"model": unwrap(model).state_dict(), "epoch": epoch, "test_acc": test_acc, "args": vars(args)}, out_dir / "best.pt")
        seconds = time.time() - t0
        torch.save(
            {
                "model": unwrap(model).state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "rng": rng_state(),
                "loader_generator": generator.get_state(),
                "args": vars(args),
            },
            out_dir / "last.pt",
        )
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{lr:.6g}", f"{train_loss:.4f}", f"{train_acc:.3f}", f"{test_loss:.4f}", f"{test_acc:.3f}", f"{best_acc:.3f}", f"{seconds:.1f}"]
            )
        print(
            f"epoch {epoch}/{args.epochs} | lr {lr:.4g} | train loss {train_loss:.4f} acc {train_acc:.2f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.2f} | best {best_acc:.2f} (epoch {best_epoch}) | {seconds:.1f}s",
            flush=True,
        )

    summary = {
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": test_acc,
        "epochs": args.epochs,
        "seconds": round(time.time() - t_start, 1),
        "device": str(device),
        "num_parameters": n_params,
        "args": vars(args),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done: best test acc {best_acc:.2f} (epoch {best_epoch}), final {test_acc:.2f}; results in {out_dir}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
