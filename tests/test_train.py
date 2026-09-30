import csv
import importlib.util
import json

import pytest
import torch

import iwcl
from conftest import ROOT
from iwcl.models import build_model


@pytest.fixture(scope="module")
def train_cli():
    spec = importlib.util.spec_from_file_location("train_cli", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_args(out_dir, *extra):
    return [
        "--dataset", "fake", "--fake-size", "16", "--model", "wideresnet", "--depth", "10", "--widen-factor", "1",
        "--augmentation", "cutout", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
        "--log-interval", "0", "--out-dir", str(out_dir), *extra,
    ]


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_one_optimisation_step_reduces_the_loss():
    torch.manual_seed(0)
    model = build_model("wideresnet", num_classes=10, depth=10, widen_factor=1)
    criterion = iwcl.InstanceWiseCenterLoss(alpha=0.5, distance="mse")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    views = torch.rand(2, 8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))

    def loss_value():
        return criterion(torch.stack([model(v)[0] for v in views]), labels)

    before = loss_value()
    optimizer.zero_grad()
    before.backward()
    optimizer.step()
    with torch.no_grad():
        after = loss_value()
    assert after.item() < before.item()


def test_cli_trains_and_writes_outputs(train_cli, tmp_path):
    summary = train_cli.main(_tiny_args(tmp_path / "run", "--epochs", "2"))
    out = tmp_path / "run"
    for name in ("args.json", "metrics.csv", "summary.json", "last.pt", "best.pt"):
        assert (out / name).exists(), name
    rows = _rows(out / "metrics.csv")
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert json.loads((out / "summary.json").read_text())["best_test_acc"] == summary["best_test_acc"]


@pytest.mark.parametrize("method", ["baseline", "center_loss", "triplet_loss", "augmix"])
def test_cli_comparison_methods(train_cli, tmp_path, method):
    summary = train_cli.main(_tiny_args(tmp_path / method, "--epochs", "1", "--method", method))
    assert summary["args"]["num_views"] == (3 if method == "augmix" else 2)
    assert summary["args"]["method"] == ("jsd" if method == "augmix" else method)


def test_resume_reproduces_an_uninterrupted_run(train_cli, tmp_path):
    train_cli.main(_tiny_args(tmp_path / "full", "--epochs", "2"))
    train_cli.main(_tiny_args(tmp_path / "part", "--epochs", "1"))
    train_cli.main(_tiny_args(tmp_path / "part", "--epochs", "2", "--resume", str(tmp_path / "part" / "last.pt")))
    full, part = _rows(tmp_path / "full" / "metrics.csv"), _rows(tmp_path / "part" / "metrics.csv")
    keys = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"]
    assert [[r[k] for k in keys] for r in part] == [[r[k] for k in keys] for r in full]


def test_learning_rate_schedule_matches_release(train_cli):
    # MultiStepLR.step(epoch) at the start of each 1-indexed epoch: decay *at* the milestone
    lrs = [train_cli.lr_at_epoch(e, 0.1, [60, 120, 180], 0.1) for e in (1, 59, 60, 119, 120, 180, 200)]
    assert lrs == pytest.approx([0.1, 0.1, 0.01, 0.01, 0.001, 0.0001, 0.0001])


def test_paper_defaults_and_yaml_config(train_cli, tmp_path):
    args = train_cli.parse_args([])
    assert (args.dataset, args.model, args.augmentation, args.method) == ("cifar100", "wideresnet", "cutout", "proposed")
    assert (args.num_views, args.alpha, args.distance, args.stopgrad, args.ic_reduction) == (2, 0.5, "mse", True, "mean")
    assert (args.epochs, args.lr, args.momentum, args.weight_decay, args.batch_size) == (200, 0.1, 0.9, 5e-4, 256)
    assert (args.milestones, args.gamma, args.resnext_base_width) == ([60, 120, 180], 0.1, 32)
    assert train_cli.parse_args(["--model", "resnet18"]).alpha == 0.7   # lambda_IC of the paper
    assert train_cli.parse_args(["--model", "resnext"]).alpha == 0.5
    assert train_cli.parse_args(["--model", "resnet18", "--lambda-ic", "0.1"]).alpha == 0.1
    config = tmp_path / "cfg.yaml"
    config.write_text("num_views: 3\nlambda_ic: 0.25\nstopgrad: false\n")
    args = train_cli.parse_args(["--config", str(config), "--no-stopgrad"])
    assert (args.num_views, args.alpha, args.stopgrad) == (3, 0.25, False)
    assert train_cli.parse_args(["--config", str(config), "--alpha", "0.75"]).alpha == 0.75
    config.write_text("not_an_option: 1\n")
    with pytest.raises(SystemExit):
        train_cli.parse_args(["--config", str(config)])


def test_summarize_runs_groups_seeds(train_cli, tmp_path, capsys):
    for seed in ("1", "2"):
        train_cli.main(_tiny_args(tmp_path / "runs" / f"s{seed}", "--epochs", "1", "--seed", seed))
    spec = importlib.util.spec_from_file_location("summarize_runs", ROOT / "scripts" / "summarize_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capsys.readouterr()
    module.main([str(tmp_path / "runs")])
    table = capsys.readouterr().out.strip().splitlines()
    assert len(table) == 3 and " ± " in table[2]           # header, separator, one setting with 2 seeds
    assert "| fake | wideresnet | cutout | proposed | mse | 0.5 | 2 | 0 | 2 |" in table[2]


def test_example_config_matches_the_defaults(train_cli):
    defaults = vars(train_cli.parse_args([]))
    from_config = vars(train_cli.parse_args(["--config", str(ROOT / "configs" / "example.yaml")]))
    assert {k: v for k, v in from_config.items() if k != "config"} == {k: v for k, v in defaults.items() if k != "config"}
