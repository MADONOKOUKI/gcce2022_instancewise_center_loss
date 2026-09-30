import importlib.util
import random
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "archive" / "release_2023"


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture
def load_release():
    """Import a module of the archived release code by path (skip if unavailable)."""

    def _load(relpath: str, argv=None):
        path = RELEASE / relpath
        if not path.exists():
            pytest.skip(f"{path} not available")
        if relpath.endswith("contrastive_center_loss.py") and "logzero" not in sys.modules:
            noop = lambda *a, **k: None  # noqa: E731 - the release module only configures a logger
            sys.modules["logzero"] = types.SimpleNamespace(loglevel=noop, formatter=noop, logfile=noop, logger=None)
        spec = importlib.util.spec_from_file_location("release_" + relpath.replace("/", "_")[:-3], path)
        module = importlib.util.module_from_spec(spec)
        old_argv = sys.argv
        try:
            if argv is not None:
                sys.argv = argv
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec.loader.exec_module(module)
        except Exception as exc:  # e.g. a deprecated API removed in a newer library version
            pytest.skip(f"cannot import archived {relpath}: {exc}")
        finally:
            sys.argv = old_argv
        return module

    return _load
