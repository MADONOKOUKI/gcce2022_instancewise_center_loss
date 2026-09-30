#!/usr/bin/env python
"""Summarise the archived training logs into ``archive/results_summary.csv``.

The logs under ``archive/results*/`` were written by the original research
scripts (``archive/main_*.py``). Every script prints, per epoch,
``===> epoch: e/T`` followed by a ``train:`` and a ``test:`` progress bar whose
entries end with ``Acc: x% (correct/seen)``; a finished run ends with
``===> BEST ACC. PERFORMANCE: x%`` (the maximum test accuracy over epochs).

For every ``*.txt`` log this script records

* the path, experiment folder and run name,
* the setting parsed from the file name (see ``parse_setting``),
* the number of completed epochs and the number of training images per epoch,
* ``best_acc_reported``: the value printed on the ``BEST ACC. PERFORMANCE`` line
  (empty if the run did not finish),
* ``best_test_acc_parsed``: the maximum of the per-epoch test accuracies,
* ``last_epoch_test_acc``: the test accuracy of the last completed epoch.

These are exploratory runs from the development of the method (the archived scripts
train a WideResNet-28 on CIFAR-100). They are *not* the experiments of the paper.

Usage::

    python scripts/summarize_archived_logs.py            # writes archive/results_summary.csv
    python scripts/summarize_archived_logs.py --out other.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

EPOCH_RE = re.compile(rb"===> epoch: (\d+)/(\d+)")
# one progress-bar entry: " 157/157 [====>]  Step: ... | Acc: 10.970% (1097/10000)"
# = batch counter (done/total), accuracy, (correct/seen images)
PROGRESS_RE = re.compile(rb"(\d+)/(\d+) \[[=>.]*\][^\n]*?Acc: ([0-9.]+)% \((\d+)/(\d+)\)")
BEST_RE = re.compile(rb"BEST ACC\. PERFORMANCE: ([0-9.]+)%")

FIELDS = [
    "path",
    "experiment_folder",
    "run_name",
    "script",
    "script_archived",
    "setting",
    "num_views",
    "epochs_total",
    "epochs_completed",
    "train_images_per_epoch",
    "best_acc_reported",
    "best_test_acc_parsed",
    "last_epoch_test_acc",
    "status",
]


ARCHIVED_SCRIPTS = sorted((p.stem for p in (ROOT / "archive").glob("main*.py")), key=len, reverse=True)


def parse_setting(run_name: str) -> Dict[str, str]:
    """Setting parsed from a log file name, using the conventions of ``archive/exec_*.sh``.

    * ``<script>_num_<K>.txt``: ``python <script>.py --num_imgs K`` (K augmented views),
      e.g. ``exec_avg_detach.sh``; ``main_avg_kl_num_<K>.txt`` is written by ``main_avg.py``
      (``exec_avg.sh``).
    * ``<prefix>_<K>_<N>_<M>_<mean>[_tag].txt``: ``--num_imgs K --N N --M M --mean mean``
      (RandAugment N/M), as in ``exec_avg_randaug.sh``, which writes
      ``results_exp3/.../main_avg_<K>_<N>_<M>_<mean>.txt`` from ``main_avg_randaug.py``.
    * Anything else: the longest archived script name that prefixes the file name, and the
      remaining suffix verbatim.
    """
    name, debug = run_name, run_name.startswith("debug_")
    if debug:
        name = name[len("debug_"):]
    out = {"script": "", "setting": "", "num_views": ""}

    m = re.fullmatch(r"(main_(?:no)?avg(?:_randaug)?)_(\d+)_(\d+)_(\d+)_(0|1|True|False)(?:_(.+))?", name)
    m2 = re.fullmatch(r"(main(?:_[a-z0-9]+)*?)_num_(\d+)", name)
    m3 = re.fullmatch(r"(main_(?:no)?avg_randaug)_(\d+)", name)
    if m:
        base, k, n, mag, mean, tag = m.groups()
        out["script"] = "main_avg_randaug" if base == "main_avg" else base
        out["setting"] = f"num_imgs={k}, N={n}, M={mag}, mean={mean}" + (f"; tag: {tag}" if tag else "")
        out["num_views"] = k
    elif m2:
        base, k = m2.groups()
        out["script"], tag = ("main_avg", "kl") if base == "main_avg_kl" else (base, "")
        out["setting"] = f"num_imgs={k}" + (f"; tag: {tag}" if tag else "")
        out["num_views"] = k
    elif m3:
        out["script"], out["num_views"] = m3.group(1), m3.group(2)
        out["setting"] = f"num_imgs={m3.group(2)}"
    else:
        script = next((s for s in ARCHIVED_SCRIPTS if name == s or name.startswith(s + "_")), name)
        out["script"] = script
        suffix = name[len(script):].strip("_")
        out["setting"] = f"unparsed suffix: {suffix}" if suffix else "script defaults"
    if debug:
        out["setting"] = "debug run; " + out["setting"]
    return out


def _last_progress(chunk: bytes) -> Optional[re.Match]:
    last = None
    for last in PROGRESS_RE.finditer(chunk):
        pass
    return last


def summarise(path: Path) -> Dict[str, str]:
    data = path.read_bytes().replace(b"\r", b"\n")
    epochs = list(EPOCH_RE.finditer(data))
    epochs_total = epochs[0].group(2).decode() if epochs else ""

    test_accs: List[float] = []
    train_images = ""
    for i, ep in enumerate(epochs):
        end = epochs[i + 1].start() if i + 1 < len(epochs) else len(data)
        chunk = data[ep.end():end]
        pos = chunk.rfind(b"\ntest:")
        if pos < 0:
            continue
        if i == 0:
            m = _last_progress(chunk[:pos])
            if m is not None and m.group(1) == m.group(2):
                train_images = m.group(5).decode()
        m = _last_progress(chunk[pos:])
        # a test pass is complete when its last progress entry reached the final batch
        if m is not None and m.group(1) == m.group(2):
            test_accs.append(float(m.group(3)))

    best = BEST_RE.search(data)
    rel = path.relative_to(ROOT).as_posix()
    parts = path.relative_to(ROOT / "archive").parts
    info = parse_setting(path.stem)
    status = "complete" if best else ("no epoch finished" if not test_accs else "incomplete")
    return {
        "path": rel,
        "experiment_folder": "/".join(parts[:-1]),
        "run_name": path.stem,
        "script": info["script"] + ".py",
        "script_archived": "yes" if (ROOT / "archive" / (info["script"] + ".py")).exists() else "no",
        "setting": info["setting"],
        "num_views": info["num_views"],
        "epochs_total": epochs_total,
        "epochs_completed": str(len(test_accs)),
        "train_images_per_epoch": train_images,
        "best_acc_reported": best.group(1).decode() if best else "",
        "best_test_acc_parsed": f"{max(test_accs):.3f}" if test_accs else "",
        "last_epoch_test_acc": f"{test_accs[-1]:.3f}" if test_accs else "",
        "status": status,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "archive" / "results_summary.csv"))
    args = parser.parse_args(argv)

    logs = sorted((ROOT / "archive").glob("results*/**/*.txt"))
    rows = []
    for path in logs:
        row = summarise(path)
        rows.append(row)
        print(f"{row['status']:>17}  best={row['best_acc_reported'] or '-':>7}  {row['path']}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_complete = sum(r["status"] == "complete" for r in rows)
    print(f"wrote {len(rows)} rows ({n_complete} complete runs) to {args.out}")


if __name__ == "__main__":
    main()
