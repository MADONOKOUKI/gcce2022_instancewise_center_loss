#!/usr/bin/env python
"""Aggregate ``train.py`` runs over seeds: mean +- std of the test accuracy per setting.

    python scripts/summarize_runs.py runs            # markdown table on stdout
    python scripts/summarize_runs.py runs --csv summary.csv

Runs that differ only in ``--seed`` (and in runtime options such as ``--out-dir`` or
``--num-workers``) form one setting. Both the best test accuracy over the epochs and the
final-epoch test accuracy are reported; the paper reports the mean over three runs without
saying which of the two, and the standard deviation is the sample standard deviation.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNTIME_KEYS = {
    "seed", "out_dir", "resume", "log_interval", "num_workers", "device", "data_root",
    "download", "config", "data_parallel", "test_batch_size",
}
SHOWN = ["dataset", "model", "augmentation", "method", "distance", "alpha", "num_views", "subset_per_class"]


def mean_std(values):
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="?", default="runs", help="directory containing the run folders")
    parser.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = parser.parse_args(argv)

    groups = defaultdict(list)
    for path in sorted(Path(args.runs).glob("**/summary.json")):
        summary = json.loads(path.read_text())
        run_args = summary["args"]
        key = tuple(sorted((k, json.dumps(v)) for k, v in run_args.items() if k not in RUNTIME_KEYS))
        groups[key].append(summary)
    if not groups:
        sys.exit(f"no summary.json found under {args.runs}")

    rows = []
    for key, summaries in groups.items():
        run_args = summaries[0]["args"]
        row = {k: run_args.get(k, "") for k in SHOWN}
        if run_args["method"] != "proposed":
            row["distance"], row["alpha"] = "", ""
        row["runs"] = len(summaries)
        row["best_test_acc"] = mean_std([s["best_test_acc"] for s in summaries])
        row["final_test_acc"] = mean_std([s["final_test_acc"] for s in summaries])
        rows.append(row)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in SHOWN))

    header = SHOWN + ["runs", "best_test_acc", "final_test_acc"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(row[k]) for k in header) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
