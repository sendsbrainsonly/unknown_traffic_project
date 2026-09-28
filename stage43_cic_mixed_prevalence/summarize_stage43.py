#!/usr/bin/env python3
"""Create compact class-level Stage 43 summaries from preserved run rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(name: str, rows: list[dict]) -> None:
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_unknown() -> list[dict]:
    groups = defaultdict(list)
    for row in read_rows("per_unknown_class_metrics.csv"):
        key = (row["experiment"], row["setting"], row["method"], row["unknown_class"])
        groups[key].append(row)
    output = []
    for (experiment, setting, method, unknown_class), rows in sorted(groups.items()):
        result = {
            "experiment": experiment,
            "setting": setting,
            "method": method,
            "unknown_class": unknown_class,
            "runs": len(rows),
            "unknown_samples_per_run": int(rows[0]["unknown_samples"]),
        }
        for name in ("auroc", "auprc", "ufar", "unknown_recall"):
            values = np.asarray([float(row[name]) for row in rows])
            result[f"{name}_mean"] = float(values.mean())
            result[f"{name}_std"] = float(values.std(ddof=1))
            result[f"{name}_q025"] = float(np.quantile(values, 0.025))
            result[f"{name}_q975"] = float(np.quantile(values, 0.975))
        output.append(result)
    return output


def aggregate_known() -> list[dict]:
    groups = defaultdict(list)
    for row in read_rows("known_class_metrics.csv"):
        key = (row["experiment"], row["setting"], row["method"], row["known_class"])
        groups[key].append(row)
    output = []
    for (experiment, setting, method, known_class), rows in sorted(groups.items()):
        result = {
            "experiment": experiment,
            "setting": setting,
            "method": method,
            "known_class": known_class,
            "runs": len(rows),
            "samples_per_run": int(rows[0]["samples"]),
        }
        for name in ("known_frr", "known_acceptance"):
            values = np.asarray([float(row[name]) for row in rows])
            result[f"{name}_mean"] = float(values.mean())
            result[f"{name}_std"] = float(values.std(ddof=1))
            result[f"{name}_q025"] = float(np.quantile(values, 0.025))
            result[f"{name}_q975"] = float(np.quantile(values, 0.975))
        output.append(result)
    return output


def main() -> None:
    unknown = aggregate_unknown()
    known = aggregate_known()
    write_rows("per_unknown_class_summary.csv", unknown)
    write_rows("known_class_summary.csv", known)
    print(
        {
            "status": "PASS",
            "per_unknown_class_summary_rows": len(unknown),
            "known_class_summary_rows": len(known),
        }
    )


if __name__ == "__main__":
    main()
