#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from stage21_common import OUT, RUN_ROOT, write_csv, write_json


def main() -> None:
    rows = []
    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            path = RUN_ROOT / dataset / f"seed{seed}" / "results.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    write_csv(OUT / "all_run_results.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["encoder"])].append(row)
    summary_rows = []
    for (dataset, encoder), values in sorted(grouped.items()):
        item = {"dataset": dataset, "encoder": encoder, "method": values[0]["method"], "runs": len(values)}
        for metric in ("test_accuracy", "test_macro_f1", "test_weighted_f1"):
            scores = np.asarray([float(row[metric]) for row in values])
            item[f"mean_{metric}"] = float(scores.mean())
            item[f"std_{metric}"] = float(scores.std(ddof=0))
        summary_rows.append(item)
    write_csv(OUT / "summary_results.csv", summary_rows)
    primary = [row for row in summary_rows if row["encoder"] == "E3"]
    write_json(OUT / "aggregate_summary.json", {"status": "SUCCESS", "formal_runs": 4, "result_rows": len(rows), "primary_method": "OURS-E3-T8", "primary_results": primary})
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
