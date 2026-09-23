#!/usr/bin/env python3
"""Check support-weighted F1 reconstruction on all currently completed runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_results import weighted_f1_from_per_class  # noqa: E402


def main() -> None:
    probe = pd.DataFrame({"f1": [1.0, 0.0], "val_support": [9, 1]})
    assert abs(weighted_f1_from_per_class(probe) - 0.9) < 1e-12

    rows = []
    for result_path in ROOT.glob("runs/f[0-3]/*/result.json"):
        result = json.loads(result_path.read_text())
        per_class = pd.read_csv(result_path.parent / "per_class_metrics.csv")
        reconstructed_macro = float(per_class["f1"].mean())
        assert abs(reconstructed_macro - float(result["val_macro_f1"])) < 1e-12
        result["val_weighted_f1"] = weighted_f1_from_per_class(per_class)
        assert 0.0 <= result["val_weighted_f1"] <= 1.0
        rows.append(result)

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["feature", "setting"])
        .agg(
            runs=("protocol_id", "size"),
            macro_mean=("val_macro_f1", "mean"),
            macro_std=("val_macro_f1", "std"),
            weighted_mean=("val_weighted_f1", "mean"),
            weighted_std=("val_weighted_f1", "std"),
        )
        .reset_index()
    )
    print(f"WEIGHTED_F1_REPORTING_PASS runs={len(frame)}")
    print(summary.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
