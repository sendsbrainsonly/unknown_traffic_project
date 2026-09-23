#!/usr/bin/env python3
"""Verify appendix-only K=3 correspondence outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/component_correspondence_audit/k3_sensitivity"


def main() -> None:
    required = {
        "tf_component_assignments_k3.parquet",
        "od_component_assignments_k3.parquet",
        "cross_encoder_correspondence_k3.csv",
        "cross_encoder_confusion_k3.csv",
        "train_val_stability_k3.csv",
        "flow_id_assertions_k3.csv",
        "k3_sensitivity_summary.md",
        "run_metadata.json",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).is_file())
    if missing:
        raise AssertionError(f"missing K=3 outputs: {missing}")
    tf = pd.read_parquet(OUTPUT / "tf_component_assignments_k3.parquet")
    od = pd.read_parquet(OUTPUT / "od_component_assignments_k3.parquet")
    if len(tf) != 124581 or len(od) != 124581:
        raise AssertionError("K=3 assignment row count mismatch")
    if set(tf.component_id) != {0, 1, 2} or set(od.component_id) != {0, 1, 2}:
        raise AssertionError("K=3 component IDs incomplete")
    if set(tf.flow_id) != set(od.flow_id):
        raise AssertionError("K=3 cross-encoder flow IDs differ")
    cross = pd.read_csv(OUTPUT / "cross_encoder_correspondence_k3.csv")
    if len(cross) != 8 or len(cross[cross.split == "val"]) != 4:
        raise AssertionError("K=3 correspondence rows incomplete")
    metadata = json.loads((OUTPUT / "run_metadata.json").read_text())
    if metadata["K"] != 3 or not metadata["main_k2_gate_unchanged"].startswith("C."):
        raise AssertionError("K=3 appendix changed the main protocol/Gate")
    if metadata["test_loaded_or_used"] or metadata["encoder_retrained"]:
        raise AssertionError("forbidden K=3 operation recorded")
    for encoder in ("trafficformer", "opendetect"):
        for check in metadata[encoder]["checks"].values():
            if check["validation_absolute_difference"] > 1e-6:
                raise AssertionError(f"{encoder}: validation NLL reproduction failed")
    print(json.dumps({"status": "PASS", "assignment_rows": len(tf), "correspondence_rows": len(cross)}))


if __name__ == "__main__":
    main()
