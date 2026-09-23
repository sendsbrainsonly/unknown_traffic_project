#!/usr/bin/env python3
"""Independent structural and arithmetic verification of Stage 5.5 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    required = [
        "dgsb_gate_audit/gate_transition_counts.csv",
        "dgsb_gate_audit/global_gate_effectiveness.csv",
        "dgsb_gate_audit/class_consistency.csv",
        "dgsb_gate_audit/global_local_class_confusion.csv",
        "dgsb_gate_audit/audit_summary.md",
        "cstnet_input_audit/raw_input_manifest.csv",
        "cstnet_input_audit/domain_leakage_audit.csv",
        "cstnet_input_audit/token_leakage_summary.csv",
        "cstnet_input_audit/input_byte_source_mapping.csv",
        "cstnet_input_audit/sampled_open_detect_inputs.npz",
        "cstnet_input_audit/endpoint_input_audit.md",
        "calibration_sample_audit/class_split_counts.csv",
        "calibration_sample_audit/class_calibration_reliability.csv",
        "calibration_sample_audit/component_count_stress_table.csv",
        "calibration_sample_audit/fallback_risk_summary.csv",
        "calibration_sample_audit/audit_summary.md",
        "final_preflight_summary.md",
        "run_metadata.json",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing Stage 5.5 outputs: {missing}")
    metadata = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or any(
        row["status"] != "PASS" for row in metadata["hash_verification"]
    ):
        raise RuntimeError("completion/hash verification failed")
    if metadata["fit_operations_performed"]:
        raise RuntimeError("unexpected fit operation")
    if metadata["cstnet_training_executed"] or metadata["cstnet_formal_mu_extraction_executed"]:
        raise RuntimeError("forbidden CSTNET execution flag")
    if metadata["cstnet_unknown_scores_accessed"] or metadata["ustc_unknown_inputs_accessed"]:
        raise RuntimeError("forbidden Unknown access flag")
    if metadata["forbidden_ustc_unknown_files_read"]:
        raise RuntimeError("forbidden USTC Unknown input ledger entry")
    forbidden_names = {"test_unknown_mu.parquet", "frozen_final_predictions.parquet"}
    if any(Path(value).name.lower() in forbidden_names for value in metadata["accessed_input_files"]):
        raise RuntimeError("forbidden path present in input ledger")

    gates = pd.read_csv(OUT / "dgsb_gate_audit/gate_transition_counts.csv")
    if len(gates) != 6:
        raise RuntimeError("expected 3 settings x 2 Known splits")
    summed = gates[["both_pass", "global_fail_local_pass", "global_pass_local_fail", "both_fail"]].sum(axis=1)
    if not np.array_equal(summed.to_numpy(), gates["n_samples"].to_numpy()):
        raise RuntimeError("gate transition counts do not partition each split")
    effects = pd.read_csv(OUT / "dgsb_gate_audit/global_gate_effectiveness.csv")
    if not np.allclose(
        effects["global_exclusive_rejection_rate"],
        effects["global_exclusive_rejection_count"] / effects["n_samples"],
        rtol=0,
        atol=1e-15,
    ):
        raise RuntimeError("global-exclusive rejection arithmetic failed")
    consistency = pd.read_csv(OUT / "dgsb_gate_audit/class_consistency.csv")
    if not np.array_equal(
        consistency["same_class_count"] + consistency["different_class_count"], consistency["n"]
    ):
        raise RuntimeError("class consistency counts do not partition each split")

    manifest = pd.read_csv(OUT / "cstnet_input_audit/raw_input_manifest.csv")
    leakage = pd.read_csv(OUT / "cstnet_input_audit/domain_leakage_audit.csv")
    if len(manifest) > 600 or manifest["class_name"].nunique() > 30:
        raise RuntimeError("bounded CSTNET sample contract violated")
    if set(manifest["sample_id"]) != set(leakage["sample_id"]):
        raise RuntimeError("input/leakage sample identities differ")
    for fold in ("low", "medium", "high"):
        if set(manifest[f"{fold}_role"]) != {"known", "unknown"}:
            raise RuntimeError(f"{fold}: sample does not cover both roles")
    arrays = np.load(OUT / "cstnet_input_audit/sampled_open_detect_inputs.npz")
    valid = manifest[manifest["input_valid"].astype(bool)]
    if arrays["images"].shape != (len(valid), 32, 32):
        raise RuntimeError("saved Open-Detect input tensor shape mismatch")

    counts = pd.read_csv(OUT / "calibration_sample_audit/class_split_counts.csv")
    reliability = pd.read_csv(OUT / "calibration_sample_audit/class_calibration_reliability.csv")
    if len(counts) != len(reliability):
        raise RuntimeError("calibration row mismatch")
    expected = np.select(
        [counts["validation_count"] >= 50, counts["validation_count"] >= 30, counts["validation_count"] >= 20],
        ["GOOD", "ACCEPTABLE", "WEAK"],
        default="UNRELIABLE",
    )
    if not np.array_equal(expected, reliability["reliability"].to_numpy()):
        raise RuntimeError("calibration reliability boundary mismatch")
    stress = pd.read_csv(OUT / "calibration_sample_audit/component_count_stress_table.csv")
    if set(stress["component_weight_scenario"]) != {"50/50", "80/20", "90/10", "95/5"}:
        raise RuntimeError("component stress scenarios changed")
    print(json.dumps({
        "status": "PASS",
        "final_gate": metadata["final_gate"],
        "hash_checks": len(metadata["hash_verification"]),
        "sampled_pcaps": len(manifest),
        "valid_inputs": len(valid),
        "dgsb_cells": len(gates),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

