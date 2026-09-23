#!/usr/bin/env python3
"""Independent completion verifier for Stage 14D."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stage14d_common import (
    ARTIFACT_ROOT,
    METHODS,
    ROOT,
    detection_metrics,
    load_protocols,
    read_json,
    robust_normalize,
    threshold_p95,
    verify_checkpoint_freeze,
    verify_stage14b,
)


REQUIRED = (
    "stage14d_run_results.csv",
    "stage14d_setting_summary.csv",
    "stage14d_paired_comparison.csv",
    "stage14d_per_class_analysis.csv",
    "stage14d_score_distributions.csv",
    "stage14d_open_set_report.md",
    "completion_verification.json",
)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    freeze = verify_stage14b()
    checkpoints = verify_checkpoint_freeze()
    protocols = load_protocols()
    failures: list[str] = []
    for name in REQUIRED:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing/empty {name}")
    for protocol_id in protocols:
        run_dir = ARTIFACT_ROOT / protocol_id
        if not (run_dir / "SUCCESS").is_file():
            failures.append(f"missing SUCCESS {protocol_id}")
            continue
        result = read_json(run_dir / "result.json")
        if result.get("native_validation_parity", {}).get("status") != "PASS":
            failures.append(f"native validation parity failed {protocol_id}")
        if result.get("stage14b_before") != freeze or result.get("stage14b_after") != freeze:
            failures.append(f"Stage 14B mismatch {protocol_id}")
        operations = result.get("operations", {})
        forbidden = {
            "new_encoder_training": False,
            "unknown_support_samples": 0,
            "unknown_normalization_samples": 0,
            "unknown_threshold_samples": 0,
            "test_parameter_selection": False,
        }
        for key, expected in forbidden.items():
            if operations.get(key) != expected:
                failures.append(f"forbidden operation {protocol_id}: {key}={operations.get(key)}")
        if result.get("checkpoint_sha256_after") != checkpoints[protocol_id]["checkpoint_sha256"]:
            failures.append(f"checkpoint hash mismatch after run {protocol_id}")
        with np.load(run_dir / "score_arrays.npz", allow_pickle=False) as arrays:
            params = result["normalization"]
            expected_m2 = {
                role: 0.5 * robust_normalize(arrays[f"{role}_global"], params["M2_global"])
                + 0.5 * robust_normalize(arrays[f"{role}_local"], params["M2_local"])
                for role in ("validation", "known_test", "unknown_test")
            }
            for role, values in expected_m2.items():
                if not np.allclose(values, arrays[f"{role}_m2"], rtol=0, atol=1e-12):
                    failures.append(f"M2 formula mismatch {protocol_id}:{role}")
            for method in METHODS:
                suffix = method.lower()
                validation = arrays[f"validation_{suffix}"]
                known = arrays[f"known_test_{suffix}"]
                unknown = arrays[f"unknown_test_{suffix}"]
                recomputed = detection_metrics(validation, known, unknown)
                observed = result["metrics"][method]
                for metric in ("threshold", "auroc", "auprc", "ufar", "known_frr", "binary_f1"):
                    if abs(float(recomputed[metric]) - float(observed[metric])) > 1e-12:
                        failures.append(f"metric mismatch {protocol_id}:{method}:{metric}")
                if float(observed["threshold"]) != threshold_p95(validation):
                    failures.append(f"P95 threshold mismatch {protocol_id}:{method}")
        sample_rows = csv_rows(run_dir / "sample_scores.csv")
        expected_score_rows = 3 * (
            int(result["samples"]["validation"])
            + int(result["samples"]["known_test"])
            + int(result["samples"]["unknown_test"])
        )
        if len(sample_rows) != expected_score_rows:
            failures.append(f"sample score row count mismatch {protocol_id}")
        score_keys = {(row["role"], row["flow_uid"], row["method"]) for row in sample_rows}
        if len(score_keys) != len(sample_rows):
            failures.append(f"duplicate sample score key {protocol_id}")
        if len(csv_rows(run_dir / "per_class_analysis.csv")) != 30:
            failures.append(f"per-class row count mismatch {protocol_id}")
    if (ROOT / "stage14d_run_results.csv").is_file():
        rows = csv_rows(ROOT / "stage14d_run_results.csv")
        expected_pairs = {(protocol_id, method) for protocol_id in protocols for method in METHODS}
        actual_pairs = {(row["protocol_id"], row["method"]) for row in rows}
        if actual_pairs != expected_pairs or len(rows) != 45:
            failures.append(f"run result grid mismatch: rows={len(rows)}")
    if (ROOT / "stage14d_setting_summary.csv").is_file() and len(csv_rows(ROOT / "stage14d_setting_summary.csv")) != 12:
        failures.append("setting summary row count mismatch")
    if (ROOT / "stage14d_paired_comparison.csv").is_file() and len(csv_rows(ROOT / "stage14d_paired_comparison.csv")) != 12:
        failures.append("paired comparison row count mismatch")
    if (ROOT / "stage14d_per_class_analysis.csv").is_file() and len(csv_rows(ROOT / "stage14d_per_class_analysis.csv")) != 450:
        failures.append("global per-class row count mismatch")
    completion = read_json(ROOT / "completion_verification.json") if (ROOT / "completion_verification.json").is_file() else {}
    if completion.get("conclusion") not in {
        "DES_V1_EXTERNAL_CONFIRMED",
        "DECOUPLING_CONFIRMED_LOCAL_NOT_CONFIRMED",
        "DECOUPLING_ONLY_PARTIALLY_CONFIRMED",
        "EXTERNAL_VALIDATION_FAILED",
    }:
        failures.append("invalid final conclusion")
    if failures:
        raise RuntimeError("; ".join(failures))
    print("PASS: Stage 14D completion verified independently")


if __name__ == "__main__":
    main()
