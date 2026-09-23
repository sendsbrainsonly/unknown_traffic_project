#!/usr/bin/env python3
"""Fresh, read-only scientific verification of the completed 15-run grid."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from common import OUTPUTS_ROOT, RUNS_ROOT, load_protocols, sha256_file, verify_freeze


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    failures: list[str] = []
    freeze = verify_freeze()
    protocols = load_protocols()
    input_audit = json.loads((OUTPUTS_ROOT / "input_audit.json").read_text(encoding="utf-8"))
    if input_audit["status"] != "PASS" or len(input_audit["runs"]) != 15:
        failures.append("input audit is not PASS for 15 runs")
    result_rows: dict[str, dict[str, str]] = {}
    with (OUTPUTS_ROOT / "stage14c_native_run_results.csv").open(encoding="utf-8", newline="") as handle:
        result_rows = {row["protocol_id"]: row for row in csv.DictReader(handle)}
    if set(result_rows) != set(protocols):
        failures.append("run-results CSV protocol grid mismatch")

    run_checks: list[dict[str, object]] = []
    for protocol_id, protocol in sorted(protocols.items()):
        run_dir = RUNS_ROOT / protocol_id
        required = [
            run_dir / "COMPLETED", run_dir / "result.json", run_dir / "history.csv",
            run_dir / "per_class_metrics.json", run_dir / "validation_confusion_matrix.npy",
            run_dir / "best_checkpoint.pt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.append(f"{protocol_id}: missing artifacts {missing}")
            continue
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        with (run_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        per_class = json.loads((run_dir / "per_class_metrics.json").read_text(encoding="utf-8"))
        matrix = np.load(run_dir / "validation_confusion_matrix.npy", allow_pickle=False)
        local_failures: list[str] = []
        if result["status"] != "success" or result["anomaly"]:
            local_failures.append("status/anomaly")
        if len(history) != 100 or int(result["epochs_completed"]) != 100:
            local_failures.append("not exactly 100 completed epochs")
        if any(not math.isfinite(float(row[key])) for row in history for key in (
            "train_accuracy", "train_total", "val_accuracy", "val_total"
        )):
            local_failures.append("non-finite epoch metric")
        validation_accuracies = [float(row["val_accuracy"]) for row in history]
        first_best = validation_accuracies.index(max(validation_accuracies)) + 1
        if int(result["best_epoch"]) != first_best:
            local_failures.append("best epoch is not first maximum validation Accuracy")
        checkpoint_hash = sha256_file(run_dir / "best_checkpoint.pt")
        if checkpoint_hash != result["checkpoint_sha256"]:
            local_failures.append("checkpoint SHA256 mismatch")
        if result["freeze_before"] != freeze or result["freeze_after"] != freeze:
            local_failures.append("per-run freeze mismatch")
        if any(int(result[field]) != 0 for field in (
            "unknown_samples_used_in_training", "unknown_samples_used_in_validation",
            "known_test_samples_used",
        )):
            local_failures.append("model-visible data boundary violation")
        expected_classes = len(protocol["known_applications"])
        if len(per_class) != expected_classes or matrix.shape != (expected_classes, expected_classes):
            local_failures.append("per-class/confusion shape mismatch")
        supports = np.asarray([int(row["support"]) for row in per_class], dtype=np.int64)
        f1_values = np.asarray([float(row["f1"]) for row in per_class], dtype=np.float64)
        reconstructed_macro = float(f1_values.mean())
        reconstructed_weighted = float(np.average(f1_values, weights=supports))
        reconstructed_accuracy = float(np.trace(matrix) / matrix.sum())
        if int(supports.sum()) != int(result["validation_samples"]) or int(matrix.sum()) != int(result["validation_samples"]):
            local_failures.append("validation support mismatch")
        if not close(reconstructed_accuracy, float(result["validation_accuracy"])):
            local_failures.append("Accuracy reconstruction mismatch")
        if not close(reconstructed_macro, float(result["validation_macro_f1"])):
            local_failures.append("Macro-F1 reconstruction mismatch")
        if not close(reconstructed_weighted, float(result["validation_weighted_f1"])):
            local_failures.append("Weighted-F1 reconstruction mismatch")
        csv_row = result_rows.get(protocol_id, {})
        for csv_name, json_name in (
            ("validation_accuracy", "validation_accuracy"),
            ("validation_macro_f1", "validation_macro_f1"),
            ("validation_weighted_f1", "validation_weighted_f1"),
        ):
            if not csv_row or not close(float(csv_row[csv_name]), float(result[json_name])):
                local_failures.append(f"run-results CSV mismatch: {csv_name}")
        if local_failures:
            failures.extend(f"{protocol_id}: {failure}" for failure in local_failures)
        run_checks.append({
            "protocol_id": protocol_id,
            "history_epochs": len(history),
            "best_epoch": int(result["best_epoch"]),
            "checkpoint_sha256_verified": checkpoint_hash == result["checkpoint_sha256"],
            "accuracy_reconstructed": reconstructed_accuracy,
            "macro_f1_reconstructed": reconstructed_macro,
            "weighted_f1_reconstructed": reconstructed_weighted,
            "status": "PASS" if not local_failures else "FAIL",
        })

    with (OUTPUTS_ROOT / "stage14c_native_per_class_recall.csv").open(encoding="utf-8", newline="") as handle:
        per_class_csv_count = sum(1 for _ in csv.DictReader(handle))
    if per_class_csv_count != 105:
        failures.append(f"expected 105 per-class rows, found {per_class_csv_count}")
    with (OUTPUTS_ROOT / "stage14c_native_setting_summary.csv").open(encoding="utf-8", newline="") as handle:
        setting_rows = list(csv.DictReader(handle))
    if [row["setting"] for row in setting_rows] != ["Low", "Medium", "High", "Overall"]:
        failures.append("setting summary rows mismatch")
    source = (Path(__file__).with_name("train_one.py")).read_text(encoding="utf-8").lower()
    forbidden_array_paths = ("test_images.npy", "test_labels.npy", "unknown_images.npy", "unknown_labels.npy")
    if any(token in source for token in forbidden_array_paths):
        failures.append("training runner contains a forbidden Test/Unknown array path")
    verification = {
        "status": "PASS" if not failures else "FAIL",
        "freeze": freeze,
        "protocols_verified": len(run_checks),
        "formal_failures": 0,
        "unknown_training_samples": 0,
        "unknown_validation_samples": 0,
        "known_test_samples": 0,
        "per_class_csv_rows": per_class_csv_count,
        "setting_summary_rows": len(setting_rows),
        "run_checks": run_checks,
        "failures": failures,
    }
    (OUTPUTS_ROOT / "completion_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: verification[key] for key in (
        "status", "protocols_verified", "formal_failures", "unknown_training_samples",
        "unknown_validation_samples", "known_test_samples", "per_class_csv_rows",
        "setting_summary_rows", "failures",
    )}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
