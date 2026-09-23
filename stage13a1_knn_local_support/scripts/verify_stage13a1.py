#!/usr/bin/env python3
"""Terminal acceptance audit for Stage13A-1."""

from __future__ import annotations

import json

import numpy as np

from stage13_common import CONFIG_PATH, METHODS, STAGE_ROOT, SUMMARY, csv_rows, read_json, write_json


OUTPUT = SUMMARY / "completion_verification.json"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    config = read_json(CONFIG_PATH)
    failures: list[str] = []
    runs = 0
    checkpoints: set[str] = set()
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            required = ["SUCCESS", "config.json", "input_hashes.json", "thresholds.json", "score_arrays.npz", "results.json", "RESULTS.md", "manifest.json"]
            check(all((run_dir / name).is_file() for name in required), f"Incomplete run: {run_dir}", failures)
            check(not (run_dir / "FAILURE.json").exists(), f"Failure marker exists: {run_dir}", failures)
            if not (run_dir / "results.json").is_file():
                continue
            result = read_json(run_dir / "results.json")
            check(set(result["methods"]) == set(METHODS), f"Method set mismatch: {run_dir}", failures)
            check(result["parity"]["status"] == "PASS", f"Centroid parity failed: {run_dir}", failures)
            check(result["parity"]["centroid_metric_max_abs_error"] <= 1e-10, f"Metric parity exceeded: {run_dir}", failures)
            operations = result["operations"]
            check(not operations["new_encoder_training"], f"Encoder training recorded: {run_dir}", failures)
            check(operations["optimizer"] is None and not operations["backward"], f"Training operation recorded: {run_dir}", failures)
            check(not operations["checkpoint_update"], f"Checkpoint update recorded: {run_dir}", failures)
            check(operations["self_neighbor_excluded_for_train"], f"Self exclusion absent: {run_dir}", failures)
            check(operations["allowed_k"] == [5, 10], f"Unexpected k: {run_dir}", failures)
            check(not operations["score_fusion"], f"Fusion recorded: {run_dir}", failures)
            check(operations["external_test_datasets_read"] == [], f"External Test access recorded: {run_dir}", failures)
            check(not operations["cipherspectrum_test_read"], f"CipherSpectrum Test access recorded: {run_dir}", failures)
            check(not operations["frozen_experiment_modified"], f"Frozen modification recorded: {run_dir}", failures)
            thresholds = read_json(run_dir / "thresholds.json")
            check(set(thresholds) == set(METHODS), f"Threshold method set mismatch: {run_dir}", failures)
            check(all(value["source"] == "Known Validation score P95 only" for value in thresholds.values()), f"Threshold source violation: {run_dir}", failures)
            with np.load(run_dir / "score_arrays.npz", allow_pickle=False) as arrays:
                check("unknown_test_knn10" in arrays and "validation_knn5" in arrays, f"Score evidence missing: {run_dir}", failures)
            checkpoints.add(result["frozen_inputs"]["checkpoint_sha256"])
            runs += 1

    required_summary = ["run_level_results.csv", "scenario_summary.csv", "knn10_vs_centroid.csv", "knn5_sensitivity.csv", "final_gate.md", "final_gate.json", "provenance_verification.json"]
    check(all((SUMMARY / name).is_file() for name in required_summary), "Required summary output missing", failures)
    run_rows = csv_rows(SUMMARY / "run_level_results.csv") if (SUMMARY / "run_level_results.csv").is_file() else []
    scenario_rows = csv_rows(SUMMARY / "scenario_summary.csv") if (SUMMARY / "scenario_summary.csv").is_file() else []
    knn10_rows = csv_rows(SUMMARY / "knn10_vs_centroid.csv") if (SUMMARY / "knn10_vs_centroid.csv").is_file() else []
    knn5_rows = csv_rows(SUMMARY / "knn5_sensitivity.csv") if (SUMMARY / "knn5_sensitivity.csv").is_file() else []
    check(len(run_rows) == 45, f"Expected 45 method rows, got {len(run_rows)}", failures)
    check(len(scenario_rows) == 9, f"Expected 9 scenario rows, got {len(scenario_rows)}", failures)
    check(len(knn10_rows) == 15 and len(knn5_rows) == 15, "Expected 15 paired rows per comparison", failures)
    check(set(row["method"] for row in run_rows) == set(METHODS), "Unexpected method in run table", failures)
    check(all(row["new_encoder_training"] == "False" for row in run_rows), "Training flag in run table", failures)

    provenance = read_json(SUMMARY / "provenance_verification.json") if (SUMMARY / "provenance_verification.json").is_file() else {}
    check(provenance.get("status") == "PASS" and provenance.get("phase") == "after", "Final provenance not PASS/after", failures)
    check(provenance.get("frozen_runs_verified") == 15, "Provenance did not verify 15 runs", failures)
    check(provenance.get("unique_checkpoint_hashes") == 15, "Provenance did not verify 15 checkpoints", failures)
    check(not provenance.get("frozen_experiment_modified", True), "Provenance reports frozen change", failures)
    check(provenance.get("external_test_datasets_read") == [], "Provenance reports external Test access", failures)
    check(not provenance.get("cipherspectrum_test_read", True), "Provenance reports CipherSpectrum Test access", failures)

    gate = read_json(SUMMARY / "final_gate.json") if (SUMMARY / "final_gate.json").is_file() else {}
    check(gate.get("final_gate") in {"GO", "CONDITIONAL_GO", "NO_GO"}, "Invalid final gate", failures)
    check(not gate.get("new_encoder_training", True), "Gate reports encoder training", failures)
    check(gate.get("external_test_datasets_read") == [], "Gate reports external Test access", failures)
    check(not gate.get("frozen_experiment_modified", True), "Gate reports frozen modification", failures)
    check(not gate.get("next_stage_started", True), "Gate reports next stage", failures)
    check(runs == 15 and len(checkpoints) == 15, "Run/checkpoint count mismatch", failures)

    status = "PASS" if not failures else "FAIL"
    report = {
        "status": status,
        "formal_runs": runs,
        "unique_checkpoint_hashes": len(checkpoints),
        "method_rows": len(run_rows),
        "scenario_rows": len(scenario_rows),
        "knn10_comparison_rows": len(knn10_rows),
        "knn5_comparison_rows": len(knn5_rows),
        "new_encoder_training": False,
        "external_test_datasets_read": [],
        "cipherspectrum_test_read": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
        "final_gate": gate.get("final_gate"),
        "failures": failures,
    }
    write_json(OUTPUT, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

