#!/usr/bin/env python3
"""Terminal acceptance audit for Stage 13A-2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage13a2_common import (
    CONFIG_PATH,
    METHODS,
    STAGE_ROOT,
    SUMMARY,
    csv_rows,
    fixed_gl_fusion,
    read_json,
    robust_parameters,
    sha256_file,
    write_json,
)


OUTPUT = SUMMARY / "completion_verification.json"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    config = read_json(CONFIG_PATH)
    failures: list[str] = []
    runs = 0
    checkpoints: set[str] = set()
    provenance_path = SUMMARY / "provenance_verification.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else {}
    frozen_runs = provenance.get("frozen_hashes_before", {}).get("runs", {})

    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_name = f"fold{fold}_seed{seed}"
            key = f"{scenario}/{run_name}"
            run_dir = STAGE_ROOT / "artifacts" / scenario / run_name
            required = [
                "SUCCESS", "config.json", "input_hashes.json", "normalization_stats.json",
                "thresholds.json", "score_arrays.npz", "results.json", "RESULTS.md", "manifest.json",
            ]
            check(all((run_dir / name).is_file() for name in required), f"Incomplete run: {run_dir}", failures)
            check(not (run_dir / "FAILURE.json").exists(), f"Failure marker exists: {run_dir}", failures)
            if not (run_dir / "results.json").is_file() or key not in frozen_runs:
                continue
            result = read_json(run_dir / "results.json")
            normalization = read_json(run_dir / "normalization_stats.json")
            frozen = frozen_runs[key]
            source = Path(frozen["stage13a1_run_dir"])
            check(set(result["methods"]) == set(METHODS), f"Method set mismatch: {run_dir}", failures)
            check(result["parity"]["status"] == "PASS", f"Baseline parity failed: {run_dir}", failures)
            check(result["parity"]["centroid_metric_max_abs_error"] <= 1e-12, f"Centroid parity exceeded: {run_dir}", failures)
            check(result["parity"]["knn10_metric_max_abs_error"] <= 1e-12, f"kNN-10 parity exceeded: {run_dir}", failures)
            operations = result["operations"]
            check(not operations["new_encoder_training"] and not operations["encoder_inference"], f"Encoder execution recorded: {run_dir}", failures)
            check(operations["optimizer"] is None and not operations["backward"], f"Training operation recorded: {run_dir}", failures)
            check(not operations["checkpoint_update"], f"Checkpoint update recorded: {run_dir}", failures)
            check(operations["normalization_fit_split"] == "Known Validation only", f"Normalization source violation: {run_dir}", failures)
            check(not operations["unknown_calibration"] and not operations["test_threshold_tuning"], f"Calibration violation: {run_dir}", failures)
            check(not operations["weight_search"] and operations["allowed_weights"] == [0.5, 0.5], f"Fusion weight violation: {run_dir}", failures)
            check(not any(operations[name] for name in ("learned_fusion", "logvar", "margin", "covariance", "k2")), f"Forbidden method term recorded: {run_dir}", failures)
            check(operations["external_test_datasets_read"] == [], f"External Test access recorded: {run_dir}", failures)
            check(not operations["cipherspectrum_test_read"], f"CipherSpectrum Test access recorded: {run_dir}", failures)
            check(not operations["frozen_experiment_modified"], f"Frozen modification recorded: {run_dir}", failures)
            check(normalization["fit_split"] == "Known Validation only" and not normalization["unknown_used"] and not normalization["test_used"], f"Normalization metadata violation: {run_dir}", failures)
            thresholds = read_json(run_dir / "thresholds.json")
            check(set(thresholds) == set(METHODS), f"Threshold method set mismatch: {run_dir}", failures)
            check(all(value["source"] == "Known Validation score P95 only" and value["numpy_method"] == "higher" for value in thresholds.values()), f"Threshold source violation: {run_dir}", failures)

            for name in ("results.json", "input_hashes.json", "score_arrays.npz"):
                check(sha256_file(source / name) == frozen["artifacts"][name], f"Frozen source hash changed: {source / name}", failures)
            check(sha256_file(Path(frozen["checkpoint_path"])) == frozen["checkpoint_sha256"], f"Checkpoint hash changed: {run_dir}", failures)
            with np.load(source / "score_arrays.npz", allow_pickle=False) as source_arrays, np.load(run_dir / "score_arrays.npz", allow_pickle=False) as output_arrays:
                pc = robust_parameters(source_arrays["validation_centroid"], float(config["normalization"]["eps"]))
                pk = robust_parameters(source_arrays["validation_knn10"], float(config["normalization"]["eps"]))
                check(all(np.isclose(normalization["centroid"][name], pc[name]) for name in ("median", "mad", "eps", "denominator")), f"Centroid normalization mismatch: {run_dir}", failures)
                check(all(np.isclose(normalization["knn10"][name], pk[name]) for name in ("median", "mad", "eps", "denominator")), f"kNN normalization mismatch: {run_dir}", failures)
                for role in ("validation", "known_test", "unknown_test"):
                    zc, zk, gl = fixed_gl_fusion(source_arrays[f"{role}_centroid"], source_arrays[f"{role}_knn10"], pc, pk)
                    check(np.array_equal(output_arrays[f"{role}_centroid"], source_arrays[f"{role}_centroid"]), f"Centroid score copy mismatch: {run_dir}/{role}", failures)
                    check(np.array_equal(output_arrays[f"{role}_knn10"], source_arrays[f"{role}_knn10"]), f"kNN score copy mismatch: {run_dir}/{role}", failures)
                    check(np.allclose(output_arrays[f"{role}_zc"], zc, rtol=0, atol=1e-12), f"Zc mismatch: {run_dir}/{role}", failures)
                    check(np.allclose(output_arrays[f"{role}_zk"], zk, rtol=0, atol=1e-12), f"Zk mismatch: {run_dir}/{role}", failures)
                    check(np.allclose(output_arrays[f"{role}_gl"], gl, rtol=0, atol=1e-12), f"GL mismatch: {run_dir}/{role}", failures)
            checkpoints.add(result["frozen_inputs"]["checkpoint_sha256"])
            runs += 1

    required_summary = ["run_level_results.csv", "scenario_summary.csv", "gl_vs_centroid.csv", "final_gate.md", "final_gate.json", "provenance_verification.json"]
    check(all((SUMMARY / name).is_file() for name in required_summary), "Required summary output missing", failures)
    run_rows = csv_rows(SUMMARY / "run_level_results.csv") if (SUMMARY / "run_level_results.csv").is_file() else []
    scenario_rows = csv_rows(SUMMARY / "scenario_summary.csv") if (SUMMARY / "scenario_summary.csv").is_file() else []
    comparison_rows = csv_rows(SUMMARY / "gl_vs_centroid.csv") if (SUMMARY / "gl_vs_centroid.csv").is_file() else []
    check(len(run_rows) == 45, f"Expected 45 method rows, got {len(run_rows)}", failures)
    check(len(scenario_rows) == 9, f"Expected 9 scenario rows, got {len(scenario_rows)}", failures)
    check(len(comparison_rows) == 15, f"Expected 15 paired rows, got {len(comparison_rows)}", failures)
    check(set(row["method"] for row in run_rows) == set(METHODS), "Unexpected method in run table", failures)
    check(all(row["new_encoder_training"] == "False" for row in run_rows), "Training flag in run table", failures)
    check(provenance.get("status") == "PASS" and provenance.get("phase") == "after", "Final provenance not PASS/after", failures)
    check(provenance.get("frozen_runs_verified") == 15 and provenance.get("unique_checkpoint_hashes") == 15, "Provenance run/checkpoint count mismatch", failures)
    check(not provenance.get("frozen_experiment_modified", True), "Provenance reports frozen change", failures)
    check(provenance.get("external_test_datasets_read") == [], "Provenance reports external Test access", failures)
    gate = read_json(SUMMARY / "final_gate.json") if (SUMMARY / "final_gate.json").is_file() else {}
    check(gate.get("final_gate") in {"GO", "CONDITIONAL_GO", "NO_GO"}, "Invalid final gate", failures)
    check(not gate.get("new_encoder_training", True), "Gate reports encoder training", failures)
    check(gate.get("external_test_datasets_read") == [], "Gate reports external Test access", failures)
    check(not gate.get("frozen_experiment_modified", True), "Gate reports frozen modification", failures)
    check(not gate.get("next_stage_started", True), "Gate reports next stage", failures)
    check(runs == 15 and len(checkpoints) == 15, "Run/checkpoint count mismatch", failures)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "formal_runs": runs,
        "unique_checkpoint_hashes": len(checkpoints),
        "method_rows": len(run_rows),
        "scenario_rows": len(scenario_rows),
        "comparison_rows": len(comparison_rows),
        "normalization_fit_split": "Known Validation only",
        "fusion_weights": [0.5, 0.5],
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

