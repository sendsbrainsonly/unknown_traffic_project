#!/usr/bin/env python3
"""Terminal Stage 11B acceptance audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from stage11b_common import CONFIG_PATH, METHODS, STAGE_ROOT, read_json, sha256_file, write_json


SUMMARY = STAGE_ROOT / "outputs" / "summary"
OUTPUT = SUMMARY / "completion_verification.json"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = read_json(CONFIG_PATH)
    failures: list[str] = []
    formal_runs = 0
    checkpoint_hashes: set[str] = set()
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            required = [
                "SUCCESS", "config.json", "results.json", "checkpoint.sha256",
                "support_models.joblib", "thresholds.json", "sample_outputs.npz",
                "per_class_support_analysis.csv", "unknown_absorption_analysis.csv",
                "RESULTS.md", "manifest.json",
            ]
            check(all((run_dir / name).is_file() for name in required), f"Incomplete formal run: {run_dir}", failures)
            check(not (run_dir / "FAILURE.json").exists(), f"Formal failure marker exists: {run_dir}", failures)
            if not (run_dir / "results.json").is_file():
                continue
            result = read_json(run_dir / "results.json")
            frozen = result["frozen_encoder"]
            check(result["mode"] == "FORMAL", f"Non-formal result in campaign: {run_dir}", failures)
            check(result["representation_parity"]["status"] == "PASS", f"Parity failed: {run_dir}", failures)
            check(result["representation_parity"]["max_abs_error"] <= 1e-10, f"Parity tolerance exceeded: {run_dir}", failures)
            check(set(result["evaluation"]["detectors"]) == set(METHODS), f"Detector set changed: {run_dir}", failures)
            check(frozen["checkpoint_sha256_before"] == frozen["checkpoint_sha256_after"], f"Checkpoint changed: {run_dir}", failures)
            check(frozen["model_state_sha256_before"] == frozen["model_state_sha256_after"], f"Model state changed: {run_dir}", failures)
            check(frozen["optimizer"] is None, f"Optimizer exists: {run_dir}", failures)
            check(not frozen["backward"] and not frozen["loss_update"] and not frozen["checkpoint_update"], f"Training operation recorded: {run_dir}", failures)
            check(not frozen["prototype_update"] and not frozen["new_encoder_training"], f"Prototype/encoder update recorded: {run_dir}", failures)
            check(not result["cipher_spectrum_stage9_sample_level_test_used"], f"CipherSpectrum usage recorded: {run_dir}", failures)
            check(not result["frozen_source_modified"], f"Frozen modification recorded: {run_dir}", failures)
            thresholds = read_json(run_dir / "thresholds.json")
            check(set(thresholds) == set(METHODS), f"Threshold detector set changed: {run_dir}", failures)
            check(all(value["source"] == "Known Validation anomaly score P95 only" for value in thresholds.values()), f"Threshold source violation: {run_dir}", failures)
            support = joblib.load(run_dir / "support_models.joblib")
            check(set(support["R2_R3_frozen_density"]["models"]) == {"K1", "K2"}, f"Density method set changed: {run_dir}", failures)
            check(len(support["R2_R3_frozen_density"]["models"]["K1"]) == int(config["scenarios"][scenario]["known_count"]), f"K1 class count mismatch: {run_dir}", failures)
            check(len(support["R2_R3_frozen_density"]["models"]["K2"]) == int(config["scenarios"][scenario]["known_count"]), f"K2 class count mismatch: {run_dir}", failures)
            with np.load(run_dir / "sample_outputs.npz", allow_pickle=False) as arrays:
                check("unknown_test_r0_scores" in arrays and "unknown_test_r3_scores" in arrays, f"Sample evidence incomplete: {run_dir}", failures)
            checkpoint_hashes.add(frozen["checkpoint_sha256_before"])
            formal_runs += 1

    required_summary = [
        "provenance_verification.json", "run_level_results.csv", "scenario_summary.csv",
        "primary_r2_vs_r0.csv", "r1_vs_r0.csv", "r2_vs_r1.csv", "r3_vs_r2.csv",
        "per_class_support_analysis.csv", "unknown_absorption_analysis.csv",
        "unknown_absorption_hub_summary.csv", "bootstrap_summary.csv",
        "gap_predictiveness.csv", "mechanism_case.md", "final_gate.md", "final_gate.json",
    ]
    check(all((SUMMARY / name).is_file() for name in required_summary), "Required summary file missing", failures)
    run_rows = csv_rows(SUMMARY / "run_level_results.csv") if (SUMMARY / "run_level_results.csv").is_file() else []
    class_rows = csv_rows(SUMMARY / "per_class_support_analysis.csv") if (SUMMARY / "per_class_support_analysis.csv").is_file() else []
    absorption_rows = csv_rows(SUMMARY / "unknown_absorption_analysis.csv") if (SUMMARY / "unknown_absorption_analysis.csv").is_file() else []
    bootstrap_rows = csv_rows(SUMMARY / "bootstrap_summary.csv") if (SUMMARY / "bootstrap_summary.csv").is_file() else []
    check(len(run_rows) == 60, f"Expected 60 detector rows, got {len(run_rows)}", failures)
    check(len(class_rows) == 255, f"Expected 255 class rows, got {len(class_rows)}", failures)
    check(len(absorption_rows) == 2900, f"Expected 2900 absorption rows, got {len(absorption_rows)}", failures)
    check(len(bootstrap_rows) == 36, f"Expected 36 bootstrap rows, got {len(bootstrap_rows)}", failures)
    check(set(row["method"] for row in run_rows) == set(METHODS), "Unexpected detector in run table", failures)
    check(all(row["new_encoder_training"] == "False" for row in run_rows), "Training flag in run table", failures)
    for name in ("primary_r2_vs_r0.csv", "r1_vs_r0.csv", "r2_vs_r1.csv", "r3_vs_r2.csv"):
        path = SUMMARY / name
        check(path.is_file() and len(csv_rows(path)) == 15, f"Expected 15 paired rows: {name}", failures)
    provenance = read_json(SUMMARY / "provenance_verification.json") if (SUMMARY / "provenance_verification.json").is_file() else {}
    check(provenance.get("status") == "PASS" and provenance.get("phase") == "after", "Final provenance is not PASS/after", failures)
    check(provenance.get("v6_checkpoints_verified") == 15, "Provenance did not verify 15 checkpoints", failures)
    check(not provenance.get("frozen_experiment_modified", True), "Final provenance reports frozen change", failures)
    gate = read_json(SUMMARY / "final_gate.json") if (SUMMARY / "final_gate.json").is_file() else {}
    check(gate.get("final_gate") in {"GO", "CONDITIONAL_GO", "NO_GO"}, "Invalid final gate", failures)
    check(gate.get("mechanism") in {"D1", "D2", "D3", "D4", "D5", "D6"}, "Invalid mechanism", failures)
    check(not gate.get("new_encoder_training", True), "Gate reports encoder training", failures)
    check(not gate.get("cipherspectrum_stage9_sample_level_test_used", True), "Gate reports CipherSpectrum Test use", failures)
    check(not gate.get("next_stage_started", True), "Gate reports next stage started", failures)
    check(not (STAGE_ROOT.parent / "stage11c_decoupled_support_method").exists(), "Unexpected next-stage directory exists", failures)
    required_headings = [
        "Goal", "Stage11A NO-GO", "Why Decouple Training and Detection", "Frozen B0 Encoders",
        "R0 Native", "R1 Empirical Centroid", "R2 Full Gaussian K1", "R3 Full Gaussian K2",
        "Validation-Only Calibration", "Five-Seed Evaluation", "Known Classification",
        "Unknown Detection", "Absorption Analysis", "Mechanism Interpretation", "Final Gate",
        "Limitations", "Next Step",
    ]
    readme = (STAGE_ROOT / "README.md").read_text(encoding="utf-8")
    check(all(f"## {heading}" in readme for heading in required_headings), "README section missing", failures)
    status = "PASS" if not failures else "FAIL"
    report = {
        "status": status,
        "formal_frozen_runs": formal_runs,
        "unique_checkpoint_hashes": len(checkpoint_hashes),
        "detector_rows": len(run_rows),
        "class_rows": len(class_rows),
        "absorption_rows": len(absorption_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "new_encoder_training": False,
        "cipherspectrum_stage9_sample_level_test_used": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
        "final_gate": gate.get("final_gate"),
        "mechanism": gate.get("mechanism"),
        "failures": failures,
    }
    write_json(OUTPUT, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
