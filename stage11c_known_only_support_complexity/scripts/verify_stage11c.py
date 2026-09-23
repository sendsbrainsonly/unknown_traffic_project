#!/usr/bin/env python3
"""Semantic completion checks for Stage 11C."""

from __future__ import annotations

import json

from stage11c_common import SCENARIOS, SEEDS, STAGE_ROOT, SUMMARY_ROOT, output_run_dir, read_csv, read_json, write_json


def main() -> None:
    failures: list[str] = []
    formal = 0
    for scenario in SCENARIOS:
        for fold, seed in enumerate(SEEDS):
            run = output_run_dir(scenario, fold, seed)
            if not (run / "SUCCESS").is_file():
                failures.append(f"Missing SUCCESS: {run}")
                continue
            result = read_json(run / "results.json")
            checks = {
                "unknown feature": result.get("unknown_feature_used") is False,
                "Known Test feature": result.get("known_test_feature_used") is False,
                "Unknown Test feature": result.get("unknown_test_feature_used") is False,
                "new encoder training": result.get("new_encoder_training") is False,
                "new detector fitting": result.get("new_detector_fitting") is False,
                "new threshold fitting": result.get("new_threshold_fitting") is False,
                "frozen K1 reuse": result.get("frozen_k1_reused_without_refit") is True,
                "bootstrap B": result.get("bootstrap_iterations") == 100,
                "next stage": result.get("next_stage_started") is False,
            }
            for name, passed in checks.items():
                if not passed:
                    failures.append(f"{name} contract failed: {run}")
            formal += 1

    required = [
        "provenance_verification.json",
        "known_only_features.csv",
        "class_level_geometry.csv",
        "covariance_stability.csv",
        "validation_fit_comparison.csv",
        "validation_classification_comparison.csv",
        "coverage_stability.csv",
        "feature_outcome_correlations.csv",
        "loso_analysis.csv",
        "candidate_rule_results.csv",
        "a1_failure_diagnosis.md",
        "a2_covariance_benefit_diagnosis.md",
        "a3_diagnosis.md",
        "mechanism_case.md",
        "final_gate.md",
    ]
    for name in required:
        if not (SUMMARY_ROOT / name).is_file():
            failures.append(f"Missing output: {name}")
    counts = {}
    if not failures:
        counts = {
            "known_only_features": len(read_csv(SUMMARY_ROOT / "known_only_features.csv")),
            "class_level_geometry": len(read_csv(SUMMARY_ROOT / "class_level_geometry.csv")),
            "covariance_stability": len(read_csv(SUMMARY_ROOT / "covariance_stability.csv")),
            "validation_fit": len(read_csv(SUMMARY_ROOT / "validation_fit_comparison.csv")),
            "validation_classification": len(read_csv(SUMMARY_ROOT / "validation_classification_comparison.csv")),
            "coverage": len(read_csv(SUMMARY_ROOT / "coverage_stability.csv")),
            "candidate_rules": len(read_csv(SUMMARY_ROOT / "candidate_rule_results.csv")),
            "loso": len(read_csv(SUMMARY_ROOT / "loso_analysis.csv")),
        }
        expected = {"known_only_features": 15, "class_level_geometry": 510, "covariance_stability": 255, "validation_fit": 255, "validation_classification": 15, "coverage": 510, "candidate_rules": 72, "loso": 72}
        for key, value in expected.items():
            if counts[key] != value:
                failures.append(f"Unexpected {key} row count: {counts[key]} != {value}")

    provenance = read_json(SUMMARY_ROOT / "provenance_verification.json") if (SUMMARY_ROOT / "provenance_verification.json").is_file() else {}
    if provenance.get("status") != "PASS" or provenance.get("phase") != "after" or provenance.get("frozen_experiment_modified"):
        failures.append("Final provenance is not clean PASS/after")
    gate = read_json(SUMMARY_ROOT / "final_gate.json") if (SUMMARY_ROOT / "final_gate.json").is_file() else {}
    if gate.get("final_gate") not in ("SIGNAL_FOUND", "WEAK_SIGNAL", "NO_SIGNAL"):
        failures.append("Invalid final gate")
    if gate.get("mechanism") not in ("C1", "C2", "C3", "C4", "C5"):
        failures.append("Invalid mechanism")
    for key in ("unknown_feature_used_for_criterion", "known_test_feature_used_for_criterion", "cipherspectrum_stage9_sample_level_test_used", "new_training", "new_detector_fitting", "frozen_experiment_modified", "next_stage_started"):
        if gate.get(key) is not False:
            failures.append(f"Gate boundary failed: {key}")

    headings = ["Goal", "Stage11B Motivation", "Why A1/A2/A3 Differ", "Known-Only Constraint", "Covariance Geometry", "Covariance Stability", "Validation Density Fit", "Validation Classification", "Margin Analysis", "Coverage Analysis", "Candidate Rules", "Leave-One-Scenario-Out", "A1 Diagnosis", "A2 Diagnosis", "A3 Diagnosis", "Mechanism", "Final Gate", "Limitations", "Next Step"]
    readme = (STAGE_ROOT / "README.md").read_text(encoding="utf-8")
    for heading in headings:
        if f"## {heading}" not in readme:
            failures.append(f"README missing heading: {heading}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "formal_runs": formal,
        **counts,
        "new_training": False,
        "unknown_feature_used_for_criterion": False,
        "cipherspectrum_stage9_sample_level_test_used": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
        "final_gate": gate.get("final_gate"),
        "mechanism": gate.get("mechanism"),
        "failures": failures,
    }
    write_json(SUMMARY_ROOT / "completion_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
