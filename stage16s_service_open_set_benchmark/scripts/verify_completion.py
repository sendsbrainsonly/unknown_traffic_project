#!/usr/bin/env python3
"""Fail-closed completion verification for Stage 16S."""

from __future__ import annotations

from pathlib import Path

from stage16s_common import CANONICAL, METHODS, OUT, RUNS, SEEDS, SERVICES, protocol_id, read_csv, read_json, sha256_file


def main() -> None:
    failures = []
    metric_rows = read_csv(OUT / "service_open_set_six_metrics.csv")
    if len(metric_rows) != 72:
        failures.append(f"metric rows={len(metric_rows)}")
    paired = read_csv(OUT / "paired_vs_opendetect.csv")
    if len(paired) != 54:
        failures.append(f"paired rows={len(paired)}")
    for service in SERVICES:
        pid = protocol_id(service)
        for seed in SEEDS:
            run = CANONICAL / pid / f"seed{seed}"
            if not (run / "SUCCESS").is_file():
                failures.append(f"missing SUCCESS {pid}/{seed}")
                continue
            result = read_json(run / "result.json")
            if set(result["methods"]) != set(METHODS):
                failures.append(f"method set {pid}/{seed}")
            if any(result[name] != 0 for name in (
                "unknown_training_samples", "unknown_validation_samples", "unknown_support_samples",
                "unknown_normalization_samples", "unknown_threshold_samples", "known_test_selection_samples",
            )):
                failures.append(f"Unknown/Test leak marker {pid}/{seed}")
            if sha256_file(Path(result["checkpoint_path"])) != result["checkpoint_sha256"]:
                failures.append(f"checkpoint hash {pid}/{seed}")
            if sha256_file(run / "sample_scores.csv") != result["score_file_sha256"]:
                failures.append(f"score hash {pid}/{seed}")
    completion = read_json(OUT / "completion_verification.json")
    if completion["successful_runs"] != 18 or not completion["strict_unknown_free"]:
        failures.append("completion summary invalid")
    before = read_json(OUT / "protected_asset_hashes_before.json")
    after = read_json(OUT / "protected_asset_hashes_after.json")
    if before != after:
        failures.append("protected hashes changed")
    for path, expected in after["files"].items():
        if sha256_file(Path(path)) != expected:
            failures.append(f"protected asset changed {path}")
    required = [
        "method_identity_and_lineage.md", "existing_implementation_reuse_audit.md", "service_open_set_feasibility.md",
        "service_unknown_protocol_manifest.csv", "known_unknown_membership_audit.csv", "training_configs.json",
        "shared_encoder_checkpoints.csv", "known_classifier_results.csv", "od_native_scores.csv", "des_v0_scores.csv",
        "des_v1_scores.csv", "h1_scores.csv", "known_val_thresholds.csv", "service_open_set_six_metrics.csv",
        "paired_vs_opendetect.csv", "per_unknown_service_results.csv", "per_known_service_results.csv",
        "unknown_confusion_analysis.csv", "known_rejection_analysis.csv", "seed_stability.csv",
        "failure_and_blocker_log.md", "stage16s_report.md", "RESULTS.md", "completion_verification.json",
    ]
    for name in required:
        if not (OUT / name).is_file():
            failures.append(f"missing {name}")
    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    print("PASS")
    print("formal_runs=18/18 methods=4 metric_rows=72 paired_rows=54")
    print("strict_unknown_free=PASS shared_encoder=PASS protected_assets=UNCHANGED")


if __name__ == "__main__":
    main()
