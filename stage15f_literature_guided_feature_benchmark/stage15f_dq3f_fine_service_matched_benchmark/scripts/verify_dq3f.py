#!/usr/bin/env python3
"""Independent fail-closed completion verifier for DQ-3F."""

from __future__ import annotations

from dq3f_common import OUT, SEEDS, SERVICES, read_csv, read_json, sha256_file


def main() -> int:
    required = [
        "dq3f_manifest_parity.csv", "dq3f_label_audit.md", "dq3f_training_configs.json",
        "fine_f1_results.csv", "fine_f1_predictions.csv", "coarse_f2_partial_results.csv",
        "coarse_f2_partial_predictions.csv", "service_f3_results.csv", "service_f3_predictions.csv",
        "matched_subset_comparison.csv", "fine_to_service_error_decomposition.csv",
        "per_class_metrics.csv", "per_service_metrics.csv", "fine_confusion_matrix.csv",
        "service_confusion_matrix.csv", "capture_dependency_audit.md", "capture_level_results.csv",
        "seed_stability.csv", "paired_error_analysis.csv", "dq3f_report.md", "RESULTS.md",
        "completion_verification.json", "checkpoint_hashes.csv", "protected_asset_hashes_before.json",
        "protected_asset_hashes_after.json",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"required outputs missing: {missing}")
    completion = read_json(OUT / "completion_verification.json")
    assert completion["status"] == "PASS"
    assert completion["formal_runs"] == completion["successful_runs"] == 6
    assert completion["known_test_feature_values_used"] == 0
    assert completion["unknown_test_feature_values_used"] == 0
    assert completion["protected_asset_hash_status"] == "PASS"
    assert completion["checkpoint_hashes_verified"] == 6
    assert completion["dq4_to_dq7"] == "NOT_RUN"
    assert completion["des_h1_modified"] is False
    before = read_json(OUT / "protected_asset_hashes_before.json")
    after = read_json(OUT / "protected_asset_hashes_after.json")
    assert before["files"] == after["files"]
    fine = read_csv(OUT / "fine_f1_predictions.csv")
    service = read_csv(OUT / "service_f3_predictions.csv")
    assert len(fine) == len(service) == 335 * len(SEEDS)
    for seed in SEEDS:
        f_ids = [row["flow_id"] for row in fine if int(row["seed"]) == seed]
        s_ids = [row["flow_id"] for row in service if int(row["seed"]) == seed]
        assert f_ids == s_ids and len(set(f_ids)) == 335
    hashes = read_csv(OUT / "checkpoint_hashes.csv")
    assert len(hashes) == 6
    for row in hashes:
        assert sha256_file(OUT / "runs" / row["task"] / f"seed{row['seed']}" / "model_best.pt") == row["checkpoint_sha256"]
    service_metrics = read_csv(OUT / "per_service_metrics.csv")
    for seed in SEEDS:
        assert {row["class_name"] for row in service_metrics if row["task"] == "F3" and int(row["seed"]) == seed} == set(SERVICES)
    print({
        "verification": "PASS", "required_outputs": len(required), "runs": 6,
        "fine_predictions": len(fine), "service_predictions": len(service),
        "checkpoint_hashes": len(hashes), "primary_gate": completion["primary_gate"],
        "capture_gate": completion["capture_gate"], "test_feature_values_used": 0,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
