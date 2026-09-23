#!/usr/bin/env python3
"""Fail-closed completion verifier for Stage 15F-0."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import ROOT, sha256_file, write_json


REQUIRED = (
    "literature_primary_source_notes.md",
    "literature_feature_registry.csv",
    "feature_group_definitions.json",
    "dataset_feature_availability.csv",
    "packet_window_coverage.csv",
    "encryption_visibility_audit.csv",
    "feature_lineage_audit.md",
    "stage15f1_preregistered_matrix.csv",
    "stage15f_report.md",
    "RESULTS.md",
    "manifest.json",
    "window_cache_parity.json",
    "stage15f0_aggregation_audit.json",
    "frozen_asset_hashes_before_initial.json",
    "frozen_asset_hashes_before.json",
    "frozen_asset_hashes_after.json",
)

FORBIDDEN_FUTURE_RESULTS = (
    "single_feature_results.csv",
    "per_class_feature_results.csv",
    "feature_complementarity.csv",
    "multiview_ablation.csv",
    "paired_vs_native.csv",
    "open_set_six_metrics.csv",
)


def load_json(name: str):
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_rows(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    checks: dict[str, object] = {}
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    checks["required_outputs_present"] = not missing
    checks["missing_required_outputs"] = missing

    literature = csv_rows("literature_feature_registry.csv")
    checks["literature_work_count"] = len(literature)
    checks["literature_19_of_19"] = len(literature) == 19
    evidence_blob = " ".join(" ".join(row.values()) for row in literature)
    checks["evidence_tags_present"] = all(
        tag in evidence_blob
        for tag in ("PAPER_CONFIRMED", "CODE_CONFIRMED", "INFERRED", "UNKNOWN")
    )

    feature_config = load_json("feature_group_definitions.json")
    feature_ids = [row["id"] for row in feature_config["feature_groups"]]
    expected_ids = ["B0", "B1", "B2", "B3", "B4", "T1", "T2", "S1", "S2", "M1", "M2", "M3", "P1"]
    checks["feature_configuration_count"] = len(feature_ids)
    checks["feature_ids_exact"] = feature_ids == expected_ids

    availability = csv_rows("dataset_feature_availability.csv")
    checks["datasets"] = sorted(row["dataset"] for row in availability)
    checks["datasets_4_of_4"] = checks["datasets"] == ["iscx_tor", "iscx_vpn", "ustc", "vnat"]
    checks["availability_unknown_test_usage_zero"] = all(int(row["unknown_test_feature_values_used"]) == 0 for row in availability)
    checks["availability_known_test_usage_zero"] = all(int(row["known_test_feature_values_used"]) == 0 for row in availability)

    coverage = csv_rows("packet_window_coverage.csv")
    checks["coverage_row_count"] = len(coverage)
    checks["coverage_windows"] = sorted({int(row["window_packets"]) for row in coverage})
    checks["coverage_roles"] = sorted({row["role_scope"] for row in coverage})
    checks["coverage_complete"] = (
        len(coverage) == 2928
        and checks["coverage_windows"] == [8, 16, 32, 64]
        and checks["coverage_roles"] == ["known_train", "known_train_validation_union", "known_validation"]
    )

    aggregation = load_json("stage15f0_aggregation_audit.json")
    parity = load_json("window_cache_parity.json")
    checks["aggregation_status"] = aggregation.get("status")
    checks["aggregation_unknown_test_usage"] = aggregation.get("unknown_test_feature_values_used")
    checks["aggregation_known_test_usage"] = aggregation.get("known_test_feature_values_used")
    checks["unmatched_membership_counts"] = aggregation.get("unmatched_membership_counts")
    checks["window_cache_parity_status"] = parity.get("status")
    checks["strict_unknown_free"] = (
        aggregation.get("status") == "PASS"
        and aggregation.get("unknown_test_feature_values_used") == 0
        and aggregation.get("known_test_feature_values_used") == 0
        and all(value == 0 for value in aggregation.get("unmatched_membership_counts", {}).values())
    )
    checks["cache_parity_pass"] = parity.get("status") == "PASS"

    before_initial = load_json("frozen_asset_hashes_before_initial.json")
    before = load_json("frozen_asset_hashes_before.json")
    after = load_json("frozen_asset_hashes_after.json")
    initial_files = before_initial["files"]
    before_files = before["files"]
    after_files = after["files"]
    initial_overlap_unchanged = all(path in after_files and record == after_files[path] for path, record in initial_files.items())
    checks["initial_protected_file_count"] = len(initial_files)
    checks["final_protected_file_count"] = len(before_files)
    checks["initial_overlap_unchanged"] = initial_overlap_unchanged
    checks["final_hash_scope_unchanged"] = before_files == after_files and len(before_files) == 193

    future_present = [name for name in FORBIDDEN_FUTURE_RESULTS if (ROOT / name).exists()]
    checks["future_result_files_absent"] = not future_present
    checks["unexpected_future_result_files"] = future_present
    checkpoint_files = [str(path.relative_to(ROOT)) for pattern in ("*.pt", "*.pth", "*.bin", "*.ckpt") for path in ROOT.rglob(pattern)]
    checks["checkpoint_files_created"] = checkpoint_files
    checks["no_training_checkpoint_created"] = not checkpoint_files

    prereg = csv_rows("stage15f1_preregistered_matrix.csv")
    checks["preregistered_rows"] = len(prereg)
    checks["preregistered_rows_not_run_or_blocked"] = all(
        row["status"] in {"PREREGISTERED_NOT_RUN", "PREREGISTERED_BLOCKED"} for row in prereg
    )

    required_boolean_checks = (
        "required_outputs_present",
        "literature_19_of_19",
        "evidence_tags_present",
        "feature_ids_exact",
        "datasets_4_of_4",
        "availability_unknown_test_usage_zero",
        "availability_known_test_usage_zero",
        "coverage_complete",
        "strict_unknown_free",
        "cache_parity_pass",
        "initial_overlap_unchanged",
        "final_hash_scope_unchanged",
        "future_result_files_absent",
        "no_training_checkpoint_created",
        "preregistered_rows_not_run_or_blocked",
    )
    status = "PASS" if all(checks[name] is True for name in required_boolean_checks) else "FAIL"
    # `manifest.json` is the inventory container itself and changes when the
    # preservation tool refreshes artifact hashes; exclude it here to avoid a
    # circular hash dependency while still requiring its presence above.
    artifacts = {
        name: {"bytes": (ROOT / name).stat().st_size, "sha256": sha256_file(ROOT / name)}
        for name in REQUIRED
        if (ROOT / name).is_file() and name != "manifest.json"
    }
    payload = {
        "stage": "Stage 15F-0",
        "status": status,
        "training_runs": 0,
        "gpu_workloads": 0,
        "stage15f_1_to_5": "NOT_RUN",
        "checks": checks,
        "artifact_sha256": artifacts,
    }
    write_json(ROOT / "completion_verification.json", payload)
    print(json.dumps({"status": status, "checks": checks}, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
