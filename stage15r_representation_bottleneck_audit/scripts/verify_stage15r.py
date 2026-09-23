#!/usr/bin/env python3
"""Verify the completed Stage 15R bundle without opening Test feature values."""

from __future__ import annotations

import json
from pathlib import Path

from common import ROOT, read_json, sha256_file, write_json


REQUIRED_ALWAYS = (
    "dataset_protocol_audit.csv",
    "feature_pipeline_audit.md",
    "packet_flow_statistics.csv",
    "input_shape_and_lineage_audit.csv",
    "pilot_closed_set_results.csv",
    "paired_vs_native.csv",
    "per_class_results.csv",
    "confusion_analysis.csv",
    "encryption_regime_audit.csv",
    "stage15r_representation_report.md",
    "RESULTS.md",
    "manifest.json",
    "pilot_gate.json",
    "frozen_asset_hashes_before.json",
    "frozen_asset_hashes_after.json",
)


def check(name: str, passed: bool, evidence: object) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main() -> None:
    checks: list[dict[str, object]] = []
    missing = [name for name in REQUIRED_ALWAYS if not (ROOT / name).is_file()]
    checks.append(check("required_artifacts_exist", not missing, {"missing": missing}))

    status = read_json(ROOT / "stage15r0_status.json")
    checks.append(check("stage15r0_audit_pass", status.get("status") == "PASS", status))
    checks.append(check(
        "audit_is_test_free",
        not status.get("known_test_used_for_feature_or_model_selection", True)
        and not status.get("unknown_test_used_for_feature_or_model_selection", True),
        {
            "known_test_used": status.get("known_test_used_for_feature_or_model_selection"),
            "unknown_test_used": status.get("unknown_test_used_for_feature_or_model_selection"),
        },
    ))

    results = []
    for path in sorted((ROOT / "pilot_runs").glob("e[0-3]/*/*/result.json")):
        row = read_json(path)
        row["_path"] = str(path.relative_to(ROOT))
        results.append(row)
    expected_pairs = {
        (experiment, item["dataset"], item["protocol_id"])
        for experiment in ("E0", "E1", "E2", "E3")
        for item in read_json(ROOT / "config.json")["pilot_protocols"]
    }
    observed_pairs = {(row["experiment"], row["dataset"], row["protocol_id"]) for row in results}
    checks.append(check(
        "all_20_preregistered_pilots_exist",
        observed_pairs == expected_pairs and len(results) == 20,
        {"expected": 20, "observed": len(results), "missing": sorted(expected_pairs - observed_pairs), "extra": sorted(observed_pairs - expected_pairs)},
    ))
    nonzero_test = [
        {"path": row["_path"], "known_test": row.get("known_test_samples_used"), "unknown_test": row.get("unknown_test_samples_used")}
        for row in results
        if int(row.get("known_test_samples_used", -1)) != 0 or int(row.get("unknown_test_samples_used", -1)) != 0
    ]
    checks.append(check("all_pilots_strict_unknown_free", not nonzero_test, {"violations": nonzero_test}))
    failed = [row["_path"] for row in results if not str(row.get("status", "")).startswith("PASS")]
    checks.append(check("all_pilot_results_pass", not failed, {"failed": failed}))

    before = read_json(ROOT / "frozen_asset_hashes_before.json")
    after = read_json(ROOT / "frozen_asset_hashes_after.json")
    same = before.get("files") == after.get("files")
    checks.append(check(
        "protected_frozen_assets_unchanged",
        same,
        {"before_file_count": before.get("file_count"), "after_file_count": after.get("file_count"), "identical": same},
    ))

    gate = read_json(ROOT / "pilot_gate.json")
    clear = list(gate.get("full15_candidates", []))
    full15_exists = (ROOT / "full15_closed_set_results.csv").exists()
    correct_full15_state = full15_exists if clear else not full15_exists
    checks.append(check(
        "full15_output_matches_preregistered_gate",
        correct_full15_state,
        {"clear_candidates": clear, "gate_status": gate.get("full15_status"), "full15_file_exists": full15_exists},
    ))

    checkpoint_issues = []
    for row in results:
        if row["experiment"] == "E0":
            continue
        path = Path(row.get("checkpoint_path", ""))
        if not path.is_file() or sha256_file(path) != row.get("checkpoint_sha256"):
            checkpoint_issues.append(row["_path"])
    checks.append(check("new_checkpoint_hashes_verify", not checkpoint_issues, {"violations": checkpoint_issues}))

    passed = all(item["passed"] for item in checks)
    payload = {
        "stage": "Stage 15R — Representation Bottleneck Audit",
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "known_test_samples_used_for_selection": 0,
        "unknown_test_samples_used": 0,
        "des_or_hybrid_modified": False,
    }
    write_json(ROOT / "completion_verification.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
