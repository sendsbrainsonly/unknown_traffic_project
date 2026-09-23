#!/usr/bin/env python3
"""Fail-closed completion checks for Stage 14C-3."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    required = [
        "training_config_diff.md",
        "data_alignment_audit.csv",
        "feature_distribution_audit.csv",
        "training_dynamics.csv",
        "per_class_comparison.csv",
        "component_ablation.csv",
        "input_gradient_audit.csv",
        "stage14c3_failure_diagnosis.md",
        "RESULTS.md",
        "manifest.json",
    ]
    for name in required:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing_or_empty:{name}")

    tables = {
        "data_alignment_audit.csv": 30,
        "feature_distribution_audit.csv": 90,
        "training_dynamics.csv": 2026,
        "per_class_comparison.csv": 315,
        "component_ablation.csv": 20,
        "input_gradient_audit.csv": 6,
    }
    rows_by_table: dict[str, int] = {}
    for name, expected in tables.items():
        rows = read_csv(name)
        rows_by_table[name] = len(rows)
        if len(rows) != expected:
            failures.append(f"row_count:{name}:{len(rows)}!={expected}")

    alignment = read_csv("data_alignment_audit.csv")
    if any(row["status"] != "PASS" for row in alignment):
        failures.append("data_alignment_not_all_pass")

    ablation = read_csv("component_ablation.csv")
    formal = [row for row in ablation if row["source"] == "new_known_only_ablation"]
    if len(formal) != 10 or any(row["status"] != "success" for row in formal):
        failures.append("formal_ablation_runs_not_10_of_10_success")
    if any(int(row["known_test_samples_used"]) != 0 for row in ablation):
        failures.append("known_test_used_in_ablation")
    if any(int(row["unknown_test_samples_used"]) != 0 for row in ablation):
        failures.append("unknown_test_used_in_ablation")

    gradients = read_csv("input_gradient_audit.csv")
    if any(int(row["known_test_samples_read"]) != 0 for row in gradients):
        failures.append("known_test_used_in_gradient_audit")
    if any(int(row["unknown_samples_read"]) != 0 for row in gradients):
        failures.append("unknown_used_in_gradient_audit")

    static = json.loads((ROOT / "static_audit_summary.json").read_text(encoding="utf-8"))
    if static["status"] != "PASS":
        failures.append("static_audit_not_pass")
    for key in ("known_test_samples_read", "unknown_test_samples_read"):
        if static[key] != 0:
            failures.append(f"static_{key}_nonzero")
    if static["des_executed"] is not False:
        failures.append("des_executed")

    protocol = json.loads((STAGE14B / "vnat_open_set_protocol.json").read_text(encoding="utf-8"))
    recorded = json.loads((STAGE14B / "outputs" / "freeze_hashes.json").read_text(encoding="utf-8"))
    if protocol["freeze_hash"] != EXPECTED_FREEZE or recorded["freeze_hash"] != EXPECTED_FREEZE:
        failures.append("freeze_hash_changed")
    if sha256_file(STAGE14B / "vnat_open_set_protocol.json") != recorded["vnat_open_set_protocol_sha256"]:
        failures.append("protocol_file_hash_changed")
    if sha256_file(STAGE14B / "vnat_split_manifest.csv") != recorded["vnat_split_manifest_sha256"]:
        failures.append("split_file_hash_changed")

    for variant in ("D1", "D4", "D5", "D6", "D7"):
        for protocol_id in ("medium_seed2025", "medium_seed2026"):
            run_dir = ROOT / "ablation_runs" / variant / protocol_id
            if not (run_dir / "COMPLETED").is_file() or not (run_dir / "result.json").is_file():
                failures.append(f"incomplete_run:{variant}:{protocol_id}")

    report = (ROOT / "stage14c3_failure_diagnosis.md").read_text(encoding="utf-8")
    if "ROOT_CAUSE_IDENTIFIED" not in report:
        failures.append("missing_final_diagnosis")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "success":
        failures.append("manifest_not_success")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "required_files": len(required),
        "table_rows": rows_by_table,
        "formal_ablation_runs": len(formal),
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
        "freeze_hash": protocol["freeze_hash"],
        "final_diagnosis": "ROOT_CAUSE_IDENTIFIED",
    }
    (ROOT / "completion_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
