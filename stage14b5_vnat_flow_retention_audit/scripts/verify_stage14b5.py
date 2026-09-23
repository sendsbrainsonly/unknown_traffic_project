#!/usr/bin/env python3
"""Independent completion checks for the Stage 14B.5 audit bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "stage14b5_vnat_flow_retention_audit"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_PROTOCOL_SHA = "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced"
EXPECTED_SPLIT_SHA = "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        ROOT / "stage14b5_flow_lineage.csv",
        ROOT / "stage14b5_filter_breakdown.csv",
        ROOT / "stage14b5_class_retention.csv",
        ROOT / "stage14b5_flow_audit.md",
        ROOT / "artifacts" / "official_connection_comparison.csv",
        ROOT / "artifacts" / "corrupt_prefix_flow_counts.json",
        ROOT / "artifacts" / "freeze_hash_before.json",
        ROOT / "artifacts" / "freeze_hash_after.json",
        ROOT / "artifacts" / "audit_summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    assert not missing, f"missing/empty outputs: {missing}"

    lineage = rows(required[0])
    filters = rows(required[1])
    classes = rows(required[2])
    official = rows(required[4])
    assert len(lineage) == 12
    assert lineage[-1]["stage"] == "stage14b_clean_pool"
    assert int(lineage[-1]["output_count"]) == 23449
    diagnostic = next(row for row in lineage if row["stage"] == "corrupt_pcap_flow_exclusion_diagnostic")
    assert int(diagnostic["input_count"]) - int(diagnostic["removed_count"]) == 23454
    assert int(diagnostic["removed_count"]) == 10555
    assert len(filters) == 14
    assert sum(int(row["removed_flow_count"]) for row in classes) == 5
    assert sum(int(row["final_flow_count"]) for row in classes) == 23449
    assert len(classes) == 10
    assert len(official) == 165
    assert sum(int(row["official_connection_count"]) for row in official) == 33711

    before = json.loads(required[6].read_text(encoding="utf-8"))
    after = json.loads(required[7].read_text(encoding="utf-8"))
    summary = json.loads(required[8].read_text(encoding="utf-8"))
    assert before["freeze_hash"] == after["freeze_hash"] == EXPECTED_FREEZE
    assert after["matches_before"] is True
    assert summary["freeze_unchanged"] is True
    assert summary["verdict"] == "PASS_WITH_EXPLAINED_DIFFERENCE"
    assert summary["stage14a_flows"] - summary["duplicate_pcap_flows_removed"] - summary["cross_application_rows_removed"] == 23449

    protocol = STAGE14B / "vnat_open_set_protocol.json"
    split = STAGE14B / "vnat_split_manifest.csv"
    assert sha256(protocol) == EXPECTED_PROTOCOL_SHA == after["vnat_open_set_protocol_sha256"]
    assert sha256(split) == EXPECTED_SPLIT_SHA == after["vnat_split_manifest_sha256"]
    assert EXPECTED_FREEZE in required[3].read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["verdict"] == "PASS_WITH_EXPLAINED_DIFFERENCE"
    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        path = PROJECT / relative if relative.parts[0] == ".tmux-task" else ROOT / relative
        assert path.is_file(), path
        assert path.stat().st_size == artifact["size_bytes"], path
        assert sha256(path) == artifact["sha256"], path
    print("STAGE14B5_VERIFICATION=PASS")
    print("FINAL_CLEAN_FLOWS=23449")
    print(f"FREEZE_HASH={EXPECTED_FREEZE}")


if __name__ == "__main__":
    main()
