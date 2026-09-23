#!/usr/bin/env python3
"""Independent consistency checks for the completed Stage 14A audit outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ELIGIBLE = {"rdp", "rsync", "scp", "sftp", "skype", "ssh", "youtube", "zoiper"}
EXPECTED_EXCLUDED = {"netflix", "vimeo"}
REQUIRED_OUTPUTS = (
    "vnat_manifest.csv",
    "vnat_class_statistics.csv",
    "eligible_classes.json",
    "excluded_classes.json",
    "stage14a_vnat_audit.md",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for name in REQUIRED_OUTPUTS:
        path = ROOT / name
        assert path.is_file() and path.stat().st_size > 0, f"missing/empty output: {path}"

    manifest = read_csv(ROOT / "vnat_manifest.csv")
    stats = read_csv(ROOT / "vnat_class_statistics.csv")
    eligible = json.loads((ROOT / "eligible_classes.json").read_text(encoding="utf-8"))
    excluded = json.loads((ROOT / "excluded_classes.json").read_text(encoding="utf-8"))
    details = json.loads((ROOT / "outputs" / "audit_details.json").read_text(encoding="utf-8"))
    evidence_manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    duplicate_files = read_csv(ROOT / "outputs" / "duplicate_files.csv")
    duplicate_flows = read_csv(ROOT / "outputs" / "duplicate_flows.csv")

    assert len(manifest) == 172
    pcaps = [row for row in manifest if row["file_type"] == "pcap"]
    assert len(pcaps) == 165
    assert sum(row["readable"] == "True" for row in pcaps) == 163
    failed = {row["file_name"] for row in pcaps if row["readable"] == "False"}
    assert failed == {"nonvpn_scp_long_capture1.pcap", "vpn_skype-chat_capture6.pcap"}
    assert all(len(row["sha256"]) == 64 for row in manifest)
    for field in ("application", "vpn_status", "capture_id", "source_group_id", "strict_base_group_id", "group_id"):
        assert field in pcaps[0], f"missing manifest field: {field}"

    assert len(stats) == 10
    assert sum(int(row["flow_count"]) for row in stats) == 23454
    assert all(row["has_vpn_and_nonvpn"] == "True" for row in stats)
    assert all(int(row["vpn_pcap_count"]) > 0 and int(row["nonvpn_pcap_count"]) > 0 for row in stats)

    eligible_names = {row["application"] for row in eligible["eligible_classes"]}
    excluded_names = {row["application"] for row in excluded["excluded_classes"]}
    assert eligible_names == EXPECTED_ELIGIBLE
    assert excluded_names == EXPECTED_EXCLUDED
    assert eligible_names.isdisjoint(excluded_names)
    assert eligible_names | excluded_names == {row["application"] for row in stats}
    stat_by_class = {row["application"]: row for row in stats}
    assert all(int(stat_by_class[name]["effective_group_count"]) >= 3 for name in eligible_names)
    assert all(int(stat_by_class[name]["effective_group_count"]) < 3 for name in excluded_names)

    assert len(duplicate_files) == 1
    captures_by_id = {row["capture_id"]: row for row in pcaps}
    for duplicate in duplicate_files:
        members = duplicate["capture_ids"].split(";")
        assert len({captures_by_id[member]["group_id"] for member in members}) == 1
    assert len(duplicate_flows) == 6
    cross_application_rows = sum(
        any(peer != row["application"] for peer in row["peer_applications"].split(";"))
        for row in duplicate_flows
    )
    assert cross_application_rows == 4

    for forbidden_flag in ("unknown_class_selected", "settings_generated", "model_training", "open_detect_run", "des_run", "unknown_detection_results_read"):
        assert details[forbidden_flag] is False, forbidden_flag
    assert details["metadata_failures"] == []
    assert details["pcap_count"] == 165 and details["successful_pcap_count"] == 163
    assert evidence_manifest["status"] == "success_with_quality_issues"
    for artifact in evidence_manifest["artifacts"]:
        if "sha256" in artifact:
            assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]

    by_base_group: dict[str, set[str]] = defaultdict(set)
    for row in pcaps:
        by_base_group[row["strict_base_group_id"]].add(row["group_id"])
    assert all(len(values) == 1 for values in by_base_group.values())

    print(json.dumps({
        "verification": "PASS",
        "dataset_files": len(manifest),
        "pcaps_readable": "163/165",
        "flows": 23454,
        "applications": 10,
        "eligible": sorted(eligible_names),
        "excluded": sorted(excluded_names),
        "exact_duplicate_pcap_groups": len(duplicate_files),
        "cross_application_duplicate_flow_rows": cross_application_rows,
        "forbidden_actions": "all false",
    }, indent=2))


if __name__ == "__main__":
    main()
