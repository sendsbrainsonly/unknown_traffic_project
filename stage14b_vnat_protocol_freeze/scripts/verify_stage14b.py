#!/usr/bin/env python3
"""Verify Stage 14B freeze integrity and Strict Unknown-Free invariants."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from freeze_vnat_protocol import APPLICATIONS, protocol_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = (
        ROOT / "vnat_final_class_audit.csv",
        ROOT / "vnat_open_set_protocol.json",
        ROOT / "vnat_split_manifest.csv",
        ROOT / "stage14b_protocol_freeze.md",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)

    class_audit = read_csv(ROOT / "vnat_final_class_audit.csv")
    protocol = json.loads((ROOT / "vnat_open_set_protocol.json").read_text(encoding="utf-8"))
    freeze_hashes = json.loads((ROOT / "outputs" / "freeze_hashes.json").read_text(encoding="utf-8"))
    exclusions = read_csv(ROOT / "outputs" / "integrity_exclusions.csv")

    assert [row["application"] for row in class_audit] == APPLICATIONS
    assert all(row["status"] == "retained" for row in class_audit)
    assert sum(int(row["cleaned_flow_count"]) for row in class_audit) == 23449
    assert sum(int(row["removed_cross_application_duplicate_flows"]) for row in class_audit) == 4
    assert sum(int(row["removed_duplicate_pcap_flows"]) for row in class_audit) == 1
    assert all(row["nonempty_8_1_1"] == "True" for row in class_audit)

    assert protocol["freeze_status"] == "FROZEN_READY_FOR_STAGE_14C"
    assert protocol_hash(protocol) == protocol["freeze_hash"]
    assert protocol["retained_applications"] == APPLICATIONS
    assert protocol["excluded_applications"] == []
    assert protocol["clean_flow_count"] == 23449
    assert len(protocol["protocols"]) == 15
    assert protocol["verification"]["status"] == "PASS"
    assert protocol["split_manifest"]["rows"] == 351735
    assert sha256_file(ROOT / "vnat_split_manifest.csv") == protocol["split_manifest"]["sha256"]
    for relative_path, digest in protocol["source_hashes"].items():
        assert sha256_file(PROJECT_ROOT / relative_path) == digest, relative_path

    by_seed = defaultdict(dict)
    coverage = defaultdict(Counter)
    protocol_by_id = {}
    for record in protocol["protocols"]:
        protocol_by_id[record["protocol_id"]] = record
        unknown = set(record["unknown_applications"])
        known = set(record["known_applications"])
        assert unknown.isdisjoint(known)
        assert unknown | known == set(APPLICATIONS)
        expected_unknown = {"Low": 2, "Medium": 3, "High": 4}[record["setting"]]
        assert len(unknown) == expected_unknown
        by_seed[record["seed"]][record["setting"]] = unknown
        coverage[record["setting"]].update(unknown)
    for values in by_seed.values():
        assert values["Low"] < values["Medium"] < values["High"]
    assert set(coverage["Low"].values()) == {1}
    assert set(coverage["High"].values()) == {2}

    excluded_captures = {
        row["capture_id"]
        for row in exclusions
        if row["exclusion_type"] in {"failed_pcap", "exact_duplicate_pcap"}
    }
    excluded_flow_keys = {
        (row["capture_id"], row["flow_id"])
        for row in exclusions
        if row["exclusion_type"] == "cross_application_duplicate_flow"
    }
    assert len(excluded_flow_keys) == 4

    row_counts = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vpn_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    flow_ids: dict[str, set[str]] = defaultdict(set)
    known_split_consistency: dict[tuple[int, str, str], str] = {}
    known_class_split_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    with (ROOT / "vnat_split_manifest.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protocol_id = row["protocol_id"]
            record = protocol_by_id[protocol_id]
            unknown = set(record["unknown_applications"])
            row_counts[protocol_id] += 1
            split_counts[protocol_id][row["split"]] += 1
            vpn_counts[protocol_id][(row["split"], row["vpn_status"])] += 1
            assert row["flow_uid"] not in flow_ids[protocol_id]
            flow_ids[protocol_id].add(row["flow_uid"])
            assert row["capture_id"] not in excluded_captures
            assert (row["capture_id"], row["source_flow_id"]) not in excluded_flow_keys
            if row["application"] in unknown:
                assert row["class_role"] == "unknown" and row["split"] == "unknown_test"
            else:
                assert row["class_role"] == "known" and row["split"] in {"train", "validation", "test"}
                key = (int(row["protocol_seed"]), row["application"], row["flow_uid"])
                previous = known_split_consistency.setdefault(key, row["split"])
                assert previous == row["split"]
                known_class_split_counts[protocol_id][(row["application"], row["split"])] += 1

    assert sum(row_counts.values()) == 351735
    for protocol_id, record in protocol_by_id.items():
        assert row_counts[protocol_id] == 23449
        assert len(flow_ids[protocol_id]) == 23449
        for split in ("train", "validation", "test", "unknown_test"):
            expected = record["split_statistics"][split]
            assert split_counts[protocol_id][split] == expected["flows"]
            assert vpn_counts[protocol_id][(split, "vpn")] == expected["vpn_flows"]
            assert vpn_counts[protocol_id][(split, "nonvpn")] == expected["nonvpn_flows"]
        for application in record["known_applications"]:
            for split in ("train", "validation", "test"):
                assert known_class_split_counts[protocol_id][(application, split)] > 0

    assert freeze_hashes["freeze_hash"] == protocol["freeze_hash"]
    assert freeze_hashes["vnat_open_set_protocol_sha256"] == sha256_file(ROOT / "vnat_open_set_protocol.json")
    assert freeze_hashes["vnat_split_manifest_sha256"] == sha256_file(ROOT / "vnat_split_manifest.csv")
    for forbidden, value in protocol["forbidden_actions"].items():
        assert value is False, forbidden

    print(json.dumps({
        "verification": "PASS",
        "freeze_hash": protocol["freeze_hash"],
        "classes_retained": 10,
        "classes_excluded": 0,
        "clean_flows": 23449,
        "protocols": 15,
        "split_manifest_rows": 351735,
        "strict_unknown_free": "PASS",
        "capture_disjoint_known_split": False,
        "forbidden_actions": "all false",
    }, indent=2))


if __name__ == "__main__":
    main()
