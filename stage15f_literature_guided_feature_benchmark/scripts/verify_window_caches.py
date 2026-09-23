#!/usr/bin/env python3
"""Verify Stage 15F window caches against frozen Stage 15R aggregate lineage."""

from __future__ import annotations

import csv
from pathlib import Path

from common import PROJECT_ROOT, ROOT, iter_csv, read_json, sha256_file, write_json


def compare(dataset: str) -> dict[str, object]:
    new_path = ROOT / "window_cache" / dataset / "flow_window_metrics.csv.gz"
    old_path = PROJECT_ROOT / "stage15r_representation_bottleneck_audit" / "feature_cache" / dataset / "flow_statistics.csv"
    new_iter = iter_csv(new_path)
    with old_path.open(encoding="utf-8", newline="") as handle:
        old_iter = iter(csv.DictReader(handle))
        old = next(old_iter, None)
        checked = 0
        packet_mismatch = 0
        byte_mismatch = 0
        duration_mismatch = 0
        max_duration_abs_diff = 0.0
        previous_uid = ""
        for new in new_iter:
            uid = new["flow_uid"]
            if uid <= previous_uid:
                raise RuntimeError(f"{dataset}: new cache is not strictly sorted")
            previous_uid = uid
            while old is not None and old["flow_uid"] < uid:
                old = next(old_iter, None)
            if old is None or old["flow_uid"] != uid:
                raise RuntimeError(f"{dataset}: {uid} absent from frozen Stage15R lineage cache")
            checked += 1
            packet_mismatch += int(int(new["packet_count"]) != int(old["packet_count"]))
            byte_mismatch += int(int(new["total_frame_bytes"]) != int(old["total_bytes"]))
            diff = abs(float(new["duration_seconds"]) - float(old["duration_seconds"]))
            max_duration_abs_diff = max(max_duration_abs_diff, diff)
            duration_mismatch += int(diff > 1e-6)
    audit = read_json(ROOT / "window_cache" / dataset / "cache_audit.json")
    assert isinstance(audit, dict)
    passed = (
        checked == int(audit["recovered_flows"])
        and packet_mismatch == 0
        and byte_mismatch == 0
        and duration_mismatch == 0
    )
    return {
        "dataset": dataset,
        "status": "PASS" if passed else "FAIL",
        "checked_known_train_validation_flows": checked,
        "packet_count_mismatches": packet_mismatch,
        "total_frame_byte_mismatches": byte_mismatch,
        "duration_mismatches_tolerance_1e-6": duration_mismatch,
        "max_duration_absolute_difference_seconds": max_duration_abs_diff,
        "comparison_scope": "integrity-only comparison for Stage15F target UIDs; no legacy Test metric is aggregated or reported",
        "new_cache_sha256": sha256_file(new_path),
        "frozen_stage15r_cache_sha256": sha256_file(old_path),
    }


def main() -> None:
    rows = [compare(dataset) for dataset in ("ustc", "iscx_vpn", "iscx_tor")]
    if any(row["status"] != "PASS" for row in rows):
        write_json(ROOT / "window_cache_parity.json", {"status": "FAIL", "datasets": rows})
        raise RuntimeError(rows)
    vnat = read_json(ROOT / "window_cache" / "vnat" / "cache_audit.json")
    assert isinstance(vnat, dict)
    if vnat["status"] != "PASS" or int(vnat.get("packet_count_mismatches", -1)) != 0:
        raise RuntimeError("VNAT cache did not pass its exact frozen packet-count check")
    write_json(
        ROOT / "window_cache_parity.json",
        {
            "status": "PASS",
            "datasets": rows,
            "vnat": {
                "status": "PASS",
                "checked_known_train_validation_flows": vnat["recovered_flows"],
                "packet_count_mismatches": vnat["packet_count_mismatches"],
                "comparison": "exact Stage14B packet_count for every Stage15F target UID",
            },
            "known_test_or_unknown_test_values_used_for_scientific_selection": False,
        },
    )
    print("window cache parity PASS")


if __name__ == "__main__":
    main()
