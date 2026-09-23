#!/usr/bin/env python3
"""Independently verify generated CipherSpectrum label manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from cipherspectrum_labeling.scripts.build_label_manifest import (
    CAPTURE_GROUPS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_DIR,
    locate_payload_root,
    resolve_group_root,
    sha256_file,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def recompute_dataset_fingerprint(payload_root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for group in CAPTURE_GROUPS:
        group_root = resolve_group_root(payload_root, group)
        paths = sorted(
            path
            for path in group_root.glob("*/*.pcap")
            if path.is_file() and not path.name.startswith("._")
        )
        for path in paths:
            relative_path = path.relative_to(payload_root).as_posix()
            stat = path.stat()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(b"\n")
            file_count += 1
            total_bytes += stat.st_size
    return digest.hexdigest(), file_count, total_bytes


def count_manifest(path: Path) -> dict[str, object]:
    count = 0
    ids: set[str] = set()
    paths: set[str] = set()
    classes: Counter[str] = Counter()
    official = 0
    local = 0
    parse_errors = 0
    bad_magic = 0
    review = 0
    group_mismatches = 0
    official_group_mismatches = 0
    manual_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "flow_id",
            "relative_path",
            "canonical_label",
            "official40_eligible",
            "local41_candidate",
            "manual_review_required",
            "pcap_magic_valid",
            "parse_status",
            "split_group_id",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            count += 1
            ids.add(row["flow_id"])
            paths.add(row["relative_path"])
            classes[row["canonical_label"]] += 1
            official += row["official40_eligible"] == "true"
            local += row["local41_candidate"] == "true"
            review += row["manual_review_required"] == "true"
            group_mismatches += row["capture_group_status"] == "MISPLACED_GROUP"
            official_group_mismatches += (
                row["capture_group_status"] == "MISPLACED_GROUP"
                and row["official40_eligible"] == "true"
            )
            if row["manual_review_required"] == "true":
                manual_ids.add(row["flow_id"])
            parse_errors += row["parse_status"] != "OK"
            bad_magic += row["pcap_magic_valid"] != "true"
            if row["parse_status"] == "OK" and not row["split_group_id"]:
                raise RuntimeError(f"Missing split_group_id for {row['flow_id']}")
    if len(ids) != count:
        raise RuntimeError(f"Duplicate flow_id values: {count - len(ids)}")
    if len(paths) != count:
        raise RuntimeError(f"Duplicate relative_path values: {count - len(paths)}")
    return {
        "rows": count,
        "unique_flow_ids": len(ids),
        "unique_paths": len(paths),
        "class_counts": dict(sorted(classes.items())),
        "official40_eligible": official,
        "local41_candidate": local,
        "manual_review_required": review,
        "parse_errors": parse_errors,
        "bad_magic": bad_magic,
        "group_mismatches": group_mismatches,
        "official_group_mismatches": official_group_mismatches,
        "flow_ids": ids,
        "manual_ids": manual_ids,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def verify(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload_root = locate_payload_root(dataset_root)
    if output_dir == payload_root or payload_root in output_dir.parents:
        raise RuntimeError("Output directory is inside raw dataset")

    all_stats = count_manifest(output_dir / "cipherspectrum_label_manifest.csv")
    official_stats = count_manifest(
        output_dir / "cipherspectrum_official40_manifest.csv"
    )
    review_stats = count_manifest(output_dir / "manual_review_inventory.csv")
    all_ids = all_stats.pop("flow_ids")
    official_ids = official_stats.pop("flow_ids")
    review_ids = review_stats.pop("flow_ids")
    all_manual_ids = all_stats.pop("manual_ids")
    official_stats.pop("manual_ids")
    review_stats.pop("manual_ids")

    checks = {
        "all_count_matches_metadata": all_stats["rows"] == metadata["total_files"],
        "official_count_matches_metadata": (
            official_stats["rows"] == metadata["official40_flows"]
        ),
        "review_count_matches_metadata": (
            review_stats["rows"] == metadata["manual_review_rows"]
        ),
        "official_is_subset": official_ids <= all_ids,
        "review_is_subset": review_ids <= all_ids,
        "review_file_matches_all_marked_rows": review_ids == all_manual_ids,
        "all_rows_partitioned": len(official_ids | review_ids) == len(all_ids),
        "official_manifest_only_contains_eligible_rows": (
            official_stats["official40_eligible"] == official_stats["rows"]
        ),
        "review_manifest_only_contains_marked_rows": (
            review_stats["manual_review_required"] == review_stats["rows"]
        ),
        "official_review_overlap_is_packaging_mismatch": (
            len(official_ids & review_ids)
            == metadata["official40_capture_group_mismatches"]
            == all_stats["official_group_mismatches"]
        ),
        "all_parse_clean": all_stats["parse_errors"] == 0,
        "all_pcap_magic_valid": all_stats["bad_magic"] == 0,
        "local_labels_balanced": len(set(all_stats["class_counts"].values())) == 1,
        "official_labels_balanced": (
            len(set(official_stats["class_counts"].values())) == 1
        ),
        "local_class_count_41": len(all_stats["class_counts"]) == 41,
        "official_class_count_40": len(official_stats["class_counts"]) == 40,
    }

    fingerprint, raw_count, raw_bytes = recompute_dataset_fingerprint(payload_root)
    checks.update(
        {
            "raw_fingerprint_unchanged": (
                fingerprint == metadata["dataset_fingerprint_sha256"]
            ),
            "raw_count_unchanged": raw_count == metadata["total_files"],
            "raw_size_unchanged": raw_bytes == metadata["total_bytes"],
        }
    )
    hash_checks = {}
    for name, expected in metadata["output_sha256"].items():
        hash_checks[name] = sha256_file(output_dir / name) == expected
    checks["all_output_hashes_match"] = all(hash_checks.values())

    passed = all(checks.values())
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "output_hash_checks": hash_checks,
        "all_manifest": all_stats,
        "official40_manifest": official_stats,
        "manual_review_inventory": review_stats,
        "raw_dataset": {
            "payload_root": str(payload_root),
            "file_count": raw_count,
            "total_bytes": raw_bytes,
            "fingerprint_sha256": fingerprint,
        },
    }
    write_json_atomic(output_dir / "verification_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify(args.dataset_root, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
