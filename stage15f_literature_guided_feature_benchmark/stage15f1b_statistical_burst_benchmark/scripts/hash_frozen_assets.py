#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    PROJECT, ROOT, STAGE12, STAGE14C_INPUT, STAGE14D, STAGE15F0,
    STAGE15F1A, STAGE15R, STAGE3, read_json, sha256_file, write_json,
)


def add_tree(paths: set[Path], root: Path) -> None:
    if root.is_file():
        paths.add(root.resolve())
    elif root.is_dir():
        paths.update(path.resolve() for path in root.rglob("*") if path.is_file())


def protected_paths() -> list[Path]:
    paths: set[Path] = set()
    inherited = STAGE15F0 / "frozen_asset_hashes_after.json"
    if inherited.exists():
        payload = read_json(inherited)
        entries = payload.get("files", payload.get("hashes", {}))
        raw_paths = entries.keys() if isinstance(entries, dict) else [row["path"] for row in entries]
        for raw_path in raw_paths:
            path = Path(raw_path)
            if path.is_file():
                paths.add(path.resolve())
    for path in (
        STAGE15R / "config.json",
        STAGE15R / "RESULTS.md",
        STAGE15R / "pilot_results.csv",
        STAGE15F0 / "feature_group_definitions.json",
        STAGE15F0 / "packet_window_coverage.csv",
        STAGE15F0 / "window_cache_parity.json",
        STAGE15F1A / "config.json",
        STAGE15F1A / "window_pilot_results.csv",
        STAGE15F1A / "window_per_class_results.csv",
        STAGE15F1A / "window_confusion_analysis.csv",
        STAGE15F1A / "window_error_complementarity.csv",
        STAGE15F1A / "completion_verification.json",
        STAGE12 / "configs" / "stage12_config.json",
        PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_open_set_protocol.json",
        PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv",
        STAGE14C_INPUT / "flow_image_cache" / "cache_manifest.csv",
        STAGE3 / "outputs" / "A-2" / "training_config.json",
        PROJECT / "stage3_protocol" / "split_manifest.csv",
    ):
        if path.is_file():
            paths.add(path.resolve())
    for root in (
        STAGE15R / "pilot_runs" / "e3",
        STAGE15R / "feature_cache",
        STAGE15F0 / "window_cache",
        STAGE15F1A / "runs" / "T16",
        STAGE12 / "artifacts" / "iscx_vpn" / "protocol" / "medium",
        STAGE12 / "artifacts" / "iscx_tor" / "protocol" / "medium",
        STAGE12 / "runs" / "iscx_vpn" / "medium" / "seed2022",
        STAGE12 / "runs" / "iscx_tor" / "medium" / "seed2022",
        STAGE14C_INPUT / "runs" / "medium_seed2025" / "inputs",
        STAGE14C_INPUT / "runs" / "medium_seed2026" / "inputs",
        STAGE14D / "artifacts" / "medium_seed2025",
        STAGE14D / "artifacts" / "medium_seed2026",
    ):
        add_tree(paths, root)
    for dataset in ("iscx_vpn", "iscx_tor"):
        for path in (
            STAGE12 / "protocol" / dataset / "split_manifest.csv",
            STAGE12 / "protocol" / dataset / "preprocessing_manifest.json",
        ):
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    rows = [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in protected_paths()]
    current = {row["path"]: row["sha256"] for row in rows}
    comparison = {"requested": bool(args.compare), "status": "NOT_REQUESTED", "added": [], "removed": [], "changed": []}
    if args.compare:
        previous_rows = read_json(Path(args.compare))["files"]
        previous = {row["path"]: row["sha256"] for row in previous_rows}
        comparison = {
            "requested": True,
            "status": "PASS" if previous == current else "FAIL",
            "added": sorted(current.keys() - previous.keys()),
            "removed": sorted(previous.keys() - current.keys()),
            "changed": sorted(path for path in current.keys() & previous.keys() if current[path] != previous[path]),
        }
    write_json(Path(args.output), {"file_count": len(rows), "files": rows, "comparison": comparison})
    print(json.dumps({"file_count": len(rows), "comparison": comparison}, sort_keys=True))
    if args.compare and comparison["status"] != "PASS":
        raise SystemExit("protected asset hash comparison failed")


if __name__ == "__main__":
    main()
