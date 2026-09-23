#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, STAGE15F0, STAGE15R, read_json, sha256_file, write_json


def protected_paths() -> list[Path]:
    paths: set[Path] = set()
    inherited = STAGE15F0 / "frozen_asset_hashes_after.json"
    if inherited.exists():
        data = read_json(inherited)
        entries = data.get("files", data.get("hashes", data))
        if isinstance(entries, dict):
            candidates = entries.keys()
        else:
            candidates = [row.get("path") for row in entries if isinstance(row, dict)]
        for raw in candidates:
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT.parent / path
            if path.is_file():
                paths.add(path.resolve())
    for rel in (
        "config.json", "RESULTS.md", "stage15r_report.md",
        "pilot_results.csv", "per_class_results.csv", "frozen_asset_hashes_after.json",
    ):
        path = STAGE15R / rel
        if path.is_file():
            paths.add(path.resolve())
    for path in (STAGE15R / "pilot_runs" / "e2").glob("**/*"):
        if path.is_file():
            paths.add(path.resolve())
    for rel in (
        "feature_registry.csv", "packet_window_coverage.csv", "stage15f0_report.md",
        "RESULTS.md", "manifest.json", "frozen_asset_hashes_after.json",
    ):
        path = STAGE15F0 / rel
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
        prior_rows = read_json(Path(args.compare))["files"]
        prior = {row["path"]: row["sha256"] for row in prior_rows}
        comparison = {
            "requested": True,
            "status": "PASS" if prior == current else "FAIL",
            "added": sorted(current.keys() - prior.keys()),
            "removed": sorted(prior.keys() - current.keys()),
            "changed": sorted(path for path in current.keys() & prior.keys() if current[path] != prior[path]),
        }
    payload = {"file_count": len(rows), "files": rows, "comparison": comparison}
    write_json(Path(args.output), payload)
    print(json.dumps({"file_count": len(rows), "comparison": comparison}, sort_keys=True))
    if args.compare and comparison["status"] != "PASS":
        raise SystemExit("protected asset hash comparison failed")


if __name__ == "__main__":
    main()
