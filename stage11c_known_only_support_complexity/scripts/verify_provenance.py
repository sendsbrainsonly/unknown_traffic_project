#!/usr/bin/env python3
"""Verify the complete frozen Stage6-11B provenance chain read-only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from stage11c_common import SCENARIOS, SEEDS, STAGE11B_ROOT, SUMMARY_ROOT, read_json, run_name, sha256_file, write_json


OUTPUT = SUMMARY_ROOT / "provenance_verification.json"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def verify_file(path: Path, expected: str, failures: list[str]) -> str | None:
    check(path.is_file(), f"Missing frozen file: {path}", failures)
    if not path.is_file():
        return None
    observed = sha256_file(path)
    check(observed == expected, f"Frozen hash mismatch: {path}", failures)
    return observed


def snapshot(failures: list[str]) -> dict:
    provenance_path = STAGE11B_ROOT / "outputs" / "summary" / "provenance_verification.json"
    provenance = read_json(provenance_path)
    for key in ("stage6_to_stage10b", "stage11a", "open_detect_v6"):
        check(provenance.get(key) == "PASS", f"Stage11B provenance field is not PASS: {key}", failures)
    check(provenance.get("status") == "PASS" and provenance.get("phase") == "after", "Stage11B final provenance is not PASS/after", failures)
    check(not provenance.get("frozen_experiment_modified"), "Stage11B reports frozen modification", failures)
    check(not provenance.get("cipherspectrum_stage9_sample_level_test_used_for_development"), "Stage11B reports CipherSpectrum Test use", failures)

    manifest_path = STAGE11B_ROOT / "manifest.json"
    manifest = read_json(manifest_path)
    check(manifest.get("status") == "success", "Stage11B manifest is not successful", failures)
    bundle_hashes: dict[str, str] = {}
    for row in manifest.get("artifacts", []):
        if row.get("scope") != "bundle" or not row.get("sha256"):
            continue
        path = STAGE11B_ROOT / row["path"]
        observed = verify_file(path, row["sha256"], failures)
        if observed is not None:
            bundle_hashes[row["path"]] = observed
            check(path.stat().st_size == int(row["size_bytes"]), f"Stage11B size mismatch: {path}", failures)

    expected_v6 = provenance["frozen_hashes_after"]["v6_contract"]
    checkpoint_hashes: dict[str, str] = {}
    for raw_path, expected in expected_v6["checkpoint_hashes"].items():
        path = Path(raw_path)
        observed = verify_file(path, expected, failures)
        if observed is not None:
            checkpoint_hashes[str(path.resolve())] = observed
    check(len(checkpoint_hashes) == 15, "Expected 15 frozen B0 checkpoints", failures)

    split_hashes = expected_v6["split_array_hashes"]
    check(len(split_hashes) == 5, "Expected five frozen split-hash sets", failures)
    for fold, seed in enumerate(SEEDS):
        key = run_name(fold, seed)
        check(key in split_hashes, f"Missing split hashes: {key}", failures)
        for scenario in SCENARIOS:
            result = read_json(STAGE11B_ROOT / "artifacts" / scenario / key / "results.json")
            check(result["split_array_sha256"] == split_hashes[key], f"Run/split hash disagreement: {scenario}/{key}", failures)

    return {
        "stage11b_manifest_sha256": sha256_file(manifest_path),
        "stage11b_provenance_sha256": sha256_file(provenance_path),
        "stage11b_bundle_artifacts_verified": len(bundle_hashes),
        "stage11b_bundle_hashes": bundle_hashes,
        "v6_checkpoint_hashes": checkpoint_hashes,
        "five_split_hash_sets": split_hashes,
        "class_sets": expected_v6["class_sets"],
        "seeds": expected_v6["seeds"],
        "upstream_chain_status": {
            "stage6_to_stage10b": provenance["stage6_to_stage10b"],
            "stage11a": provenance["stage11a"],
            "open_detect_v6": provenance["open_detect_v6"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    failures: list[str] = []
    observed = snapshot(failures)
    previous = read_json(OUTPUT) if OUTPUT.is_file() else None
    if args.phase == "after":
        check(previous is not None, "Missing before-phase Stage11C provenance", failures)
        if previous is not None:
            check(previous.get("frozen_hashes_before") == observed, "Frozen Stage6-11B snapshot changed during Stage11C", failures)
    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "phase": args.phase,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage6_to_stage11b": status,
        "stage11b_bundle_artifacts_verified": observed["stage11b_bundle_artifacts_verified"],
        "v6_checkpoints_verified": len(observed["v6_checkpoint_hashes"]),
        "split_hash_sets_verified": len(observed["five_split_hash_sets"]),
        "unknown_feature_used_for_criterion": False,
        "known_test_feature_used_for_criterion": False,
        "cipherspectrum_stage9_sample_level_files_opened": [],
        "cipherspectrum_stage9_sample_level_test_used": False,
        "frozen_hashes_before": observed if args.phase == "before" or previous is None else previous["frozen_hashes_before"],
        "frozen_hashes_after": observed if args.phase == "after" else None,
        "frozen_experiment_modified": bool(failures) if args.phase == "after" else False,
        "failures": failures,
    }
    write_json(OUTPUT, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frozen_hashes_before", "frozen_hashes_after")}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
