#!/usr/bin/env python3
"""Snapshot and verify Stage11B inputs without modifying frozen assets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from stage13_common import (
    CONFIG_PATH,
    STAGE11B_ROOT,
    SUMMARY,
    read_json,
    sha256_file,
    write_json,
)


OUTPUT = SUMMARY / "provenance_verification.json"


def artifact_hash(manifest: dict[str, object], name: str) -> str:
    matches = [item for item in manifest.get("artifacts", []) if item.get("path") == name]
    if len(matches) != 1 or not matches[0].get("sha256"):
        raise AssertionError(f"Manifest does not uniquely hash {name}")
    return str(matches[0]["sha256"])


def snapshot() -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    provenance_path = STAGE11B_ROOT / "outputs" / "summary" / "provenance_verification.json"
    provenance = read_json(provenance_path)
    if provenance.get("status") != "PASS" or provenance.get("phase") != "after":
        raise AssertionError("Stage11B final provenance is not PASS/after")
    if provenance.get("frozen_experiment_modified"):
        raise AssertionError("Stage11B reports frozen source modification")

    runs: dict[str, object] = {}
    checkpoint_hashes: set[str] = set()
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            key = f"{scenario}/fold{fold}_seed{seed}"
            run_dir = STAGE11B_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            required = ["SUCCESS", "manifest.json", "config.json", "results.json", "sample_outputs.npz"]
            if not all((run_dir / name).is_file() for name in required):
                raise FileNotFoundError(f"Incomplete Stage11B source run: {run_dir}")
            manifest = read_json(run_dir / "manifest.json")
            result = read_json(run_dir / "results.json")
            if manifest.get("status") != "success":
                raise AssertionError(f"Stage11B source manifest is not success: {run_dir}")
            observed: dict[str, str] = {}
            for name in ("SUCCESS", "config.json", "results.json", "sample_outputs.npz"):
                digest = sha256_file(run_dir / name)
                if digest != artifact_hash(manifest, name):
                    raise AssertionError(f"Stage11B artifact drift: {run_dir / name}")
                observed[name] = digest
            frozen = result["frozen_encoder"]
            checkpoint_path = Path(frozen["source_checkpoint"])
            checkpoint_digest = sha256_file(checkpoint_path)
            if checkpoint_digest != frozen["checkpoint_sha256_before"]:
                raise AssertionError(f"Frozen checkpoint drift: {checkpoint_path}")
            if frozen["checkpoint_sha256_before"] != frozen["checkpoint_sha256_after"]:
                raise AssertionError(f"Stage11B checkpoint parity failed: {run_dir}")
            if frozen.get("new_encoder_training"):
                raise AssertionError(f"Stage11B unexpectedly trained encoder: {run_dir}")
            checkpoint_hashes.add(checkpoint_digest)
            runs[key] = {
                "stage11b_run_dir": str(run_dir.resolve()),
                "stage11b_manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "artifacts": observed,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_digest,
                "split_array_sha256": result["split_array_sha256"],
                "known_classes": result["classes"]["known"],
                "unknown_classes": result["classes"]["unknown"],
            }
    if len(runs) != 15 or len(checkpoint_hashes) != 15:
        raise AssertionError("Expected 15 distinct frozen Stage11B B0 runs")
    return {
        "stage11b_root_manifest_sha256": sha256_file(STAGE11B_ROOT / "manifest.json"),
        "stage11b_provenance_sha256": sha256_file(provenance_path),
        "stage11b_provenance_status": provenance["status"],
        "runs": runs,
        "unique_checkpoint_hashes": len(checkpoint_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    current = snapshot()
    previous = read_json(OUTPUT) if OUTPUT.is_file() else None
    failures: list[str] = []
    if args.phase == "after":
        if previous is None or previous.get("phase") != "before":
            failures.append("Missing before-phase provenance snapshot")
        elif previous.get("frozen_hashes_before") != current:
            failures.append("Frozen Stage11B/checkpoint snapshot changed during Stage13A-1")
    status = "PASS" if not failures else "FAIL"
    report = {
        "status": status,
        "phase": args.phase,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_runs_verified": len(current["runs"]),
        "unique_checkpoint_hashes": current["unique_checkpoint_hashes"],
        "stage11b_provenance_status": current["stage11b_provenance_status"],
        "frozen_hashes_before": current if args.phase == "before" or previous is None else previous["frozen_hashes_before"],
        "frozen_hashes_after": current if args.phase == "after" else None,
        "new_encoder_training": False,
        "external_test_datasets_read": [],
        "cipherspectrum_test_read": False,
        "frozen_experiment_modified": bool(failures),
        "failures": failures,
    }
    write_json(OUTPUT, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

