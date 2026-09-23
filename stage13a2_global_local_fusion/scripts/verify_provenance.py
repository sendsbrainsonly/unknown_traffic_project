#!/usr/bin/env python3
"""Snapshot and verify all frozen Stage 13A-1 inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from stage13a2_common import (
    CONFIG_PATH,
    STAGE13A1_ROOT,
    SUMMARY,
    read_json,
    sha256_file,
    write_json,
)


OUTPUT = SUMMARY / "provenance_verification.json"
REQUIRED_SCORE_KEYS = {
    "validation_centroid",
    "known_test_centroid",
    "unknown_test_centroid",
    "validation_knn10",
    "known_test_knn10",
    "unknown_test_knn10",
    "known_test_labels_reindexed",
    "unknown_test_labels_original",
    "known_predictions_reindexed",
}


def artifact_hash(manifest: dict[str, object], name: str) -> str:
    matches = [item for item in manifest.get("artifacts", []) if item.get("path") == name]
    if len(matches) != 1 or not matches[0].get("sha256"):
        raise AssertionError(f"Manifest does not uniquely hash {name}")
    return str(matches[0]["sha256"])


def snapshot() -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    source_root = Path(config["stage13a1_root"])
    if source_root.resolve() != STAGE13A1_ROOT.resolve():
        raise AssertionError("Configured Stage13A-1 root is not the expected frozen source")
    root_manifest = read_json(source_root / "manifest.json")
    if root_manifest.get("status") != "success":
        raise AssertionError("Stage13A-1 root manifest is not successful")
    source_provenance_path = source_root / "outputs" / "summary" / "provenance_verification.json"
    source_provenance = read_json(source_provenance_path)
    if source_provenance.get("status") != "PASS" or source_provenance.get("phase") != "after":
        raise AssertionError("Stage13A-1 final provenance is not PASS/after")
    if source_provenance.get("frozen_experiment_modified"):
        raise AssertionError("Stage13A-1 reports frozen source modification")

    runs: dict[str, object] = {}
    checkpoint_hashes: set[str] = set()
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_name = f"fold{fold}_seed{seed}"
            key = f"{scenario}/{run_name}"
            run_dir = source_root / "artifacts" / scenario / run_name
            required = [
                "SUCCESS",
                "manifest.json",
                "config.json",
                "input_hashes.json",
                "results.json",
                "score_arrays.npz",
                "thresholds.json",
            ]
            if not all((run_dir / name).is_file() for name in required):
                raise FileNotFoundError(f"Incomplete Stage13A-1 source run: {run_dir}")
            manifest = read_json(run_dir / "manifest.json")
            result = read_json(run_dir / "results.json")
            frozen = read_json(run_dir / "input_hashes.json")
            if manifest.get("status") != "success" or result.get("status") != "SUCCESS":
                raise AssertionError(f"Stage13A-1 source is not successful: {run_dir}")
            if result.get("parity", {}).get("status") != "PASS":
                raise AssertionError(f"Stage13A-1 parity failed: {run_dir}")
            operations = result.get("operations", {})
            if operations.get("new_encoder_training") or operations.get("external_test_datasets_read"):
                raise AssertionError(f"Forbidden Stage13A-1 operation recorded: {run_dir}")
            if operations.get("frozen_experiment_modified"):
                raise AssertionError(f"Stage13A-1 frozen source modification recorded: {run_dir}")

            observed: dict[str, str] = {}
            for name in ("SUCCESS", "config.json", "input_hashes.json", "results.json", "score_arrays.npz", "thresholds.json"):
                digest = sha256_file(run_dir / name)
                if digest != artifact_hash(manifest, name):
                    raise AssertionError(f"Stage13A-1 artifact drift: {run_dir / name}")
                observed[name] = digest
            if result["frozen_inputs"] != frozen:
                raise AssertionError(f"Stage13A-1 input hash record mismatch: {run_dir}")
            checkpoint_path = Path(frozen["checkpoint_path"])
            checkpoint_digest = sha256_file(checkpoint_path)
            if checkpoint_digest != frozen["checkpoint_sha256"]:
                raise AssertionError(f"Frozen checkpoint drift: {checkpoint_path}")
            stage11b_run = Path(frozen["stage11b_run"])
            sample_outputs = stage11b_run / "sample_outputs.npz"
            if sha256_file(sample_outputs) != frozen["sample_outputs_sha256"]:
                raise AssertionError(f"Frozen Stage11B representation drift: {sample_outputs}")
            with np.load(run_dir / "score_arrays.npz", allow_pickle=False) as arrays:
                if not REQUIRED_SCORE_KEYS.issubset(arrays.files):
                    raise AssertionError(f"Required frozen score missing: {run_dir}")
                score_shapes = {name: list(arrays[name].shape) for name in sorted(REQUIRED_SCORE_KEYS)}
            checkpoint_hashes.add(checkpoint_digest)
            runs[key] = {
                "stage13a1_run_dir": str(run_dir.resolve()),
                "stage13a1_manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "artifacts": observed,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_digest,
                "stage11b_run_dir": str(stage11b_run.resolve()),
                "stage11b_sample_outputs_sha256": frozen["sample_outputs_sha256"],
                "split_array_sha256": frozen["split_array_sha256"],
                "known_classes": frozen["known_classes"],
                "unknown_classes": frozen["unknown_classes"],
                "score_shapes": score_shapes,
            }
    if len(runs) != 15 or len(checkpoint_hashes) != 15:
        raise AssertionError("Expected 15 distinct frozen Stage13A-1 runs/checkpoints")
    return {
        "stage13a1_root_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "stage13a1_provenance_sha256": sha256_file(source_provenance_path),
        "stage13a1_provenance_status": source_provenance["status"],
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
            failures.append("Frozen Stage13A-1/checkpoint snapshot changed during Stage13A-2")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "phase": args.phase,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_runs_verified": len(current["runs"]),
        "unique_checkpoint_hashes": current["unique_checkpoint_hashes"],
        "stage13a1_provenance_status": current["stage13a1_provenance_status"],
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

