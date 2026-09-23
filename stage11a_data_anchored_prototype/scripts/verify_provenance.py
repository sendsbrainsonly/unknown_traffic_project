#!/usr/bin/env python3
"""Verify Stage 6-10B sentinels and the frozen Open-Detect v6 baseline.

The verifier intentionally does not open any CipherSpectrum Stage 9 sample
array.  It reads only previously frozen provenance/sentinel metadata and USTC
v6 development artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from stage11a_common import (
    OPEN_DETECT_ROOT,
    STAGE_ROOT,
    UNKNOWN_ROOT,
    array_digest,
    load_npz,
    read_json,
    sha256_file,
    write_json,
)


OUTPUT = STAGE_ROOT / "outputs" / "summary" / "provenance_verification.json"
STAGE10A_MANIFEST = (
    UNKNOWN_ROOT / "stage10a_opendetect_protocol_audit" / "sources" / "source_manifest.json"
)
STAGE10B_ROOT = UNKNOWN_ROOT / "stage10b_opendetect_score_decomposition"
STAGE10B_PROVENANCE = STAGE10B_ROOT / "outputs" / "summary" / "provenance_verification.json"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def hash_rows(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    failures: list[str] = []
    stage10b_provenance = read_json(STAGE10B_PROVENANCE)
    check(stage10b_provenance.get("status") == "PASS", "Stage10B prior provenance status is not PASS", failures)
    check(stage10b_provenance.get("frozen_sentinels_unchanged") is True, "Stage6-9 prior sentinels were not unchanged", failures)
    check(stage10b_provenance.get("stage10a", {}).get("status") == "PASS", "Stage10A prior provenance status is not PASS", failures)

    sentinel_expected = stage10b_provenance["frozen_sentinel_hashes_after"]
    sentinel_observed: dict[str, str] = {}
    for relative, expected in sentinel_expected.items():
        path = UNKNOWN_ROOT / relative
        check(path.is_file(), f"Missing frozen sentinel: {path}", failures)
        if path.is_file():
            observed = sha256_file(path)
            sentinel_observed[relative] = observed
            check(observed == expected, f"Frozen sentinel hash mismatch: {path}", failures)

    source_manifest = read_json(STAGE10A_MANIFEST)
    source_observed: dict[str, str] = {}
    for row in source_manifest["files"] + source_manifest["v6_checkpoints"]:
        path = Path(row["path"])
        check(path.is_file(), f"Missing Stage10A/Open-Detect frozen source: {path}", failures)
        if path.is_file():
            observed = sha256_file(path)
            source_observed[str(path.resolve())] = observed
            check(observed == row["sha256"], f"Stage10A/Open-Detect source hash mismatch: {path}", failures)
            check(path.stat().st_size == int(row["size_bytes"]), f"Frozen source size mismatch: {path}", failures)

    config = read_json(STAGE_ROOT / "configs" / "stage11a_config.json")
    v6_root = Path(config["v6_frozen_root"])
    split_root = Path(config["v6_split_root"])
    split_manifest_path = split_root / "campaign_manifest.json"
    split_manifest = read_json(split_manifest_path)
    check(split_manifest.get("status") == "complete", "v6 split manifest is not complete", failures)
    check(split_manifest.get("pool_sha256") == "9e08cfab6cf913eba4611a859af326ad5a21e61a0d4dede22b5571c5238012b8", "v6 pool hash changed", failures)
    for fold_row in split_manifest["folds"]:
        for kind in ("flow_overlap", "exact_image_overlap"):
            check(all(int(value) == 0 for value in fold_row[kind].values()), f"Non-zero {kind} in fold {fold_row['fold']}", failures)

    expected_checkpoint_hashes = {
        str(Path(row["path"]).resolve()): row["sha256"] for row in source_manifest["v6_checkpoints"]
    }
    fixed = config["training"]
    v6_runs_verified = 0
    split_array_hashes: dict[str, dict[str, str]] = {}
    for fold, seed in enumerate(config["seeds"]):
        fold_dir = split_root / f"fold{fold}_seed{seed}"
        train_x, train_y = load_npz(fold_dir / "ustc_train.npz")
        val_x, val_y = load_npz(fold_dir / "ustc_validation.npz")
        test_x, test_y = load_npz(fold_dir / "ustc_test.npz")
        observed_arrays = {
            "train": array_digest(train_x, train_y),
            "validation": array_digest(val_x, val_y),
            "test": array_digest(test_x, test_y),
        }
        split_array_hashes[f"fold{fold}_seed{seed}"] = observed_arrays
        for scenario, scenario_cfg in config["scenarios"].items():
            run_dir = v6_root / scenario / f"fold{fold}_seed{seed}"
            required = ["SUCCESS", "run_config.json", "train_log.jsonl", "model_best.pt", "test_metrics.json"]
            check(all((run_dir / name).is_file() for name in required), f"Incomplete frozen v6 run: {run_dir}", failures)
            if not all((run_dir / name).is_file() for name in required):
                continue
            run_cfg = read_json(run_dir / "run_config.json")
            check(run_cfg["split"] == scenario_cfg["split"], f"v6 split mismatch: {run_dir}", failures)
            check(run_cfg["seed"] == seed, f"v6 seed mismatch: {run_dir}", failures)
            check(run_cfg["split_array_sha256"] == observed_arrays, f"v6 array digest mismatch: {run_dir}", failures)
            contract = {
                "epochs": fixed["epochs"],
                "batch_size": fixed["batch_size"],
                "eval_batch_size": fixed["eval_batch_size"],
                "workers": fixed["workers"],
                "lr": fixed["learning_rate"],
                "lamda": fixed["lambda"],
                "latent_dim": fixed["latent_dim"],
                "eval_seed": fixed["eval_seed"],
                "early_stop_patience": fixed["early_stop_patience"],
                "early_stop_min_epoch": fixed["early_stop_min_epoch"],
                "early_stop_min_delta": fixed["early_stop_min_delta"],
                "training_protocol": "corrected-paper",
            }
            for key, expected in contract.items():
                check(run_cfg[key] == expected, f"v6 contract mismatch {key}: {run_dir}", failures)
            checkpoint = (run_dir / "model_best.pt").resolve()
            check(str(checkpoint) in expected_checkpoint_hashes, f"v6 checkpoint absent from Stage10A manifest: {checkpoint}", failures)
            if str(checkpoint) in expected_checkpoint_hashes:
                check(sha256_file(checkpoint) == expected_checkpoint_hashes[str(checkpoint)], f"v6 checkpoint drift: {checkpoint}", failures)
            v6_runs_verified += 1

    stage10b_summary_paths = [
        STAGE10B_ROOT / "README.md",
        STAGE10B_ROOT / "RESULTS.md",
        STAGE10B_ROOT / "DIAGNOSIS_REPORT.md",
        STAGE10B_ROOT / "manifest.json",
        STAGE10B_PROVENANCE,
    ]
    stage10b_summary_paths.extend(sorted((STAGE10B_ROOT / "outputs" / "summary").glob("*")))
    stage10b_summary_paths = sorted(set(path for path in stage10b_summary_paths if path.is_file()))
    stage10b_summary_hashes = hash_rows(stage10b_summary_paths)
    frozen_snapshot = {
        "stage6_to_stage9_sentinels": sentinel_observed,
        "stage10a_and_opendetect_sources": source_observed,
        "stage10b_summary_only": stage10b_summary_hashes,
        "v6_split_manifest": sha256_file(split_manifest_path),
    }

    previous = read_json(OUTPUT) if OUTPUT.is_file() else None
    if args.phase == "after":
        check(previous is not None, "Missing before-phase provenance record", failures)
        if previous is not None:
            check(previous.get("frozen_hashes_before") == frozen_snapshot, "Frozen source snapshot changed during Stage11A", failures)
    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "phase": args.phase,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage6_to_stage10b": status,
        "open_detect_v6": status,
        "stage10b_prior_provenance_status": stage10b_provenance.get("status"),
        "stage10b_access_scope": "summary files only",
        "cipherspectrum_stage9_sample_level_files_opened": [],
        "cipherspectrum_stage9_sample_level_test_used_for_development": False,
        "stage6_to_stage9_sentinels_verified": len(sentinel_observed),
        "stage10a_manifest_items_verified": len(source_observed),
        "v6_checkpoints_verified": len(expected_checkpoint_hashes),
        "v6_runs_verified": v6_runs_verified,
        "v6_split_array_hashes": split_array_hashes,
        "flow_and_exact_image_disjoint_verified": True,
        "frozen_hashes_before": (
            frozen_snapshot
            if args.phase == "before" or previous is None
            else previous["frozen_hashes_before"]
        ),
        "frozen_hashes_after": frozen_snapshot if args.phase == "after" else None,
        "frozen_experiment_modified": bool(failures) if args.phase == "after" else False,
        "failures": failures,
    }
    write_json(OUTPUT, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
