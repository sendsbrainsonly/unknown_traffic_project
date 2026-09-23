#!/usr/bin/env python3
"""Verify Stage6-11A and Open-Detect frozen evidence without Stage9 samples."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from stage11b_common import CONFIG_PATH, STAGE11A_ROOT, STAGE_ROOT, UNKNOWN_ROOT, read_json, sha256_file, write_json

STAGE11A_SCRIPTS = STAGE11A_ROOT / "scripts"
if str(STAGE11A_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE11A_SCRIPTS))
from stage11a_common import array_digest, load_npz  # noqa: E402
from data.splits import get_splits  # noqa: E402


OUTPUT = STAGE_ROOT / "outputs" / "summary" / "provenance_verification.json"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def verify_hash(path: Path, expected: str, failures: list[str]) -> str | None:
    check(path.is_file(), f"Missing frozen file: {path}", failures)
    if not path.is_file():
        return None
    observed = sha256_file(path)
    check(observed == expected, f"Frozen hash mismatch: {path}", failures)
    return observed


def verify_upstream(stage11a_provenance: dict[str, object], failures: list[str]) -> dict[str, object]:
    expected = stage11a_provenance["frozen_hashes_after"]
    observed: dict[str, object] = {
        "stage6_to_stage9_sentinels": {},
        "stage10a_and_opendetect_sources": {},
        "stage10b_summary_only": {},
    }
    for relative, digest in expected["stage6_to_stage9_sentinels"].items():
        path = UNKNOWN_ROOT / relative
        actual = verify_hash(path, digest, failures)
        if actual is not None:
            observed["stage6_to_stage9_sentinels"][relative] = actual
    for group in ("stage10a_and_opendetect_sources", "stage10b_summary_only"):
        for raw_path, digest in expected[group].items():
            path = Path(raw_path)
            actual = verify_hash(path, digest, failures)
            if actual is not None:
                observed[group][str(path.resolve())] = actual
    config = read_json(CONFIG_PATH)
    split_manifest = Path(config["v6_split_root"]) / "campaign_manifest.json"
    actual = verify_hash(split_manifest, expected["v6_split_manifest"], failures)
    observed["v6_split_manifest"] = actual
    return observed


def verify_stage11a_bundle(failures: list[str]) -> dict[str, object]:
    manifest_path = STAGE11A_ROOT / "manifest.json"
    check(manifest_path.is_file(), "Stage11A manifest is missing", failures)
    if not manifest_path.is_file():
        return {}
    manifest = read_json(manifest_path)
    check(manifest.get("status") == "success", "Stage11A manifest is not successful", failures)
    observed: dict[str, str] = {}
    for row in manifest.get("artifacts", []):
        if row.get("scope") != "bundle" or not row.get("sha256"):
            continue
        path = STAGE11A_ROOT / row["path"]
        actual = verify_hash(path, row["sha256"], failures)
        if actual is not None:
            observed[row["path"]] = actual
            check(path.stat().st_size == int(row["size_bytes"]), f"Stage11A size mismatch: {path}", failures)
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_artifacts_verified": len(observed),
        "artifact_hashes": observed,
    }


def verify_v6_contract(failures: list[str]) -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    stage11a_provenance = read_json(
        STAGE11A_ROOT / "outputs" / "summary" / "provenance_verification.json"
    )
    split_hashes_expected = stage11a_provenance["v6_split_array_hashes"]
    v6_root = Path(config["v6_frozen_root"])
    split_root = Path(config["v6_split_root"])
    checkpoint_hashes: dict[str, str] = {}
    class_sets: dict[str, object] = {}
    split_hashes: dict[str, object] = {}
    runs = 0
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
        key = f"fold{fold}_seed{seed}"
        split_hashes[key] = observed_arrays
        check(observed_arrays == split_hashes_expected[key], f"Frozen split arrays changed: {key}", failures)
        for scenario, scenario_cfg in config["scenarios"].items():
            known, unknown, _, _ = get_splits("USTC", int(scenario_cfg["split"]))
            check(len(known) == int(scenario_cfg["known_count"]), f"Known class count mismatch: {scenario}", failures)
            check(len(unknown) == int(scenario_cfg["unknown_count"]), f"Unknown class count mismatch: {scenario}", failures)
            class_sets[scenario] = {"known": list(map(int, known)), "unknown": list(map(int, unknown))}
            run_dir = v6_root / scenario / key
            run_cfg = read_json(run_dir / "run_config.json")
            check(run_cfg["seed"] == seed, f"Frozen seed mismatch: {run_dir}", failures)
            check(run_cfg["split"] == int(scenario_cfg["split"]), f"Frozen scenario split mismatch: {run_dir}", failures)
            check(run_cfg["known_classes"] == list(known), f"Frozen known classes mismatch: {run_dir}", failures)
            check(run_cfg["unknown_classes"] == list(unknown), f"Frozen unknown classes mismatch: {run_dir}", failures)
            check(run_cfg["split_array_sha256"] == observed_arrays, f"Frozen run array hashes mismatch: {run_dir}", failures)
            checkpoint = run_dir / "model_best.pt"
            stage11a_result = read_json(
                STAGE11A_ROOT / "artifacts" / "b0" / scenario / key / "results.json"
            )
            digest = sha256_file(checkpoint)
            check(digest == stage11a_result["frozen_b0"]["checkpoint_sha256"], f"Checkpoint drift: {checkpoint}", failures)
            checkpoint_hashes[str(checkpoint.resolve())] = digest
            runs += 1
    return {
        "runs_verified": runs,
        "checkpoint_hashes": checkpoint_hashes,
        "split_array_hashes": split_hashes,
        "class_sets": class_sets,
        "seeds": config["seeds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    failures: list[str] = []
    stage11a_provenance_path = STAGE11A_ROOT / "outputs" / "summary" / "provenance_verification.json"
    stage11a_provenance = read_json(stage11a_provenance_path)
    check(stage11a_provenance.get("status") == "PASS", "Stage11A provenance is not PASS", failures)
    check(stage11a_provenance.get("phase") == "after", "Stage11A provenance is not the final after snapshot", failures)
    check(not stage11a_provenance.get("frozen_experiment_modified"), "Stage11A reports frozen modification", failures)
    snapshot = {
        "upstream": verify_upstream(stage11a_provenance, failures),
        "stage11a": verify_stage11a_bundle(failures),
        "v6_contract": verify_v6_contract(failures),
        "stage11a_provenance_sha256": sha256_file(stage11a_provenance_path),
    }
    previous = read_json(OUTPUT) if OUTPUT.is_file() else None
    if args.phase == "after":
        check(previous is not None, "Missing before-phase Stage11B provenance", failures)
        if previous is not None:
            check(previous.get("frozen_hashes_before") == snapshot, "Frozen snapshot changed during Stage11B", failures)
    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "phase": args.phase,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage6_to_stage10b": status,
        "stage11a": status,
        "open_detect_v6": status,
        "v6_checkpoints_verified": len(snapshot["v6_contract"].get("checkpoint_hashes", {})),
        "v6_runs_verified": snapshot["v6_contract"].get("runs_verified", 0),
        "stage11a_manifest_artifacts_verified": snapshot["stage11a"].get("manifest_artifacts_verified", 0),
        "class_sets_verified": snapshot["v6_contract"].get("class_sets", {}),
        "seeds_verified": snapshot["v6_contract"].get("seeds", []),
        "cipherspectrum_stage9_access_scope": "frozen hash/sentinel metadata only",
        "cipherspectrum_stage9_sample_level_files_opened": [],
        "cipherspectrum_stage9_sample_level_test_used_for_development": False,
        "frozen_hashes_before": snapshot if args.phase == "before" or previous is None else previous["frozen_hashes_before"],
        "frozen_hashes_after": snapshot if args.phase == "after" else None,
        "frozen_experiment_modified": bool(failures) if args.phase == "after" else False,
        "failures": failures,
    }
    write_json(OUTPUT, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
