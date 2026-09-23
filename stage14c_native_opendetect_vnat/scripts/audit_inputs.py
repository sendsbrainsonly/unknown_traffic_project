#!/usr/bin/env python3
"""Fail-closed audit of model-visible inputs without loading Test/Unknown arrays."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from common import (
    NATIVE_CONFIG, OPENDETECT_ROOT, OUTPUTS_ROOT, ROOT, load_protocols,
    run_input_dir, sha256_file, verify_freeze,
)


MODEL_VISIBLE_FILES = (
    "train_images.npy", "train_labels.npy", "validation_images.npy",
    "validation_labels.npy", "input_audit.json", "input_manifest.csv",
    "label_map.json",
)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(OPENDETECT_ROOT / "code"), *args], text=True
    ).strip()


def main() -> None:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    freeze = verify_freeze()
    protocols = load_protocols()
    runs: list[dict[str, object]] = []
    for protocol_id, protocol in sorted(protocols.items()):
        input_dir = run_input_dir(protocol_id)
        audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
        label_map = json.loads((input_dir / "label_map.json").read_text(encoding="utf-8"))
        train_x = np.load(input_dir / "train_images.npy", mmap_mode="r", allow_pickle=False)
        train_y = np.load(input_dir / "train_labels.npy", mmap_mode="r", allow_pickle=False)
        val_x = np.load(input_dir / "validation_images.npy", mmap_mode="r", allow_pickle=False)
        val_y = np.load(input_dir / "validation_labels.npy", mmap_mode="r", allow_pickle=False)
        known = list(map(str, protocol["known_applications"]))
        unknown = list(map(str, protocol["unknown_applications"]))
        failures: list[str] = []
        if set(known) & set(unknown):
            failures.append("Known/Unknown class overlap")
        if audit.get("status") != "PASS" or audit.get("freeze_hash") != freeze["freeze_hash"]:
            failures.append("source input audit/freeze mismatch")
        for field in (
            "unknown_samples_used_in_training", "unknown_samples_used_in_validation",
            "known_test_samples_used", "train_validation_flow_overlap",
        ):
            if int(audit.get(field, -1)) != 0:
                failures.append(f"{field} is not zero")
        expected_train = int(protocol["split_statistics"]["train"]["flows"])
        expected_val = int(protocol["split_statistics"]["validation"]["flows"])
        if train_x.shape != (expected_train, 32, 32) or train_y.shape != (expected_train,):
            failures.append(f"train shape mismatch: {train_x.shape}/{train_y.shape}")
        if val_x.shape != (expected_val, 32, 32) or val_y.shape != (expected_val,):
            failures.append(f"validation shape mismatch: {val_x.shape}/{val_y.shape}")
        if train_x.dtype != np.uint8 or val_x.dtype != np.uint8:
            failures.append("images are not uint8")
        expected_labels = list(range(len(known)))
        if sorted(map(int, np.unique(train_y))) != expected_labels:
            failures.append("train label coverage mismatch")
        if sorted(map(int, np.unique(val_y))) != expected_labels:
            failures.append("validation label coverage mismatch")
        if label_map.get("class_to_local") != {name: i for i, name in enumerate(known)}:
            failures.append("label map does not match frozen Known class order")
        runs.append({
            "protocol_id": protocol_id,
            "setting": protocol["setting"],
            "protocol_seed": int(protocol["seed"]),
            "known_classes": known,
            "unknown_classes": unknown,
            "train_samples": expected_train,
            "validation_samples": expected_val,
            "model_visible_files": {
                name: sha256_file(input_dir / name) for name in MODEL_VISIBLE_FILES
            },
            "unknown_samples_used_in_training": 0,
            "unknown_samples_used_in_validation": 0,
            "known_test_samples_used": 0,
            "failures": failures,
            "status": "PASS" if not failures else "FAIL",
        })
    code_files = [
        OPENDETECT_ROOT / "code" / "train.py",
        OPENDETECT_ROOT / "code" / "model.py",
        OPENDETECT_ROOT / "code" / "utils.py",
        OPENDETECT_ROOT / "reproduction" / "run_reproduction.py",
    ]
    source = {
        "opendetect_code_git_commit": git_value("rev-parse", "HEAD"),
        "opendetect_code_git_status_short": git_value("status", "--short"),
        "files": {str(path): sha256_file(path) for path in code_files},
    }
    result = {
        "status": "PASS" if all(run["status"] == "PASS" for run in runs) else "FAIL",
        "freeze": freeze,
        "model_visible_roles": ["known:train", "known:validation"],
        "explicitly_not_loaded": ["known:test", "unknown:test"],
        "native_config": NATIVE_CONFIG,
        "runs": runs,
    }
    (OUTPUTS_ROOT / "input_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUTS_ROOT / "source_code_hashes.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUTS_ROOT / "native_config.json").write_text(
        json.dumps(NATIVE_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "runs": len(runs), "freeze": freeze}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

