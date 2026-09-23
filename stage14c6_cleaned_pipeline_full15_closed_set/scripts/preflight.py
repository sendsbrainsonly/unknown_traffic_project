#!/usr/bin/env python3
"""Fail-closed preflight for the frozen Stage 14C-6 formal grid."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
F2 = PROJECT / "stage14c5_feature_representation_audit"
NATIVE = PROJECT / "stage14c_native_opendetect_vnat"
TRAIN_SOURCE = PROJECT / "stage14c4_our_method_training_cleanup/scripts/train_cleaned.py"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_HASHES = {
    STAGE14B / "vnat_open_set_protocol.json": "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced",
    STAGE14B / "vnat_split_manifest.csv": "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e",
    TRAIN_SOURCE: "a72b0e52be8c10917c057b49fea58a228253a867007e7390097c27e98cab8655",
}
PROTOCOLS = [
    f"{setting}_seed{seed}"
    for setting in ("low", "medium", "high")
    for seed in range(2022, 2027)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input hash changed: {path}: {actual}")

    frozen = json.loads((STAGE14B / "vnat_open_set_protocol.json").read_text(encoding="utf-8"))
    if frozen["freeze_hash"] != EXPECTED_FREEZE:
        raise RuntimeError("Stage 14B freeze hash changed")
    protocol_by_id = {row["protocol_id"]: row for row in frozen["protocols"]}
    if set(protocol_by_id) != set(PROTOCOLS) or len(protocol_by_id) != 15:
        raise RuntimeError("frozen protocol set is not the expected 15-run grid")

    input_rows: list[dict[str, object]] = []
    for protocol_id in PROTOCOLS:
        protocol = protocol_by_id[protocol_id]
        input_dir = F2 / "runs" / "f2" / protocol_id / "inputs"
        audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
        label_map = json.loads((input_dir / "label_map.json").read_text(encoding="utf-8"))
        manifest_hash = sha256(input_dir / "input_manifest.csv")
        train_images = np.load(input_dir / "train_images.npy", mmap_mode="r", allow_pickle=False)
        train_labels = np.load(input_dir / "train_labels.npy", mmap_mode="r", allow_pickle=False)
        val_images = np.load(input_dir / "validation_images.npy", mmap_mode="r", allow_pickle=False)
        val_labels = np.load(input_dir / "validation_labels.npy", mmap_mode="r", allow_pickle=False)
        failures: list[str] = []
        if audit.get("status") != "PASS":
            failures.append("input_audit_not_pass")
        if audit.get("unknown_samples_loaded") != 0:
            failures.append("unknown_samples_loaded")
        if audit.get("known_test_samples_loaded") != 0:
            failures.append("known_test_samples_loaded")
        if audit.get("derived_manifest_sha256") != manifest_hash:
            failures.append("input_manifest_hash_mismatch")
        if set(label_map["class_to_local"]) != set(protocol["known_applications"]):
            failures.append("known_class_map_mismatch")
        if len(train_images) != len(train_labels) or len(val_images) != len(val_labels):
            failures.append("image_label_count_mismatch")
        if len(train_labels) != protocol["split_statistics"]["train"]["flows"]:
            failures.append("train_count_mismatch")
        if len(val_labels) != protocol["split_statistics"]["validation"]["flows"]:
            failures.append("validation_count_mismatch")
        if failures:
            raise RuntimeError(f"{protocol_id} input gate failed: {failures}")
        input_rows.append({
            "protocol_id": protocol_id,
            "setting": protocol["setting"],
            "seed": int(protocol["seed"]),
            "known_classes": "|".join(protocol["known_applications"]),
            "unknown_classes": "|".join(protocol["unknown_applications"]),
            "train_samples": len(train_labels),
            "validation_samples": len(val_labels),
            "feature_shape": "x".join(map(str, train_images.shape[1:])),
            "input_manifest_sha256": manifest_hash,
            "known_test_samples_loaded": 0,
            "unknown_samples_loaded": 0,
            "status": "PASS",
        })

    native_path = NATIVE / "outputs/stage14c_native_run_results.csv"
    with native_path.open(encoding="utf-8") as handle:
        native_rows = list(csv.DictReader(handle))
    if len(native_rows) != 15 or {row["protocol_id"] for row in native_rows} != set(PROTOCOLS):
        raise RuntimeError("Native Open-Detect paired results are incomplete")
    if any(row["status"] != "success" for row in native_rows):
        raise RuntimeError("Native Open-Detect contains failed runs")

    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "input_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)

    disk = shutil.disk_usage(PROJECT)
    config = {
        "status": "FROZEN_READY_TO_RUN",
        "freeze_hash": EXPECTED_FREEZE,
        "protocols": PROTOCOLS,
        "runs": 15,
        "feature": "F2",
        "architecture": "current own-method architecture imported through immutable Stage 14C-4 implementation",
        "loss": "0.005*(rec+kld+ent)+0.995*dis",
        "epochs": 100,
        "batch_size": 128,
        "early_stopping": False,
        "optimizer": "Adam(beta1=0.9,beta2=0.999,weight_decay=0)",
        "learning_rate": 0.001,
        "scheduler": "MultiStepLR milestones=[50,80] gamma=0.1",
        "prototype_reset_epochs": [51, 81],
        "prototype_reset": "in-place copy; optimizer parameter link retained; prototype optimizer state cleared",
        "checkpoint_selection": "highest harmonic mean of Known Validation Accuracy and Macro-F1",
        "training_seed": "frozen protocol seed",
        "numerical_protection": "upper logvar clamp at 20",
        "weak_run_definition": "val_macro_f1 < 0.50 OR val_accuracy < 0.60 OR NaN/crash",
        "conclusion_rule": {
            "CLOSED_SET_NEEDS_FURTHER_FIX": "any integrity violation, Test/Unknown/DES use, incomplete/NaN/crashed run, or unexplained catastrophic behavior",
            "CLOSED_SET_PASS_WITH_HARD_PROTOCOLS": "all integrity gates pass but reproducible low performance is concentrated in identifiable frozen Known class compositions",
            "CLOSED_SET_PASS": "all integrity gates pass with no weak or composition-specific hard protocols",
        },
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
        "stage14c4_training_source_sha256": EXPECTED_HASHES[TRAIN_SOURCE],
        "native_results_path": str(native_path),
        "native_results_sha256": sha256(native_path),
        "disk_free_bytes_before_training": disk.free,
        "estimated_checkpoint_storage_bytes": 11_000_000_000,
    }
    (ROOT / "frozen_training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "freeze_hash": EXPECTED_FREEZE,
        "protocols": 15,
        "f2_input_gates_pass": 15,
        "native_paired_results_available": 15,
        "known_test_samples_loaded": 0,
        "unknown_samples_loaded": 0,
        "des_executed": False,
        "disk_free_bytes": disk.free,
    }
    (ROOT / "preflight.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
