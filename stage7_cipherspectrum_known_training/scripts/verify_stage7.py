#!/usr/bin/env python3
"""Verify all frozen Stage 7 Known-only formal training bundles."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import (
    SPLIT_MANIFEST,
    STAGE6_OUTPUTS,
    STAGE7_ROOT,
    VALID_SETTINGS,
    load_fold,
    sha256_file,
    verify_stage6_protocol_hashes,
)


RUN_DATE = "20260913"
EXPECTED_KNOWN = {"low": 38, "medium": 34, "high": 30}
EXPECTED_UNKNOWN = {"low": 2, "medium": 6, "high": 10}
EXPECTED_SEED = 2022
EXPECTED_LATENT_DIMENSION = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
        f"{label}: {actual} != {expected}",
    )


def verify_bundle_artifacts(run_dir: Path, manifest: dict[str, object]) -> int:
    artifacts = manifest.get("artifacts", [])
    require(bool(artifacts), f"{run_dir.name}: empty artifact inventory")
    for artifact in artifacts:
        path = run_dir / str(artifact["path"])
        require(path.is_file(), f"{run_dir.name}: missing artifact {path}")
        require(
            sha256_file(path) == artifact["sha256"],
            f"{run_dir.name}: artifact hash mismatch: {artifact['path']}",
        )
    return len(artifacts)


def verify_setting(setting: str) -> dict[str, object]:
    run_dir = STAGE7_ROOT / f"runs/formal-{setting}-{RUN_DATE}"
    fold = load_fold(setting)
    audit = load_json(run_dir / "inputs/input_audit.json")
    config = load_json(run_dir / "training_config.json")
    selection = load_json(run_dir / "training_checkpoint_selection.json")
    manifest = load_json(run_dir / "manifest.json")

    known_count = EXPECTED_KNOWN[setting]
    unknown_count = EXPECTED_UNKNOWN[setting]
    train_count = int(fold["known_train_count"])
    validation_count = int(fold["known_validation_count"])

    require(manifest["status"] == "success", f"{setting}: bundle not successful")
    require(audit["status"] == "PASS", f"{setting}: input gate failed")
    require(audit["setting"] == setting and audit["mode"] == "formal", f"{setting}: input identity mismatch")
    require(config["setting"] == setting and config["mode"] == "formal", f"{setting}: config identity mismatch")
    require(selection["setting"] == setting and selection["mode"] == "formal", f"{setting}: selection identity mismatch")

    require(len(fold["known_classes"]) == known_count, f"{setting}: frozen Known class count mismatch")
    require(len(fold["unknown_classes"]) == unknown_count, f"{setting}: frozen Unknown class count mismatch")
    require(audit["known_classes"] == fold["known_classes"], f"{setting}: Known class order mismatch")
    require(audit["unknown_classes"] == fold["unknown_classes"], f"{setting}: Unknown classes mismatch")
    require(config["known_classes"] == fold["known_classes"], f"{setting}: trained Known classes mismatch")
    require(config["unknown_classes_excluded"] == fold["unknown_classes"], f"{setting}: excluded Unknown classes mismatch")

    require(int(audit["stage6_protocol_hashes_verified"]) == 16, f"{setting}: Stage 6 verification count mismatch")
    require(audit["stage6_split_manifest_sha256"] == sha256_file(SPLIT_MANIFEST), f"{setting}: split manifest changed")
    require(
        audit["stage6_fold_sha256"] == sha256_file(STAGE6_OUTPUTS / f"splits/{setting}_fold.json"),
        f"{setting}: fold changed",
    )
    require(audit["input_manifest_sha256"] == sha256_file(run_dir / "inputs/input_manifest.csv"), f"{setting}: input manifest changed")

    require(int(audit["selected_role_counts"]["KNOWN_TRAIN"]) == train_count, f"{setting}: train audit count mismatch")
    require(int(audit["selected_role_counts"]["KNOWN_VALIDATION"]) == validation_count, f"{setting}: validation audit count mismatch")
    require(int(audit["known_train_pcap_files_opened"]) == train_count, f"{setting}: train PCAP count mismatch")
    require(int(audit["known_validation_pcap_files_opened"]) == validation_count, f"{setting}: validation PCAP count mismatch")
    require(int(audit["pcap_files_opened"]) == train_count + validation_count, f"{setting}: total PCAP count mismatch")
    require(int(audit["input_failures"]) == 0, f"{setting}: input failures present")
    require(int(audit["known_test_pcap_files_opened"]) == 0, f"{setting}: Known Test PCAP opened")
    require(int(audit["unknown_pcap_files_opened"]) == 0, f"{setting}: Unknown PCAP opened")
    require(int(audit["unknown_sample_ids_intersect_selected"]) == 0, f"{setting}: Unknown ID selected")
    require(audit["unknown_inference_executed"] is False, f"{setting}: Unknown inference recorded")

    require(int(config["seed"]) == EXPECTED_SEED, f"{setting}: seed mismatch")
    require(int(config["train_samples"]) == train_count, f"{setting}: train config count mismatch")
    require(int(config["validation_samples"]) == validation_count, f"{setting}: validation config count mismatch")
    require(int(config["num_classes"]) == known_count, f"{setting}: num_classes mismatch")
    require(int(config["prototype_count"]) == known_count, f"{setting}: prototype count mismatch")
    require(int(config["latent_dimension"]) == EXPECTED_LATENT_DIMENSION, f"{setting}: latent dimension mismatch")
    require(int(config["unknown_samples_loaded"]) == 0, f"{setting}: Unknown loaded")
    require(int(config["known_test_samples_loaded"]) == 0, f"{setting}: Known Test loaded")
    require(config["warm_start"] is False, f"{setting}: warm start was used")
    require(str(config["physical_gpu"]) not in {"1", "2"}, f"{setting}: forbidden physical GPU used")

    require(int(selection["unknown_samples_used"]) == 0, f"{setting}: Unknown used in selection")
    require(int(selection["known_test_samples_used"]) == 0, f"{setting}: Known Test used in selection")
    require(selection["unknown_inference_executed"] is False, f"{setting}: Unknown inference executed")
    require(selection["selection_data"] == "Known Validation only", f"{setting}: selection data mismatch")
    require(int(selection["early_stopping_patience"]) == 5, f"{setting}: patience mismatch")
    require(int(selection["stop_epoch"]) - int(selection["best_epoch"]) == 5, f"{setting}: early-stop trace mismatch")
    require(int(selection["stability_guard_cumulative_activations"]) == 0, f"{setting}: numerical guard activated")

    with (run_dir / "training_metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    require(len(metrics) == int(selection["stop_epoch"]), f"{setting}: metric row count mismatch")
    require(int(metrics[-1]["epoch"]) == int(selection["stop_epoch"]), f"{setting}: final metric epoch mismatch")
    for row in metrics:
        for key, value in row.items():
            if key.startswith("train_") or key.startswith("val_"):
                require(math.isfinite(float(value)), f"{setting}: non-finite {key} at epoch {row['epoch']}")
    best_row = max(metrics, key=lambda row: float(row["val_composite_score"]))
    require(int(best_row["epoch"]) == int(selection["best_epoch"]), f"{setting}: best epoch mismatch")
    close(float(best_row["val_accuracy"]), float(selection["val_accuracy"]), f"{setting}: accuracy")
    close(float(best_row["val_macro_f1"]), float(selection["val_macro_f1"]), f"{setting}: macro-F1")
    close(float(best_row["val_composite_score"]), float(selection["combined_score"]), f"{setting}: composite")
    harmonic = 2.0 * float(selection["val_accuracy"]) * float(selection["val_macro_f1"]) / (
        float(selection["val_accuracy"]) + float(selection["val_macro_f1"])
    )
    close(harmonic, float(selection["combined_score"]), f"{setting}: harmonic score")

    input_dir = run_dir / "inputs"
    train_images = np.load(input_dir / "train_images.npy", mmap_mode="r", allow_pickle=False)
    validation_images = np.load(input_dir / "validation_images.npy", mmap_mode="r", allow_pickle=False)
    train_labels = np.load(input_dir / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    validation_labels = np.load(input_dir / "validation_labels.npy", mmap_mode="r", allow_pickle=False)
    require(train_images.shape == (train_count, 32, 32), f"{setting}: train image shape mismatch")
    require(validation_images.shape == (validation_count, 32, 32), f"{setting}: validation image shape mismatch")
    require(len(train_labels) == train_count and len(validation_labels) == validation_count, f"{setting}: label count mismatch")
    require(set(np.unique(train_labels)) == set(range(known_count)), f"{setting}: train label domain mismatch")
    require(set(np.unique(validation_labels)) == set(range(known_count)), f"{setting}: validation label domain mismatch")

    checkpoint_path = run_dir / "artifacts/training_best_checkpoint.pt"
    require(selection["checkpoint_sha256"] == sha256_file(checkpoint_path), f"{setting}: checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(int(checkpoint["epoch"]) == int(selection["best_epoch"]), f"{setting}: checkpoint epoch mismatch")
    require(int(checkpoint["best_epoch"]) == int(selection["best_epoch"]), f"{setting}: checkpoint best epoch mismatch")
    require(int(checkpoint["config"]["num_classes"]) == known_count, f"{setting}: checkpoint class count mismatch")
    require(
        tuple(checkpoint["model_state_dict"]["prototypes"].shape) == (known_count, EXPECTED_LATENT_DIMENSION),
        f"{setting}: checkpoint prototype shape mismatch",
    )
    close(float(checkpoint["best_validation_composite"]), float(selection["combined_score"]), f"{setting}: checkpoint score")

    artifact_count = verify_bundle_artifacts(run_dir, manifest)
    return {
        "setting": setting,
        "known_classes": known_count,
        "unknown_classes_excluded": unknown_count,
        "train_samples": train_count,
        "validation_samples": validation_count,
        "best_epoch": int(selection["best_epoch"]),
        "stop_epoch": int(selection["stop_epoch"]),
        "val_accuracy": float(selection["val_accuracy"]),
        "val_macro_f1": float(selection["val_macro_f1"]),
        "val_composite": float(selection["combined_score"]),
        "checkpoint_sha256": str(selection["checkpoint_sha256"]),
        "bundle_artifacts_verified": artifact_count,
        "known_test_samples_opened_or_used": 0,
        "unknown_samples_opened_or_used": 0,
        "unknown_inference_executed": False,
        "status": "PASS",
    }


def main() -> None:
    protocol_count = len(verify_stage6_protocol_hashes())
    results = [verify_setting(setting) for setting in VALID_SETTINGS]
    hashes = {result["checkpoint_sha256"] for result in results}
    require(len(hashes) == len(VALID_SETTINGS), "formal setting checkpoints are not independent")
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage6_protocol_hashes_verified": protocol_count,
                "formal_settings_verified": len(results),
                "checkpoints_distinct": True,
                "known_test_samples_opened_or_used": 0,
                "unknown_samples_opened_or_used": 0,
                "unknown_inference_executed": False,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
