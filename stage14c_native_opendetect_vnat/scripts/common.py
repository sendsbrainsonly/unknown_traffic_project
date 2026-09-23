#!/usr/bin/env python3
"""Shared immutable paths and configuration for the native Open-Detect run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STAGE14B_ROOT = PROJECT_ROOT / "stage14b_vnat_protocol_freeze"
SOURCE_INPUT_ROOT = PROJECT_ROOT / "stage14c_vnat_encoder_training" / "runs"
OPENDETECT_ROOT = PROJECT_ROOT.parent / "Open-Detect"
PROTOCOL_PATH = STAGE14B_ROOT / "vnat_open_set_protocol.json"
SPLIT_MANIFEST_PATH = STAGE14B_ROOT / "vnat_split_manifest.csv"
FREEZE_HASH_PATH = STAGE14B_ROOT / "outputs" / "freeze_hashes.json"
RUNS_ROOT = ROOT / "runs"
OUTPUTS_ROOT = ROOT / "outputs"

EXPECTED_FREEZE_HASH = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_PROTOCOL_SHA256 = "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced"
EXPECTED_SPLIT_SHA256 = "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e"
SETTINGS = ("Low", "Medium", "High")
PROTOCOL_SEEDS = (2022, 2023, 2024, 2025, 2026)
TRAINING_SEED = 2022

NATIVE_CONFIG = {
    "arch": "resnet18",
    "channels": 1,
    "latent_dim": 128,
    "temp_inter": 1.0,
    "temp_intra": 1.0,
    "lambda": 0.005,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "epochs": 100,
    "train_batch_size": 128,
    "validation_batch_size": 64,
    "workers": 4,
    "dataloader_tensor_sharing_strategy": "default_file_descriptor",
    "dataloader_temp_path_addressing": "project-local directory via /proc/self/fd dirfd alias",
    "lr_milestones": [50, 80],
    "lr_gamma": 0.1,
    "prototype_reset_zero_based_epochs": [50, 80],
    "checkpoint_selection": "maximum Known Validation Accuracy",
    "early_stopping": False,
    "training_seed": TRAINING_SEED,
    "training_protocol": "released-code",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze() -> dict[str, str]:
    recorded = json.loads(FREEZE_HASH_PATH.read_text(encoding="utf-8"))
    actual = {
        "freeze_hash": str(recorded["freeze_hash"]),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
    }
    expected = {
        "freeze_hash": EXPECTED_FREEZE_HASH,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
    }
    if actual != expected:
        raise RuntimeError(f"Stage 14B freeze mismatch: expected={expected}, actual={actual}")
    return actual


def load_protocols() -> dict[str, dict[str, object]]:
    verify_freeze()
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if document.get("freeze_hash") != EXPECTED_FREEZE_HASH:
        raise RuntimeError("protocol document freeze hash mismatch")
    protocols = {str(item["protocol_id"]): item for item in document["protocols"]}
    expected = {
        f"{setting.lower()}_seed{seed}"
        for setting in SETTINGS
        for seed in PROTOCOL_SEEDS
    }
    if set(protocols) != expected:
        raise RuntimeError(f"unexpected frozen protocol grid: {sorted(protocols)}")
    return protocols


def run_input_dir(protocol_id: str) -> Path:
    return SOURCE_INPUT_ROOT / protocol_id / "inputs"
