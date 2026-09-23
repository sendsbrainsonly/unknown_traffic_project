#!/usr/bin/env python3
"""Shared paths and integrity gates for Stage 14C.5."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STAGE14C_ROOT = PROJECT_ROOT / "stage14c_vnat_encoder_training"
STAGE14C_SCRIPTS = STAGE14C_ROOT / "scripts"
_stage14c_spec = importlib.util.spec_from_file_location(
    "stage14c_frozen_common", STAGE14C_SCRIPTS / "common.py"
)
if _stage14c_spec is None or _stage14c_spec.loader is None:
    raise ImportError("cannot load Stage 14C frozen common module")
stage14c_common = importlib.util.module_from_spec(_stage14c_spec)
_stage14c_spec.loader.exec_module(stage14c_common)

RAW_CACHE = ROOT / "raw_feature_cache"
RUNS_ROOT = ROOT / "runs"
CHECKPOINT_ROOT = ROOT / "checkpoints"
CONFUSION_ROOT = ROOT / "confusion_matrices"
FEATURES = ("f0", "f1", "f2", "f3")
NEW_FEATURES = ("f1", "f2", "f3")
STAT_NAMES = (
    "packet_count",
    "total_bytes",
    "duration_seconds",
    "packet_size_mean",
    "packet_size_std",
    "iat_mean_seconds",
    "iat_std_seconds",
    "up_down_packet_ratio",
)
EXPECTED_STAGE14C_RUNS = 15


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs() -> dict[str, object]:
    freeze = stage14c_common.verify_freeze()
    summary_path = STAGE14C_ROOT / "stage14c_training_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_STAGE14C_RUNS or any(row["status"] != "success" for row in rows):
        raise RuntimeError("Stage 14C baseline is not a complete 15/15 success grid")
    checkpoint_hashes: dict[str, str] = {}
    for row in rows:
        path = Path(row["checkpoint_path"])
        actual = sha256_file(path)
        if actual != row["checkpoint_sha256"]:
            raise RuntimeError(f"Stage 14C checkpoint hash mismatch: {path}")
        protocol_id = f"{row['setting'].lower()}_seed{row['seed']}"
        checkpoint_hashes[protocol_id] = actual
    return {
        **freeze,
        "stage14c_summary_sha256": sha256_file(summary_path),
        "stage14c_checkpoint_hashes": checkpoint_hashes,
    }


def protocol_ids() -> list[str]:
    return sorted(stage14c_common.protocols_by_id())


def source_input_dir(protocol_id: str) -> Path:
    return STAGE14C_ROOT / "runs" / protocol_id / "inputs"


def load_known_input_rows(protocol_id: str) -> list[dict[str, str]]:
    path = source_input_dir(protocol_id) / "input_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["class_role"] != "known" for row in rows):
        raise RuntimeError(f"{protocol_id}: non-Known row in Stage 14C input manifest")
    if {row["split"] for row in rows} - {"train", "validation"}:
        raise RuntimeError(f"{protocol_id}: Test/Unknown row in Stage 14C input manifest")
    return rows


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-9, 1.0, iqr)
    return np.asarray(median), np.asarray(iqr)


def robust_quantize(values: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    z = np.clip((np.asarray(values, dtype=np.float64) - median) / iqr, -4.0, 4.0)
    return np.rint((z + 4.0) * (255.0 / 8.0)).astype(np.uint8)


def transformed_statistics(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    out = raw.copy()
    out[..., 0] = np.log1p(np.maximum(raw[..., 0], 0.0))
    out[..., 1] = np.log1p(np.maximum(raw[..., 1], 0.0))
    out[..., 2] = np.log1p(np.maximum(raw[..., 2], 0.0) * 1e6)
    out[..., 3] = np.log1p(np.maximum(raw[..., 3], 0.0))
    out[..., 4] = np.log1p(np.maximum(raw[..., 4], 0.0))
    out[..., 5] = np.log1p(np.maximum(raw[..., 5], 0.0) * 1e6)
    out[..., 6] = np.log1p(np.maximum(raw[..., 6], 0.0) * 1e6)
    out[..., 7] = np.log(np.maximum(raw[..., 7], 1e-12))
    return out


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
