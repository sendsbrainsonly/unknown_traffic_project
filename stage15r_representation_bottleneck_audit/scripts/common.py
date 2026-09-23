#!/usr/bin/env python3
"""Shared paths and deterministic helpers for Stage 15R."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
OPENDETECT_ROOT = PROJECT_ROOT.parent / "Open-Detect"
STAGE12_ROOT = PROJECT_ROOT / "stage12_dual_external_validation"
STAGE14B_ROOT = PROJECT_ROOT / "stage14b_vnat_protocol_freeze"
STAGE14C_INPUT_ROOT = PROJECT_ROOT / "stage14c_vnat_encoder_training"
STAGE14C_NATIVE_ROOT = PROJECT_ROOT / "stage14c_native_opendetect_vnat"
STAGE14C5_ROOT = PROJECT_ROOT / "stage14c5_feature_representation_audit"
USTC_AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
STAGE3_PROTOCOL_ROOT = PROJECT_ROOT / "stage3_protocol"
STAGE3_ROOT = PROJECT_ROOT / "stage3_unknown_utility"
CONFIG_PATH = ROOT / "config.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def percentile_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "p1": float("nan"), "p99": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p1": float(np.quantile(finite, 0.01)),
        "p99": float(np.quantile(finite, 0.99)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


PACKET_BINS = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-10", 5, 10),
    ("11-20", 11, 20),
    (">20", 21, None),
)


def packet_bin(count: int) -> str:
    for label, lower, upper in PACKET_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    return "0"
