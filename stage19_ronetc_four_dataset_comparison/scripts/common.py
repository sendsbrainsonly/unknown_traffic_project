#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
RONETC = PROJECT.parent / "RoNeTC" / "code"
STAGE12 = PROJECT / "stage12_dual_external_validation"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
STAGE14C = PROJECT / "stage14c_vnat_encoder_training"
STAGE15R = PROJECT / "stage15r_representation_bottleneck_audit"
USTC_AUDIT = PROJECT / "opendetect_ustc_encoder_audit"
CONFIG_PATH = ROOT / "config.json"
PROTOCOL_MANIFEST = ROOT / "protocol_manifest.csv"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config() -> dict:
    return read_json(CONFIG_PATH)


def protocol_key(dataset: str, protocol_id: str) -> str:
    return f"{dataset}__{protocol_id}"


def cache_dir(dataset: str) -> Path:
    return ROOT / "byte_cache" / dataset


def quantile_higher(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))
