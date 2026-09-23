#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE16 = PROJECT / "stage16s_service_open_set_benchmark"
MANIFEST = STAGE16 / "service_unknown_protocol_manifest.csv"
CACHE = OUT / "feature_cache" / "service3065_historical_inputs.npz"
SERVICES = ("Email", "Streaming")
SEEDS = (2022, 2023, 2024)
ENCODERS = ("E0", "E1", "E2", "E3")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    values = list(rows)
    if fieldnames is None:
        if not values:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def protocol_id(service: str) -> str:
    return "loso_" + service.lower().replace("-", "_")


def protocol_rows(service: str) -> list[dict[str, str]]:
    pid = protocol_id(service)
    rows = [row for row in read_csv(MANIFEST) if row["protocol_id"] == pid]
    if len(rows) != 3065:
        raise RuntimeError(f"{pid}: expected 3065 rows, got {len(rows)}")
    return rows


def percentile95(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="higher"))

