#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
CONFIG = OUT / "config.json"
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
STAGE21 = PROJECT / "stage21_coarse_service_ours_e3_benchmark"
CLOSED_MANIFEST = STAGE20 / "closed_service_manifest.csv"
CACHE_ROOT = STAGE21 / "feature_cache"
PRETRAINED_MODEL = PROJECT / "tf_runtime" / "code" / "models" / "pretrained_model.bin"
RUN_ROOT = OUT / "runs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    values = list(rows)
    if fieldnames is None:
        if not values:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_rows(dataset: str) -> list[dict[str, str]]:
    return [row for row in read_csv(CLOSED_MANIFEST) if row["dataset"] == dataset]


def cache_path(dataset: str) -> Path:
    return CACHE_ROOT / dataset / "e3_t8_inputs.npz"


def protected_assets() -> dict[str, Path]:
    return {
        "stage20_closed_manifest": CLOSED_MANIFEST,
        "stage21_vpn_cache": cache_path("iscx_vpn"),
        "stage21_tor_cache": cache_path("iscx_tor"),
        "official_pretrained_model": PRETRAINED_MODEL,
    }
