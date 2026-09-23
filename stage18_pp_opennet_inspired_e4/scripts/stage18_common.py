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
STAGE17 = PROJECT / "stage17_encoder_recovery_and_open_set_pilot"
STAGE16 = PROJECT / "stage16s_service_open_set_benchmark"
CACHE = STAGE17 / "feature_cache" / "service3065_historical_inputs.npz"
MANIFEST = STAGE16 / "service_unknown_protocol_manifest.csv"
CONFIG = OUT / "configs" / "stage18_config.json"
CACHE_SHA256 = "aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff"
SERVICES = ("Email", "Streaming")
SEEDS = (2022, 2023, 2024)
VARIANT_ORDER = (
    "E4_FULL",
    "E4_TIME",
    "E4_LENGTH",
    "E4_TIME_LENGTH",
    "E4_NO_MS",
    "E4_NO_RNN",
    "E4_BASE",
)
SCORES = ("feature_distance", "msp", "energy")
ACCEPTANCE_TARGETS = (0.90, 0.95, 0.99)


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


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="higher"))


def variant_spec(name: str) -> dict[str, Any]:
    specs = read_json(CONFIG)["variants"]
    if name not in specs:
        raise KeyError(name)
    return specs[name]


def extract_packet_features(cache: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalized [log length, log IAT-us, direction] and mask."""
    x = cache["fig_x"].astype(np.float32, copy=False)
    mask = cache["fig_mask"].astype(bool, copy=False)
    length = np.log1p(np.maximum(x[:, :, 1], 0.0))
    relative_ts = np.maximum(x[:, :, 2], 0.0)
    iat = np.zeros_like(relative_ts, dtype=np.float32)
    iat[:, 1:] = np.maximum(relative_ts[:, 1:] - relative_ts[:, :-1], 0.0)
    iat *= mask
    log_iat = np.log1p(iat * 1_000_000.0)
    direction = x[:, :, 0]
    result = np.stack([length, log_iat, direction], axis=-1).astype(np.float32)
    result[~mask] = 0.0
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite packet features")
    return result, mask


def fit_robust_scaler(train_x: np.ndarray, train_mask: np.ndarray) -> dict[str, list[float]]:
    valid = train_x[train_mask]
    median = np.median(valid[:, :2], axis=0)
    q25 = np.quantile(valid[:, :2], 0.25, axis=0)
    q75 = np.quantile(valid[:, :2], 0.75, axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-8] = 1.0
    return {"median": median.tolist(), "iqr": iqr.tolist()}


def apply_robust_scaler(x: np.ndarray, mask: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    out = x.copy()
    median = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    out[:, :, :2] = np.clip((out[:, :, :2] - median) / iqr, -8.0, 8.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def select_variant_channels(x: np.ndarray, feature_mode: str) -> np.ndarray:
    mapping = {
        "time": (1,),
        "length": (0,),
        "time_length": (0, 1),
        "time_length_direction": (0, 1, 2),
    }
    if feature_mode not in mapping:
        raise KeyError(feature_mode)
    return x[:, :, mapping[feature_mode]].astype(np.float32)


def load_frozen_protocol(service: str) -> dict[str, Any]:
    if sha256_file(CACHE) != CACHE_SHA256:
        raise RuntimeError("Stage17 cache hash mismatch")
    cache = np.load(CACHE, allow_pickle=False)
    pos = {str(flow): i for i, flow in enumerate(cache["flow_ids"])}
    rows = protocol_rows(service)
    by_role = {role: [] for role in ("known_train", "known_validation", "known_test", "unknown_test")}
    for row in rows:
        by_role[row["role"]].append(row)
    for role in by_role:
        by_role[role].sort(key=lambda item: int(item["role_row_index"]))
    if any(row["service_label"] == service for role in ("known_train", "known_validation", "known_test") for row in by_role[role]):
        raise RuntimeError("unknown service leaked into a Known role")
    if any(row["service_label"] != service for row in by_role["unknown_test"]):
        raise RuntimeError("unknown role contains another service")
    indices = {role: np.asarray([pos[row["flow_id"]] for row in role_rows], dtype=np.int64) for role, role_rows in by_role.items()}
    labels = {
        role: np.asarray([int(row["local_label"]) for row in role_rows], dtype=np.int64)
        for role, role_rows in by_role.items()
        if role != "unknown_test"
    }
    raw_x, mask = extract_packet_features(cache)
    return {
        "cache": cache,
        "raw_x": raw_x,
        "mask": mask,
        "rows": by_role,
        "indices": indices,
        "labels": labels,
    }
