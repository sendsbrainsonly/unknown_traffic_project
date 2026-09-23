#!/usr/bin/env python3
"""Shared frozen definitions for Stage 16S Service-level open-set evaluation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
DQ3F = PROJECT / "stage15f_literature_guided_feature_benchmark" / "stage15f_dq3f_fine_service_matched_benchmark"
DQ4 = PROJECT / "stage15f_literature_guided_feature_benchmark" / "stage15f_dq4_flow_selection_filtering_attribution"
STAGE14D = PROJECT / "stage14d_vnat_frozen_open_set_evaluation"
STAGE15B = PROJECT / "stage15b_known_only_hybrid_detector"
OPEN_DETECT = PROJECT.parent / "Open-Detect"
MANIFEST = OUT / "service_unknown_protocol_manifest.csv"
CONFIG = OUT / "training_configs.json"
RUNS = OUT / "runs"
CANONICAL = OUT / "canonical_evaluation"
SERVICES = ("Chat", "Email", "File-Transfer", "P2P", "Streaming", "VoIP")
SEEDS = (2022, 2023, 2024)
METHODS = ("OD-Native", "DES-v0", "DES-v1", "H1")


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(lines: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dq3f_common():
    path = DQ3F / "scripts" / "dq3f_common.py"
    spec = importlib.util.spec_from_file_location("stage16s_dq3f_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE_CACHE: dict[str, dict[str, Any]] | None = None


def base_splits() -> dict[str, dict[str, Any]]:
    global _BASE_CACHE
    if _BASE_CACHE is not None:
        return _BASE_CACHE
    common = load_dq3f_common()
    _BASE_CACHE = {
        "known_train": common.load_matched_split("known_train"),
        "known_validation": common.load_matched_split("known_validation"),
    }
    return _BASE_CACHE


def protocol_id(unknown_service: str) -> str:
    return "loso_" + unknown_service.lower().replace("-", "_")


def protocol_rows(selected_protocol: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv(MANIFEST) if row["protocol_id"] == selected_protocol]
    if len(rows) != 3065:
        raise RuntimeError(f"{selected_protocol}: expected 3065 rows, got {len(rows)}")
    return rows


def load_role(selected_protocol: str, role: str) -> dict[str, Any]:
    rows = [row for row in protocol_rows(selected_protocol) if row["role"] == role]
    rows.sort(key=lambda row: int(row["role_row_index"]))
    if not rows:
        raise RuntimeError(f"{selected_protocol}: empty {role}")
    splits = base_splits()
    arrays = []
    metadata = []
    for row in rows:
        source = splits[row["base_role"]]
        index = int(row["base_row_index"])
        meta = source["metadata"][index]
        if meta["flow_id"] != row["flow_id"] or meta["service_label"] != row["service_label"]:
            raise RuntimeError(f"{selected_protocol}: source parity failed for {row['flow_id']}")
        arrays.append(source["data"][index])
        metadata.append(row)
    data = np.stack(arrays)
    labels = np.asarray([int(row["local_label"]) for row in rows], dtype=np.int64)
    return {"data": data, "labels": labels, "metadata": metadata}


def known_classes(selected_protocol: str) -> list[str]:
    values = sorted({row["service_label"] for row in protocol_rows(selected_protocol) if row["class_role"] == "known"})
    if len(values) != 5:
        raise RuntimeError(f"{selected_protocol}: expected five Known Services, got {values}")
    return values


def protected_paths() -> list[Path]:
    paths = [
        DQ3F / "dq3f_training_configs.json",
        DQ3F / "scripts" / "dq3f_common.py",
        DQ3F / "scripts" / "train_dq3f.py",
        DQ3F / "completion_verification.json",
        DQ4 / "dq4_subset_manifest.csv",
        PROJECT / "stage16_coarse_service_opendetect_benchmark" / "method_identity_audit.md",
        OPEN_DETECT / "code" / "model.py",
        OPEN_DETECT / "code" / "utils.py",
        OPEN_DETECT / "reproduction" / "corrected_model.py",
        OPEN_DETECT / "reproduction" / "run_reproduction.py",
        STAGE14D / "scripts" / "stage14d_common.py",
        STAGE14D / "scripts" / "evaluate_protocol.py",
        STAGE15B / "scripts" / "run_stage15b.py",
        STAGE15B / "config.json",
    ]
    paths.extend(sorted((OPEN_DETECT / "code" / "networks").rglob("*.py")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected inputs missing: {missing}")
    return sorted(set(path.resolve() for path in paths))


def protected_hashes() -> dict[str, Any]:
    paths = protected_paths()
    return {"file_count": len(paths), "files": {str(path): sha256_file(path) for path in paths}}
