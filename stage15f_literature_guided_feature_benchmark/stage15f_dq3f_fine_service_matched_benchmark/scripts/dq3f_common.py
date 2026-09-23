#!/usr/bin/env python3
"""Shared frozen definitions for the DQ-3F matched-sample benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
DQ3R = PROJECT / "stage15f_literature_guided_feature_benchmark" / "stage15f_dq3r_service_label_reconstruction"
STAGE12 = PROJECT / "stage12_dual_external_validation"
OPEN_DETECT = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/Open-Detect")
PROVENANCE = DQ3R / "service_label_provenance.csv"
SPLIT_MANIFEST = STAGE12 / "protocol" / "iscx_vpn" / "split_manifest.csv"
SETTING_DIR = STAGE12 / "artifacts" / "iscx_vpn" / "protocol" / "medium"
STAGE12_CONFIG = STAGE12 / "configs" / "stage12_config.json"
TRAINING_CONFIG = DQ3R / "dq3f_preregistered_config.json"
SEEDS = [2022, 2023, 2024]
SERVICES = ["Chat", "Email", "File-Transfer", "P2P", "Streaming", "VoIP"]


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(data: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (np.ascontiguousarray(data), np.ascontiguousarray(target)):
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


def provenance_rows() -> list[dict[str, str]]:
    rows = read_csv(PROVENANCE)
    if len(rows) != 3065:
        raise RuntimeError(f"DQ-3R provenance count changed: {len(rows)} != 3065")
    return rows


def source_rows(role: str) -> list[dict[str, str]]:
    rows = [
        row for row in read_csv(SPLIT_MANIFEST)
        if row["setting"] == "medium" and row["role"] == role
    ]
    return sorted(rows, key=lambda row: (row["canonical_class"], row["flow_id_sha256"]))


def load_source_npz(role: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    name = {"known_train": "known_train.npz", "known_validation": "known_validation.npz"}[role]
    path = SETTING_DIR / name
    bundle = np.load(path, allow_pickle=False)
    data = bundle["data"]
    target = bundle["target"].astype(np.int64, copy=False)
    rows = source_rows(role)
    if data.shape != (len(rows), 32, 32) or data.dtype != np.uint8 or target.shape != (len(rows),):
        raise RuntimeError(f"source array/manifest mismatch for {role}: {data.shape}, {target.shape}, {len(rows)}")
    expected = np.asarray([int(row["eligible_label"]) for row in rows], dtype=np.int64)
    if not np.array_equal(target, expected):
        raise RuntimeError(f"source target order mismatch for {role}")
    protocol = read_json(SETTING_DIR / "protocol.json")
    if array_digest(data, target) != protocol["roles"][role]["array_sha256"]:
        raise RuntimeError(f"source array digest mismatch for {role}")
    return data, target, rows


def load_matched_split(role: str) -> dict:
    split_name = {"known_train": "known_train", "known_validation": "known_validation"}[role]
    provenance = [row for row in provenance_rows() if row["split_role"] == split_name]
    provenance_by_id = {row["flow_id"]: row for row in provenance}
    if len(provenance_by_id) != len(provenance):
        raise RuntimeError(f"duplicate DQ-3R flow_id in {role}")
    data, original_target, rows = load_source_npz(role)
    row_index = {row["flow_id_sha256"]: index for index, row in enumerate(rows)}
    if not set(provenance_by_id).issubset(row_index):
        missing = sorted(set(provenance_by_id) - set(row_index))
        raise RuntimeError(f"DQ-3R flow IDs absent from Stage12 {role}: {missing[:3]}")
    # Preserve the exact Stage12 source-array order after filtering.
    indices = [index for index, row in enumerate(rows) if row["flow_id_sha256"] in provenance_by_id]
    metadata = []
    for index in indices:
        row = rows[index]
        prov = provenance_by_id[row["flow_id_sha256"]]
        if row["canonical_class"] != prov["application_label"] or row["source_file"] != prov["source_file"]:
            raise RuntimeError(f"label/source mismatch for {row['flow_id_sha256']}")
        metadata.append({**row, **prov, "source_array_index": index})
    return {
        "role": role,
        "data": data[np.asarray(indices, dtype=np.int64)],
        "original_target": original_target[np.asarray(indices, dtype=np.int64)],
        "metadata": metadata,
        "source_total": len(rows),
        "indices": indices,
    }


def label_space(train: dict, validation: dict, task: str) -> tuple[list[str], dict[str, int]]:
    field = {"fine": "application_label", "service": "service_label"}[task]
    values = sorted({row[field] for split in (train, validation) for row in split["metadata"]})
    if task == "service" and values != SERVICES:
        raise RuntimeError(f"Service label space changed: {values}")
    return values, {name: index for index, name in enumerate(values)}


def encoded_targets(split: dict, task: str, mapping: dict[str, int]) -> np.ndarray:
    field = {"fine": "application_label", "service": "service_label"}[task]
    return np.asarray([mapping[row[field]] for row in split["metadata"]], dtype=np.int64)


def singleton_application_service_map() -> tuple[dict[str, str], dict[str, list[str]]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in provenance_rows():
        values[row["application_label"]].add(row["service_label"])
    singleton = {name: next(iter(services)) for name, services in values.items() if len(services) == 1}
    ambiguous = {name: sorted(services) for name, services in values.items() if len(services) != 1}
    return singleton, ambiguous


def support_counts() -> dict[str, Counter]:
    result = {service: Counter() for service in SERVICES}
    for row in provenance_rows():
        result[row["service_label"]][row["split_role"]] += 1
        result[row["service_label"]][f"capture::{row['capture_id']}"] += 1
    return result


def protected_input_paths() -> list[Path]:
    dq3r_names = [
        "service_label_provenance.csv", "service_label_mapping_audit.md",
        "service_train_val_support.csv", "capture_group_feasibility.md",
        "coarse_open_set_semantics.md", "dq3f_preregistered_protocol.md",
        "dq3f_preregistered_config.json", "completion_verification.json",
    ]
    fixed = [
        *(DQ3R / name for name in dq3r_names),
        SPLIT_MANIFEST, SETTING_DIR / "protocol.json", SETTING_DIR / "known_train.npz",
        SETTING_DIR / "known_validation.npz", STAGE12_CONFIG,
        STAGE12 / "scripts" / "train_pretest.py", STAGE12 / "scripts" / "stage12_common.py",
        OPEN_DETECT / "reproduction" / "corrected_model.py",
        OPEN_DETECT / "reproduction" / "run_reproduction.py",
        OPEN_DETECT / "code" / "model.py", OPEN_DETECT / "code" / "utils.py",
    ]
    fixed.extend(sorted((OPEN_DETECT / "code" / "networks").rglob("*.py")))
    missing = [str(path) for path in fixed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected input missing: {missing}")
    return sorted(set(path.resolve() for path in fixed))


def hash_protected_inputs() -> dict:
    return {
        "file_count": len(protected_input_paths()),
        "files": {str(path): sha256_file(path) for path in protected_input_paths()},
    }

