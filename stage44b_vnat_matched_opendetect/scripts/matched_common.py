#!/usr/bin/env python3
"""Immutable Stage 44 roles and the released Open-Detect image cache."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE44 = PROJECT / "stage44_vnat_coarse_service_loso_open_set"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
IMAGE_CACHE = PROJECT / "stage14c_vnat_encoder_training" / "flow_image_cache"
FOLDS = ("communication", "file_transfer", "remote_access", "streaming")
SEED = 2022


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_freeze() -> dict[str, str]:
    frozen = json.loads((STAGE44 / "protocol_freeze_verification.json").read_text(encoding="utf-8"))
    cache = json.loads((IMAGE_CACHE / "cache_audit.json").read_text(encoding="utf-8"))
    if frozen["status"] != "PASS" or frozen["unique_flows"] != 23_449:
        raise RuntimeError("Stage 44 protocol is not frozen and complete")
    if cache["status"] != "PASS" or cache["clean_flow_count"] != 23_449:
        raise RuntimeError("Open-Detect flow image cache is incomplete")
    actual = {
        "freeze_hash": sha256(STAGE44 / "vnat_loso_protocol.json"),
        "role_manifest_sha256": sha256(STAGE44 / "vnat_loso_role_manifest.csv"),
        "stage14b_source_manifest_sha256": sha256(STAGE14B / "vnat_split_manifest.csv"),
        "image_cache_sha256": sha256(IMAGE_CACHE / "images.npy"),
        "image_uid_cache_sha256": sha256(IMAGE_CACHE / "flow_uids.npy"),
    }
    expected = {
        "freeze_hash": frozen["protocol_sha256"],
        "role_manifest_sha256": frozen["role_manifest_sha256"],
        "stage14b_source_manifest_sha256": frozen["source_manifest_sha256"],
        "image_cache_sha256": cache["images_sha256"],
        "image_uid_cache_sha256": cache["flow_uids_sha256"],
    }
    if actual != expected:
        raise RuntimeError(f"frozen source hash mismatch: actual={actual} expected={expected}")
    return actual


def load_protocols() -> dict[str, dict[str, object]]:
    document = json.loads((STAGE44 / "vnat_loso_protocol.json").read_text(encoding="utf-8"))
    if document["seed"] != SEED or set(document["folds"]) != set(FOLDS):
        raise RuntimeError("Stage 44 fold/seed mismatch")
    result: dict[str, dict[str, object]] = {}
    for fold in FOLDS:
        source = document["folds"][fold]
        result[fold] = {
            "protocol_id": fold,
            "setting": "ServiceLOSO",
            "seed": SEED,
            "known_applications": source["known_services"],
            "unknown_applications": [fold],
        }
    return result


def role_rows(fold: str) -> dict[str, list[dict[str, str]]]:
    if fold not in FOLDS:
        raise KeyError(fold)
    source = json.loads((STAGE44 / "protocols" / fold / "protocol.json").read_text(encoding="utf-8"))
    path = STAGE44 / "protocols" / fold / "role_manifest.csv"
    if sha256(path) != source["role_manifest_sha256"]:
        raise RuntimeError(f"{fold}: frozen role-manifest hash mismatch")
    all_rows = read_csv(path)
    if len(all_rows) != 23_449 or len({row["flow_uid"] for row in all_rows}) != 23_449:
        raise RuntimeError(f"{fold}: incomplete or duplicate role manifest")
    roles = {
        role: sorted((row for row in all_rows if row["role"] == role), key=lambda row: row["flow_uid"])
        for role in ("known_train", "known_validation", "known_test", "unknown_test")
    }
    if {role: len(rows) for role, rows in roles.items()} != source["counts"]:
        raise RuntimeError(f"{fold}: role counts differ from frozen source")
    for field in ("flow_uid", "group_id", "capture_id"):
        sets = [{row[field] for row in roles[role]} for role in ("known_train", "known_validation", "known_test")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError(f"{fold}: {field} overlap between Known splits")
    if any(row["service"] == fold for role in ("known_train", "known_validation", "known_test") for row in roles[role]):
        raise RuntimeError(f"{fold}: Unknown service leaked into Known roles")
    if {row["service"] for row in roles["unknown_test"]} != {fold}:
        raise RuntimeError(f"{fold}: Unknown Test service mismatch")
    return roles


def image_index() -> dict[str, int]:
    uids = np.load(IMAGE_CACHE / "flow_uids.npy", allow_pickle=False)
    if len(uids) != 23_449:
        raise RuntimeError("Open-Detect image cache has the wrong flow count")
    result = {str(uid): index for index, uid in enumerate(uids)}
    if len(result) != 23_449:
        raise RuntimeError("Open-Detect image cache contains duplicate IDs")
    images = np.load(IMAGE_CACHE / "images.npy", mmap_mode="r", allow_pickle=False)
    if images.shape != (23_449, 1024) or images.dtype != np.uint8:
        raise RuntimeError("Open-Detect image cache shape/dtype mismatch")
    return result


def balanced_indices(ids: list[str], count: int, namespace: str) -> np.ndarray:
    ranked = sorted(range(len(ids)), key=lambda index: hashlib.sha256(
        f"2022:{namespace}:{ids[index]}".encode("utf-8")
    ).hexdigest())
    return np.asarray(sorted(ranked[:count]), dtype=np.int64)
