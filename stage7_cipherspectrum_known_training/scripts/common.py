#!/usr/bin/env python3
"""Shared frozen paths and integrity checks for CipherSpectrum Known-only training."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scapy.utils import PcapReader


STAGE7_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE7_ROOT.parent
STAGE6_ROOT = PROJECT_ROOT / "stage6_cipherspectrum_protocol"
STAGE6_OUTPUTS = STAGE6_ROOT / "outputs"
PROTOCOL_HASH_MANIFEST = STAGE6_OUTPUTS / "protocol/protocol_hashes.sha256"
SPLIT_MANIFEST = STAGE6_OUTPUTS / "splits/split_manifest.csv"
AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
VALID_SETTINGS = ("low", "medium", "high")
TRAINING_ROLES = ("KNOWN_TRAIN", "KNOWN_VALIDATION")
ROLE_TO_PREFIX = {
    "KNOWN_TRAIN": "train",
    "KNOWN_VALIDATION": "validation",
}

sys.path.insert(0, str(AUDIT_ROOT))
from adapters.opendetect_preprocessing import (  # noqa: E402
    BYTES_PER_PACKET,
    FLOW_BYTES,
    IMAGE_SIDE,
    PACKETS_PER_FLOW,
    encode_packet,
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage6_protocol_hashes() -> dict[str, str]:
    """Verify every Stage 6 protocol artifact before any training input is read."""

    if not PROTOCOL_HASH_MANIFEST.is_file():
        raise RuntimeError(f"missing Stage 6 hash manifest: {PROTOCOL_HASH_MANIFEST}")
    verified: dict[str, str] = {}
    protocol_root = PROTOCOL_HASH_MANIFEST.parent.resolve()
    output_root = STAGE6_OUTPUTS.resolve()
    for raw_line in PROTOCOL_HASH_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, relative = raw_line.split(maxsplit=1)
        path = (protocol_root / relative).resolve()
        if not path.is_relative_to(output_root):
            raise RuntimeError(f"hash target escapes Stage 6 outputs: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Stage 6 hash mismatch: {relative}")
        verified[str(path)] = actual
    if len(verified) != 16:
        raise RuntimeError(f"expected 16 Stage 6 protocol hashes, got {len(verified)}")
    return verified


def load_fold(setting: str) -> dict[str, object]:
    if setting not in VALID_SETTINGS:
        raise ValueError(f"unknown setting: {setting}")
    verify_stage6_protocol_hashes()
    path = STAGE6_OUTPUTS / f"splits/{setting}_fold.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["setting"] != setting:
        raise RuntimeError("fold setting mismatch")
    known = [str(value) for value in payload["known_classes"]]
    unknown = [str(value) for value in payload["unknown_classes"]]
    if set(known) & set(unknown):
        raise RuntimeError("Known and Unknown classes overlap")
    if len(known) + len(unknown) != 40:
        raise RuntimeError("fold does not cover the frozen 40 classes")
    if payload["known_unknown_group_overlap"] != 0:
        raise RuntimeError("frozen fold has Known/Unknown group overlap")
    if any(int(value) for value in payload["known_group_overlap"].values()):
        raise RuntimeError("frozen fold has Known split group overlap")
    return payload


def label_maps(fold: dict[str, object]) -> tuple[dict[str, int], dict[int, str]]:
    known_classes = [str(value) for value in fold["known_classes"]]
    class_to_local = {name: index for index, name in enumerate(known_classes)}
    return class_to_local, {index: name for name, index in class_to_local.items()}


def load_setting_rows(setting: str) -> tuple[list[dict[str, str]], Counter[str]]:
    """Read metadata for one setting without opening any PCAP."""

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    with SPLIT_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] != setting:
                continue
            counts[row["role"]] += 1
            rows.append(row)
    if sum(counts.values()) != 120_000:
        raise RuntimeError(f"{setting}: expected 120000 manifest rows, got {sum(counts.values())}")
    return rows, counts


def select_training_rows(
    setting: str,
    mode: str,
    smoke_train_per_class: int,
    smoke_validation_per_class: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """Select only Known Train/Validation rows; Unknown paths are never returned."""

    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    fold = load_fold(setting)
    known = set(str(value) for value in fold["known_classes"])
    unknown = set(str(value) for value in fold["unknown_classes"])
    rows, role_counts = load_setting_rows(setting)
    selected: dict[str, list[dict[str, str]]] = {}
    rng = np.random.default_rng(seed)
    for role in TRAINING_ROLES:
        candidates = [row for row in rows if row["role"] == role]
        if any(row["class_name"] not in known for row in candidates):
            raise RuntimeError(f"{setting}/{role}: non-Known class selected")
        if any(row["class_name"] in unknown for row in candidates):
            raise RuntimeError(f"{setting}/{role}: Unknown class selected")
        if mode == "smoke":
            per_class = (
                smoke_train_per_class
                if role == "KNOWN_TRAIN"
                else smoke_validation_per_class
            )
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in candidates:
                grouped[row["class_name"]].append(row)
            sampled: list[dict[str, str]] = []
            for class_name in fold["known_classes"]:
                choices = sorted(grouped[str(class_name)], key=lambda row: row["sample_id"])
                if len(choices) < per_class:
                    raise RuntimeError(f"{setting}/{role}/{class_name}: insufficient smoke rows")
                indices = sorted(
                    int(value)
                    for value in rng.choice(len(choices), size=per_class, replace=False)
                )
                sampled.extend(choices[index] for index in indices)
            candidates = sampled
        selected[role] = sorted(
            candidates, key=lambda row: (row["class_name"], row["sample_id"])
        )
    selected_ids = {
        row["sample_id"] for role_rows in selected.values() for row in role_rows
    }
    if len(selected_ids) != sum(len(values) for values in selected.values()):
        raise RuntimeError("duplicate sample selected across training roles")
    unknown_ids = {
        row["sample_id"] for row in rows if row["role"] == "UNKNOWN_TEST"
    }
    if selected_ids & unknown_ids:
        raise RuntimeError("selected training rows overlap Unknown Test")
    metadata = {
        "setting": setting,
        "mode": mode,
        "known_classes": list(fold["known_classes"]),
        "unknown_classes": list(fold["unknown_classes"]),
        "source_role_counts": dict(sorted(role_counts.items())),
        "selected_role_counts": {
            role: len(values) for role, values in selected.items()
        },
        "unknown_sample_ids_intersect_selected": len(selected_ids & unknown_ids),
    }
    return selected, metadata


def encode_pcap_first_eight(path: Path) -> tuple[np.ndarray, int]:
    """Encode at most eight packets without loading a whole PCAP into memory."""

    frames: list[bytes] = []
    with PcapReader(str(path)) as reader:
        for packet in reader:
            frames.append(bytes(packet))
            if len(frames) == PACKETS_PER_FLOW:
                break
    if not frames:
        raise ValueError("empty PCAP")
    flat = np.zeros(FLOW_BYTES, dtype=np.uint8)
    for packet_index, raw_frame in enumerate(frames):
        start = packet_index * BYTES_PER_PACKET
        flat[start : start + BYTES_PER_PACKET] = encode_packet(raw_frame)
    return flat.reshape(IMAGE_SIDE, IMAGE_SIDE), len(frames)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def string_set_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
