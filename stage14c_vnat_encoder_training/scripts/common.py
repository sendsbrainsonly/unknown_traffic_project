#!/usr/bin/env python3
"""Shared frozen paths and integrity gates for VNAT Stage 14C."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


STAGE14C_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE14C_ROOT.parent
STAGE14B_ROOT = PROJECT_ROOT / "stage14b_vnat_protocol_freeze"
PROTOCOL_PATH = STAGE14B_ROOT / "vnat_open_set_protocol.json"
SPLIT_MANIFEST = STAGE14B_ROOT / "vnat_split_manifest.csv"
FREEZE_HASH_PATH = STAGE14B_ROOT / "outputs" / "freeze_hashes.json"
CACHE_ROOT = STAGE14C_ROOT / "flow_image_cache"
RUNS_ROOT = STAGE14C_ROOT / "runs"
CHECKPOINT_ROOT = STAGE14C_ROOT / "checkpoints"
AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect" / "code"
EXPECTED_FREEZE_HASH = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_PROTOCOL_SHA256 = "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced"
EXPECTED_SPLIT_SHA256 = "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e"
PROTOCOL_SEEDS = (2022, 2023, 2024, 2025, 2026)
SETTINGS = ("Low", "Medium", "High")

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
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_set_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_freeze() -> dict[str, str]:
    recorded = json.loads(FREEZE_HASH_PATH.read_text(encoding="utf-8"))
    state = {
        "freeze_hash": str(recorded["freeze_hash"]),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
    }
    expected = {
        "freeze_hash": EXPECTED_FREEZE_HASH,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
    }
    if state != expected:
        raise RuntimeError(f"Stage 14B freeze mismatch: expected={expected}, actual={state}")
    return state


def load_protocol_document() -> dict[str, object]:
    verify_freeze()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload["freeze_hash"] != EXPECTED_FREEZE_HASH:
        raise RuntimeError("protocol document freeze hash mismatch")
    protocols = payload.get("protocols", [])
    if len(protocols) != 15:
        raise RuntimeError(f"expected 15 frozen protocols, found {len(protocols)}")
    identities = {(item["setting"], int(item["seed"])) for item in protocols}
    expected = {(setting, seed) for setting in SETTINGS for seed in PROTOCOL_SEEDS}
    if identities != expected:
        raise RuntimeError("frozen protocol setting/seed grid changed")
    return payload


def protocols_by_id() -> dict[str, dict[str, object]]:
    payload = load_protocol_document()
    return {str(item["protocol_id"]): item for item in payload["protocols"]}


def load_protocol(protocol_id: str) -> dict[str, object]:
    protocols = protocols_by_id()
    if protocol_id not in protocols:
        raise KeyError(protocol_id)
    protocol = protocols[protocol_id]
    known = set(map(str, protocol["known_applications"]))
    unknown = set(map(str, protocol["unknown_applications"]))
    if known & unknown:
        raise RuntimeError(f"{protocol_id}: Known/Unknown class overlap")
    if len(known | unknown) != 10:
        raise RuntimeError(f"{protocol_id}: incomplete application coverage")
    return protocol


def iter_split_rows(protocol_id: str | None = None) -> Iterator[dict[str, str]]:
    with SPLIT_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if protocol_id is None or row["protocol_id"] == protocol_id:
                yield row


def canonical_tuple(src: str, sport: str, dst: str, dport: str, protocol: str) -> str:
    first, second = sorted(((src, sport), (dst, dport)))
    return f"{protocol}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"


def tshark_flow_id(values: list[str]) -> str | None:
    if len(values) != 12:
        raise ValueError(f"expected 12 tshark fields, got {len(values)}")
    (
        ip4_src, ip6_src, ip4_dst, ip6_dst, ip4_proto, ip6_next,
        tcp_sport, tcp_dport, udp_sport, udp_dport, tcp_stream, udp_stream,
    ) = values
    src, dst = ip4_src or ip6_src, ip4_dst or ip6_dst
    if not src or not dst:
        return None
    if tcp_stream:
        return f"tcp:{tcp_stream}"
    if udp_stream:
        return f"udp:{udp_stream}"
    protocol = f"ip:{ip4_proto or ip6_next or 'unknown'}"
    text = canonical_tuple(src, "", dst, "", protocol)
    return f"other:{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_maps(protocol: dict[str, object]) -> tuple[dict[str, int], dict[int, str]]:
    known = list(map(str, protocol["known_applications"]))
    class_to_local = {name: index for index, name in enumerate(known)}
    return class_to_local, {index: name for name, index in class_to_local.items()}

