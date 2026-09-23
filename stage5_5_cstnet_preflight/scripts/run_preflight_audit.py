#!/usr/bin/env python3
"""Run the Stage 5.5 CSTNET final preflight without training or Unknown scoring.

The executable has three deliberately separated read-only audit paths:

* replay frozen Stage 5 DGSB scores on USTC Known Validation/Test only;
* encode a bounded CSTNET PCAP sample with the already-audited Open-Detect
  byte adapter and inspect only those resulting 32x32 inputs;
* summarize the already-frozen CSTNET class splits and deterministic K2
  component-size stress scenarios.

No estimator exposes ``fit`` in this module and no USTC Unknown artifact is
allowed in the input ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import PcapReader


STAGE55_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE55_ROOT.parent
STAGE5_ROOT = PROJECT_ROOT / "stage5_dual_gate_boundary"
STAGE3_ROOT = PROJECT_ROOT / "stage3_unknown_utility"
OPENDETECT_AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
CSTNET_ROOT = Path(
    "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CSTNET-TLS1.3"
)
OUTPUT_ROOT = STAGE55_ROOT / "outputs"
RULE_ROOT = STAGE5_ROOT / "outputs" / "rule_freeze"
PROTOCOL_ROOT = STAGE5_ROOT / "outputs" / "cstnet_protocol"
SETTINGS = ("A-1", "A-2", "A-3")
FOLDS = ("low", "medium", "high")
SAMPLE_SEED = 20260913
MAX_CLASSES = 30
MAX_PCAPS_PER_CLASS = 20
MASS_UNRELIABLE_RATE = 0.25

sys.path.insert(0, str(STAGE5_ROOT / "scripts"))
from run_known_only_sanity import score_k2, sha256_file, transform  # noqa: E402

sys.path.insert(0, str(OPENDETECT_AUDIT_ROOT))
from adapters.opendetect_preprocessing import (  # noqa: E402
    BYTES_PER_PACKET,
    FLOW_BYTES,
    HEADER_BYTES,
    IMAGE_SIDE,
    PACKETS_PER_FLOW,
    PAYLOAD_BYTES,
    encode_packet,
)


GENERIC_DOMAIN_LABELS = {
    "www", "com", "net", "org", "cn", "io", "ru", "la", "co", "gov", "edu",
}
DOMAIN_PATTERN = re.compile(
    rb"(?i)(?:[a-z0-9](?:[a-z0-9-]{0,62})\.)+"
    rb"(?:com|net|org|cn|io|ru|la|so|co|gov|edu)(?:\.[a-z]{2})?"
)
FORBIDDEN_USTC_INPUT_NAMES = {
    "test_unknown_mu.parquet",
    "frozen_final_predictions.parquet",
    "posthoc_unknown_predictions.parquet",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class InputLedger:
    """Record every executable input and reject forbidden USTC Unknown artifacts."""

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def add(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved.name.lower() in FORBIDDEN_USTC_INPUT_NAMES:
            raise RuntimeError(f"forbidden USTC Unknown/result input: {resolved}")
        self.paths.append(resolved)
        return resolved

    def unique(self) -> list[Path]:
        return sorted(set(self.paths))


def read_json(path: Path, ledger: InputLedger) -> object:
    return json.loads(ledger.add(path).read_text(encoding="utf-8"))


def verify_sha256_list(path: Path, ledger: InputLedger) -> list[dict[str, object]]:
    root = path.parent
    rows: list[dict[str, object]] = []
    for line in ledger.add(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        target = ledger.add(root / relative)
        actual = sha256_file(target)
        rows.append(
            {
                "manifest": str(path.resolve()),
                "file": str(target),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual == expected else "FAIL",
            }
        )
    return rows


def verify_frozen_inputs(ledger: InputLedger) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    hash_rows = verify_sha256_list(RULE_ROOT / "dgsb_rule.sha256", ledger)
    hash_rows.extend(verify_sha256_list(PROTOCOL_ROOT / "split_hashes.sha256", ledger))
    if not hash_rows or any(row["status"] != "PASS" for row in hash_rows):
        raise RuntimeError("frozen rule/protocol SHA-256 verification failed")

    rule = read_json(RULE_ROOT / "dgsb_rule.json", ledger)
    protocol = read_json(PROTOCOL_ROOT / "cstnet_open_set_protocol.json", ledger)
    metadata = read_json(RULE_ROOT / "known_only_run_metadata.json", ledger)
    expected_known = {
        (str(row["setting"]), Path(str(row["file"])).name): str(row["sha256"])
        for row in metadata["verified_known_asset_hashes"]
    }
    required = (
        "standard_scaler.joblib",
        "pca64.joblib",
        "multi_full_k2_models.joblib",
        "val_known_mu.parquet",
        "test_known_mu.parquet",
    )
    for setting in SETTINGS:
        artifact_root = STAGE3_ROOT / "artifacts" / setting
        for filename in required:
            target = ledger.add(artifact_root / filename)
            expected = expected_known.get((setting, filename))
            actual = sha256_file(target)
            status = "PASS" if expected is not None and actual == expected else "FAIL"
            hash_rows.append(
                {
                    "manifest": str((RULE_ROOT / "known_only_run_metadata.json").resolve()),
                    "file": str(target),
                    "expected_sha256": expected or "MISSING",
                    "actual_sha256": actual,
                    "status": status,
                }
            )
    if any(row["status"] != "PASS" for row in hash_rows):
        raise RuntimeError("frozen Known asset SHA-256 verification failed")
    if rule["model"] != {
        "K": 2,
        "covariance_type": "full",
        "pca_dimension": 64,
        "refit_performed": False,
        "reg_covar": 0.001,
    }:
        raise RuntimeError("frozen DGSB model contract changed")
    if rule["ustc_unknown_inputs_accessed"] != 0:
        raise RuntimeError("frozen rule has nonzero USTC Unknown access")
    if protocol["eligibility"]["rule"] != "sample_count >= 100":
        raise RuntimeError("frozen CSTNET eligibility rule changed")
    for fold in FOLDS:
        item = protocol["class_holdout_folds"][fold]
        if not item["group_disjoint"] or int(item["active_group_overlap_count"]) != 0:
            raise RuntimeError(f"{fold}: frozen group split no longer passes")
    return rule, protocol, hash_rows


def dgsb_audit(rule: dict[str, object], ledger: InputLedger) -> dict[str, object]:
    gate_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    consistency_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for setting in SETTINGS:
        artifact_root = STAGE3_ROOT / "artifacts" / setting
        scaler = joblib.load(ledger.add(artifact_root / "standard_scaler.joblib"))
        pca = joblib.load(ledger.add(artifact_root / "pca64.joblib"))
        models = joblib.load(ledger.add(artifact_root / "multi_full_k2_models.joblib"))
        frozen = rule["ustc_known_only_setting_thresholds"][setting]
        tau_global = float(frozen["tau_global_dual"])
        threshold_map = {
            (str(row["class"]), int(row["component"])): float(row["threshold"])
            for row in frozen["component_thresholds"]
        }

        for split_label, filename in (
            ("Known Validation", "val_known_mu.parquet"),
            ("Known Test", "test_known_mu.parquet"),
        ):
            frame = pd.read_parquet(ledger.add(artifact_root / filename))
            if set(frame["known_or_unknown"].astype(str)) != {"Known"}:
                raise RuntimeError(f"{setting}/{split_label}: non-Known input row")
            values = transform(frame, scaler, pca)
            names, class_scores, local_scores = score_k2(models, values)
            local_thresholds = np.asarray(
                [threshold_map[(name, component)] for name in names for component in range(2)],
                dtype=np.float64,
            )
            global_scores = class_scores.max(axis=1)
            global_pass = global_scores >= tau_global
            local_winner = np.argmax(local_scores, axis=1)
            local_pass = (
                local_scores[np.arange(len(local_scores)), local_winner]
                >= local_thresholds[local_winner]
            )
            both_pass = global_pass & local_pass
            global_fail_local_pass = (~global_pass) & local_pass
            global_pass_local_fail = global_pass & (~local_pass)
            both_fail = (~global_pass) & (~local_pass)
            n = len(frame)
            counts = {
                "both_pass": int(both_pass.sum()),
                "global_fail_local_pass": int(global_fail_local_pass.sum()),
                "global_pass_local_fail": int(global_pass_local_fail.sum()),
                "both_fail": int(both_fail.sum()),
            }
            gate_rows.append(
                {
                    "setting": setting,
                    "split": split_label,
                    "n_samples": n,
                    **counts,
                    **{f"{key}_ratio": value / n for key, value in counts.items()},
                }
            )
            effect_rows.append(
                {
                    "setting": setting,
                    "split": split_label,
                    "n_samples": n,
                    "global_exclusive_rejection_count": counts["global_fail_local_pass"],
                    "global_exclusive_rejection_rate": counts["global_fail_local_pass"] / n,
                    "local_exclusive_rejection_count": counts["global_pass_local_fail"],
                    "local_exclusive_rejection_rate": counts["global_pass_local_fail"] / n,
                    "both_fail_count": counts["both_fail"],
                    "both_fail_rate": counts["both_fail"] / n,
                    "global_gate_has_independent_effect": counts["global_fail_local_pass"] > 0,
                }
            )

            global_winner = np.argmax(class_scores, axis=1)
            global_names = np.asarray(names, dtype=object)[global_winner]
            local_names = np.asarray(names, dtype=object)[local_winner // 2]
            same = global_names == local_names
            mismatch_rate = float((~same).mean())
            if mismatch_rate < 0.01:
                category = "BASICALLY_SAME_CLASS"
            elif mismatch_rate <= 0.05:
                category = "MODERATE_MISMATCH"
            else:
                category = "STRUCTURAL_MISMATCH"
            consistency_rows.append(
                {
                    "setting": setting,
                    "split": split_label,
                    "n": n,
                    "same_class_count": int(same.sum()),
                    "different_class_count": int((~same).sum()),
                    "different_class_rate": mismatch_rate,
                    "mismatch_category": category,
                }
            )
            pairs = Counter(zip(global_names.tolist(), local_names.tolist()))
            for (global_name, local_name), count in sorted(pairs.items()):
                confusion_rows.append(
                    {
                        "setting": setting,
                        "split": split_label,
                        "y_global": global_name,
                        "y_local": local_name,
                        "count": count,
                        "ratio": count / n,
                        "same_class": global_name == local_name,
                    }
                )

    out = OUTPUT_ROOT / "dgsb_gate_audit"
    write_csv(out / "gate_transition_counts.csv", gate_rows)
    write_csv(out / "global_gate_effectiveness.csv", effect_rows)
    write_csv(out / "class_consistency.csv", consistency_rows)
    write_csv(out / "global_local_class_confusion.csv", confusion_rows)
    all_nonzero = all(bool(row["global_gate_has_independent_effect"]) for row in effect_rows)
    max_mismatch = max(float(row["different_class_rate"]) for row in consistency_rows)
    lines = [
        "# DGSB Gate Contribution Audit",
        "",
        "Only frozen USTC Known Validation and Known Test latent files were read. No USTC Unknown file was read, no estimator was fitted, and all Stage 5 thresholds were replayed unchanged.",
        "",
        "## Gate Effectiveness",
        "",
        "| Setting | Split | Global-exclusive rejection | Local-exclusive rejection | Both fail |",
        "|---|---|---:|---:|---:|",
    ]
    for row in effect_rows:
        lines.append(
            f"| {row['setting']} | {row['split']} | {row['global_exclusive_rejection_count']} "
            f"({100 * float(row['global_exclusive_rejection_rate']):.4f}%) | "
            f"{row['local_exclusive_rejection_count']} ({100 * float(row['local_exclusive_rejection_rate']):.4f}%) | "
            f"{row['both_fail_count']} ({100 * float(row['both_fail_rate']):.4f}%) |"
        )
    lines.extend(
        [
            "",
            f"Global Gate has a nonzero independent effect in every setting/split cell: **{'YES' if all_nonzero else 'NO'}**.",
            "",
            "## Global-vs-Local Class Consistency",
            "",
            "| Setting | Split | Different count | Different rate | Category |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in consistency_rows:
        lines.append(
            f"| {row['setting']} | {row['split']} | {row['different_class_count']} | "
            f"{100 * float(row['different_class_rate']):.4f}% | {row['mismatch_category']} |"
        )
    lines.extend(
        [
            "",
            f"Maximum observed class mismatch: **{100 * max_mismatch:.4f}%**.",
        ]
    )
    (out / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "global_gate_nonzero_all_cells": all_nonzero,
        "max_mismatch_rate": max_mismatch,
        "gate_rows": gate_rows,
        "effect_rows": effect_rows,
        "consistency_rows": consistency_rows,
    }


def parse_client_hello_body(body: bytes) -> list[str] | None:
    """Return SNI values from one complete TLS ClientHello body, or None if malformed."""
    try:
        pos = 2 + 32
        session_len = body[pos]
        pos += 1 + session_len
        cipher_len = int.from_bytes(body[pos : pos + 2], "big")
        pos += 2 + cipher_len
        compression_len = body[pos]
        pos += 1 + compression_len
        extensions_len = int.from_bytes(body[pos : pos + 2], "big")
        pos += 2
        end = pos + extensions_len
        if end > len(body):
            return None
        names: list[str] = []
        while pos + 4 <= end:
            extension_type = int.from_bytes(body[pos : pos + 2], "big")
            extension_len = int.from_bytes(body[pos + 2 : pos + 4], "big")
            pos += 4
            extension_end = pos + extension_len
            if extension_end > end:
                return None
            if extension_type == 0:
                if extension_len < 2:
                    return None
                list_len = int.from_bytes(body[pos : pos + 2], "big")
                cursor = pos + 2
                list_end = cursor + list_len
                if list_end > extension_end:
                    return None
                while cursor + 3 <= list_end:
                    name_type = body[cursor]
                    name_len = int.from_bytes(body[cursor + 1 : cursor + 3], "big")
                    cursor += 3
                    name_end = cursor + name_len
                    if name_end > list_end:
                        return None
                    if name_type == 0:
                        names.append(body[cursor:name_end].decode("ascii").lower())
                    cursor = name_end
            pos = extension_end
        return names
    except (IndexError, UnicodeDecodeError):
        return None


def parse_tls_sni(blobs: Iterable[bytes]) -> tuple[bool, list[str]]:
    """Deterministically scan complete TLS records for complete ClientHello messages."""
    parser_success = False
    names: set[str] = set()
    for blob in blobs:
        index = 0
        while index + 9 <= len(blob):
            if blob[index] != 0x16 or blob[index + 1] != 0x03:
                index += 1
                continue
            record_len = int.from_bytes(blob[index + 3 : index + 5], "big")
            record_end = index + 5 + record_len
            if record_len <= 0 or record_end > len(blob):
                index += 1
                continue
            payload = blob[index + 5 : record_end]
            cursor = 0
            while cursor + 4 <= len(payload):
                handshake_type = payload[cursor]
                handshake_len = int.from_bytes(payload[cursor + 1 : cursor + 4], "big")
                handshake_end = cursor + 4 + handshake_len
                if handshake_end > len(payload):
                    break
                if handshake_type == 1:
                    parsed = parse_client_hello_body(payload[cursor + 4 : handshake_end])
                    if parsed is not None:
                        parser_success = True
                        names.update(parsed)
                cursor = handshake_end
            index = record_end
    return parser_success, sorted(names)


def contiguous_tcp_blobs(segments: dict[tuple[object, ...], list[tuple[int, bytes]]]) -> list[bytes]:
    blobs: list[bytes] = []
    for values in segments.values():
        current = bytearray()
        next_seq: int | None = None
        for seq, payload in sorted(values, key=lambda item: item[0]):
            if not payload:
                continue
            if next_seq is None or seq > next_seq:
                if current:
                    blobs.append(bytes(current))
                current = bytearray(payload)
                next_seq = seq + len(payload)
            elif seq == next_seq:
                current.extend(payload)
                next_seq += len(payload)
            else:
                overlap = next_seq - seq
                if overlap < len(payload):
                    current.extend(payload[overlap:])
                    next_seq += len(payload) - overlap
        if current:
            blobs.append(bytes(current))
    return blobs


def select_sample_classes(
    eligible: Sequence[str], role_map: dict[tuple[str, str], str], seed: int = SAMPLE_SEED
) -> tuple[list[str], dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    for fold in FOLDS:
        for role in ("known", "unknown"):
            candidates = sorted(name for name in eligible if role_map[(fold, name)] == role)
            order = rng.permutation(len(candidates))
            picked = next((candidates[int(i)] for i in order if candidates[int(i)] not in selected), None)
            if picked is None:
                raise RuntimeError(f"cannot cover sampling role {fold}/{role}")
            selected.append(picked)
            reasons[picked].append(f"coverage:{fold}:{role}")
    remaining = sorted(set(eligible) - set(selected))
    for index in rng.permutation(len(remaining)):
        if len(selected) >= min(MAX_CLASSES, len(eligible)):
            break
        picked = remaining[int(index)]
        selected.append(picked)
        reasons[picked].append("deterministic_fill")
    return selected, dict(reasons)


@dataclass
class PcapEncoding:
    image: np.ndarray | None
    packets_found: int
    packets_used: int
    source_bytes_retained: int
    input_valid: bool
    failure_reason: str
    raw_blob: bytes
    raw_tcp_blobs: list[bytes]
    retained_tcp_blobs: list[bytes]
    ip_fields_checked: int
    ip_fields_masked: int
    port_fields_checked: int
    port_fields_visible: int
    tls_header_visible: bool
    mapping_rows: list[dict[str, object]]


def encode_pcap(path: Path, sample_id: str) -> PcapEncoding:
    raw_frames: list[bytes] = []
    first_frames: list[bytes] = []
    raw_segments: dict[tuple[object, ...], list[tuple[int, bytes]]] = defaultdict(list)
    retained_segments: dict[tuple[object, ...], list[tuple[int, bytes]]] = defaultdict(list)
    packets_found = 0
    with PcapReader(str(path)) as reader:
        for packet in reader:
            frame = bytes(packet)
            packets_found += 1
            raw_frames.append(frame)
            if len(first_frames) < PACKETS_PER_FLOW:
                first_frames.append(frame)
            if IP in packet and TCP in packet and Raw in packet:
                ip = packet[IP]
                tcp = packet[TCP]
                payload = bytes(packet[Raw])
                key = (str(ip.src), str(ip.dst), int(tcp.sport), int(tcp.dport))
                raw_segments[key].append((int(tcp.seq), payload))

    image = np.zeros(FLOW_BYTES, dtype=np.uint8)
    mapping_rows: list[dict[str, object]] = []
    source_bytes_retained = 0
    ip_checked = ip_masked = port_checked = port_visible = 0
    tls_header_visible = False
    try:
        if not first_frames:
            raise ValueError("empty PCAP")
        for packet_index, raw_frame in enumerate(first_frames):
            packet = Ether(raw_frame)
            if IP not in packet:
                raise ValueError(f"packet {packet_index} has no IPv4 layer")
            encoded = encode_packet(raw_frame)
            start = packet_index * BYTES_PER_PACKET
            image[start : start + BYTES_PER_PACKET] = encoded
            ip = packet[IP]
            ip_checked += 1
            if bytes(encoded[12:20]) == b"\x00" * 8:
                ip_masked += 1
            ihl = int(encoded[0] & 0x0F) * 4
            if (TCP in packet or UDP in packet) and ihl + 4 <= HEADER_BYTES:
                layer = packet[TCP] if TCP in packet else packet[UDP]
                expected_ports = int(layer.sport).to_bytes(2, "big") + int(layer.dport).to_bytes(2, "big")
                port_checked += 1
                if bytes(encoded[ihl : ihl + 4]) == expected_ports:
                    port_visible += 1

            masked = packet.copy()
            masked[IP].src = "0.0.0.0"
            masked[IP].dst = "0.0.0.0"
            header_hex = bytes(masked[IP]).hex()
            payload = bytes(packet[Raw]) if Raw in packet else b""
            payload_hex = payload.hex()
            if payload_hex:
                header_hex = header_hex.replace(payload_hex, "")
            header_source = bytes.fromhex(header_hex)
            header_retained = min(len(header_source), HEADER_BYTES)
            payload_retained = min(len(payload), PAYLOAD_BYTES)
            source_bytes_retained += header_retained + payload_retained
            mapping_rows.extend(
                [
                    {
                        "sample_id": sample_id,
                        "packet_index": packet_index,
                        "region": "header",
                        "input_offset_start": start,
                        "input_offset_end_exclusive": start + HEADER_BYTES,
                        "source": "masked IPv4 serialization with Scapy Raw bytes removed",
                        "source_bytes_retained": header_retained,
                        "zero_padding_bytes": HEADER_BYTES - header_retained,
                    },
                    {
                        "sample_id": sample_id,
                        "packet_index": packet_index,
                        "region": "payload",
                        "input_offset_start": start + HEADER_BYTES,
                        "input_offset_end_exclusive": start + BYTES_PER_PACKET,
                        "source": "first 48 bytes of Scapy Raw payload",
                        "source_bytes_retained": payload_retained,
                        "zero_padding_bytes": PAYLOAD_BYTES - payload_retained,
                    },
                ]
            )
            if payload_retained:
                retained = payload[:payload_retained]
                tls_header_visible = tls_header_visible or bool(
                    len(retained) >= 3 and retained[0] in (0x14, 0x15, 0x16, 0x17) and retained[1] == 0x03
                )
                if TCP in packet:
                    tcp = packet[TCP]
                    key = (str(ip.src), str(ip.dst), int(tcp.sport), int(tcp.dport))
                    retained_segments[key].append((int(tcp.seq), retained))
        for packet_index in range(len(first_frames), PACKETS_PER_FLOW):
            start = packet_index * BYTES_PER_PACKET
            mapping_rows.append(
                {
                    "sample_id": sample_id,
                    "packet_index": packet_index,
                    "region": "flow_padding",
                    "input_offset_start": start,
                    "input_offset_end_exclusive": start + BYTES_PER_PACKET,
                    "source": "zero padding for absent packet",
                    "source_bytes_retained": 0,
                    "zero_padding_bytes": BYTES_PER_PACKET,
                }
            )
        return PcapEncoding(
            image=image.reshape(IMAGE_SIDE, IMAGE_SIDE),
            packets_found=packets_found,
            packets_used=len(first_frames),
            source_bytes_retained=source_bytes_retained,
            input_valid=True,
            failure_reason="",
            raw_blob=b"".join(raw_frames),
            raw_tcp_blobs=contiguous_tcp_blobs(raw_segments),
            retained_tcp_blobs=contiguous_tcp_blobs(retained_segments),
            ip_fields_checked=ip_checked,
            ip_fields_masked=ip_masked,
            port_fields_checked=port_checked,
            port_fields_visible=port_visible,
            tls_header_visible=tls_header_visible,
            mapping_rows=mapping_rows,
        )
    except Exception as exc:
        return PcapEncoding(
            image=None,
            packets_found=packets_found,
            packets_used=min(len(first_frames), PACKETS_PER_FLOW),
            source_bytes_retained=source_bytes_retained,
            input_valid=False,
            failure_reason=f"{type(exc).__name__}:{exc}",
            raw_blob=b"".join(raw_frames),
            raw_tcp_blobs=contiguous_tcp_blobs(raw_segments),
            retained_tcp_blobs=[],
            ip_fields_checked=ip_checked,
            ip_fields_masked=ip_masked,
            port_fields_checked=port_checked,
            port_fields_visible=port_visible,
            tls_header_visible=tls_header_visible,
            mapping_rows=[],
        )


def class_tokens(name: str) -> list[str]:
    values = re.split(r"[.\-_]+", name.lower())
    return sorted({value for value in values if value not in GENERIC_DOMAIN_LABELS and len(value) >= 4})


def cstnet_input_audit(
    protocol: dict[str, object], ledger: InputLedger
) -> dict[str, object]:
    manifest_path = ledger.add(PROTOCOL_ROOT / "split_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    eligible = [str(name) for name in protocol["eligibility"]["eligible_classes"]]
    role_map: dict[tuple[str, str], str] = {}
    for fold in FOLDS:
        subset = manifest[manifest["fold"] == fold]
        for name in eligible:
            roles = set(subset.loc[subset["class_name"] == name, "class_role"].astype(str))
            if len(roles) != 1:
                raise RuntimeError(f"{fold}/{name}: ambiguous class role {roles}")
            role_map[(fold, name)] = roles.pop()
    selected, reasons = select_sample_classes(eligible, role_map)
    coverage = {
        f"{fold}_{role}": any(role_map[(fold, name)] == role for name in selected)
        for fold in FOLDS
        for role in ("known", "unknown")
    }
    if not all(coverage.values()):
        raise RuntimeError(f"sample role coverage failed: {coverage}")

    rng = np.random.default_rng(SAMPLE_SEED)
    unique_paths = (
        manifest.loc[manifest["eligible_class"] == True, ["class_name", "relative_path"]]  # noqa: E712
        .drop_duplicates()
        .groupby("class_name")["relative_path"]
        .apply(list)
        .to_dict()
    )
    samples: list[tuple[str, str]] = []
    for name in selected:
        paths = sorted(str(value) for value in unique_paths[name])
        count = min(MAX_PCAPS_PER_CLASS, len(paths))
        indices = sorted(int(value) for value in rng.choice(len(paths), size=count, replace=False))
        samples.extend((name, paths[index]) for index in indices)

    raw_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    valid_images: list[np.ndarray] = []
    valid_ids: list[str] = []
    endpoint_totals = Counter()

    for sample_number, (name, relative_path) in enumerate(samples):
        sample_id = f"cstnet-{sample_number:04d}"
        pcap_path = ledger.add(CSTNET_ROOT / relative_path)
        encoded = encode_pcap(pcap_path, sample_id)
        image_bytes = encoded.image.reshape(-1).tobytes() if encoded.image is not None else b""
        raw_parser_success, raw_sni_names = parse_tls_sni(encoded.raw_tcp_blobs)
        input_parser_success, input_sni_names = parse_tls_sni(encoded.retained_tcp_blobs)
        lower_raw = encoded.raw_blob.lower()
        lower_input = image_bytes.lower()
        domain_bytes = name.lower().encode("ascii")
        raw_domain_visible = domain_bytes in lower_raw
        domain_visible = bool(image_bytes) and domain_bytes in lower_input
        sni_visible_names = sorted(
            sni for sni in raw_sni_names if sni.encode("ascii", errors="ignore") in lower_input
        )
        sni_visible = bool(sni_visible_names or input_sni_names)
        hostname_visible = sni_visible or domain_visible
        tokens = class_tokens(name)
        visible_tokens = sorted(token for token in tokens if token.encode("ascii") in lower_input)
        raw_visible_tokens = sorted(token for token in tokens if token.encode("ascii") in lower_raw)
        obvious_domains = sorted(
            {match.group(0).decode("ascii", errors="ignore").lower() for match in DOMAIN_PATTERN.finditer(image_bytes)}
        )
        token_visible = bool(visible_tokens or obvious_domains)
        direct_domain_visible = domain_visible or sni_visible or hostname_visible
        total_visible = direct_domain_visible or token_visible
        matched_candidates = ([name] if domain_visible else []) + sni_visible_names + visible_tokens + obvious_domains
        matched = matched_candidates[0] if matched_candidates else ""
        match_offset = lower_input.find(matched.encode("ascii", errors="ignore")) if matched else -1

        array_index = -1
        if encoded.input_valid and encoded.image is not None:
            array_index = len(valid_images)
            valid_images.append(encoded.image)
            valid_ids.append(sample_id)
            mapping_rows.extend(encoded.mapping_rows)
        roles = {f"{fold}_role": role_map[(fold, name)] for fold in FOLDS}
        raw_rows.append(
            {
                "sample_id": sample_id,
                "class_name": name,
                "pcap": str(pcap_path),
                "relative_path": relative_path,
                "selection_reason": ";".join(reasons[name]),
                **roles,
                "packets_found": encoded.packets_found,
                "packets_used": encoded.packets_used,
                "bytes_used": encoded.packets_used * BYTES_PER_PACKET,
                "source_bytes_retained": encoded.source_bytes_retained,
                "input_tensor_bytes": FLOW_BYTES,
                "input_valid": encoded.input_valid,
                "failure_reason": encoded.failure_reason,
                "input_array_index": array_index,
                "input_sha256": hashlib.sha256(image_bytes).hexdigest() if image_bytes else "",
                "input_bytes_hex": image_bytes.hex(),
            }
        )
        leakage_rows.append(
            {
                "sample_id": sample_id,
                "class_name": name,
                "pcap": str(pcap_path),
                "raw_pcap_domain_visible": raw_domain_visible,
                "raw_pcap_class_token_visible": bool(raw_visible_tokens),
                "domain_string_visible": domain_visible,
                "sni_visible": sni_visible,
                "hostname_visible": hostname_visible,
                "class_token_visible": bool(visible_tokens),
                "obvious_textual_identifier_visible": bool(obvious_domains),
                "direct_domain_visible": direct_domain_visible,
                "domain_or_token_visible": total_visible,
                "matched_string": matched,
                "match_offset": match_offset,
                "inside_model_input": direct_domain_visible,
                "parser_success": raw_parser_success,
                "input_parser_success": input_parser_success,
                "raw_sni_values": json.dumps(raw_sni_names, ensure_ascii=False),
                "input_sni_values": json.dumps(input_sni_names, ensure_ascii=False),
                "visible_class_tokens": json.dumps(visible_tokens, ensure_ascii=False),
                "obvious_domains_in_input": json.dumps(obvious_domains, ensure_ascii=False),
                "input_valid": encoded.input_valid,
            }
        )
        endpoint_totals["valid"] += int(encoded.input_valid)
        endpoint_totals["ip_checked"] += encoded.ip_fields_checked
        endpoint_totals["ip_masked"] += encoded.ip_fields_masked
        endpoint_totals["port_checked"] += encoded.port_fields_checked
        endpoint_totals["port_visible"] += encoded.port_fields_visible
        endpoint_totals["tls_header_visible_inputs"] += int(encoded.input_valid and encoded.tls_header_visible)

    out = OUTPUT_ROOT / "cstnet_input_audit"
    write_csv(out / "raw_input_manifest.csv", raw_rows)
    write_csv(out / "domain_leakage_audit.csv", leakage_rows)
    write_csv(out / "input_byte_source_mapping.csv", mapping_rows)
    np.savez_compressed(
        out / "sampled_open_detect_inputs.npz",
        images=np.stack(valid_images).astype(np.uint8) if valid_images else np.empty((0, 32, 32), np.uint8),
        sample_ids=np.asarray(valid_ids, dtype="U32"),
    )

    valid = [row for row in leakage_rows if bool(row["input_valid"])]
    token_rows: list[dict[str, object]] = []
    for scope, name, rows in [("overall", "__ALL__", valid)] + [
        ("class", class_name, [row for row in valid if row["class_name"] == class_name])
        for class_name in selected
    ]:
        n = len(rows)
        def metric(field: str) -> tuple[int, float]:
            count = sum(bool(row[field]) for row in rows)
            return count, count / n if n else float("nan")
        domain_count, domain_rate = metric("domain_string_visible")
        sni_count, sni_rate = metric("sni_visible")
        direct_count, direct_rate = metric("direct_domain_visible")
        token_count, token_rate = metric("class_token_visible")
        total_count, total_rate = metric("domain_or_token_visible")
        token_rows.append(
            {
                "scope": scope,
                "class_name": name,
                "valid_inputs": n,
                "full_domain_visible_count": domain_count,
                "full_domain_visible_rate": domain_rate,
                "sni_visible_count": sni_count,
                "sni_visible_rate": sni_rate,
                "direct_domain_visible_count": direct_count,
                "direct_domain_visible_rate": direct_rate,
                "class_token_visible_count": token_count,
                "class_token_visible_rate": token_rate,
                "domain_or_token_visible_count": total_count,
                "domain_or_token_visible_rate": total_rate,
            }
        )
    write_csv(out / "token_leakage_summary.csv", token_rows)
    overall = token_rows[0]
    direct_rate = float(overall["direct_domain_visible_rate"])
    if direct_rate >= 0.50:
        leakage_level = "HIGH_DIRECT_DOMAIN_LEAKAGE"
    elif direct_rate >= 0.10:
        leakage_level = "MODERATE_DOMAIN_LEAKAGE"
    else:
        leakage_level = "LOW_DIRECT_DOMAIN_LEAKAGE"
    ip_status = (
        "MASKED"
        if endpoint_totals["ip_checked"] > 0
        and endpoint_totals["ip_checked"] == endpoint_totals["ip_masked"]
        else "VISIBLE"
    )
    port_status = (
        "VISIBLE" if endpoint_totals["port_visible"] > 0 else "MASKED"
    )
    endpoint_lines = [
        "# Endpoint Input Audit",
        "",
        "This audit reuses `opendetect_ustc_encoder_audit/adapters/opendetect_preprocessing.py`: first 8 packets, each represented by 80 masked-IPv4/header bytes plus 48 Scapy Raw payload bytes, then reshaped to 32x32.",
        "",
        f"- IP: **{ip_status}** ({endpoint_totals['ip_masked']}/{endpoint_totals['ip_checked']} encoded packet IP source/destination fields were zero).",
        f"- Port: **{port_status}** ({endpoint_totals['port_visible']}/{endpoint_totals['port_checked']} eligible transport headers retained exact source/destination ports).",
        f"- SNI/domain: **{100 * direct_rate:.4f}%** direct visibility among valid model inputs.",
        f"- Full class-domain visibility: **{100 * float(overall['full_domain_visible_rate']):.4f}%**.",
        f"- Parsed-SNI visibility: **{100 * float(overall['sni_visible_rate']):.4f}%**.",
        f"- TLS record/header prefix visible: **{endpoint_totals['tls_header_visible_inputs']}/{endpoint_totals['valid']}** valid inputs.",
        "",
        "The port result is a representation-level visibility finding, not a demonstrated classifier shortcut.",
    ]
    (out / "endpoint_input_audit.md").write_text("\n".join(endpoint_lines) + "\n", encoding="utf-8")
    if leakage_level == "HIGH_DIRECT_DOMAIN_LEAKAGE":
        candidate = """# Candidate CSTNET Domain Masking Protocol (Not Applied)

Status: candidate only. Stage 5.5 performed no preprocessing modification and no training.

For a separately versioned and frozen CSTNET preprocessing v2, deterministically parse TLS ClientHello bytes before the Open-Detect 80+48 byte truncation and zero all retained SNI/server-name byte ranges. Also zero deterministic plaintext class-domain/hostname matches that enter the retained payload window. Preserve byte positions and input length, record every masked range, and rerun the same leakage audit before training.

This candidate must not be silently substituted into the current frozen protocol.
"""
        (out / "domain_masking_protocol_candidate.md").write_text(candidate, encoding="utf-8")
    return {
        "sampled_classes": len(selected),
        "sampled_pcaps": len(samples),
        "valid_inputs": len(valid),
        "coverage": coverage,
        "selected_classes": selected,
        "full_domain_rate": float(overall["full_domain_visible_rate"]),
        "sni_rate": float(overall["sni_visible_rate"]),
        "direct_domain_rate": direct_rate,
        "domain_or_token_rate": float(overall["domain_or_token_visible_rate"]),
        "leakage_level": leakage_level,
        "ip_status": ip_status,
        "port_status": port_status,
        "endpoint_counts": dict(endpoint_totals),
    }


def reliability_label(validation_n: int) -> str:
    if validation_n >= 50:
        return "GOOD"
    if validation_n >= 30:
        return "ACCEPTABLE"
    if validation_n >= 20:
        return "WEAK"
    return "UNRELIABLE"


def calibration_audit(
    protocol: dict[str, object], ledger: InputLedger
) -> dict[str, object]:
    manifest = pd.read_csv(ledger.add(PROTOCOL_ROOT / "split_manifest.csv"))
    count_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    scenarios = (("50/50", 0.50), ("80/20", 0.20), ("90/10", 0.10), ("95/5", 0.05))

    for fold in FOLDS:
        fold_frame = manifest[(manifest["fold"] == fold) & (manifest["class_role"] == "known")]
        known_classes = [str(value) for value in protocol["class_holdout_folds"][fold]["known_classes"]]
        fold_counts: list[dict[str, object]] = []
        for name in known_classes:
            class_frame = fold_frame[fold_frame["class_name"] == name]
            train_n = int((class_frame["split"] == "known_train").sum())
            val_n = int((class_frame["split"] == "known_validation").sum())
            test_n = int((class_frame["split"] == "known_test").sum())
            row = {
                "fold": fold,
                "class_name": name,
                "train_count": train_n,
                "validation_count": val_n,
                "test_count": test_n,
                "known_total_after_group_purge": train_n + val_n + test_n,
            }
            count_rows.append(row)
            fold_counts.append(row)
            reliability_rows.append(
                {
                    **row,
                    "p05_empirical_rank_ceil": int(np.ceil(0.05 * val_n)),
                    "reliability": reliability_label(val_n),
                }
            )
            for scenario, small_weight in scenarios:
                expected = val_n * small_weight
                stress_rows.append(
                    {
                        "fold": fold,
                        "class_name": name,
                        "validation_count": val_n,
                        "component_weight_scenario": scenario,
                        "small_component_weight": small_weight,
                        "expected_small_component_validation_count": expected,
                        "expected_fallback_n_lt_30": expected < 30,
                    }
                )
        values = np.asarray([int(row["validation_count"]) for row in fold_counts], dtype=float)
        if int(values.sum()) != int(protocol["class_holdout_folds"][fold]["known_split_counts"]["known_validation"]):
            raise RuntimeError(f"{fold}: validation count does not match frozen protocol")
        summary_rows.append(
            {
                "fold": fold,
                "known_class_count": len(values),
                "validation_min": int(values.min()),
                "validation_p05": float(np.quantile(values, 0.05, method="linear")),
                "validation_p25": float(np.quantile(values, 0.25, method="linear")),
                "validation_median": float(np.quantile(values, 0.50, method="linear")),
                "validation_p75": float(np.quantile(values, 0.75, method="linear")),
                "validation_max": int(values.max()),
                "classes_n_lt_10": int((values < 10).sum()),
                "classes_n_lt_20": int((values < 20).sum()),
                "classes_n_lt_30": int((values < 30).sum()),
                "classes_n_lt_50": int((values < 50).sum()),
                "unreliable_rate": float((values < 20).mean()),
                "mass_unreliable_threshold": MASS_UNRELIABLE_RATE,
                "mass_unreliable": float((values < 20).mean()) >= MASS_UNRELIABLE_RATE,
            }
        )
        for scenario, small_weight in scenarios:
            matching = [
                row for row in stress_rows
                if row["fold"] == fold and row["component_weight_scenario"] == scenario
            ]
            fallback_count = sum(bool(row["expected_fallback_n_lt_30"]) for row in matching)
            risk_rows.append(
                {
                    "fold": fold,
                    "component_weight_scenario": scenario,
                    "small_component_weight": small_weight,
                    "known_class_count": len(matching),
                    "classes_expected_fallback_n_lt_30": fallback_count,
                    "expected_fallback_rate": fallback_count / len(matching),
                    "interpretation": "sample-size stress only; not an observed GMM component result",
                }
            )

    out = OUTPUT_ROOT / "calibration_sample_audit"
    write_csv(out / "class_split_counts.csv", count_rows)
    write_csv(out / "class_calibration_reliability.csv", reliability_rows)
    write_csv(out / "component_count_stress_table.csv", stress_rows)
    write_csv(out / "fallback_risk_summary.csv", risk_rows)
    mass_unreliable = any(bool(row["mass_unreliable"]) for row in summary_rows)
    revise = (
        mass_unreliable
        or any(int(row["classes_n_lt_30"]) > 0 for row in summary_rows)
        or any(float(row["expected_fallback_rate"]) >= 0.50 for row in risk_rows)
    )
    recommendation = "REVISE_BEFORE_TRAINING" if revise else "KEEP"
    lines = [
        "# Calibration Sample Sufficiency Audit",
        "",
        "This is a class-count and hypothetical component-weight stress audit. No CSTNET encoder, scaler, PCA, GMM, component assignment, score, or threshold was produced.",
        "",
        "## Known Validation Distribution",
        "",
        "| Fold | Classes | min | P05 | P25 | median | P75 | max | n<20 | n<30 | n<50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['fold']} | {row['known_class_count']} | {row['validation_min']} | "
            f"{float(row['validation_p05']):.2f} | {float(row['validation_p25']):.2f} | "
            f"{float(row['validation_median']):.2f} | {float(row['validation_p75']):.2f} | "
            f"{row['validation_max']} | {row['classes_n_lt_20']} | {row['classes_n_lt_30']} | "
            f"{row['classes_n_lt_50']} |"
        )
    lines.extend(
        [
            "",
            "## Component Fallback Stress",
            "",
            "| Fold | Weight | Classes expected n<30 | Rate |",
            "|---|---|---:|---:|",
        ]
    )
    for row in risk_rows:
        lines.append(
            f"| {row['fold']} | {row['component_weight_scenario']} | "
            f"{row['classes_expected_fallback_n_lt_30']}/{row['known_class_count']} | "
            f"{100 * float(row['expected_fallback_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Frozen minimum-count=100 recommendation: **{recommendation}**.",
            "",
            "If revision is required, Stage 5.5 does not choose between: (A) increasing the eligibility minimum; or (B) keeping the class set while defining a larger Known-only calibration pool. Either change requires a separately frozen protocol v2 before training.",
        ]
    )
    (out / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "fold_summaries": summary_rows,
        "fallback_risk": risk_rows,
        "mass_unreliable": mass_unreliable,
        "minimum_count_recommendation": recommendation,
        "protocol_v2_recommended": revise,
    }


def final_gate(
    hashes_pass: bool,
    dgsb: dict[str, object],
    cstnet: dict[str, object],
    calibration: dict[str, object],
    group_split_pass: bool,
) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not hashes_pass:
        blockers.append("frozen hash failure")
    if not bool(dgsb["global_gate_nonzero_all_cells"]):
        blockers.append("Global Gate has zero independent effect in at least one setting/split")
    if float(dgsb["max_mismatch_rate"]) > 0.05:
        blockers.append("Global-vs-Local class mismatch exceeds 5%")
    if cstnet["leakage_level"] == "HIGH_DIRECT_DOMAIN_LEAKAGE":
        blockers.append("HIGH_DIRECT_DOMAIN_LEAKAGE")
    elif cstnet["leakage_level"] == "MODERATE_DOMAIN_LEAKAGE":
        warnings.append("moderate direct domain leakage")
    if bool(calibration["mass_unreliable"]):
        blockers.append("mass Known-class Validation n<20")
    elif any(row["classes_n_lt_30"] for row in calibration["fold_summaries"]):
        warnings.append("some class-level calibration counts are weak")
    if not group_split_pass:
        blockers.append("frozen group split audit failure")
    if any(float(row["expected_fallback_rate"]) >= 0.50 for row in calibration["fallback_risk"]):
        warnings.append("component n<30 fallback risk is high in stress scenarios")
    if cstnet["port_status"] == "VISIBLE":
        warnings.append("transport ports remain visible in the model input")
    if blockers:
        return "NOT_READY", blockers, warnings
    if warnings:
        return "READY_WITH_WARNINGS", blockers, warnings
    return "READY", blockers, warnings


def write_final_summary(
    hash_rows: Sequence[dict[str, object]],
    dgsb: dict[str, object],
    cstnet: dict[str, object],
    calibration: dict[str, object],
    gate: str,
    blockers: Sequence[str],
    warnings: Sequence[str],
) -> None:
    lines = [
        "# Stage 5.5 Final Preflight Summary",
        "",
        f"Final Preflight Gate: **{gate}**",
        "",
        "## Frozen Integrity",
        "",
        f"- SHA-256 checks: {sum(row['status'] == 'PASS' for row in hash_rows)}/{len(hash_rows)} PASS.",
        "- Frozen Stage 5 rule/protocol files were not modified.",
        "",
        "## Q1: DGSB Global Gate Contribution",
        "",
        f"- Nonzero Global-exclusive rejection in every A-1/A-2/A-3 Known Validation/Test cell: **{'YES' if dgsb['global_gate_nonzero_all_cells'] else 'NO'}**.",
        "",
        "## Q2: Global-vs-Local Class Consistency",
        "",
        f"- Maximum mismatch rate: **{100 * float(dgsb['max_mismatch_rate']):.4f}%**; structural mismatch (>5%): **{'YES' if float(dgsb['max_mismatch_rate']) > 0.05 else 'NO'}**.",
        "",
        "## Q3: CSTNET Actual Input Leakage",
        "",
        f"- Sample: {cstnet['sampled_pcaps']} PCAPs, {cstnet['valid_inputs']} valid actual 32x32 inputs.",
        f"- Full class-domain visibility: {100 * float(cstnet['full_domain_rate']):.4f}%.",
        f"- Parsed-SNI visibility: {100 * float(cstnet['sni_rate']):.4f}%.",
        f"- Direct domain/hostname/SNI visibility: {100 * float(cstnet['direct_domain_rate']):.4f}% ({cstnet['leakage_level']}).",
        f"- Domain/token union visibility: {100 * float(cstnet['domain_or_token_rate']):.4f}%.",
        f"- IP: {cstnet['ip_status']}; Port: {cstnet['port_status']}.",
        "",
        "## Q4: Calibration Sufficiency",
        "",
        f"- minimum_total_samples_per_class=100: **{calibration['minimum_count_recommendation']}**.",
        f"- Protocol v2 recommended before training: **{'YES' if calibration['protocol_v2_recommended'] else 'NO'}**.",
        "",
        "## Blocking Issues",
        "",
    ]
    lines.extend([f"- {item}" for item in blockers] or ["- None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "No CSTNET training, formal mu_x extraction, scaler/PCA/GMM fit, boundary calibration, Unknown inference, UFAR, AUROC, or AUPRC was performed. USTC Unknown artifacts were not read.",
        ]
    )
    (OUTPUT_ROOT / "final_preflight_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--max-classes", type=int, default=MAX_CLASSES)
    parser.add_argument("--max-pcaps-per-class", type=int, default=MAX_PCAPS_PER_CLASS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.sample_seed, args.max_classes, args.max_pcaps_per_class) != (
        SAMPLE_SEED,
        MAX_CLASSES,
        MAX_PCAPS_PER_CLASS,
    ):
        raise RuntimeError("Stage 5.5 sampling contract is fixed at seed=20260913, 30 classes, 20 PCAP/class")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    ledger = InputLedger()
    rule, protocol, hash_rows = verify_frozen_inputs(ledger)
    dgsb = dgsb_audit(rule, ledger)
    cstnet = cstnet_input_audit(protocol, ledger)
    calibration = calibration_audit(protocol, ledger)
    group_split_pass = all(
        bool(protocol["class_holdout_folds"][fold]["group_disjoint"])
        and int(protocol["class_holdout_folds"][fold]["active_group_overlap_count"]) == 0
        for fold in FOLDS
    )
    hashes_pass = all(row["status"] == "PASS" for row in hash_rows)
    gate, blockers, warnings = final_gate(
        hashes_pass, dgsb, cstnet, calibration, group_split_pass
    )
    write_final_summary(hash_rows, dgsb, cstnet, calibration, gate, blockers, warnings)
    metadata = {
        "run_id": "stage5_5_cstnet_preflight_20260913_v1",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "status": "completed",
        "claim_scope": "PREFLIGHT_DIAGNOSTIC_ONLY",
        "environment": {
            "python": platform.python_version(),
            "prefix": sys.prefix,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sampling": {
            "seed": SAMPLE_SEED,
            "maximum_classes": MAX_CLASSES,
            "maximum_pcaps_per_class": MAX_PCAPS_PER_CLASS,
            "selected_classes": cstnet["selected_classes"],
            "role_coverage": cstnet["coverage"],
        },
        "hash_verification": hash_rows,
        "group_split_audit": "PASS" if group_split_pass else "FAIL",
        "dgsb": {key: value for key, value in dgsb.items() if key != "gate_rows"},
        "cstnet_input": cstnet,
        "calibration": calibration,
        "final_gate": gate,
        "blocking_issues": blockers,
        "warnings": warnings,
        "accessed_input_files": [str(path) for path in ledger.unique()],
        "forbidden_ustc_unknown_files_read": [],
        "fit_operations_performed": [],
        "cstnet_training_executed": False,
        "cstnet_formal_mu_extraction_executed": False,
        "cstnet_unknown_scores_accessed": False,
        "ustc_unknown_inputs_accessed": 0,
        "metrics_not_computed": ["UFAR", "AUROC", "AUPRC"],
        "preprocessing_source": {
            "adapter": str((OPENDETECT_AUDIT_ROOT / "adapters/opendetect_preprocessing.py").resolve()),
            "adapter_sha256": sha256_file(OPENDETECT_AUDIT_ROOT / "adapters/opendetect_preprocessing.py"),
            "contract": "first 8 packets; 80 header + 48 payload bytes; IPv4 source/destination zeroed; 32x32 uint8",
        },
        "commands": [
            "python stage5_5_cstnet_preflight/scripts/run_preflight_audit.py",
            "pytest -q stage5_5_cstnet_preflight/tests",
            "python stage5_5_cstnet_preflight/tests/verify_preflight.py",
        ],
    }
    (OUTPUT_ROOT / "run_metadata.json").write_text(stable_json(metadata), encoding="utf-8")
    print(stable_json({
        "status": metadata["status"],
        "hashes_pass": hashes_pass,
        "sampled_pcaps": cstnet["sampled_pcaps"],
        "valid_inputs": cstnet["valid_inputs"],
        "direct_domain_rate": cstnet["direct_domain_rate"],
        "max_mismatch_rate": dgsb["max_mismatch_rate"],
        "mass_unreliable": calibration["mass_unreliable"],
        "final_gate": gate,
        "blocking_issues": blockers,
    }))


if __name__ == "__main__":
    main()
