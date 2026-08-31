# -*- coding: utf-8 -*-
"""TrafficFormer-compatible input generation from Stage 0 flow PKLs.

This module is intentionally additive: it does not change the canonical Stage 0
flow splitter.  The primary ``compatible_min1`` policy accepts every non-empty
flow, while ``strict_min3`` provides a short-flow-filtered comparison cohort.

Only observed packets are serialized.  Missing packets are *not* synthesized;
the downstream TrafficFormer reader remains responsible for token-level padding.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


OFFICIAL_TRAFFICFORMER_REPOSITORY = "https://github.com/IDP-code/TrafficFormer"
OFFICIAL_TRAFFICFORMER_COMMIT = "6d0ba64d82e74fb130c6c7301ef20885dbfbdf29"


@dataclass(frozen=True)
class TrafficFormerFormat:
    """Pre-tokenizer format controls for one TrafficFormer flow sample."""

    policy: str = "compatible_min1"
    min_packets: int = 1
    max_packets: int = 5
    ethernet_offset_bytes: int = 14
    bytes_per_packet: int = 64
    add_sep_before_packet: bool = True
    seq_length: int = 320

    def validate(self) -> None:
        if self.min_packets < 1:
            raise ValueError("min_packets must be >= 1; empty flows are invalid")
        if self.max_packets < self.min_packets:
            raise ValueError("max_packets must be >= min_packets")
        if self.ethernet_offset_bytes < 0:
            raise ValueError("ethernet_offset_bytes must be >= 0")
        if self.bytes_per_packet < 1:
            raise ValueError("bytes_per_packet must be >= 1")
        if self.seq_length < 1:
            raise ValueError("seq_length must be >= 1")


@dataclass(frozen=True)
class EncodedFlow:
    """Observable metadata for one generated TrafficFormer text sample."""

    text: str
    packet_count: int
    used_packet_count: int
    token_count: int


def official_bigram_generation(packet_hex: str) -> str:
    """Return TrafficFormer's overlapping two-byte bigram text.

    This is a clean-room expression of ``data_generation/utils.py`` at
    ``OFFICIAL_TRAFFICFORMER_COMMIT``.  For bytes ``45 00 00 3c`` the result is
    ``"4500 0000 003c "``.  The trailing space is retained for byte-compatible
    pre-tokenizer text.
    """

    if len(packet_hex) % 2:
        raise ValueError("packet_hex must contain a whole number of bytes")
    try:
        bytes.fromhex(packet_hex)
    except ValueError as exc:
        raise ValueError("packet_hex contains non-hexadecimal characters") from exc

    octets = [packet_hex[i : i + 2].lower() for i in range(0, len(packet_hex), 2)]
    if len(octets) < 2:
        return ""
    return "".join(f"{octets[i]}{octets[i + 1]} " for i in range(len(octets) - 1))


def encode_flow(
    packets: Sequence[Sequence[Any]],
    fmt: TrafficFormerFormat,
) -> Optional[EncodedFlow]:
    """Encode one stored bidirectional flow without inventing packet content.

    Stage 0 packets have the schema
    ``(timestamp, caplen, wirelen, direction, raw_frame)``.  Their stored order
    is preserved.  ``None`` means the selected policy filtered the flow.
    """

    fmt.validate()
    packet_count = len(packets)
    if packet_count < fmt.min_packets:
        return None

    parts = []
    token_count = 0
    selected = packets[: fmt.max_packets]
    for packet_index, packet in enumerate(selected):
        if len(packet) < 5:
            raise ValueError(
                f"packet {packet_index} does not match "
                "(timestamp, caplen, wirelen, direction, raw_frame)"
            )
        frame = packet[4]
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise TypeError(f"packet {packet_index} raw_frame must be bytes-like")
        frame_bytes = bytes(frame)
        packet_hex = frame_bytes[
            fmt.ethernet_offset_bytes : fmt.ethernet_offset_bytes + fmt.bytes_per_packet
        ].hex()
        bigrams = official_bigram_generation(packet_hex)
        if fmt.add_sep_before_packet:
            parts.append("[SEP] ")
        parts.append(bigrams)
        token_count += len(bigrams.split())

    return EncodedFlow(
        text="".join(parts),
        packet_count=packet_count,
        used_packet_count=len(selected),
        token_count=token_count,
    )


def packet_count_bucket(packet_count: int) -> str:
    if packet_count <= 0:
        return "empty"
    if packet_count == 1:
        return "1"
    if packet_count == 2:
        return "2"
    if packet_count <= 4:
        return "3-4"
    return ">=5"


def _parse_generated_pkl_name(path: Path) -> Tuple[str, str, str]:
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        raise ValueError(
            f"unexpected Stage 0 PKL name {path.name!r}; expected split__class__stem.pkl"
        )
    return parts[0], parts[1], parts[2]


def _discover_pkl_files(flows_dir: Path) -> Sequence[Path]:
    paths = sorted(flows_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no Stage 0 flow PKLs found under {flows_dir}")
    return paths


def _new_class_stats() -> Dict[str, int]:
    return {
        "total_flows": 0,
        "retained_flows": 0,
        "dropped_flows": 0,
        "packet_count_1": 0,
        "packet_count_2": 0,
        "packet_count_3_4": 0,
        "packet_count_ge5": 0,
        "packet_count_empty": 0,
    }


def _increment_bucket(stats: Dict[str, int], packet_count: int) -> None:
    bucket = packet_count_bucket(packet_count)
    key = {
        "1": "packet_count_1",
        "2": "packet_count_2",
        "3-4": "packet_count_3_4",
        ">=5": "packet_count_ge5",
        "empty": "packet_count_empty",
    }[bucket]
    stats[key] += 1


def _validate_payload(
    path: Path,
    payload: Mapping[str, Any],
    expected_split: str,
    expected_class: str,
) -> Mapping[str, Sequence[Sequence[Any]]]:
    if payload.get("split") != expected_split:
        raise ValueError(
            f"{path.name}: split metadata {payload.get('split')!r} != {expected_split!r}"
        )
    if payload.get("class_name") != expected_class:
        raise ValueError(
            f"{path.name}: class metadata {payload.get('class_name')!r} != {expected_class!r}"
        )
    packets = payload.get("packets")
    if not isinstance(packets, Mapping):
        raise ValueError(f"{path.name}: missing mapping field 'packets'")
    return packets


def generate_dataset(
    flows_dir: Path,
    output_dir: Path,
    fmt: TrafficFormerFormat,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate official-shape TSV plus an auditable flow mapping.

    The output files are staged under ``output_dir`` and installed with atomic
    file replacement.  Existing outputs are preserved unless ``overwrite`` is
    explicitly true.
    """

    fmt.validate()
    flows_dir = Path(flows_dir)
    output_dir = Path(output_dir)
    pkl_paths = _discover_pkl_files(flows_dir)

    parsed_names = {path: _parse_generated_pkl_name(path) for path in pkl_paths}
    class_names = sorted({class_name for _, class_name, _ in parsed_names.values()})
    class_to_id = {class_name: index for index, class_name in enumerate(class_names)}

    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "tsv": "trafficformer_all.tsv",
        "flow_map": "flow_map.csv",
        "label_map": "label_map.json",
        "summary": "generation_summary.json",
        "retention": "retention_by_class.csv",
    }
    existing = [output_dir / name for name in filenames.values() if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing outputs: {joined}")

    class_stats = {class_name: _new_class_stats() for class_name in class_names}
    seen_flow_ids = set()
    retained_total = 0
    total_flows = 0

    with tempfile.TemporaryDirectory(prefix=".trafficformer-stage-", dir=output_dir) as stage:
        stage_dir = Path(stage)
        tsv_path = stage_dir / filenames["tsv"]
        map_path = stage_dir / filenames["flow_map"]

        map_fields = [
            "sample_index",
            "tsv_line_number",
            "flow_id",
            "label_id",
            "split",
            "class_name",
            "source_file",
            "source_pkl",
            "packet_count",
            "used_packet_count",
            "packet_count_bucket",
            "token_count",
            "is_short_flow",
            "policy",
            "text_sha256",
        ]

        with tsv_path.open("w", newline="", encoding="utf-8") as tsv_fh, map_path.open(
            "w", newline="", encoding="utf-8"
        ) as map_fh:
            tsv_writer = csv.writer(tsv_fh, delimiter="\t", lineterminator="\n")
            map_writer = csv.DictWriter(map_fh, fieldnames=map_fields)
            tsv_writer.writerow(["label", "text_a"])
            map_writer.writeheader()

            for pkl_path in pkl_paths:
                expected_split, expected_class, _ = parsed_names[pkl_path]
                with pkl_path.open("rb") as fh:
                    payload = pickle.load(fh)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{pkl_path.name}: expected a mapping payload")
                packets_by_flow = _validate_payload(
                    pkl_path, payload, expected_split, expected_class
                )
                source_file = str(payload.get("source_file", ""))

                for flow_id in sorted(packets_by_flow):
                    if flow_id in seen_flow_ids:
                        raise ValueError(f"duplicate flow_id across PKLs: {flow_id}")
                    seen_flow_ids.add(flow_id)
                    packets = packets_by_flow[flow_id]
                    if not isinstance(packets, Sequence):
                        raise ValueError(f"{pkl_path.name}/{flow_id}: packets must be a sequence")

                    packet_count = len(packets)
                    stats = class_stats[expected_class]
                    stats["total_flows"] += 1
                    total_flows += 1
                    _increment_bucket(stats, packet_count)

                    encoded = encode_flow(packets, fmt)
                    if encoded is None:
                        stats["dropped_flows"] += 1
                        continue

                    stats["retained_flows"] += 1
                    sample_index = retained_total
                    retained_total += 1
                    label_id = class_to_id[expected_class]
                    tsv_writer.writerow([label_id, encoded.text])
                    map_writer.writerow(
                        {
                            "sample_index": sample_index,
                            "tsv_line_number": sample_index + 2,
                            "flow_id": flow_id,
                            "label_id": label_id,
                            "split": expected_split,
                            "class_name": expected_class,
                            "source_file": source_file,
                            "source_pkl": pkl_path.name,
                            "packet_count": encoded.packet_count,
                            "used_packet_count": encoded.used_packet_count,
                            "packet_count_bucket": packet_count_bucket(encoded.packet_count),
                            "token_count": encoded.token_count,
                            "is_short_flow": int(encoded.packet_count < 3),
                            "policy": fmt.policy,
                            "text_sha256": hashlib.sha256(
                                encoded.text.encode("utf-8")
                            ).hexdigest(),
                        }
                    )

        label_payload = {
            "class_to_id": class_to_id,
            "id_to_class": {str(value): key for key, value in class_to_id.items()},
        }
        (stage_dir / filenames["label_map"]).write_text(
            json.dumps(label_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summary = {
            "policy": fmt.policy,
            "format": asdict(fmt),
            "official_reference": {
                "repository": OFFICIAL_TRAFFICFORMER_REPOSITORY,
                "commit": OFFICIAL_TRAFFICFORMER_COMMIT,
            },
            "source_pkl_count": len(pkl_paths),
            "class_count": len(class_names),
            "total_flows": total_flows,
            "retained_flows": retained_total,
            "dropped_flows": total_flows - retained_total,
            "retention_ratio": retained_total / total_flows if total_flows else 0.0,
            "padding": "downstream token-level tail padding; no synthetic packets",
            "rifa_applied": False,
        }
        (stage_dir / filenames["summary"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        retention_fields = [
            "class_name",
            "label_id",
            "total_flows",
            "retained_flows",
            "dropped_flows",
            "retention_ratio",
            "packet_count_1",
            "packet_count_2",
            "packet_count_3_4",
            "packet_count_ge5",
            "packet_count_empty",
        ]
        with (stage_dir / filenames["retention"]).open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=retention_fields)
            writer.writeheader()
            for class_name in class_names:
                stats = class_stats[class_name]
                total = stats["total_flows"]
                writer.writerow(
                    {
                        "class_name": class_name,
                        "label_id": class_to_id[class_name],
                        **stats,
                        "retention_ratio": (
                            stats["retained_flows"] / total if total else 0.0
                        ),
                    }
                )

        for name in filenames.values():
            os.replace(stage_dir / name, output_dir / name)

    return summary


def format_from_config(config: Mapping[str, Any], policy: str) -> TrafficFormerFormat:
    """Build and validate a format object from the additive YAML config."""

    format_cfg = dict(config.get("format", {}))
    policies = config.get("policies", {})
    if policy not in policies:
        raise KeyError(f"unknown policy {policy!r}; available: {sorted(policies)}")
    policy_cfg = dict(policies[policy])
    fmt = TrafficFormerFormat(
        policy=policy,
        min_packets=int(policy_cfg["min_packets"]),
        max_packets=int(format_cfg.get("max_packets", 5)),
        ethernet_offset_bytes=int(format_cfg.get("ethernet_offset_bytes", 14)),
        bytes_per_packet=int(format_cfg.get("bytes_per_packet", 64)),
        add_sep_before_packet=bool(format_cfg.get("add_sep_before_packet", True)),
        seq_length=int(format_cfg.get("seq_length", 320)),
    )
    fmt.validate()
    return fmt
