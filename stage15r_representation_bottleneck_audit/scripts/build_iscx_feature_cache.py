#!/usr/bin/env python3
"""Replay the frozen Stage 12 sessionization to recover exact flow statistics.

The source PCAPs are opened read-only.  Only flow hashes already present in the
frozen Stage 12 split manifest are emitted.  No labels or Unknown scores are
used to fit or select a representation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import ROOT, STAGE12_ROOT, read_json, sha256_file, write_csv, write_json


FIELDS = (
    "frame.time_epoch", "frame.len", "frame.protocols",
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
    "tcp.len", "udp.length",
)
MAX_PACKETS = 8


@dataclass
class FlowState:
    sequence: int = 0
    session_id: str | None = None
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


@dataclass
class Accumulator:
    flow_hash: str
    dataset: str
    class_name: str
    source_file: str
    domain_state: str
    official_category: str
    packet_count: int = 0
    total_bytes: int = 0
    payload_bytes: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    previous_timestamp: float | None = None
    length_mean: float = 0.0
    length_m2: float = 0.0
    iat_count: int = 0
    iat_mean: float = 0.0
    iat_m2: float = 0.0
    origin: tuple[str, str] | None = None
    forward_packets: int = 0
    reverse_packets: int = 0
    direction_changes: int = 0
    previous_direction: int = 0
    sequence_lengths: list[float] = field(default_factory=list)
    sequence_iats: list[float] = field(default_factory=list)
    sequence_directions: list[int] = field(default_factory=list)
    protocol_tokens: set[str] = field(default_factory=set)

    def add(self, timestamp: float, frame_length: int, payload_length: int, src: tuple[str, str], dst: tuple[str, str], protocols: str) -> None:
        if self.origin is None:
            self.origin = src
        direction = 1 if src == self.origin else (-1 if dst == self.origin else 0)
        if direction > 0:
            self.forward_packets += 1
        elif direction < 0:
            self.reverse_packets += 1
        if self.previous_direction and direction and direction != self.previous_direction:
            self.direction_changes += 1
        if direction:
            self.previous_direction = direction
        self.packet_count += 1
        self.total_bytes += frame_length
        self.payload_bytes += max(payload_length, 0)
        delta = frame_length - self.length_mean
        self.length_mean += delta / self.packet_count
        self.length_m2 += delta * (frame_length - self.length_mean)
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        iat = 0.0 if self.previous_timestamp is None else max(0.0, timestamp - self.previous_timestamp)
        if self.previous_timestamp is not None:
            self.iat_count += 1
            delta_iat = iat - self.iat_mean
            self.iat_mean += delta_iat / self.iat_count
            self.iat_m2 += delta_iat * (iat - self.iat_mean)
        self.previous_timestamp = timestamp
        self.last_timestamp = timestamp
        if len(self.sequence_lengths) < MAX_PACKETS:
            self.sequence_lengths.append(float(frame_length))
            self.sequence_iats.append(float(iat))
            self.sequence_directions.append(int(direction))
        self.protocol_tokens.update(token.lower() for token in protocols.split(":") if token)

    def row(self) -> dict[str, object]:
        duration = 0.0 if self.first_timestamp is None or self.last_timestamp is None else max(0.0, self.last_timestamp - self.first_timestamp)
        length_std = math.sqrt(max(self.length_m2 / self.packet_count, 0.0)) if self.packet_count else 0.0
        iat_std = math.sqrt(max(self.iat_m2 / self.iat_count, 0.0)) if self.iat_count else 0.0
        if self.domain_state == "VPN":
            regime = "VPN tunnel"
        elif self.domain_state == "Tor":
            regime = "Tor"
        elif any(token.startswith("quic") for token in self.protocol_tokens):
            regime = "TLS/QUIC"
        elif any(token.startswith(("tls", "ssl")) for token in self.protocol_tokens):
            regime = "TLS/QUIC"
        elif any(token.startswith("ssh") for token in self.protocol_tokens):
            regime = "SSH"
        elif self.protocol_tokens & {"http", "ftp", "smtp", "imap", "pop"}:
            regime = "visible plaintext"
        else:
            regime = "undetermined"
        return {
            "dataset": self.dataset,
            "flow_uid": self.flow_hash,
            "class_name": self.class_name,
            "source_file": self.source_file,
            "domain_state": self.domain_state,
            "official_category": self.official_category,
            "packet_count": self.packet_count,
            "total_bytes": self.total_bytes,
            "payload_bytes": self.payload_bytes,
            "duration_seconds": duration,
            "packet_length_mean": self.length_mean,
            "packet_length_std": length_std,
            "iat_mean_seconds": self.iat_mean,
            "iat_std_seconds": iat_std,
            "forward_packets": self.forward_packets,
            "reverse_packets": self.reverse_packets,
            "forward_reverse_ratio": (self.forward_packets + 1.0) / (self.reverse_packets + 1.0),
            "direction_changes": self.direction_changes,
            "padding_ratio_packet_slots": max(0, MAX_PACKETS - min(self.packet_count, MAX_PACKETS)) / MAX_PACKETS,
            "truncated_after_packet_8": self.packet_count > MAX_PACKETS,
            "encryption_regime": regime,
            "protocol_tokens": ":".join(sorted(self.protocol_tokens)),
        }


def value(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def endpoint(values: list[str]) -> tuple[tuple[str, str], tuple[str, str], str] | None:
    src = value(values, 3) or value(values, 4)
    dst = value(values, 5) or value(values, 6)
    tcp_s, tcp_d = value(values, 7), value(values, 8)
    udp_s, udp_d = value(values, 9), value(values, 10)
    if tcp_s and tcp_d:
        return (src, tcp_s), (dst, tcp_d), "tcp"
    if udp_s and udp_d:
        return (src, udp_s), (dst, udp_d), "udp"
    return None


def flag(value_: str) -> bool:
    return value_.lower() in {"1", "true", "set"}


def load_targets(dataset: str) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    manifest_path = STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    capture_meta: dict[tuple[str, str], dict[str, str]] = {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["flow_id_sha256"]
            prior = rows.setdefault(uid, row)
            if prior["canonical_class"] != row["canonical_class"] or prior["source_file"] != row["source_file"]:
                raise RuntimeError(f"inconsistent frozen flow metadata: {uid}")
            capture_meta[(row["canonical_class"], Path(row["source_file"]).name)] = row
    return rows, capture_meta


def scan(dataset: str) -> None:
    output = ROOT / "feature_cache" / dataset
    if output.exists():
        raise RuntimeError(f"refusing to overwrite cache: {output}")
    output.mkdir(parents=True)
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    targets, capture_meta = load_targets(dataset)
    config = read_json(STAGE12_ROOT / "configs" / "stage12_config.json")
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    manifest = read_json(STAGE12_ROOT / "protocol" / dataset / "preprocessing_manifest.json")
    accumulators: dict[str, Accumulator] = {}
    capture_audit: list[dict[str, object]] = []
    for class_spec in sorted(manifest["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            pcap = Path(raw_path)
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            relative = next((row["source_file"] for (name, base), row in capture_meta.items() if name == class_name and base == pcap.name), "")
            meta = capture_meta.get((class_name, pcap.name))
            if meta is None:
                continue
            command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
            for field_name in FIELDS:
                command.extend(["-e", field_name])
            states: dict[tuple[str, str, str, str, str], FlowState] = {}
            packet_rows = 0
            matched_rows = 0
            stderr_path = stderr_root / f"{class_name}_{capture_index:03d}_{pcap.stem}.log"
            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
                assert process.stdout is not None
                for line in process.stdout:
                    packet_rows += 1
                    values = line.rstrip("\r\n").split("\t")
                    ep = endpoint(values)
                    if ep is None:
                        continue
                    src, dst, protocol = ep
                    ordered = tuple(sorted((src, dst)))
                    key = (capture_id, protocol, ordered[0][0], ordered[0][1], ordered[1][0] + ":" + ordered[1][1])
                    timestamp = float(value(values, 0))
                    state = states.setdefault(key, FlowState())
                    new_syn = protocol == "tcp" and flag(value(values, 11)) and not flag(value(values, 12))
                    timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                    if state.session_id is None or state.closed or timed_out or syn_starts_new:
                        flow_token = f"{protocol}|{ordered[0][0]}:{ordered[0][1]}|{ordered[1][0]}:{ordered[1][1]}"
                        state.session_id = f"{capture_id}|{flow_token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                    uid = str(state.session_hash)
                    if uid in targets:
                        target = targets[uid]
                        if uid not in accumulators:
                            accumulators[uid] = Accumulator(uid, dataset, class_name, relative, target["domain_state"], target["official_category"])
                        frame_length = int(value(values, 1) or 0)
                        tcp_payload = int(value(values, 15) or 0)
                        udp_total = int(value(values, 16) or 0)
                        payload_length = tcp_payload if protocol == "tcp" else max(udp_total - 8, 0)
                        accumulators[uid].add(timestamp, frame_length, payload_length, src, dst, value(values, 2))
                        matched_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and (flag(value(values, 13)) or flag(value(values, 14))):
                        state.closed = True
                returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"tshark failed ({returncode}) for {pcap}; see {stderr_path}")
            capture_audit.append({"dataset": dataset, "class_name": class_name, "capture_index": capture_index, "source_file": relative, "pcap_path": str(pcap), "packet_rows": packet_rows, "matched_packet_rows": matched_rows, "stderr_path": str(stderr_path), "pcap_sha256": sha256_file(pcap)})
            print(json.dumps({"event": "capture_complete", "dataset": dataset, "class": class_name, "capture": capture_index, "matched_flows": len(accumulators), "target_flows": len(targets)}), flush=True)
    missing = sorted(set(targets) - set(accumulators))
    if missing:
        write_json(output / "failure.json", {"status": "FAIL", "missing_count": len(missing), "missing_preview": missing[:100]})
        raise RuntimeError(f"failed to recover {len(missing)} frozen flows")
    ordered = sorted(accumulators)
    rows = [accumulators[uid].row() for uid in ordered]
    lengths = np.zeros((len(ordered), MAX_PACKETS), dtype=np.float32)
    iats = np.zeros_like(lengths)
    directions = np.zeros((len(ordered), MAX_PACKETS), dtype=np.int8)
    mask = np.zeros((len(ordered), MAX_PACKETS), dtype=np.uint8)
    for index, uid in enumerate(ordered):
        acc = accumulators[uid]
        used = len(acc.sequence_lengths)
        lengths[index, :used] = acc.sequence_lengths
        iats[index, :used] = acc.sequence_iats
        directions[index, :used] = acc.sequence_directions
        mask[index, :used] = 1
    write_csv(output / "flow_statistics.csv", rows)
    write_csv(output / "capture_audit.csv", capture_audit)
    np.save(output / "flow_uids.npy", np.asarray(ordered, dtype="U64"), allow_pickle=False)
    np.save(output / "packet_lengths.npy", lengths, allow_pickle=False)
    np.save(output / "packet_iat_seconds.npy", iats, allow_pickle=False)
    np.save(output / "packet_directions.npy", directions, allow_pickle=False)
    np.save(output / "packet_mask.npy", mask, allow_pickle=False)
    write_json(output / "cache_audit.json", {
        "status": "PASS", "dataset": dataset, "target_flows": len(targets), "recovered_flows": len(rows),
        "known_test_or_unknown_test_values_used_for_selection": False,
        "source_split_manifest": str((STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv").resolve()),
        "source_split_manifest_sha256": sha256_file(STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv"),
        "sessionization": {"direction": "bidirectional", "timeout_seconds": timeout, "tcp_syn_fin_rst_boundaries": True},
        "sequence_packets": MAX_PACKETS,
        "flow_statistics_sha256": sha256_file(output / "flow_statistics.csv"),
        "flow_uids_sha256": sha256_file(output / "flow_uids.npy"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    scan(args.dataset)


if __name__ == "__main__":
    main()
