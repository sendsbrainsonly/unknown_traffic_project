#!/usr/bin/env python3
"""Replay Stage 12 PCAPs with the frozen sessionization to audit packet windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

from common import FLOW_WINDOW_FIELDS, PROJECT_ROOT, ROOT, WindowAccumulator, read_json, sha256_file, write_csv, write_json


STAGE12 = PROJECT_ROOT / "stage12_dual_external_validation"


@dataclass
class FlowState:
    sequence: int = 0
    session_id: str | None = None
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


def load_targets(dataset: str) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    path = STAGE12 / "protocol" / dataset / "split_manifest.csv"
    targets: dict[str, dict[str, str]] = {}
    capture_meta: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] not in {"known_train", "known_validation"}:
                continue
            uid = row["flow_id_sha256"]
            prior = targets.setdefault(uid, row)
            if prior["canonical_class"] != row["canonical_class"] or prior["source_file"] != row["source_file"]:
                raise RuntimeError(f"inconsistent Stage 12 target: {uid}")
            capture_meta[(row["canonical_class"], Path(row["source_file"]).name)] = row
    return targets, capture_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    output = ROOT / "window_cache" / dataset
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    targets, capture_meta = load_targets(dataset)
    config = read_json(STAGE12 / "configs" / "stage12_config.json")
    assert isinstance(config, dict)
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    prep_manifest_path = STAGE12 / "protocol" / dataset / "preprocessing_manifest.json"
    prep = read_json(prep_manifest_path)
    assert isinstance(prep, dict)
    accumulators: dict[str, WindowAccumulator] = {}
    capture_rows: list[dict[str, object]] = []
    for class_spec in sorted(prep["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            pcap = Path(raw_path)
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            meta = capture_meta.get((class_name, pcap.name))
            if meta is None:
                continue
            states: dict[tuple[str, str], FlowState] = {}
            packet_rows = matched_rows = discarded_rows = 0
            with PcapReader(str(pcap)) as reader:
                for packet in reader:
                    packet_rows += 1
                    if IP not in packet or (TCP not in packet and UDP not in packet):
                        discarded_rows += 1
                        continue
                    ip = packet[IP]
                    if TCP in packet:
                        transport, protocol = packet[TCP], "tcp"
                    else:
                        transport, protocol = packet[UDP], "udp"
                    source = (str(ip.src), str(int(transport.sport)))
                    destination = (str(ip.dst), str(int(transport.dport)))
                    endpoints = tuple(sorted((source, destination)))
                    flow_token = f"{protocol}|{endpoints[0][0]}:{endpoints[0][1]}|{endpoints[1][0]}:{endpoints[1][1]}"
                    key = (capture_id, flow_token)
                    timestamp = float(packet.time)
                    state = states.setdefault(key, FlowState())
                    flags = int(packet[TCP].flags) if TCP in packet else 0
                    new_syn = protocol == "tcp" and bool(flags & 0x02) and not bool(flags & 0x10)
                    timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                    if state.session_id is None or state.closed or timed_out or syn_starts_new:
                        state.session_id = f"{capture_id}|{flow_token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                    uid = str(state.session_hash)
                    if uid in targets:
                        if uid not in accumulators:
                            target = targets[uid]
                            accumulators[uid] = WindowAccumulator(
                                uid,
                                class_name,
                                str(target["source_file"]),
                                str(target["domain_state"]),
                            )
                        protocols = ":".join(layer.__name__.lower() for layer in packet.layers())
                        accumulators[uid].add(
                            timestamp,
                            len(bytes(packet)),
                            len(bytes(ip)),
                            len(bytes(transport.payload)),
                            source,
                            destination,
                            protocols,
                        )
                        matched_rows += 1
                    else:
                        discarded_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and flags & (0x01 | 0x04):
                        state.closed = True
                    if packet_rows % 1_000_000 == 0:
                        print(json.dumps({"event": "packet_progress", "dataset": dataset, "capture": pcap.name, "packets": packet_rows, "matched_flows": len(accumulators)}), flush=True)
            capture_rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "capture_index": capture_index,
                    "source_file": meta["source_file"],
                    "pcap_path": str(pcap),
                    "packet_rows": packet_rows,
                    "matched_known_train_validation_packet_rows": matched_rows,
                    "discarded_non_target_or_non_ipv4_tcp_udp_rows": discarded_rows,
                    "engine": "Scapy PcapReader; exact Stage12 sessionization",
                }
            )
            print(json.dumps({"event": "capture_complete", "dataset": dataset, "class": class_name, "capture": pcap.name, "matched_flows": len(accumulators), "target_flows": len(targets)}), flush=True)
    missing = sorted(set(targets) - set(accumulators))
    if missing:
        write_json(output / "failure.json", {"missing_count": len(missing), "preview": missing[:100]})
        raise RuntimeError(f"missing {len(missing)} {dataset} targets")
    rows = [accumulators[uid].row(dataset) for uid in sorted(accumulators)]
    metrics = output / "flow_window_metrics.csv.gz"
    write_csv(metrics, rows, FLOW_WINDOW_FIELDS)
    write_csv(output / "capture_audit.csv", capture_rows, list(capture_rows[0]))
    split_manifest = STAGE12 / "protocol" / dataset / "split_manifest.csv"
    stage15r_capture = PROJECT_ROOT / "stage15r_representation_bottleneck_audit" / "feature_cache" / dataset / "capture_audit.csv"
    write_json(
        output / "cache_audit.json",
        {
            "status": "PASS",
            "dataset": dataset,
            "scope": "union of Known Train/Validation rows in frozen low/medium/high Stage 12 protocols",
            "target_flows": len(targets),
            "recovered_flows": len(rows),
            "known_test_values_stored": 0,
            "unknown_test_values_stored": 0,
            "non_target_packet_values": "parsed only to preserve session state and discarded immediately",
            "sessionization": {"bidirectional": True, "ipv4_tcp_udp_only": True, "timeout_seconds": timeout, "tcp_syn_fin_rst_boundaries": True},
            "packet_windows": [8, 16, 32, 64],
            "byte_coverage_semantics": "captured frame bytes",
            "time_coverage_semantics": "elapsed time from first through Nth packet divided by full flow duration",
            "split_manifest_sha256": sha256_file(split_manifest),
            "preprocessing_manifest_sha256": sha256_file(prep_manifest_path),
            "stage15r_capture_audit_sha256": sha256_file(stage15r_capture),
            "metrics_sha256": sha256_file(metrics),
        },
    )


if __name__ == "__main__":
    main()
