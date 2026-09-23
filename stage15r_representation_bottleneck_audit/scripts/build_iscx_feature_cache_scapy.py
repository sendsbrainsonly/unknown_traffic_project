#!/usr/bin/env python3
"""Fast exact replay of Stage 12 flow lineage using its original Scapy semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader

from build_iscx_feature_cache import Accumulator, MAX_PACKETS, load_targets
from common import ROOT, STAGE12_ROOT, read_json, sha256_file, write_csv, write_json


@dataclass
class FlowState:
    sequence: int = 0
    session_id: str | None = None
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


def scan(dataset: str) -> None:
    output = ROOT / "feature_cache" / dataset
    if output.exists():
        raise RuntimeError(f"refusing to overwrite cache: {output}")
    output.mkdir(parents=True)
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
            meta = capture_meta.get((class_name, pcap.name))
            if meta is None:
                continue
            states: dict[tuple[str, str], FlowState] = {}
            packet_rows = matched_rows = skipped = 0
            with PcapReader(str(pcap)) as reader:
                for packet in reader:
                    packet_rows += 1
                    if IP not in packet or (TCP not in packet and UDP not in packet):
                        skipped += 1
                        continue
                    ip = packet[IP]
                    if TCP in packet:
                        transport, protocol = packet[TCP], "tcp"
                    else:
                        transport, protocol = packet[UDP], "udp"
                    source = (str(ip.src), int(transport.sport))
                    destination = (str(ip.dst), int(transport.dport))
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
                        target = targets[uid]
                        if uid not in accumulators:
                            accumulators[uid] = Accumulator(uid, dataset, class_name, meta["source_file"], target["domain_state"], target["official_category"])
                        payload_length = len(bytes(transport.payload))
                        protocols = ":".join(layer.__name__.lower() for layer in packet.layers())
                        accumulators[uid].add(timestamp, len(bytes(packet)), payload_length, source, destination, protocols)
                        matched_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and flags & (0x01 | 0x04):
                        state.closed = True
                    if packet_rows % 1_000_000 == 0:
                        print(json.dumps({"event": "packet_progress", "dataset": dataset, "class": class_name, "capture": capture_index, "packets": packet_rows, "matched_flows": len(accumulators)}), flush=True)
            capture_audit.append({"dataset": dataset, "class_name": class_name, "capture_index": capture_index, "source_file": meta["source_file"], "pcap_path": str(pcap), "packet_rows": packet_rows, "matched_packet_rows": matched_rows, "skipped_non_tcp_udp_ipv4": skipped, "pcap_sha256": sha256_file(pcap), "engine": "Scapy PcapReader; exact Stage12 flow/session semantics"})
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
        lengths[index, :used], iats[index, :used] = acc.sequence_lengths, acc.sequence_iats
        directions[index, :used], mask[index, :used] = acc.sequence_directions, 1
    write_csv(output / "flow_statistics.csv", rows)
    write_csv(output / "capture_audit.csv", capture_audit)
    np.save(output / "flow_uids.npy", np.asarray(ordered, dtype="U64"), allow_pickle=False)
    np.save(output / "packet_lengths.npy", lengths, allow_pickle=False)
    np.save(output / "packet_iat_seconds.npy", iats, allow_pickle=False)
    np.save(output / "packet_directions.npy", directions, allow_pickle=False)
    np.save(output / "packet_mask.npy", mask, allow_pickle=False)
    write_json(output / "cache_audit.json", {"status": "PASS", "dataset": dataset, "engine": "Scapy PcapReader matching Stage12 prepare_pcap_dataset.py", "target_flows": len(targets), "recovered_flows": len(rows), "known_test_or_unknown_test_values_used_for_selection": False, "source_split_manifest": str((STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv").resolve()), "source_split_manifest_sha256": sha256_file(STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv"), "sessionization": {"direction": "bidirectional", "ipv4_tcp_udp_only": True, "timeout_seconds": timeout, "tcp_syn_fin_rst_boundaries": True}, "sequence_packets": MAX_PACKETS, "flow_statistics_sha256": sha256_file(output / "flow_statistics.csv"), "flow_uids_sha256": sha256_file(output / "flow_uids.npy")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    scan(args.dataset)


if __name__ == "__main__":
    main()
