#!/usr/bin/env python3
"""Recover exact TrafficFormer/FIG inputs for the frozen Stage16S 3065-flow pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scapy.all import IP, IPv6, TCP, UDP, PcapReader

from stage17_common import CACHE, MANIFEST, OUT, PROJECT, read_csv, read_json, sha256_file, write_csv, write_json

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
from src.preprocessing.fig_graph import FigFormat, build_flow_graph
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow
from uer.utils.constants import CLS_TOKEN
from uer.utils.tokenizers import BertTokenizer


STAGE12 = PROJECT / "stage12_dual_external_validation"
PREP = STAGE12 / "protocol" / "iscx_vpn" / "preprocessing_manifest.json"
CONFIG = STAGE12 / "configs" / "stage12_config.json"
VOCAB = PROJECT / "tf_runtime" / "code" / "models" / "encryptd_vocab.txt"
@dataclass
class FlowState:
    sequence: int = 0
    session_id: str | None = None
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False
    initiator: tuple[str, str] | None = None


def endpoint(packet):
    network = packet.getlayer(IP) or packet.getlayer(IPv6)
    if network is None:
        return None
    if packet.haslayer(TCP):
        transport = packet[TCP]
        return (str(network.src), str(transport.sport)), (str(network.dst), str(transport.dport)), "tcp", transport
    if packet.haslayer(UDP):
        transport = packet[UDP]
        return (str(network.src), str(transport.sport)), (str(network.dst), str(transport.dport)), "udp", transport
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if CACHE.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {CACHE}")

    manifest_rows = read_csv(MANIFEST)
    targets = {row["flow_id"] for row in manifest_rows}
    if len(targets) != 3065:
        raise RuntimeError(f"expected 3065 unique flow ids, got {len(targets)}")
    target_meta = {}
    for row in manifest_rows:
        old = target_meta.setdefault(row["flow_id"], row)
        if old["service_label"] != row["service_label"] or old["source_file"] != row["source_file"]:
            raise RuntimeError(f"inconsistent frozen metadata for {row['flow_id']}")

    config = read_json(CONFIG)
    prep = read_json(PREP)
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    wanted_captures = {row["capture_id"] for row in target_meta.values()}
    packets: dict[str, list[tuple[float, int, int, int, bytes]]] = {}
    capture_audit = []
    stderr_dir = OUT / "feature_cache" / "tshark_stderr"
    stderr_dir.mkdir(parents=True, exist_ok=True)

    for class_spec in sorted(prep["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            pcap = Path(raw_path)
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            if capture_id not in wanted_captures:
                continue
            states = {}
            packet_rows = matched_rows = 0
            stderr_path = stderr_dir / f"{class_name}_{capture_index:03d}_{pcap.stem}.log"
            try:
                with PcapReader(str(pcap)) as reader:
                    for packet in reader:
                        packet_rows += 1
                        ep = endpoint(packet)
                        if ep is None:
                            continue
                        source, destination, protocol, transport = ep
                        ordered = tuple(sorted((source, destination)))
                        key = (capture_id, protocol, ordered[0][0], ordered[0][1], ordered[1][0] + ":" + ordered[1][1])
                        timestamp = float(packet.time)
                        state = states.setdefault(key, FlowState())
                        new_syn = protocol == "tcp" and bool(int(transport.flags) & 0x02) and not bool(int(transport.flags) & 0x10)
                        timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                        syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                        if state.session_id is None or state.closed or timed_out or syn_starts_new:
                            flow_token = f"{protocol}|{ordered[0][0]}:{ordered[0][1]}|{ordered[1][0]}:{ordered[1][1]}"
                            state.session_id = f"{capture_id}|{flow_token}|session={state.sequence}"
                            state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                            state.sequence += 1
                            state.closed = False
                            state.syn_only = new_syn
                            state.initiator = source
                        uid = str(state.session_hash)
                        if uid in targets and len(packets.setdefault(uid, [])) < 30:
                            raw = bytes(packet)
                            caplen = len(raw)
                            wirelen = int(getattr(packet, "wirelen", 0) or caplen)
                            direction = 1 if source == state.initiator else 0
                            packets[uid].append((timestamp, caplen, wirelen, direction, raw))
                            matched_rows += 1
                        state.last_timestamp = timestamp
                        if not new_syn:
                            state.syn_only = False
                        if protocol == "tcp" and bool(int(transport.flags) & (0x01 | 0x04)):
                            state.closed = True
                stderr_path.write_text("Scapy PcapReader completed without parser exception.\n", encoding="utf-8")
            except Exception as exc:
                stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                raise RuntimeError(f"Scapy PcapReader failed for {pcap}; see {stderr_path}") from exc
            capture_audit.append({"capture_id": capture_id, "pcap_path": str(pcap), "pcap_sha256": sha256_file(pcap), "packet_rows": packet_rows, "matched_target_packet_rows_capped30": matched_rows, "stderr_path": str(stderr_path)})
            print(json.dumps({"capture": capture_id, "recovered": len(packets), "targets": len(targets)}), flush=True)

    missing = sorted(targets - packets.keys())
    if missing:
        write_json(OUT / "feature_cache" / "failure.json", {"missing": missing})
        raise RuntimeError(f"missing {len(missing)} target flows")

    flow_ids = np.asarray(sorted(targets))
    n = len(flow_ids)
    token_ids = np.zeros((n, 320), dtype=np.int32)
    segments = np.zeros((n, 320), dtype=np.uint8)
    fig_x = np.zeros((n, 30, 7), dtype=np.float32)
    fig_adj = np.zeros((n, 30, 30), dtype=np.uint8)
    fig_mask = np.zeros((n, 30), dtype=np.bool_)
    packet_counts = np.zeros(n, dtype=np.int32)
    services = np.asarray([target_meta[f]["service_label"] for f in flow_ids])

    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    tf_fmt = TrafficFormerFormat()
    fig_fmt = FigFormat()
    for i, uid in enumerate(flow_ids):
        p = packets[str(uid)]
        packet_counts[i] = len(p)
        encoded = encode_flow(p, tf_fmt)
        if encoded is None:
            raise RuntimeError(f"unexpected empty flow: {uid}")
        ids = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
        token_ids[i, :len(ids)] = ids
        segments[i, :len(ids)] = 1
        graph = build_flow_graph(str(uid), p, fig_fmt)
        m = graph.node_count
        fig_x[i, :m] = np.asarray(graph.features, dtype=np.float32)
        fig_mask[i, :m] = True
        for a, b in graph.edges:
            fig_adj[i, a, b] = fig_adj[i, b, a] = 1
    if not np.isfinite(fig_x).all():
        raise RuntimeError("non-finite FIG features")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, flow_ids=flow_ids, services=services, token_ids=token_ids, segments=segments, fig_x=fig_x, fig_adj=fig_adj, fig_mask=fig_mask, packet_counts=packet_counts)
    write_csv(OUT / "feature_cache" / "capture_audit.csv", capture_audit)
    write_json(OUT / "feature_cache" / "cache_audit.json", {
        "status": "PASS", "flows": n, "missing": 0, "captures": len(capture_audit),
        "sessionization": {"bidirectional": True, "timeout_seconds": timeout, "tcp_syn_fin_rst_boundaries": True},
        "trafficformer": {"max_packets": 5, "bytes_per_packet": 64, "ethernet_offset": 14, "seq_length": 320},
        "fig": {"max_packets": 30, "node_features": 7, "edges": "historical direction-burst rule"},
        "cache_sha256": sha256_file(CACHE), "manifest_sha256": sha256_file(MANIFEST),
        "known_test_and_unknown_test_opened": True,
        "opening_reason": "post-freeze Stage17 evaluation; no values used for fitting during cache creation",
    })
    print(json.dumps({"status": "PASS", "cache": str(CACHE), "sha256": sha256_file(CACHE), "flows": n}))


if __name__ == "__main__":
    main()
