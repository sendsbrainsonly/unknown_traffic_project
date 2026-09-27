#!/usr/bin/env python3
"""Reconstruct TrafficFormer tokens and FIG graphs on frozen Stage12 Known flows."""
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

from preflight import OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
from src.preprocessing.fig_graph import FigFormat, build_flow_graph  # noqa: E402
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.tokenizers import BertTokenizer  # noqa: E402

STAGE12 = PROJECT / "stage12_dual_external_validation"
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
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    if args.phase == "test":
        ensure_all_heads_frozen()
    dataset = args.dataset
    result = OUT / "input_caches" / dataset / ("tf_fig" if args.phase == "trainval" else "tf_fig_test")
    if result.exists():
        raise RuntimeError(f"refusing to overwrite cache: {result}")
    result.mkdir(parents=True)
    split_path = STAGE12 / "protocol" / dataset / "split_manifest.csv"
    with split_path.open(newline="", encoding="utf-8") as handle:
        roles = ("known_train", "known_validation") if args.phase == "trainval" else ("known_test",)
        selected = [r for r in csv.DictReader(handle) if r["setting"] == "medium" and
                    r["role"] in roles]
    wanted = {r["flow_id_sha256"]: r for r in selected}
    if not wanted or len(wanted) != len(selected):
        raise RuntimeError("missing or duplicate frozen Known flows")
    ordered = sorted(wanted)
    index = {uid: i for i, uid in enumerate(ordered)}
    n = len(ordered)
    np.save(result / "flow_ids.npy", np.asarray(ordered, dtype="U80"), allow_pickle=False)
    token_ids = np.lib.format.open_memmap(result / "token_ids.npy", mode="w+", dtype=np.int32, shape=(n,320))
    segments = np.lib.format.open_memmap(result / "segments.npy", mode="w+", dtype=np.uint8, shape=(n,320))
    fig_x = np.lib.format.open_memmap(result / "fig_x.npy", mode="w+", dtype=np.float32, shape=(n,30,7))
    fig_adj = np.lib.format.open_memmap(result / "fig_adj.npy", mode="w+", dtype=np.uint8, shape=(n,30,30))
    fig_mask = np.lib.format.open_memmap(result / "fig_mask.npy", mode="w+", dtype=np.bool_, shape=(n,30))
    for array in (token_ids, segments, fig_x, fig_adj, fig_mask):
        array[:] = 0
    config = json.loads((STAGE12 / "configs" / "stage12_config.json").read_text())
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    pcap_root = Path(config["datasets"][dataset]["pcap_root"])
    manifest = json.loads((STAGE12 / "protocol" / dataset / "preprocessing_manifest.json").read_text())
    wanted_files = {r["source_file"] for r in selected}
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    tf_fmt, fig_fmt = TrafficFormerFormat(), FigFormat()
    found = set()
    captures = []
    for cls in sorted(manifest["classes"], key=lambda item: int(item["label"])):
        class_name = str(cls["name"])
        for capture_index, raw_path in enumerate(cls["captures"]):
            pcap = Path(raw_path)
            source_file = str(pcap.relative_to(pcap_root))
            if source_file not in wanted_files:
                continue
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            states = {}
            packets = {}
            packet_rows = matched_packet_rows = 0
            with PcapReader(str(pcap)) as reader:
                for packet in reader:
                    packet_rows += 1
                    ep = endpoint(packet)
                    if ep is None:
                        continue
                    source, destination, protocol, transport = ep
                    ordered_endpoints = tuple(sorted((source, destination)))
                    key = (capture_id, protocol, ordered_endpoints[0][0], ordered_endpoints[0][1],
                           ordered_endpoints[1][0] + ":" + ordered_endpoints[1][1])
                    timestamp = float(packet.time)
                    state = states.setdefault(key, FlowState())
                    new_syn = protocol == "tcp" and bool(int(transport.flags)&0x02) and not bool(int(transport.flags)&0x10)
                    timed_out = state.last_timestamp is not None and timestamp-state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                    if state.session_id is None or state.closed or timed_out or syn_starts_new:
                        flow_token = (f"{protocol}|{ordered_endpoints[0][0]}:{ordered_endpoints[0][1]}|"
                                      f"{ordered_endpoints[1][0]}:{ordered_endpoints[1][1]}")
                        state.session_id = f"{capture_id}|{flow_token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                        state.initiator = source
                    uid = str(state.session_hash)
                    if uid in wanted and len(packets.setdefault(uid, [])) < 30:
                        raw = bytes(packet)
                        caplen = len(raw)
                        wirelen = int(getattr(packet,"wirelen",0) or caplen)
                        direction = 1 if source == state.initiator else 0
                        packets[uid].append((timestamp,caplen,wirelen,direction,raw))
                        matched_packet_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and bool(int(transport.flags)&(0x01|0x04)):
                        state.closed = True
            for uid, flow in packets.items():
                if uid in found:
                    raise RuntimeError(f"duplicate reconstructed flow: {uid}")
                if wanted[uid]["source_file"] != source_file:
                    raise RuntimeError(f"source capture mismatch: {uid}")
                position = index[uid]
                encoded = encode_flow(flow, tf_fmt)
                if encoded is None:
                    raise RuntimeError(f"empty TrafficFormer flow: {uid}")
                ids = tokenizer.convert_tokens_to_ids([CLS_TOKEN]+tokenizer.tokenize(encoded.text))[:320]
                token_ids[position,:len(ids)] = ids
                segments[position,:len(ids)] = 1
                graph = build_flow_graph(uid, flow, fig_fmt)
                nodes = graph.node_count
                fig_x[position,:nodes] = np.asarray(graph.features,dtype=np.float32)
                fig_mask[position,:nodes] = True
                for a,b in graph.edges:
                    fig_adj[position,a,b] = fig_adj[position,b,a] = 1
                found.add(uid)
            captures.append({"pcap":str(pcap),"source_file":source_file,"packets_read":packet_rows,
                             "matched_packet_rows_capped30":matched_packet_rows,"frozen_flows":len(packets)})
            print(json.dumps({"dataset":dataset,"capture":len(captures),"found":len(found),
                              "target":n,"packets_read":packet_rows}),flush=True)
    missing = sorted(set(wanted)-found)
    if missing:
        (result/"failure.json").write_text(json.dumps({"missing_count":len(missing),"sample":missing[:100]},indent=2))
        raise RuntimeError(f"failed to recover {len(missing)} frozen flow IDs")
    for array in (token_ids,segments,fig_x,fig_adj,fig_mask):
        array.flush()
    if not np.isfinite(fig_x).all() or not np.all(fig_mask.sum(1)>0):
        raise RuntimeError("invalid FIG features")
    audit = {"status":"PASS","dataset":dataset,"protocol":"medium_seed2022",
             "known_flows":n,"known_train":sum(r["role"]=="known_train" for r in selected),
             "known_validation":sum(r["role"]=="known_validation" for r in selected),
             "known_test":sum(r["role"]=="known_test" for r in selected),
             "split_sha256":sha(split_path),"source_captures":captures,"source_capture_count":len(captures),
             "trafficformer":{"max_packets":5,"bytes_per_packet":64,"sequence_length":320},
             "fig":{"max_packets":30,"node_features":7},
             "known_test_features_materialized":n if args.phase == "test" else 0,
             "unknown_features_materialized":0,
             "raw_capture_streams_scanned_for_selected_flow_ids":True}
    (result/"cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":"PASS","dataset":dataset,"flows":n,"captures":len(captures)}),flush=True)


if __name__ == "__main__":
    main()
