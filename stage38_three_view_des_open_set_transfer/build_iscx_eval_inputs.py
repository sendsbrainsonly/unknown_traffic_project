#!/usr/bin/env python3
"""Recover Stage12 Known/Unknown Test packets after Stage38B head freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scapy.all import PcapReader
from scapy.utils import RawPcapReader

from audit_iscx_known import ROOT, PROJECT, S12, S31, sha
from run_iscx_known import lock

sys.path.insert(0, str(S31))
sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table/scripts"))
from build_iscx_tf_fig import (  # noqa: E402
    BertTokenizer, CLS_TOKEN, FigFormat, FlowState, TrafficFormerFormat,
    VOCAB, build_flow_graph, encode_flow, endpoint,
)
from build_yatc_mfr import author_mfr_packet  # noqa: E402

PROTOCOL = "medium_seed2022"
ROLES = ("known_test", "unknown_test")
STAGE12 = PROJECT / "stage12_dual_external_validation"


def frozen_test_rows(dataset: str) -> tuple[list[dict], dict]:
    unit = lock(dataset)
    run = ROOT / "runs" / dataset / PROTOCOL
    fusion = run / "T0_equal"
    if not (fusion / "SUCCESS").is_file():
        raise RuntimeError("Known-only fusion checkpoint must be selected before Test extraction")
    metrics = json.loads((fusion / "known_validation_metrics.json").read_text())
    for name, digest in metrics["checkpoint_hashes"].items():
        if sha(fusion / name) != digest:
            raise RuntimeError(f"frozen fusion checkpoint changed: {name}")
    for branch in ("trafficformer", "graph", "yatc"):
        sub = run / branch
        meta = json.loads((sub / "metrics.json").read_text())
        if not (sub / "SUCCESS").is_file() or sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"frozen branch checkpoint changed: {branch}")
    manifest = S12 / dataset / "split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["setting"] == "medium" and r["role"] in ROLES]
    counts = Counter(r["role"] for r in rows)
    if any(counts[role] != unit["role_counts"][role] for role in ROLES):
        raise RuntimeError(f"frozen Test role counts changed: {counts}")
    ids = [r["flow_id_sha256"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate frozen Test flow ID")
    if {r["canonical_class"] for r in rows if r["role"] == "unknown_test"} != set(unit["unknown_applications"]):
        raise RuntimeError("Unknown application set changed")
    return sorted(rows, key=lambda r: r["flow_id_sha256"]), unit


def save_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_tf_fig(dataset: str) -> None:
    rows, unit = frozen_test_rows(dataset)
    result = ROOT / "input_caches" / dataset / "tf_fig_eval"
    if result.exists():
        raise FileExistsError(result)
    result.mkdir(parents=True)
    wanted = {r["flow_id_sha256"]: r for r in rows}
    ids = sorted(wanted)
    index = {uid: i for i, uid in enumerate(ids)}
    n = len(ids)
    np.save(result / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    arrays = {
        "token_ids": np.lib.format.open_memmap(result / "token_ids.npy", "w+", dtype=np.int32, shape=(n, 320)),
        "segments": np.lib.format.open_memmap(result / "segments.npy", "w+", dtype=np.uint8, shape=(n, 320)),
        "fig_x": np.lib.format.open_memmap(result / "fig_x.npy", "w+", dtype=np.float32, shape=(n, 30, 7)),
        "fig_adj": np.lib.format.open_memmap(result / "fig_adj.npy", "w+", dtype=np.uint8, shape=(n, 30, 30)),
        "fig_mask": np.lib.format.open_memmap(result / "fig_mask.npy", "w+", dtype=np.bool_, shape=(n, 30)),
    }
    for array in arrays.values():
        array[:] = 0
    config = json.loads((STAGE12 / "configs/stage12_config.json").read_text())
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    pcap_root = Path(config["datasets"][dataset]["pcap_root"])
    source = json.loads((S12 / dataset / "preprocessing_manifest.json").read_text())
    files = {r["source_file"] for r in rows}
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    tf_fmt, fig_fmt = TrafficFormerFormat(), FigFormat()
    found: set[str] = set()
    captures: list[dict] = []
    for cls in sorted(source["classes"], key=lambda item: int(item["label"])):
        class_name = str(cls["name"])
        for capture_index, raw_path in enumerate(cls["captures"]):
            pcap = Path(raw_path)
            source_file = str(pcap.relative_to(pcap_root))
            if source_file not in files:
                continue
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            states: dict = {}
            packets: dict[str, list] = {}
            packet_rows = matched = 0
            with PcapReader(str(pcap)) as reader:
                for packet in reader:
                    packet_rows += 1
                    ep = endpoint(packet)
                    if ep is None:
                        continue
                    source_ep, destination, protocol, transport = ep
                    ordered = tuple(sorted((source_ep, destination)))
                    key = (capture_id, protocol, ordered[0][0], ordered[0][1],
                           ordered[1][0] + ":" + ordered[1][1])
                    timestamp = float(packet.time)
                    state = states.setdefault(key, FlowState())
                    new_syn = protocol == "tcp" and bool(int(transport.flags) & 0x02) and not bool(int(transport.flags) & 0x10)
                    timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                    if state.session_id is None or state.closed or timed_out or syn_starts_new:
                        token = (f"{protocol}|{ordered[0][0]}:{ordered[0][1]}|"
                                 f"{ordered[1][0]}:{ordered[1][1]}")
                        state.session_id = f"{capture_id}|{token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                        state.initiator = source_ep
                    uid = str(state.session_hash)
                    if uid in wanted and len(packets.setdefault(uid, [])) < 30:
                        raw = bytes(packet)
                        caplen = len(raw)
                        wirelen = int(getattr(packet, "wirelen", 0) or caplen)
                        packets[uid].append((timestamp, caplen, wirelen,
                                             1 if source_ep == state.initiator else 0, raw))
                        matched += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and bool(int(transport.flags) & (0x01 | 0x04)):
                        state.closed = True
            for uid, flow in packets.items():
                if uid in found or wanted[uid]["source_file"] != source_file:
                    raise RuntimeError(f"duplicate or mismatched reconstructed Test flow: {uid}")
                position = index[uid]
                encoded = encode_flow(flow, tf_fmt)
                if encoded is None:
                    raise RuntimeError(f"empty TrafficFormer Test flow: {uid}")
                token_ids = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
                arrays["token_ids"][position, :len(token_ids)] = token_ids
                arrays["segments"][position, :len(token_ids)] = 1
                graph = build_flow_graph(uid, flow, fig_fmt)
                nodes = graph.node_count
                arrays["fig_x"][position, :nodes] = np.asarray(graph.features, dtype=np.float32)
                arrays["fig_mask"][position, :nodes] = True
                for a, b in graph.edges:
                    arrays["fig_adj"][position, a, b] = arrays["fig_adj"][position, b, a] = 1
                found.add(uid)
            captures.append({"source_file": source_file, "packets_read": packet_rows,
                             "matched_packet_rows_capped30": matched, "frozen_flows": len(packets)})
            print(json.dumps({"dataset": dataset, "capture": len(captures), "found": len(found),
                              "target": n, "packets_read": packet_rows}), flush=True)
    missing = sorted(set(wanted) - found)
    if missing:
        save_json(result / "failure.json", {"missing_count": len(missing), "sample": missing[:100]})
        raise RuntimeError(f"failed to recover {len(missing)} frozen Test flow IDs")
    for array in arrays.values():
        array.flush()
    if not np.isfinite(arrays["fig_x"]).all() or not np.all(arrays["fig_mask"].sum(1) > 0):
        raise RuntimeError("invalid FIG Test features")
    save_json(result / "cache_audit.json", {"status": "PASS", "dataset": dataset,
        "protocol": PROTOCOL, "roles": {r: unit["role_counts"][r] for r in ROLES},
        "split_sha256": sha(S12 / dataset / "split_manifest.csv"),
        "source_capture_count": len(captures), "source_captures": captures,
        "session_timeout_seconds": timeout, "max_packets": 30,
        "trafficformer_packets": 5, "trafficformer_bytes_per_packet": 64,
        "model_and_threshold_frozen_before_read": True})


def build_mfr(dataset: str) -> None:
    rows, unit = frozen_test_rows(dataset)
    result = ROOT / "input_caches" / dataset / "yatc_mfr_eval"
    if result.exists():
        raise FileExistsError(result)
    result.mkdir(parents=True)
    pool_candidates = [p for p in (STAGE12 / "artifacts" / dataset / "preprocessing_pool").glob("raw_prepared_parallel_v*")
                       if (p / "provenance_train.jsonl").is_file()]
    if len(pool_candidates) != 1:
        raise RuntimeError(f"expected one frozen packet provenance pool, got {pool_candidates}")
    pool = pool_candidates[0]
    target = {r["flow_id_sha256"]: r for r in rows}
    refs = {}
    for part in ("train", "validation", "test"):
        with (pool / f"provenance_{part}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                uid = item["flow_id_sha256"]
                if uid not in target:
                    continue
                if uid in refs or item["image_sha256"] != target[uid]["image_sha256"]:
                    raise RuntimeError(f"duplicate or mismatched packet provenance: {uid}")
                packet_refs = item["packet_refs"][:5]
                if not packet_refs:
                    raise RuntimeError(f"empty packet refs: {uid}")
                refs[uid] = packet_refs
    if set(refs) != set(target):
        raise RuntimeError(f"missing frozen Test packet refs: {len(set(target)-set(refs))}")
    config = json.loads((STAGE12 / "configs/stage12_config.json").read_text())
    pcap_root = Path(config["datasets"][dataset]["pcap_root"])
    by_pcap = defaultdict(lambda: defaultdict(list))
    for uid, row in target.items():
        pcap = pcap_root / row["source_file"]
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        for slot, ref in enumerate(refs[uid]):
            by_pcap[pcap][int(ref["packet_number"])].append((uid, slot))
    ids = sorted(target)
    index = {uid: i for i, uid in enumerate(ids)}
    np.save(result / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    mfr = np.lib.format.open_memmap(result / "mfr.npy", "w+", dtype=np.uint8, shape=(len(ids), 40, 40))
    mfr[:] = 0
    found = Counter()
    captures = []
    for ordinal, (pcap, packet_map) in enumerate(sorted(by_pcap.items(), key=lambda x: str(x[0])), 1):
        last = max(packet_map)
        scanned = 0
        with RawPcapReader(str(pcap)) as reader:
            for number, (raw, _) in enumerate(reader, 1):
                scanned = number
                for uid, slot in packet_map.get(number, ()):
                    content = np.frombuffer(author_mfr_packet(raw), dtype=np.uint8)
                    mfr[index[uid]].reshape(-1)[slot * 320:(slot + 1) * 320] = content
                    found[uid] += 1
                if number >= last:
                    break
        captures.append({"source_file": str(pcap.relative_to(pcap_root)),
                         "last_packet_read": scanned, "last_packet_required": last,
                         "references": sum(map(len, packet_map.values()))})
        print(json.dumps({"dataset": dataset, "capture": ordinal,
                          "total_captures": len(by_pcap), "last_packet_read": scanned}), flush=True)
    missing = [uid for uid in ids if found[uid] != len(refs[uid])]
    if missing:
        save_json(result / "failure.json", {"missing_count": len(missing), "sample": missing[:50]})
        raise RuntimeError(f"MFR Test packet coverage failed for {len(missing)} flows")
    mfr.flush()
    save_json(result / "cache_audit.json", {"status": "PASS", "dataset": dataset,
        "protocol": PROTOCOL, "roles": {r: unit["role_counts"][r] for r in ROLES},
        "split_sha256": sha(S12 / dataset / "split_manifest.csv"),
        "pool_provenance": str(pool), "first_five_packet_refs": sum(map(len, refs.values())),
        "source_capture_count": len(captures), "source_captures": captures,
        "model_and_threshold_frozen_before_read": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--phase", choices=("tf_fig", "mfr"), required=True)
    args = parser.parse_args()
    if args.phase == "tf_fig":
        build_tf_fig(args.dataset)
    else:
        build_mfr(args.dataset)
