#!/usr/bin/env python3
"""Recover Stage30 TrafficFormer/FIG inputs for one frozen VNAT Known protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scapy.all import IP, IPv6, PcapReader

from preflight import OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen
from build_vnat_mfr import stream_refs

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
from src.preprocessing.fig_graph import FigFormat, build_flow_graph  # noqa: E402
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.tokenizers import BertTokenizer  # noqa: E402

MANIFEST = PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"
VOCAB = PROJECT / "tf_runtime" / "code" / "models" / "encryptd_vocab.txt"


def ethernet_compatible_frame(raw: bytes, raw_ip: bool) -> bytes:
    """TrafficFormer skips 14 Ethernet bytes; preserve raw-IP captures via a synthetic envelope."""
    if not raw:
        raise ValueError("empty frozen packet")
    version = raw[0] >> 4
    if raw_ip and version in (4, 6):
        ethertype = b"\x08\x00" if version == 4 else b"\x86\xdd"
        return b"\x00" * 12 + ethertype + raw
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("medium_seed2025", "medium_seed2026"), required=True)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    if args.phase == "test":
        ensure_all_heads_frozen()
    suffix = "tf_fig" if args.phase == "trainval" else "tf_fig_test"
    result = OUT / "input_caches" / "vnat" / args.protocol / suffix
    if result.exists():
        raise RuntimeError(f"refusing to overwrite cache: {result}")
    result.mkdir(parents=True)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        splits = ("train", "validation") if args.phase == "trainval" else ("test",)
        rows = [r for r in csv.DictReader(handle) if r["protocol_id"] == args.protocol and
                r["class_role"] == "known" and r["split"] in splits]
    targets = {row["flow_uid"]: row for row in rows}
    if not targets or len(targets) != len(rows):
        raise RuntimeError("missing or duplicate frozen Known Train/Val flow")
    ids = sorted(targets)
    pos = {uid: i for i, uid in enumerate(ids)}
    np.save(result / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    n = len(ids)
    specifications = {
        "token_ids": (np.int32, (n, 320)),
        "segments": (np.uint8, (n, 320)),
        "fig_x": (np.float32, (n, 30, 7)),
        "fig_adj": (np.uint8, (n, 30, 30)),
        "fig_mask": (np.bool_, (n, 30)),
    }
    arrays = {}
    for name, (dtype, shape) in specifications.items():
        arrays[name] = np.lib.format.open_memmap(result / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        arrays[name][:] = 0
    by_pcap = defaultdict(dict)
    for uid, row in targets.items():
        pcap = Path(row["source_pcap_path"])
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        flow_id = row["source_flow_id"]
        if flow_id in by_pcap[pcap]:
            raise RuntimeError(f"duplicate frozen source flow in capture: {pcap}/{flow_id}")
        by_pcap[pcap][flow_id] = uid
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    tf_fmt, fig_fmt = TrafficFormerFormat(), FigFormat()
    found, capture_audit = set(), []
    for number, (pcap, wanted) in enumerate(sorted(by_pcap.items(), key=lambda pair: str(pair[0])), 1):
        refs = stream_refs(pcap, set(wanted), limit=30)
        frame_to_uid = defaultdict(list)
        for flow_id, frames in refs.items():
            for frame in frames:
                frame_to_uid[frame].append(wanted[flow_id])
        last = max(frame_to_uid)
        packets = defaultdict(list)
        seen = 0
        with PcapReader(str(pcap)) as reader:
            raw_ip = reader.linktype == 101
            for frame, packet in enumerate(reader, 1):
                seen = frame
                for uid in frame_to_uid.get(frame, ()):
                    network = packet.getlayer(IP) or packet.getlayer(IPv6)
                    if network is None:
                        raise RuntimeError(f"frozen IP flow has non-IP frame {pcap}:{frame}:{uid}")
                    raw = bytes(packet)
                    initiator = packets[uid][0][5] if packets[uid] else str(network.src)
                    caplen = len(raw)
                    wirelen = int(getattr(packet, "wirelen", 0) or caplen)
                    packets[uid].append((float(packet.time), caplen, wirelen,
                                         int(str(network.src) == initiator),
                                         ethernet_compatible_frame(raw, raw_ip), initiator))
                if frame >= last:
                    break
        for flow_id, uid in wanted.items():
            selected = packets[uid]
            expected = min(30, int(targets[uid]["packet_count"]))
            if len(selected) != expected:
                raise RuntimeError(f"packet coverage mismatch {pcap}/{flow_id}: {len(selected)} != {expected}")
            if uid in found:
                raise RuntimeError(f"duplicate recovered UID {uid}")
            flow = [item[:5] for item in selected]
            i = pos[uid]
            encoded = encode_flow(flow, tf_fmt)
            if encoded is None:
                raise RuntimeError(f"empty TrafficFormer flow {uid}")
            tokens = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
            arrays["token_ids"][i, :len(tokens)] = tokens
            arrays["segments"][i, :len(tokens)] = 1
            graph = build_flow_graph(uid, flow, fig_fmt)
            count = graph.node_count
            if not 1 <= count <= 30:
                raise RuntimeError(f"invalid FIG node count {uid}: {count}")
            arrays["fig_x"][i, :count] = np.asarray(graph.features, dtype=np.float32)
            arrays["fig_mask"][i, :count] = True
            for a, b in graph.edges:
                arrays["fig_adj"][i, a, b] = arrays["fig_adj"][i, b, a] = 1
            found.add(uid)
        capture_audit.append({"pcap": str(pcap), "frozen_flows": len(wanted),
                              "last_required_frame": last, "last_read_frame": seen})
        print(json.dumps({"protocol": args.protocol, "capture": number,
                          "total_captures": len(by_pcap), "flows_seen": len(found)}), flush=True)
    if set(targets) != found:
        missing = sorted(set(targets) - found)
        (result / "failure.json").write_text(json.dumps({"missing": len(missing), "sample": missing[:50]}))
        raise RuntimeError(f"missing {len(missing)} frozen VNAT flows")
    for array in arrays.values():
        array.flush()
    audit = {"status": "PASS", "dataset": "vnat", "protocol": args.protocol,
             "known_train": sum(r["split"] == "train" for r in rows),
             "known_validation": sum(r["split"] == "validation" for r in rows),
             "known_test": sum(r["split"] == "test" for r in rows),
             "flows": n, "split_manifest_sha256": sha(MANIFEST), "capture_audit": capture_audit,
             "max_tf_packets": 5, "max_fig_packets": 30,
             "raw_ip_ethernet_envelope": "14-byte synthetic zero-address Ethernet header for DLT_RAW only; TrafficFormer strips it",
             "known_test_features_materialized": len(rows) if args.phase == "test" else 0,
             "unknown_features_materialized": 0,
             "raw_captures_scanned_for_selected_flow_ids": True}
    (result / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "protocol": args.protocol, "flows": n}), flush=True)


if __name__ == "__main__":
    main()
