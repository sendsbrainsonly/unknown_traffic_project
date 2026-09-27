#!/usr/bin/env python3
"""Recover matched CIC packet inputs for the unchanged TF/FIG/YaTC branches.

Train/Validation and Test are separate invocations.  Test is gated on frozen
branch/head hashes.  Original PCAPs and mapping Parquets are read only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cic_protocol as protocol

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CIC = ROOT / "cicids2017"
MANIFEST = ROOT / "cicids2017_three_class_manifest_v2.csv"
DATA = protocol.DATA
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime/code"))
sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table/scripts"))
sys.path.insert(0, str(PROJECT / "stage31_four_dataset_three_view_equal"))
sys.path.insert(0, str(PROJECT / "scripts"))
from src.preprocessing.fig_graph import FigFormat, build_flow_graph  # noqa: E402
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.tokenizers import BertTokenizer  # noqa: E402
from build_yatc_mfr import author_mfr_packet  # noqa: E402
from task11_cicids2017_prepare import packet_key  # noqa: E402

VOCAB = PROJECT / "tf_runtime/code/models/encryptd_vocab.txt"


def read_manifest(phase):
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    subset = [r for r in rows if r["split"] in (("train", "validation") if phase == "trainval" else ("test",))]
    if not subset or len(subset) != len({r["flow_id"] for r in subset}):
        raise RuntimeError("empty/duplicate CIC packet target")
    if {r["label"] for r in subset} != set(protocol.CLASSES):
        raise RuntimeError("not all CIC labels present in phase")
    return subset


def frozen_head_gate():
    import ustc_runner as runner
    run = CIC / "runs/ustc/A-2"
    for branch in ("trafficformer", "graph", "yatc"):
        sub = run / branch
        meta = json.loads((sub / "metrics.json").read_text())
        if not (sub / "SUCCESS").is_file() or runner.sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"CIC {branch} not frozen")
    head = run / "T0_equal"
    selected = json.loads((head / "known_validation_metrics.json").read_text())
    for name, expected in selected["checkpoint_hashes"].items():
        if runner.sha(head / name) != expected:
            raise RuntimeError(f"CIC T0 {name} changed")
    return selected


def alloc_cache(phase, rows):
    base = CIC / "input_caches/ustc/A-2"
    tf_path = base / ("tf_fig" if phase == "trainval" else "tf_fig_test")
    mfr_path = base / ("yatc_mfr" if phase == "trainval" else "yatc_mfr_test")
    if tf_path.exists() or mfr_path.exists():
        raise FileExistsError("CIC packet cache already exists; no overwrite")
    tf_path.mkdir(parents=True)
    mfr_path.mkdir(parents=True)
    ids = sorted(r["flow_id"] for r in rows)
    index = {uid: i for i, uid in enumerate(ids)}
    np.save(tf_path / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    n = len(ids)
    spec = {"token_ids": (np.int32, (n, 320)), "segments": (np.uint8, (n, 320)),
            "fig_x": (np.float32, (n, 30, 7)), "fig_adj": (np.uint8, (n, 30, 30)),
            "fig_mask": (np.bool_, (n, 30))}
    tf_arrays = {}
    for name, (dtype, shape) in spec.items():
        tf_arrays[name] = np.lib.format.open_memmap(tf_path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        tf_arrays[name][:] = 0
    mfr_arrays = {}
    mfr_index = {}
    for role, split in (("known_train", "train"), ("known_validation", "validation"), ("known_test", "test")):
        if phase == "test" and split != "test" or phase == "trainval" and split == "test":
            continue
        role_ids = sorted(r["flow_id"] for r in rows if r["split"] == split)
        mfr_index[split] = {uid: i for i, uid in enumerate(role_ids)}
        np.save(mfr_path / f"{role}_flow_ids.npy", np.asarray(role_ids, dtype="U80"), allow_pickle=False)
        mfr_arrays[split] = np.lib.format.open_memmap(mfr_path / f"{role}_mfr.npy", mode="w+",
                                                     dtype=np.uint8, shape=(len(role_ids), 40, 40))
        mfr_arrays[split][:] = 0
    return tf_path, mfr_path, tf_arrays, mfr_arrays, index, mfr_index


def direction(raw, dpkt, first_endpoint):
    frame = dpkt.ethernet.Ethernet(raw)
    net = frame.data
    if not isinstance(net, (dpkt.ip.IP, dpkt.ip6.IP6)):
        raise RuntimeError("non-IP packet in matched CIC flow")
    trans = net.data
    port = int(trans.sport) if isinstance(trans, (dpkt.tcp.TCP, dpkt.udp.UDP)) else 0
    endpoint = (bytes(net.src), port)
    return int(endpoint == first_endpoint), endpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("trainval", "test"), required=True)
    args = parser.parse_args()
    if args.phase == "test":
        frozen_head_gate()
    if protocol.digest(MANIFEST) != json.loads((ROOT / "cicids2017_protocol_v2_audit.json").read_text())["v2_manifest_sha256"]:
        raise RuntimeError("CIC sampled manifest changed")
    sys.path.insert(0, str(DATA / "cicids2017_label_mapping/vendor"))
    import dpkt  # noqa: E402

    rows = read_manifest(args.phase)
    tf_path, mfr_path, tf_arrays, mfr_arrays, tf_index, mfr_index = alloc_cache(args.phase, rows)
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    tf_format, fig_format = TrafficFormerFormat(), FigFormat()
    seen = set()
    day_audit = []

    def emit(row):
        uid = row["flow_id"]
        observed = row.pop("_observed")
        packets = row.pop("_packets")
        row.pop("_first_endpoint", None)
        if observed != int(row["packet_count"]) or len(packets) != min(30, observed):
            raise RuntimeError(f"{uid}: selected flow packet-count mismatch {observed}/{row['packet_count']}")
        i = tf_index[uid]
        encoded = encode_flow(packets, tf_format)
        if encoded is None:
            raise RuntimeError(f"TrafficFormer filtered selected flow {uid}")
        tokens = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
        tf_arrays["token_ids"][i, :len(tokens)] = tokens
        tf_arrays["segments"][i, :len(tokens)] = 1
        graph = build_flow_graph(uid, packets, fig_format)
        count = graph.node_count
        tf_arrays["fig_x"][i, :count] = np.asarray(graph.features, dtype=np.float32)
        tf_arrays["fig_mask"][i, :count] = True
        for a, b in graph.edges:
            tf_arrays["fig_adj"][i, a, b] = tf_arrays["fig_adj"][i, b, a] = 1
        split = row["split"]
        image = mfr_arrays[split][mfr_index[split][uid]].reshape(-1)
        for slot, (_, _, _, _, raw) in enumerate(packets[:5]):
            image[slot*320:(slot+1)*320] = np.frombuffer(author_mfr_packet(raw), dtype=np.uint8)
        seen.add(uid)

    try:
        for day in protocol.DAYS:
            selected = [r for r in rows if r["day"] == day]
            if not selected:
                continue
            by_key = defaultdict(list)
            for row in selected:
                row["_observed"] = 0
                row["_packets"] = []
                by_key[row["canonical_flow_key"]].append(row)
            for intervals in by_key.values():
                intervals.sort(key=lambda r: int(r["first_packet_index"]))
            positions = {key: 0 for key in by_key}
            pcaps = {r["source_pcap"] for r in selected}
            if len(pcaps) != 1:
                raise RuntimeError(f"{day}: multiple source PCAPs")
            pcap = DATA / "PCAPs" / next(iter(pcaps))
            elapsed = time.monotonic()
            packet_seen = 0
            with pcap.open("rb") as f:
                reader = dpkt.pcapng.Reader(f)
                if reader.datalink() != dpkt.pcap.DLT_EN10MB:
                    raise RuntimeError("CIC PCAP link type is not Ethernet")
                for number, (timestamp, raw) in enumerate(reader):
                    packet_seen += 1
                    key = packet_key(raw, dpkt)
                    if key is None or key not in by_key:
                        continue
                    intervals = by_key[key]
                    pos = positions[key]
                    if pos >= len(intervals):
                        continue
                    row = intervals[pos]
                    first = int(row["first_packet_index"])
                    last = int(row["last_packet_index"])
                    if number > last:
                        raise RuntimeError(f"missed CIC final packet: {row['flow_id']}")
                    if first <= number <= last:
                        _, endpoint = direction(raw, dpkt, row.get("_first_endpoint"))
                        if "_first_endpoint" not in row:
                            row["_first_endpoint"] = endpoint
                        sign, _ = direction(raw, dpkt, row["_first_endpoint"])
                        row["_observed"] += 1
                        if len(row["_packets"]) < 30:
                            row["_packets"].append((float(timestamp), len(raw), len(raw), sign, bytes(raw)))
                        if number == last:
                            emit(row)
                            positions[key] = pos + 1
                    if packet_seen % 5_000_000 == 0:
                        print(json.dumps({"phase": args.phase, "day": day, "packets": packet_seen,
                                          "flows_emitted": len(seen), "target": len(rows)}), flush=True)
                    if len(seen) == len(rows):
                        break
            unfinished = sum(len(by_key[k]) - positions[k] for k in by_key)
            if unfinished:
                raise RuntimeError(f"{day}: {unfinished} selected flow intervals unfinished")
            day_audit.append({"day": day, "source_pcap": str(pcap), "packets_scanned": packet_seen,
                              "flows_emitted": len(selected), "seconds": time.monotonic() - elapsed})
            print(json.dumps(day_audit[-1]), flush=True)
        if seen != {r["flow_id"] for r in rows}:
            raise RuntimeError(f"CIC packet extraction incomplete: {len(seen)}/{len(rows)}")
        for arr in (*tf_arrays.values(), *mfr_arrays.values()):
            arr.flush()
        if not np.isfinite(tf_arrays["fig_x"]).all():
            raise RuntimeError("nonfinite CIC FIG node features")
        audit = {"status": "PASS", "dataset": "cicids2017", "phase": args.phase,
                 "sample_manifest_sha256": protocol.digest(MANIFEST), "flows": len(rows),
                 "days": day_audit, "max_tf_packets": 5, "max_graph_packets": 30,
                 "max_yatc_packets": 5, "ipv6_mfr_adaptations": 0,
                 "known_test_features_transformed": len(rows) if args.phase == "test" else 0,
                 "unknown_samples_used": 0}
        for cache in (tf_path, mfr_path):
            (cache / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "phase": args.phase, "flows": len(rows)}), flush=True)
    except BaseException as exc:
        failure = {"status": "FAIL", "phase": args.phase, "error": repr(exc),
                   "flows_emitted": len(seen), "target": len(rows), "days": day_audit}
        (tf_path / "FAILURE.json").write_text(json.dumps(failure, indent=2) + "\n")
        (mfr_path / "FAILURE.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
