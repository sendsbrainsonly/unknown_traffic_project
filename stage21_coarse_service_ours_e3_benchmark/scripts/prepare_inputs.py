#!/usr/bin/env python3
"""Build exact Stage20-membership TrafficFormer+FIG T8 inputs.

The existing Stage19 cache supplies anonymized IP/transport/payload bytes for
all covered flows. Exact Stage12 packet timestamps supply FIG timing. A small
set of Stage20-only TOR flows is recovered from the original captures using its
frozen packet references. No label or split information enters feature values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scapy.all import RawPcapReader

from stage21_common import CACHE_ROOT, OUT, PROJECT, cache_path, dataset_rows, read_json, sha256_file, write_json

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
sys.path.insert(0, str(PROJECT / "stage19_ronetc_four_dataset_comparison" / "scripts"))
from build_byte_cache import packet_views
from src.preprocessing.fig_graph import FigFormat, build_flow_graph
from src.preprocessing.trafficformer_input import official_bigram_generation
from uer.utils.constants import CLS_TOKEN
from uer.utils.tokenizers import BertTokenizer


VOCAB = PROJECT / "tf_runtime" / "code" / "models" / "encryptd_vocab.txt"
STAGE19_CACHE = PROJECT / "stage19_ronetc_four_dataset_comparison" / "byte_cache"
STAGE12 = PROJECT / "stage12_dual_external_validation"


def load_provenance(rows: list[dict[str, str]]) -> dict[str, dict]:
    files: dict[tuple[str, str], list[dict]] = {}
    result = {}
    for row in rows:
        key = (row["source_pool"], row["source_role"])
        if key not in files:
            path = Path(key[0]) / f"provenance_{key[1]}.jsonl"
            with path.open(encoding="utf-8") as handle:
                files[key] = [json.loads(line) for line in handle if line.strip()]
        record = files[key][int(row["source_index"])]
        if record["flow_id_sha256"] != row["flow_id"]:
            raise RuntimeError(f"source provenance mismatch: {row['flow_id']}")
        result[row["flow_id"]] = record
    if len(result) != len(rows):
        raise RuntimeError("duplicate flow id in Stage20 closed manifest")
    return result


def capture_map(dataset: str) -> dict[str, Path]:
    prep = read_json(STAGE12 / "protocol" / dataset / "preprocessing_manifest.json")
    result = {}
    for class_spec in sorted(prep["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            path = Path(raw_path)
            capture_id = f"class={class_name}|capture={capture_index}|name={path.name}"
            result[capture_id] = path
    return result


def recover_missing_views(dataset: str, missing: set[str], provenance: dict[str, dict]) -> tuple[dict[str, np.ndarray], list[dict]]:
    captures = capture_map(dataset)
    wanted: dict[Path, dict[int, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    result = {flow_id: np.zeros((3, 8, 64), dtype=np.uint8) for flow_id in missing}
    for flow_id in missing:
        refs = provenance[flow_id]["packet_refs"][:8]
        for slot, ref in enumerate(refs):
            capture_id = str(ref["capture"])
            if capture_id not in captures:
                raise KeyError(f"missing capture mapping: {capture_id}")
            wanted[captures[capture_id]][int(ref["packet_number"])].append((flow_id, slot))
    audit = []
    for pcap, packet_map in sorted(wanted.items(), key=lambda item: str(item[0])):
        recovered = 0
        with RawPcapReader(str(pcap)) as reader:
            for packet_number, (raw_bytes, _metadata) in enumerate(reader, start=1):
                targets = packet_map.get(packet_number)
                if not targets:
                    continue
                views = packet_views(raw_bytes, 64)
                for flow_id, slot in targets:
                    for view_index, values in enumerate(views):
                        result[flow_id][view_index, slot] = values
                    recovered += 1
        expected = sum(map(len, packet_map.values()))
        if recovered != expected:
            raise RuntimeError(f"packet recovery mismatch: {pcap} {recovered}/{expected}")
        audit.append({"pcap": str(pcap), "pcap_sha256": sha256_file(pcap), "target_packets": expected, "recovered_packets": recovered})
    return result, audit


def ip_layer_from_views(values: np.ndarray) -> bytes:
    ip = bytes(values[0])
    transport = bytes(values[1])
    payload = bytes(values[2])
    version = ip[0] >> 4 if ip else 0
    if version == 4:
        ip_len = max(20, (ip[0] & 0x0F) * 4)
        protocol = ip[9]
    elif version == 6:
        ip_len = 40
        protocol = ip[6]
    else:
        return b""
    if protocol == 6:
        transport_len = max(20, ((transport[12] >> 4) & 0x0F) * 4) if len(transport) > 12 else 20
    elif protocol == 17:
        transport_len = 8
    else:
        transport_len = 0
    return (ip[:ip_len] + transport[:transport_len] + payload)[:64]


def packet_length(values: np.ndarray) -> int:
    ip = values[0]
    version = int(ip[0]) >> 4
    if version == 4:
        return int.from_bytes(bytes(ip[2:4]), "big") + 14
    if version == 6:
        return int.from_bytes(bytes(ip[4:6]), "big") + 40 + 14
    return 14


def directions(values: np.ndarray, count: int) -> list[int]:
    pairs = []
    for slot in range(count):
        transport = values[1, slot]
        pairs.append((int.from_bytes(bytes(transport[:2]), "big"), int.from_bytes(bytes(transport[2:4]), "big")))
    initial = pairs[0]
    output = []
    for pair in pairs:
        if pair == initial:
            output.append(1)
        elif pair == (initial[1], initial[0]):
            output.append(0)
        else:
            output.append(1 if pair[0] == initial[0] else 0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    destination = cache_path(args.dataset)
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    rows = dataset_rows(args.dataset)
    provenance = load_provenance(rows)
    stage19 = STAGE19_CACHE / args.dataset
    base_ids = np.load(stage19 / "flow_uids.npy", allow_pickle=False)
    base_views = np.load(stage19 / "views.npy", mmap_mode="r", allow_pickle=False)
    base_counts = np.load(stage19 / "packet_counts.npy", mmap_mode="r", allow_pickle=False)
    base_pos = {str(flow_id): index for index, flow_id in enumerate(base_ids)}
    flow_ids = np.asarray(sorted(provenance))
    missing = set(map(str, flow_ids)) - set(base_pos)
    recovered, recovery_audit = recover_missing_views(args.dataset, missing, provenance)

    n = len(flow_ids)
    token_ids = np.zeros((n, 320), dtype=np.int32)
    segments = np.zeros((n, 320), dtype=np.uint8)
    fig_x = np.zeros((n, 8, 7), dtype=np.float32)
    fig_adj = np.zeros((n, 8, 8), dtype=np.uint8)
    fig_mask = np.zeros((n, 8), dtype=np.bool_)
    packet_counts = np.zeros(n, dtype=np.uint8)
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    fig_format = FigFormat(max_packets=8)
    anomalies = defaultdict(int)

    for index, flow_id_obj in enumerate(flow_ids):
        flow_id = str(flow_id_obj)
        refs = provenance[flow_id]["packet_refs"][:8]
        count = len(refs)
        if count == 0:
            raise RuntimeError(f"zero packet refs: {flow_id}")
        if flow_id in base_pos:
            pos = base_pos[flow_id]
            values = np.asarray(base_views[pos])
            if int(base_counts[pos]) < count:
                raise RuntimeError(f"Stage19 packet count shorter than provenance: {flow_id}")
        else:
            values = recovered[flow_id]
        packet_counts[index] = count
        packet_dirs = directions(values, count)
        packets = []
        parts = []
        for slot, (ref, direction) in enumerate(zip(refs, packet_dirs)):
            ip_layer = ip_layer_from_views(values[:, slot])
            if not ip_layer:
                anomalies["empty_ip_layer"] += 1
            if slot < 5:
                parts.append("[SEP] ")
                parts.append(official_bigram_generation(ip_layer.hex()))
            length = packet_length(values[:, slot])
            packets.append((float(ref["timestamp"]), length, length, direction, b""))
        text = "".join(parts)
        ids = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(text))[:320]
        token_ids[index, :len(ids)] = ids
        segments[index, :len(ids)] = 1
        graph = build_flow_graph(flow_id, packets, fig_format)
        fig_x[index, :graph.node_count] = np.asarray(graph.features, dtype=np.float32)
        fig_mask[index, :graph.node_count] = True
        for left, right in graph.edges:
            fig_adj[index, left, right] = 1
            fig_adj[index, right, left] = 1

    if anomalies["empty_ip_layer"]:
        raise RuntimeError(f"empty IP-layer packets: {dict(anomalies)}")
    destination.parent.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        destination,
        flow_ids=flow_ids,
        token_ids=token_ids,
        segments=segments,
        fig_x=fig_x,
        fig_adj=fig_adj,
        fig_mask=fig_mask,
        packet_counts=packet_counts,
    )
    audit = {
        "status": "PASS",
        "dataset": args.dataset,
        "flows": n,
        "stage20_membership_missing": 0,
        "reused_stage19_flows": n - len(missing),
        "recovered_stage20_only_flows": len(missing),
        "raw_recovery_captures": len(recovery_audit),
        "input_definition": {"trafficformer_packets": 5, "fig_packets": 8, "ip_addresses": "zeroed", "seq_length": 320},
        "source_hashes": {
            "stage20_closed_manifest": sha256_file(OUT.parent / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"),
            "stage19_flow_uids": sha256_file(stage19 / "flow_uids.npy"),
            "stage19_views": sha256_file(stage19 / "views.npy"),
        },
        "cache_sha256": sha256_file(destination),
        "recovery_audit": recovery_audit,
    }
    write_json(destination.parent / "cache_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
