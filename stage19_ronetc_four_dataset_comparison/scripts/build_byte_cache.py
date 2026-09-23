#!/usr/bin/env python3
"""Reconstruct RoNeTC's three packet views for frozen flow IDs.

Raw datasets are read-only.  Derived arrays are written only under Stage19.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, IPv6, TCP, UDP, raw as scapy_raw
from scapy.utils import RawPcapReader

from common import PROJECT, PROTOCOL_MANIFEST, STAGE12, cache_dir, config, read_csv, read_json, sha256_file, write_csv, write_json


VNAT_TSHARK_FIELDS = (
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.stream", "udp.stream",
)


def canonical_tuple(src: str, sport: str, dst: str, dport: str, protocol: str) -> str:
    """Match the Stage 14A/14C bidirectional endpoint canonicalization."""
    first, second = sorted(((src, sport), (dst, dport)))
    return f"{protocol}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"


def vnat_flow_id(values: list[str]) -> str | None:
    """Match the frozen VNAT tcp/udp/other-IP flow identity exactly."""
    if len(values) != len(VNAT_TSHARK_FIELDS):
        raise ValueError(f"expected {len(VNAT_TSHARK_FIELDS)} tshark fields, got {len(values)}")
    (
        ip4_src, ip6_src, ip4_dst, ip6_dst, ip4_proto, ip6_next,
        _tcp_sport, _tcp_dport, _udp_sport, _udp_dport, tcp_stream, udp_stream,
    ) = values
    src, dst = ip4_src or ip6_src, ip4_dst or ip6_dst
    if not src or not dst:
        return None
    if tcp_stream:
        return f"tcp:{tcp_stream}"
    if udp_stream:
        return f"udp:{udp_stream}"
    protocol = f"ip:{ip4_proto or ip6_next or 'unknown'}"
    text = canonical_tuple(src, "", dst, "", protocol)
    return f"other:{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def fit(raw: bytes, width: int) -> np.ndarray:
    result = np.zeros(width, dtype=np.uint8)
    count = min(width, len(raw))
    if count:
        result[:count] = np.frombuffer(raw[:count], dtype=np.uint8)
    return result


def packet_views(raw_bytes: bytes, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros(width, dtype=np.uint8)
    try:
        packet = (IP(raw_bytes) if raw_bytes and raw_bytes[0] >> 4 == 4 else IPv6(raw_bytes) if raw_bytes and raw_bytes[0] >> 4 == 6 else Ether(raw_bytes))
        if IP in packet:
            ip = packet[IP]
        elif IPv6 in packet:
            ip = packet[IPv6]
        else:
            return zero.copy(), zero.copy(), zero.copy()
        ip_raw = bytearray(scapy_raw(ip))
        if IP in packet:
            header_len = max(20, int(getattr(ip, "ihl", 5) or 5) * 4)
            if len(ip_raw) >= 20:
                ip_raw[12:20] = b"\x00" * 8
        else:
            header_len = 40
            if len(ip_raw) >= 40:
                ip_raw[8:40] = b"\x00" * 32
        if TCP in packet:
            transport = packet[TCP]
            trans_raw = scapy_raw(transport)
            trans_len = max(20, int(getattr(transport, "dataofs", 5) or 5) * 4)
            payload = scapy_raw(transport.payload)
        elif UDP in packet:
            transport = packet[UDP]
            trans_raw = scapy_raw(transport)
            trans_len = 8
            payload = scapy_raw(transport.payload)
        else:
            return fit(bytes(ip_raw[:header_len]), width), zero.copy(), zero.copy()
        return fit(bytes(ip_raw[:header_len]), width), fit(trans_raw[:trans_len], width), fit(payload, width)
    except (ValueError, TypeError, IndexError):
        return zero.copy(), zero.copy(), zero.copy()


def open_reader(path: Path):
    # RawPcapReader dispatches to the pcapng alternative when needed and avoids
    # dissecting millions of packets that are not part of the frozen flow set.
    reader = RawPcapReader(str(path))
    return reader, reader


def allocate(dataset: str, flow_ids: list[str], packet_num: int, width: int):
    output = cache_dir(dataset)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    np.save(output / "flow_uids.npy", np.asarray(flow_ids, dtype="U80"), allow_pickle=False)
    views = np.lib.format.open_memmap(output / "views.npy", mode="w+", dtype=np.uint8, shape=(len(flow_ids), 3, packet_num, width))
    counts = np.lib.format.open_memmap(output / "packet_counts.npy", mode="w+", dtype=np.uint32, shape=(len(flow_ids),))
    return output, views, counts, {uid: i for i, uid in enumerate(flow_ids)}


def target_rows(dataset: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv(PROTOCOL_MANIFEST) if row["dataset"] == dataset]
    by_uid = {}
    for row in rows:
        prior = by_uid.setdefault(row["flow_uid"], row)
        if prior["class_name"] != row["class_name"] or prior["source_file"] != row["source_file"]:
            raise RuntimeError(f"inconsistent repeated flow {row['flow_uid']}")
    return [by_uid[uid] for uid in sorted(by_uid)]


def build_ustc(packet_num: int, width: int) -> dict:
    rows = target_rows("ustc")
    output, views, counts, index = allocate("ustc", [row["flow_uid"] for row in rows], packet_num, width)
    found = set()
    pkl_rows = []
    for path in sorted((PROJECT / "data" / "flows").glob("*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in index:
                continue
            pos = index[uid]
            matched += 1
            found.add(uid)
            counts[pos] = len(packets)
            for packet_pos, packet in enumerate(packets[:packet_num]):
                ip, transport, payload_bytes = packet_views(bytes(packet[4]), width)
                views[pos, 0, packet_pos] = ip
                views[pos, 1, packet_pos] = transport
                views[pos, 2, packet_pos] = payload_bytes
        pkl_rows.append({"path": str(path), "sha256": sha256_file(path), "matched_flows": matched})
        del payload
        gc.collect()
        print(json.dumps({"event": "pkl", "path": path.name, "matched": matched}), flush=True)
    missing = sorted(set(index) - found)
    write_csv(output / "source_audit.csv", pkl_rows)
    return finalize(output, views, counts, len(index), missing, "Stage0 PKL raw Ethernet frames")


def provenance_dir(dataset: str) -> Path:
    base = STAGE12 / "artifacts" / dataset / "preprocessing_pool"
    candidates = sorted(path for path in base.glob("raw_prepared_parallel_v*") if (path / "SUCCESS").is_file())
    if not candidates:
        raise RuntimeError(f"no successful preprocessing pool for {dataset}")
    return candidates[-1]


def pcap_path_for_iscx(dataset: str, source_file: str) -> Path:
    stage12_config = read_json(STAGE12 / "configs" / "stage12_config.json")
    return Path(stage12_config["datasets"][dataset]["pcap_root"]) / source_file


def build_iscx(dataset: str, packet_num: int, width: int) -> dict:
    rows = target_rows(dataset)
    output, views, counts, index = allocate(dataset, [row["flow_uid"] for row in rows], packet_num, width)
    target = set(index)
    metadata = {row["flow_uid"]: row for row in rows}
    refs: dict[str, list[dict]] = {}
    pool = provenance_dir(dataset)
    for path in sorted(pool.glob("provenance_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                uid = record["flow_id_sha256"]
                if uid in target:
                    refs[uid] = record["packet_refs"][:packet_num]
    missing_refs = sorted(target - set(refs))
    if missing_refs:
        raise RuntimeError(f"missing provenance refs: {missing_refs[:5]}")
    by_pcap: dict[Path, dict[int, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    for uid, packet_refs in refs.items():
        pcap = pcap_path_for_iscx(dataset, metadata[uid]["source_file"])
        counts[index[uid]] = len(packet_refs)
        for slot, ref in enumerate(packet_refs):
            by_pcap[pcap][int(ref["packet_number"])].append((uid, slot))
    capture_rows = []
    found = defaultdict(int)
    for number, (pcap, selected) in enumerate(sorted(by_pcap.items(), key=lambda item: str(item[0])), start=1):
        handle, reader = open_reader(pcap)
        total = 0
        try:
            for packet_number, (packet_bytes, _metadata) in enumerate(reader, start=1):
                total += 1
                for uid, slot in selected.get(packet_number, ()):
                    for view, value in enumerate(packet_views(packet_bytes, width)):
                        views[index[uid], view, slot] = value
                    found[uid] += 1
        finally:
            handle.close()
        capture_rows.append({"pcap": str(pcap), "sha256": sha256_file(pcap), "packets": total, "target_packet_numbers": len(selected)})
        print(json.dumps({"event": "pcap", "dataset": dataset, "index": number, "total": len(by_pcap), "path": pcap.name}), flush=True)
    missing = sorted(uid for uid in target if found[uid] != len(refs[uid]))
    write_csv(output / "source_audit.csv", capture_rows)
    return finalize(output, views, counts, len(index), missing, "Stage12 exact packet_refs")


def build_vnat(packet_num: int, width: int) -> dict:
    rows = target_rows("vnat")
    output, views, counts, index = allocate("vnat", [row["flow_uid"] for row in rows], packet_num, width)
    by_pcap: dict[Path, dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_pcap[Path(row["source_file"])][row["source_flow_id"]] = row["flow_uid"]
    capture_rows = []
    found = defaultdict(int)
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    for capture_index, (pcap, streams) in enumerate(sorted(by_pcap.items(), key=lambda item: str(item[0])), start=1):
        command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f", "-e", "frame.number"]
        for field in VNAT_TSHARK_FIELDS:
            command.extend(["-e", field])
        packet_map: dict[int, list[tuple[str, int]]] = defaultdict(list)
        slot_by_uid = defaultdict(int)
        stderr_path = stderr_root / f"{pcap.stem}.log"
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                values = line.rstrip("\r\n").split("\t")
                if len(values) < 1 + len(VNAT_TSHARK_FIELDS):
                    values += [""] * (1 + len(VNAT_TSHARK_FIELDS) - len(values))
                if len(values) != 1 + len(VNAT_TSHARK_FIELDS):
                    process.kill()
                    raise RuntimeError(f"unexpected tshark field count for {pcap}: {len(values)}")
                flow_id = vnat_flow_id(values[1:])
                uid = streams.get(flow_id)
                if uid is None or slot_by_uid[uid] >= packet_num:
                    continue
                packet_map[int(values[0])].append((uid, slot_by_uid[uid]))
                slot_by_uid[uid] += 1
            code = process.wait()
        if code != 0:
            raise RuntimeError(f"tshark failed {code}: {pcap}")
        handle, reader = open_reader(pcap)
        total = 0
        try:
            for packet_number, (packet_bytes, _metadata) in enumerate(reader, start=1):
                total += 1
                for uid, slot in packet_map.get(packet_number, ()):
                    for view, value in enumerate(packet_views(packet_bytes, width)):
                        views[index[uid], view, slot] = value
                    found[uid] += 1
        finally:
            handle.close()
        for uid in streams.values():
            counts[index[uid]] = found[uid]
        capture_rows.append({"pcap": str(pcap), "sha256": sha256_file(pcap), "packets": total, "target_flows": len(streams), "selected_packets": sum(map(len, packet_map.values()))})
        print(json.dumps({"event": "pcap", "dataset": "vnat", "index": capture_index, "total": len(by_pcap), "path": pcap.name}), flush=True)
    missing = sorted(uid for uid in index if found[uid] == 0)
    write_csv(output / "source_audit.csv", capture_rows)
    return finalize(output, views, counts, len(index), missing, "VNAT tshark stream ID plus raw-PCAP packet number")


def finalize(output: Path, views, counts, target_count: int, missing: list[str], lineage: str) -> dict:
    views.flush(); counts.flush()
    nonzero_view = np.load(output / "views.npy", mmap_mode="r", allow_pickle=False)
    packet_counts = np.load(output / "packet_counts.npy", mmap_mode="r", allow_pickle=False)
    audit = {
        "status": "PASS" if not missing else "FAIL", "lineage": lineage,
        "target_flows": target_count, "recovered_flows": target_count - len(missing), "missing_flows": len(missing),
        "zero_packet_flows": int(np.sum(packet_counts == 0)), "packet_num": int(nonzero_view.shape[2]), "byte_num": int(nonzero_view.shape[3]),
        "views_sha256": sha256_file(output / "views.npy"), "flow_uids_sha256": sha256_file(output / "flow_uids.npy"),
        "missing_preview": missing[:100], "raw_dataset_modified": False,
    }
    write_json(output / "cache_audit.json", audit)
    if missing:
        raise RuntimeError(f"{output.name}: {len(missing)} flows missing")
    print(json.dumps(audit, indent=2), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    args = parser.parse_args()
    spec = config()["input"]
    if args.dataset == "ustc":
        build_ustc(int(spec["packet_num"]), int(spec["byte_num_per_view"]))
    elif args.dataset == "vnat":
        build_vnat(int(spec["packet_num"]), int(spec["byte_num_per_view"]))
    else:
        build_iscx(args.dataset, int(spec["packet_num"]), int(spec["byte_num_per_view"]))


if __name__ == "__main__":
    main()
