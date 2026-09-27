#!/usr/bin/env python3
"""Recover exact Stage20 first-eight packet refs for two labeled adaptations.

Train/validation and test are materialized separately. No source PCAP changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, IPv6, TCP, UDP
from scapy.utils import RawPcapReader

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
STAGE12 = PROJECT / "stage12_dual_external_validation"
SOURCE = PROJECT.parent / "Trident/code/reproduction"
sys.path.insert(0, str(SOURCE))
from extract_ustc_flows import FlowStats, feature_names  # noqa: E402

FROZEN = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
ROLES = {"trainval": ("known_train", "known_validation"), "test": ("known_test",)}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def packet_data(raw: bytes) -> dict:
    try:
        pkt = IP(raw) if raw and raw[0] >> 4 == 4 else IPv6(raw) if raw and raw[0] >> 4 == 6 else Ether(raw)
    except Exception as exc:
        raise ValueError(f"packet parse failed: {type(exc).__name__}: {exc}") from exc
    ip = pkt[IP] if IP in pkt else pkt[IPv6] if IPv6 in pkt else None
    if ip is None:
        raise ValueError("frozen packet has no IPv4/IPv6 layer")
    tcp = pkt[TCP] if TCP in pkt else None
    udp = pkt[UDP] if UDP in pkt else None
    proto = int(ip.proto) if IP in pkt else int(ip.nh)
    ip_header_size = int(ip.ihl or 5) * 4 if IP in pkt else 40
    ip_bytes = bytes(ip)
    if tcp is not None:
        layer = tcp
        l4_header_size = int(tcp.dataofs or 5) * 4
        flags = int(tcp.flags)
        window = float(tcp.window)
    elif udp is not None:
        layer = udp
        l4_header_size = 8
        flags = 0
        window = None
    else:
        layer = ip.payload
        l4_header_size = 0
        flags = 0
        window = None
    l4_bytes = bytes(layer)
    header_l4 = l4_bytes[:l4_header_size]
    payload = l4_bytes[l4_header_size:] if l4_header_size else l4_bytes
    ip_header = ip_bytes[:ip_header_size]
    # Official TFE strips 12:20 and 20:24 for IPv4. The IPv6 path is an
    # explicit extension: remove source/destination bytes 8:40.
    anon_ip = ip_header[:12] + ip_header[20:] if IP in pkt else ip_header[:8]
    anon_l4 = header_l4[4:] if tcp is not None or udp is not None else header_l4
    header = anon_ip + anon_l4
    source_port = int(layer.sport) if tcp is not None or udp is not None else 0
    fragmented = bool(int(ip.frag) or int(ip.flags) & 1) if IP in pkt else False
    return {
        "header": header[:40], "payload": payload[:150],
        "protocol": proto, "source": (str(ip.src), source_port),
        "frame_len": len(raw), "payload_len": len(payload),
        "ttl": float(ip.ttl) if IP in pkt else float(ip.hlim),
        "window": window, "flags": flags, "fragmented": fragmented,
    }


def refs_for_rows(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    wanted = {(row["source_pool"], row["flow_id"]): row for row in rows}
    if len(wanted) != len(rows):
        raise RuntimeError("duplicate frozen source key")
    refs = {}
    provenance_hashes = {}
    for pool in sorted({row["source_pool"] for row in rows}):
        for path in sorted(Path(pool).glob("provenance_*.jsonl")):
            provenance_hashes[str(path)] = digest(path)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    key = pool, record["flow_id_sha256"]
                    if key not in wanted:
                        continue
                    if key in refs or record["image_sha256"] != wanted[key]["image_sha256"]:
                        raise RuntimeError(f"duplicated/mismatched provenance {key}")
                    selected = record["packet_refs"][:8]
                    if not selected or any(int(x["packet_number"]) <= 0 for x in selected):
                        raise RuntimeError(f"empty/invalid refs {key}")
                    refs[key] = selected
    if set(refs) != set(wanted):
        raise RuntimeError(f"missing {len(set(wanted) - set(refs))} provenance records")
    return {uid: value for (_, uid), value in refs.items()}, provenance_hashes


def build(dataset: str, phase: str) -> None:
    if digest(MANIFEST) != FROZEN:
        raise RuntimeError("Stage20 frozen manifest hash mismatch")
    if phase == "test":
        # Never materialize Test before all four train/validation selected
        # checkpoints for the dataset exist.
        for method in ("tfe", "trident"):
            for seed in (2022, 2023):
                selected = OUT / "runs" / method / dataset / f"seed{seed}" / "SELECTION_COMPLETE"
                if not selected.is_file():
                    raise RuntimeError(f"Test materialization premature: {selected}")
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset and r["closed_role"] in ROLES[phase]]
    rows.sort(key=lambda r: (r["closed_role"], r["flow_id"]))
    if not rows or len({r["flow_id"] for r in rows}) != len(rows):
        raise RuntimeError("empty/duplicate target flow set")
    refs, provenance_hashes = refs_for_rows(rows)
    cfg = json.loads((STAGE12 / "configs/stage12_config.json").read_text(encoding="utf-8"))
    pcap_root = Path(cfg["datasets"][dataset]["pcap_root"])
    by_pcap = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pcap = pcap_root / row["source_file"]
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        for slot, ref in enumerate(refs[row["flow_id"]]):
            by_pcap[pcap][int(ref["packet_number"])].append((row["flow_id"], slot))

    result_dir = OUT / "inputs" / dataset / phase
    if result_dir.exists():
        raise FileExistsError(f"preserve previous extraction: {result_dir}")
    result_dir.mkdir(parents=True)
    n = len(rows)
    index = {row["flow_id"]: i for i, row in enumerate(rows)}
    headers = np.lib.format.open_memmap(result_dir / "header.npy", mode="w+", dtype=np.uint16, shape=(n, 8, 40))
    payloads = np.lib.format.open_memmap(result_dir / "payload.npy", mode="w+", dtype=np.uint16, shape=(n, 8, 150))
    headers[:] = 256
    payloads[:] = 256
    feature = np.lib.format.open_memmap(result_dir / "trident86.npy", mode="w+", dtype=np.float32, shape=(n, 86))
    found = Counter()
    per_flow = defaultdict(dict)
    packet_audit = Counter()
    capture_audit = []
    for capture_no, (pcap, targets) in enumerate(sorted(by_pcap.items(), key=lambda x: str(x[0])), 1):
        limit = max(targets)
        scanned = 0
        reader = RawPcapReader(str(pcap))
        try:
            for packet_number, (raw, _meta) in enumerate(reader, 1):
                scanned = packet_number
                if packet_number in targets:
                    parsed = packet_data(raw)
                    for uid, slot in targets[packet_number]:
                        i = index[uid]
                        h, p = parsed["header"], parsed["payload"]
                        headers[i, slot, :len(h)] = list(h)
                        payloads[i, slot, :len(p)] = list(p)
                        per_flow[uid][slot] = parsed
                        found[uid] += 1
                        packet_audit[f"protocol_{parsed['protocol']}"] += 1
                        if not p:
                            packet_audit["empty_payload"] += 1
                if packet_number >= limit:
                    break
        finally:
            reader.close()
        capture_audit.append({"pcap": str(pcap), "scanned_packets": scanned, "selected_packet_numbers": len(targets)})
        print(json.dumps({"event": "capture", "dataset": dataset, "phase": phase,
                          "index": capture_no, "total": len(by_pcap), "pcap": pcap.name}), flush=True)
    for row in rows:
        uid = row["flow_id"]
        selected = refs[uid]
        if found[uid] != len(selected) or set(per_flow[uid]) != set(range(len(selected))):
            raise RuntimeError(f"incomplete frozen packet refs for flow {uid}: {found[uid]}/{len(selected)}")
        first = per_flow[uid][0]
        stats = FlowStats(first["protocol"], float(selected[0]["timestamp"]), first["source"])
        for slot, ref in enumerate(selected):
            info = per_flow[uid][slot]
            if info["protocol"] != first["protocol"]:
                raise RuntimeError(f"mixed protocols in flow {uid}")
            stats.update(float(ref["timestamp"]), info["source"], info["frame_len"], info["payload_len"],
                         info["ttl"], info["window"], info["flags"], info["fragmented"])
        feature[index[uid]] = stats.to_vector()
    headers.flush()
    payloads.flush()
    feature.flush()
    np.save(result_dir / "flow_ids.npy", np.asarray([r["flow_id"] for r in rows], dtype="U64"), allow_pickle=False)
    np.save(result_dir / "packet_counts.npy", np.asarray([len(refs[r["flow_id"]]) for r in rows], dtype=np.uint8), allow_pickle=False)
    with (result_dir / "membership.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("flow_id", "role", "service_label", "source_file"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"flow_id": row["flow_id"], "role": row["closed_role"],
                             "service_label": row["service_label"], "source_file": row["source_file"]})
    if feature.shape != (n, len(feature_names())) or not np.isfinite(feature).all():
        raise RuntimeError("Trident-86 shape/nonfinite audit failed")
    if digest(MANIFEST) != FROZEN:
        raise RuntimeError("Stage20 manifest changed during extraction")
    audit = {
        "status": "PASS", "dataset": dataset, "phase": phase, "flows": n,
        "roles": dict(Counter(r["closed_role"] for r in rows)),
        "stage20_manifest_sha256": FROZEN,
        "packet_references": int(sum(map(len, refs.values()))),
        "packet_count_distribution": dict(Counter(len(x) for x in refs.values())),
        "protocol_packet_counts": dict(packet_audit),
        "trident_feature_names": feature_names(),
        "provenance_hashes": provenance_hashes,
        "capture_audit": capture_audit,
        "file_hashes": {name: digest(result_dir / name) for name in
                        ("header.npy", "payload.npy", "trident86.npy", "flow_ids.npy",
                         "packet_counts.npy", "membership.csv")},
        "known_test_loaded_before_selection": False if phase == "trainval" else None,
        "unknown_loaded": False,
    }
    (result_dir / "input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "dataset": dataset, "phase": phase, "flows": n,
                      "packet_references": audit["packet_references"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--phase", choices=("trainval", "test"), required=True)
    a = p.parse_args()
    build(a.dataset, a.phase)
