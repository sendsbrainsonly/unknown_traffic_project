#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_utils import OUTPUT_ROOT, PROJECT_ROOT, SEED, write_csv


CIC_ROOT = PROJECT_ROOT.parents[1] / "Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)"
VENDOR = CIC_ROOT / "cicids2017_label_mapping/vendor"
SAMPLE_PATH = OUTPUT_ROOT / "cicids2017/sample_candidates.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def packet_key(raw: bytes, dpkt) -> str | None:
    return load_packet_key(raw, dpkt)


def load_packet_key(raw: bytes, dpkt) -> str | None:
    try:
        frame = dpkt.ethernet.Ethernet(raw)
        network = frame.data
        if isinstance(network, dpkt.ip.IP):
            version, protocol = 4, int(network.p)
        elif isinstance(network, dpkt.ip6.IP6):
            version, protocol = 6, int(network.nxt)
        else:
            return None
        transport = network.data
        if isinstance(transport, dpkt.tcp.TCP):
            protocol, src_port, dst_port = 6, int(transport.sport), int(transport.dport)
        elif isinstance(transport, dpkt.udp.UDP):
            protocol, src_port, dst_port = 17, int(transport.sport), int(transport.dport)
        else:
            src_port = dst_port = 0
        import ipaddress

        def endpoint(address: bytes, port: int) -> str:
            text = str(ipaddress.ip_address(address))
            return f"[{text}]:{port}" if version == 6 else f"{text}:{port}"

        src = (version, bytes(network.src), src_port)
        dst = (version, bytes(network.dst), dst_port)
        left, right = (src, dst) if src <= dst else (dst, src)
        return f"{protocol}|{endpoint(left[1], left[2])}|{endpoint(right[1], right[2])}"
    except (dpkt.UnpackError, ValueError, OSError, IndexError):
        return None


def encode_result(row: dict[str, object], preprocessing) -> dict[str, object]:
    packets = row.pop("_raw_packets", [])
    observed = int(row.pop("_observed_packet_count", 0))
    expected = int(row["packet_count"])
    result = {
        "sample_id": row["flow_id"], "label": row["label"], "day": row["day"],
        "source_pcap": row["source_pcap"], "canonical_flow_key": row["canonical_flow_key"],
        "first_packet_index": row["first_packet_index"], "last_packet_index": row["last_packet_index"],
        "expected_packet_count": expected, "observed_packet_count": observed,
        "packets_used": min(len(packets), preprocessing.PACKETS_PER_FLOW), "seed": SEED,
        "flow_reconstruction_stable": observed == expected,
        "input_valid": False, "failure_reason": "", "ip_fields_checked": 0, "ip_fields_masked": 0,
        "port_fields_checked": 0, "port_fields_visible": 0, "header_region_nonzero": False,
        "payload_region_nonzero": False,
    }
    if observed != expected:
        result["failure_reason"] = f"interval_count_mismatch:{observed}!={expected}"
        return result
    image = np.zeros(preprocessing.FLOW_BYTES, dtype=np.uint8)
    try:
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.l2 import Ether

        for index, raw in enumerate(packets[: preprocessing.PACKETS_PER_FLOW]):
            packet = Ether(raw)
            encoded = preprocessing.encode_packet(raw)
            start = index * preprocessing.BYTES_PER_PACKET
            image[start : start + preprocessing.BYTES_PER_PACKET] = encoded
            result["ip_fields_checked"] = int(result["ip_fields_checked"]) + 1
            if bytes(encoded[12:20]) == b"\x00" * 8:
                result["ip_fields_masked"] = int(result["ip_fields_masked"]) + 1
            ihl = int(encoded[0] & 0x0F) * 4
            if (TCP in packet or UDP in packet) and ihl + 4 <= preprocessing.HEADER_BYTES:
                layer = packet[TCP] if TCP in packet else packet[UDP]
                expected_ports = int(layer.sport).to_bytes(2, "big") + int(layer.dport).to_bytes(2, "big")
                result["port_fields_checked"] = int(result["port_fields_checked"]) + 1
                if bytes(encoded[ihl : ihl + 4]) == expected_ports:
                    result["port_fields_visible"] = int(result["port_fields_visible"]) + 1
            result["header_region_nonzero"] = bool(result["header_region_nonzero"] or np.any(encoded[: preprocessing.HEADER_BYTES]))
            result["payload_region_nonzero"] = bool(result["payload_region_nonzero"] or np.any(encoded[preprocessing.HEADER_BYTES :]))
            if IP not in packet:
                raise ValueError("non_ipv4_packet")
        result["input_valid"] = bool(packets)
    except Exception as exc:  # preserve every reconstruction/adapter failure in the manifest
        result["failure_reason"] = f"{type(exc).__name__}:{exc}"
    return result


def main() -> None:
    sys.path.insert(0, str(VENDOR))
    import dpkt  # type: ignore

    preprocessing = load_module(
        PROJECT_ROOT / "opendetect_ustc_encoder_audit/adapters/opendetect_preprocessing.py",
        "opendetect_preprocessing_reuse",
    )
    rows = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    for row in rows:
        row["first_packet_index"] = int(row["first_packet_index"])
        row["last_packet_index"] = int(row["last_packet_index"])
        row["packet_count"] = int(row["packet_count"])
        row["_raw_packets"] = []
        row["_observed_packet_count"] = 0
    results = []
    by_day: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_day[str(row["day"])].append(row)
    for day in sorted(by_day):
        day_rows = by_day[day]
        by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in day_rows:
            by_key[str(row["canonical_flow_key"])].append(row)
        for intervals in by_key.values():
            intervals.sort(key=lambda item: int(item["first_packet_index"]))
        positions = {key: 0 for key in by_key}
        pcap_names = {str(row["source_pcap"]) for row in day_rows}
        if len(pcap_names) != 1:
            raise ValueError(f"{day}: expected one source PCAP, found {pcap_names}")
        pcap_path = CIC_ROOT / "PCAPs" / next(iter(pcap_names))
        started = time.time()
        packet_seen = 0
        with pcap_path.open("rb") as handle:
            try:
                reader = dpkt.pcapng.Reader(handle)
                _ = reader.datalink()
            except (ValueError, dpkt.UnpackError):
                handle.seek(0)
                reader = dpkt.pcap.Reader(handle)
            for packet_index, (_, raw) in enumerate(reader):
                packet_seen += 1
                key = packet_key(raw, dpkt)
                if key is None or key not in by_key:
                    continue
                intervals = by_key[key]
                pos = positions[key]
                while pos < len(intervals) and packet_index > int(intervals[pos]["last_packet_index"]):
                    stale = intervals[pos]
                    results.append(encode_result(stale, preprocessing))
                    pos += 1
                    positions[key] = pos
                if pos >= len(intervals):
                    continue
                row = intervals[pos]
                first = int(row["first_packet_index"])
                last = int(row["last_packet_index"])
                if first <= packet_index <= last:
                    row["_observed_packet_count"] = int(row["_observed_packet_count"]) + 1
                    packet_list = row["_raw_packets"]
                    if len(packet_list) < preprocessing.PACKETS_PER_FLOW:
                        packet_list.append(bytes(raw))
                    if packet_index == last:
                        results.append(encode_result(row, preprocessing))
                        positions[key] = pos + 1
                if packet_seen % 5_000_000 == 0:
                    print(f"{day}: packets={packet_seen:,} completed={len(results)}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
                if all(positions[key_name] >= len(by_key[key_name]) for key_name in by_key):
                    break
        for key, intervals in by_key.items():
            for row in intervals[positions[key] :]:
                results.append(encode_result(row, preprocessing))
        print(f"{day}: complete packets={packet_seen:,}, selected_flows={len(day_rows)}, elapsed={time.time()-started:.1f}s", flush=True)

    fields = [
        "sample_id", "label", "day", "source_pcap", "canonical_flow_key", "first_packet_index", "last_packet_index",
        "expected_packet_count", "observed_packet_count", "packets_used", "seed", "flow_reconstruction_stable", "input_valid",
        "failure_reason", "ip_fields_checked", "ip_fields_masked", "port_fields_checked", "port_fields_visible",
        "header_region_nonzero", "payload_region_nonzero",
    ]
    write_csv(OUTPUT_ROOT / "cicids2017/raw_input_manifest.csv", sorted(results, key=lambda row: (str(row["label"]), str(row["sample_id"]))), fields)
    valid = [row for row in results if row["input_valid"]]
    stable = sum(bool(row["flow_reconstruction_stable"]) for row in results)
    ip_checked = sum(int(row["ip_fields_checked"]) for row in results)
    ip_masked = sum(int(row["ip_fields_masked"]) for row in results)
    port_checked = sum(int(row["port_fields_checked"]) for row in results)
    port_visible = sum(int(row["port_fields_visible"]) for row in results)
    headers = sum(bool(row["header_region_nonzero"]) for row in valid)
    payloads = sum(bool(row["payload_region_nonzero"]) for row in valid)
    (OUTPUT_ROOT / "cicids2017/input_endpoint_audit.md").write_text(
        "# CIC-IDS-2017 Open-Detect Input Audit\n\n"
        f"- Fixed sample seed: **{SEED}**; labels: **{len(set(str(row['label']) for row in rows))}**; sampled strict flows: **{len(rows)}**.\n"
        f"- Stable flow reconstruction: **{stable}/{len(rows)}**.\n"
        f"- Valid actual 32x32 inputs: **{len(valid)}/{len(rows)}**.\n"
        f"- IP mask check: **{ip_masked}/{ip_checked}** encoded packets masked.\n"
        f"- Port visibility check: **{port_visible}/{port_checked}** eligible transport headers retain ports.\n"
        f"- Nonzero header region: **{headers}/{len(valid)}** valid flows.\n"
        f"- Nonzero retained payload region: **{payloads}/{len(valid)}** valid flows.\n\n"
        "The source Parquet first/last packet boundaries and canonical bidirectional five-tuple were used to stream the original per-day PCAPs. No flow-level random split, model training, or inference was performed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"sampled": len(rows), "stable": stable, "valid": len(valid), "ip_masked": [ip_masked, ip_checked], "ports_visible": [port_visible, port_checked]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
