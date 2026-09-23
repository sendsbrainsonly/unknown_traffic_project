#!/usr/bin/env python3
"""Compare Scapy and tshark capture-wide five-tuple extraction for one PCAP."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import scapy.all as scapy


WORKSPACE = Path(__file__).resolve().parents[5]
TF_SCRIPTS = WORKSPACE / "Projects/TrafficFormer/reproduction/scripts"
sys.path.insert(0, str(TF_SCRIPTS))
from prepare_iscxvpn import canonical_flow_key  # noqa: E402


def compact(key: tuple) -> str:
    protocol, _proto, first, second = key
    return f"{protocol}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", type=Path)
    args = parser.parse_args()
    scapy_rows = defaultdict(lambda: {"packets": 0, "bytes": 0, "frames": []})
    with scapy.PcapReader(str(args.pcap)) as reader:
        for number, packet in enumerate(reader, 1):
            key = canonical_flow_key(packet)
            if key is None:
                continue
            name = compact(key)
            scapy_rows[name]["packets"] += 1
            scapy_rows[name]["bytes"] += len(bytes(packet))
            scapy_rows[name]["frames"].append(number)

    fields = (
        "frame.number", "frame.cap_len", "ip.src", "ip.dst",
        "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
        "ip.frag_offset", "ip.proto", "frame.protocols",
    )
    command = ["tshark", "-n", "-r", str(args.pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
    for field in fields:
        command.extend(["-e", field])
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    tshark_rows = defaultdict(lambda: {"packets": 0, "bytes": 0, "frames": [], "protocols": []})
    for line in result.stdout.splitlines():
        values = line.split("\t")
        values += [""] * (len(fields) - len(values))
        number, caplen, src, dst, ts, td, us, ud, frag, proto, protocols = values[:11]
        if frag and int(frag) > 0:
            continue
        if src and dst and ts and td:
            key = compact(("tcp", int(proto or 6), tuple(sorted(((src, int(ts)), (dst, int(td)))))[0], tuple(sorted(((src, int(ts)), (dst, int(td)))))[1]))
        elif src and dst and us and ud:
            key = compact(("udp", int(proto or 17), tuple(sorted(((src, int(us)), (dst, int(ud)))))[0], tuple(sorted(((src, int(us)), (dst, int(ud)))))[1]))
        else:
            continue
        tshark_rows[key]["packets"] += 1
        tshark_rows[key]["bytes"] += int(caplen or 0)
        tshark_rows[key]["frames"].append(int(number))
        tshark_rows[key]["protocols"].append(protocols)

    scapy_keys, tshark_keys = set(scapy_rows), set(tshark_rows)
    extra = sorted(tshark_keys - scapy_keys)
    missing = sorted(scapy_keys - tshark_keys)
    common_mismatch = sorted(
        key for key in scapy_keys & tshark_keys
        if (scapy_rows[key]["packets"], scapy_rows[key]["bytes"]) != (tshark_rows[key]["packets"], tshark_rows[key]["bytes"])
    )
    payload = {
        "pcap": str(args.pcap.resolve()),
        "scapy_flows": len(scapy_rows),
        "tshark_flows": len(tshark_rows),
        "extra_tshark_count": len(extra),
        "missing_tshark_count": len(missing),
        "common_count_or_byte_mismatch": len(common_mismatch),
        "extra_tshark": [{"key": key, **tshark_rows[key]} for key in extra[:50]],
        "missing_tshark": [{"key": key, **scapy_rows[key]} for key in missing[:50]],
        "common_mismatch": [
            {"key": key, "scapy": scapy_rows[key], "tshark": tshark_rows[key]}
            for key in common_mismatch[:50]
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
