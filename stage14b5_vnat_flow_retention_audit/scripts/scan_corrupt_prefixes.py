#!/usr/bin/env python3
"""Count flows in readable prefixes of the two truncated VNAT PCAPs.

This is a read-only diagnostic.  It deliberately accepts tshark's non-zero
exit at the truncated final packet and never feeds its results into Stage 14B.
The grouping fields and branch order mirror Stage 14A's ``scan_flows``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "stage14b5_vnat_flow_retention_audit"
DATA = Path(
    "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
    "Dataset/VNAT/VNAT/VNAT_release_1"
)
OUTPUT = AUDIT / "artifacts" / "corrupt_prefix_flow_counts.json"
STDERR_DIR = AUDIT / "artifacts" / "corrupt_prefix_stderr"
TARGETS = (
    "nonvpn_scp_long_capture1.pcap",
    "vpn_skype-chat_capture6.pcap",
)
FIELDS = (
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.stream", "udp.stream",
)


def canonical_tuple(src: str, sport: str, dst: str, dport: str, protocol: str) -> str:
    first, second = sorted(((src, sport), (dst, dport)))
    return f"{protocol}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"


def scan(path: Path) -> dict[str, object]:
    command = ["tshark", "-n", "-r", str(path), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
    for field in FIELDS:
        command.extend(["-e", field])
    flow_packets: Counter[str] = Counter()
    flow_protocol: dict[str, str] = {}
    packet_count = ip_packet_count = non_ip_packet_count = 0
    stderr_path = STDERR_DIR / f"{path.stem}.stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1024 * 1024,
        )
        assert process.stdout is not None
        for line in process.stdout:
            packet_count += 1
            values = line.rstrip("\r\n").split("\t")
            if len(values) < len(FIELDS):
                values += [""] * (len(FIELDS) - len(values))
            if len(values) != len(FIELDS):
                raise RuntimeError(f"Unexpected tshark field count at packet {packet_count}: {len(values)}")
            (
                ip4_src, ip6_src, ip4_dst, ip6_dst, ip4_proto, ip6_next,
                tcp_sport, tcp_dport, udp_sport, udp_dport, tcp_stream, udp_stream,
            ) = values
            src, dst = ip4_src or ip6_src, ip4_dst or ip6_dst
            if not src or not dst:
                non_ip_packet_count += 1
                continue
            ip_packet_count += 1
            if tcp_stream:
                flow_id, protocol = f"tcp:{tcp_stream}", "tcp"
            elif udp_stream:
                flow_id, protocol = f"udp:{udp_stream}", "udp"
            else:
                protocol = f"ip:{ip4_proto or ip6_next or 'unknown'}"
                tuple_text = canonical_tuple(src, "", dst, "", protocol)
                flow_id = f"other:{hashlib.sha256(tuple_text.encode()).hexdigest()[:24]}"
            flow_packets[flow_id] += 1
            flow_protocol[flow_id] = protocol
        returncode = process.wait()
    protocol_counts = Counter(flow_protocol.values())
    return {
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "packet_rows_emitted_before_error": packet_count,
        "ip_packet_rows": ip_packet_count,
        "non_ip_packet_rows": non_ip_packet_count,
        "readable_prefix_flow_count": len(flow_packets),
        "tcp_flow_count": protocol_counts["tcp"],
        "udp_flow_count": protocol_counts["udp"],
        "other_ip_flow_count": sum(v for k, v in protocol_counts.items() if k not in {"tcp", "udp"}),
        "single_packet_flow_count": sum(v == 1 for v in flow_packets.values()),
        "tshark_returncode": returncode,
        "tshark_stderr_path": str(stderr_path.relative_to(PROJECT)),
        "diagnostic_only": True,
        "included_in_stage14b": False,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STDERR_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Stage 14A grouping semantics over tshark-emitted readable prefix; non-zero EOF/truncation exit retained as evidence",
        "results": [scan(DATA / name) for name in TARGETS],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
