#!/usr/bin/env python3
"""Audit VNAT packet windows for the union of frozen Known Train/Validation rows."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from common import FLOW_WINDOW_FIELDS, PROJECT_ROOT, ROOT, WindowAccumulator, sha256_file, write_csv, write_json


SPLIT_MANIFEST = PROJECT_ROOT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"
BASE_FIELDS = (
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "tcp.stream", "udp.stream",
)
EXTRA_FIELDS = (
    "frame.time_epoch", "frame.len", "frame.protocols", "ip.len", "ipv6.plen", "tcp.len", "udp.length",
)


def tshark_flow_id(values: list[str]) -> str | None:
    ip4_src, ip6_src, ip4_dst, ip6_dst, ip4_proto, ip6_next, _ts, _td, _us, _ud, tcp_stream, udp_stream = values
    src, dst = ip4_src or ip6_src, ip4_dst or ip6_dst
    if not src or not dst:
        return None
    if tcp_stream:
        return f"tcp:{tcp_stream}"
    if udp_stream:
        return f"udp:{udp_stream}"
    first, second = sorted(((src, ""), (dst, "")))
    text = f"ip:{ip4_proto or ip6_next or 'unknown'}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"
    return f"other:{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def endpoints(values: list[str]) -> tuple[tuple[str, str], tuple[str, str]]:
    src, dst = values[0] or values[1], values[2] or values[3]
    sport, dport = values[6] or values[8], values[7] or values[9]
    return (src, sport), (dst, dport)


def load_targets() -> dict[str, dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    with SPLIT_MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_role"] != "known" or row["split"] not in {"train", "validation"}:
                continue
            uid = row["flow_uid"]
            prior = targets.setdefault(uid, row)
            fields = ("application", "capture_id", "source_flow_id", "source_pcap_path", "packet_count")
            if any(prior[name] != row[name] for name in fields):
                raise RuntimeError(f"inconsistent VNAT frozen metadata: {uid}")
    return targets


def main() -> None:
    output = ROOT / "window_cache" / "vnat"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    targets = load_targets()
    by_capture: dict[str, dict[str, WindowAccumulator]] = defaultdict(dict)
    capture_paths: dict[str, Path] = {}
    uid_by_capture_flow: dict[tuple[str, str], str] = {}
    for uid, row in targets.items():
        acc = WindowAccumulator(uid, row["application"], row["source_pcap_path"], row["vpn_status"])
        by_capture[row["capture_id"]][row["source_flow_id"]] = acc
        uid_by_capture_flow[(row["capture_id"], row["source_flow_id"])] = uid
        capture_paths[row["capture_id"]] = Path(row["source_pcap_path"])
    capture_rows: list[dict[str, object]] = []
    for capture_index, capture_id in enumerate(sorted(by_capture), start=1):
        pcap = capture_paths[capture_id]
        command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
        for field in BASE_FIELDS + EXTRA_FIELDS:
            command.extend(["-e", field])
        packet_rows = matched_rows = discarded_rows = 0
        stderr_path = stderr_root / f"{capture_id}.log"
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
            assert process.stdout is not None
            for line in process.stdout:
                packet_rows += 1
                values = line.rstrip("\r\n").split("\t")
                expected = len(BASE_FIELDS) + len(EXTRA_FIELDS)
                if len(values) < expected:
                    values += [""] * (expected - len(values))
                flow_id = tshark_flow_id(values[: len(BASE_FIELDS)])
                if flow_id is None or flow_id not in by_capture[capture_id]:
                    discarded_rows += 1
                    continue
                try:
                    timestamp = float(values[12])
                    frame_bytes = int(values[13])
                except ValueError as exc:
                    process.kill()
                    raise RuntimeError(f"{capture_id}: invalid timestamp/frame length at packet {packet_rows}") from exc
                protocols = values[14]
                ip_bytes = int(values[15]) if values[15] else (int(values[16]) + 40 if values[16] else None)
                transport_payload = int(values[17]) if values[17] else (max(0, int(values[18]) - 8) if values[18] else None)
                source, destination = endpoints(values)
                by_capture[capture_id][flow_id].add(
                    timestamp,
                    frame_bytes,
                    ip_bytes,
                    transport_payload,
                    source,
                    destination,
                    protocols,
                )
                matched_rows += 1
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"{capture_id}: tshark exited {returncode}; see {stderr_path}")
        capture_rows.append(
            {
                "capture_id": capture_id,
                "target_flows": len(by_capture[capture_id]),
                "pcap_packet_rows": packet_rows,
                "matched_known_train_validation_packet_rows": matched_rows,
                "discarded_non_target_packet_rows": discarded_rows,
                "source_pcap_path": str(pcap),
                "tshark_stderr": str(stderr_path),
            }
        )
        print(json.dumps({"event": "capture_complete", "capture_index": capture_index, "capture_total": len(by_capture), "capture_id": capture_id, "target_flows": len(by_capture[capture_id])}), flush=True)
    accumulators = {acc.flow_uid: acc for flows in by_capture.values() for acc in flows.values()}
    mismatches = []
    for uid, acc in accumulators.items():
        expected = int(targets[uid]["packet_count"])
        if acc.packet_count != expected:
            mismatches.append({"flow_uid": uid, "expected": expected, "observed": acc.packet_count})
    if mismatches:
        write_json(output / "failure.json", {"packet_count_mismatches": len(mismatches), "preview": mismatches[:100]})
        raise RuntimeError(f"VNAT packet-count mismatches: {len(mismatches)}")
    rows = [accumulators[uid].row("vnat") for uid in sorted(accumulators)]
    metrics = output / "flow_window_metrics.csv.gz"
    write_csv(metrics, rows, FLOW_WINDOW_FIELDS)
    write_csv(output / "capture_audit.csv", capture_rows, list(capture_rows[0]))
    stage14c5_cache_audit = PROJECT_ROOT / "stage14c5_feature_representation_audit" / "raw_feature_cache" / "cache_audit.json"
    write_json(
        output / "cache_audit.json",
        {
            "status": "PASS",
            "dataset": "vnat",
            "scope": "union of Known Train/Validation rows in all 15 frozen Stage 14B protocols",
            "target_flows": len(targets),
            "recovered_flows": len(rows),
            "known_test_values_stored": 0,
            "unknown_test_values_stored": 0,
            "non_target_packet_values": "discarded immediately; no non-target per-flow statistics retained",
            "packet_count_mismatches": 0,
            "packet_windows": [8, 16, 32, 64],
            "byte_coverage_semantics": "captured frame bytes",
            "time_coverage_semantics": "elapsed time from first through Nth packet divided by full flow duration",
            "flow_identity": "tshark tcp.stream/udp.stream within capture; exact Stage14C lineage",
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "stage14c5_cache_audit_sha256": sha256_file(stage14c5_cache_audit),
            "metrics_sha256": sha256_file(metrics),
        },
    )


if __name__ == "__main__":
    main()
