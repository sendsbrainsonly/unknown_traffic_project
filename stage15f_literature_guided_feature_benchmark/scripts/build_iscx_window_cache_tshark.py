#!/usr/bin/env python3
"""Fast tshark replay of the exact Stage 12 bidirectional session state machine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from common import FLOW_WINDOW_FIELDS, PROJECT_ROOT, ROOT, WindowAccumulator, read_json, sha256_file, write_csv, write_json


STAGE12 = PROJECT_ROOT / "stage12_dual_external_validation"
FIELDS = (
    "frame.time_epoch", "frame.len", "frame.protocols",
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
    "tcp.len", "udp.length", "ip.len", "ipv6.plen",
)


@dataclass
class FlowState:
    sequence: int = 0
    session_id: str | None = None
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


def value(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def flag(text: str) -> bool:
    return text.lower() in {"1", "true", "set"}


def endpoint(values: list[str]) -> tuple[tuple[str, str], tuple[str, str], str] | None:
    src = value(values, 3) or value(values, 4)
    dst = value(values, 5) or value(values, 6)
    tcp_s, tcp_d = value(values, 7), value(values, 8)
    udp_s, udp_d = value(values, 9), value(values, 10)
    if src and dst and tcp_s and tcp_d:
        return (src, tcp_s), (dst, tcp_d), "tcp"
    if src and dst and udp_s and udp_d:
        return (src, udp_s), (dst, udp_d), "udp"
    return None


def load_targets(dataset: str) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    path = STAGE12 / "protocol" / dataset / "split_manifest.csv"
    targets: dict[str, dict[str, str]] = {}
    capture_meta: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] not in {"known_train", "known_validation"}:
                continue
            uid = row["flow_id_sha256"]
            prior = targets.setdefault(uid, row)
            if prior["canonical_class"] != row["canonical_class"] or prior["source_file"] != row["source_file"]:
                raise RuntimeError(f"inconsistent target metadata: {uid}")
            capture_meta[(row["canonical_class"], Path(row["source_file"]).name)] = row
    return targets, capture_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    output = ROOT / "window_cache" / dataset
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    targets, capture_meta = load_targets(dataset)
    config = read_json(STAGE12 / "configs" / "stage12_config.json")
    assert isinstance(config, dict)
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    prep_path = STAGE12 / "protocol" / dataset / "preprocessing_manifest.json"
    prep = read_json(prep_path)
    assert isinstance(prep, dict)
    accumulators: dict[str, WindowAccumulator] = {}
    capture_rows: list[dict[str, object]] = []
    for class_spec in sorted(prep["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            pcap = Path(raw_path)
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            meta = capture_meta.get((class_name, pcap.name))
            if meta is None:
                continue
            command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
            for field in FIELDS:
                command.extend(["-e", field])
            states: dict[tuple[str, str, str, str, str], FlowState] = {}
            packet_rows = matched_rows = discarded_rows = 0
            stderr_path = stderr_root / f"{class_name}_{capture_index:03d}_{pcap.stem}.log"
            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
                assert process.stdout is not None
                for line in process.stdout:
                    packet_rows += 1
                    values = line.rstrip("\r\n").split("\t")
                    ep = endpoint(values)
                    if ep is None:
                        discarded_rows += 1
                        continue
                    source, destination, protocol = ep
                    ordered = tuple(sorted((source, destination)))
                    key = (capture_id, protocol, ordered[0][0], ordered[0][1], ordered[1][0] + ":" + ordered[1][1])
                    timestamp = float(value(values, 0))
                    state = states.setdefault(key, FlowState())
                    new_syn = protocol == "tcp" and flag(value(values, 11)) and not flag(value(values, 12))
                    timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                    if state.session_id is None or state.closed or timed_out or syn_starts_new:
                        flow_token = f"{protocol}|{ordered[0][0]}:{ordered[0][1]}|{ordered[1][0]}:{ordered[1][1]}"
                        state.session_id = f"{capture_id}|{flow_token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                    uid = str(state.session_hash)
                    if uid in targets:
                        if uid not in accumulators:
                            target = targets[uid]
                            accumulators[uid] = WindowAccumulator(uid, class_name, target["source_file"], target["domain_state"])
                        frame_length = int(value(values, 1) or 0)
                        tcp_payload = int(value(values, 15) or 0)
                        udp_total = int(value(values, 16) or 0)
                        transport_payload = tcp_payload if protocol == "tcp" else max(udp_total - 8, 0)
                        ip_bytes = int(value(values, 17) or 0) or (int(value(values, 18) or 0) + 40 if value(values, 18) else None)
                        accumulators[uid].add(
                            timestamp,
                            frame_length,
                            ip_bytes,
                            transport_payload,
                            source,
                            destination,
                            value(values, 2),
                        )
                        matched_rows += 1
                    else:
                        discarded_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and (flag(value(values, 13)) or flag(value(values, 14))):
                        state.closed = True
                returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"tshark failed ({returncode}) for {pcap}; see {stderr_path}")
            capture_rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "capture_index": capture_index,
                    "source_file": meta["source_file"],
                    "pcap_path": str(pcap),
                    "packet_rows": packet_rows,
                    "matched_known_train_validation_packet_rows": matched_rows,
                    "discarded_non_target_or_non_tcp_udp_rows": discarded_rows,
                    "stderr_path": str(stderr_path),
                    "engine": "tshark fields; Stage15R-verified Stage12 session state machine",
                }
            )
            print(json.dumps({"event": "capture_complete", "dataset": dataset, "class": class_name, "capture": pcap.name, "matched_flows": len(accumulators), "target_flows": len(targets)}), flush=True)
    missing = sorted(set(targets) - set(accumulators))
    if missing:
        write_json(output / "failure.json", {"missing_count": len(missing), "preview": missing[:100]})
        raise RuntimeError(f"missing {len(missing)} {dataset} targets")
    rows = [accumulators[uid].row(dataset) for uid in sorted(accumulators)]
    metrics = output / "flow_window_metrics.csv.gz"
    write_csv(metrics, rows, FLOW_WINDOW_FIELDS)
    write_csv(output / "capture_audit.csv", capture_rows, list(capture_rows[0]))
    split_path = STAGE12 / "protocol" / dataset / "split_manifest.csv"
    stage15r_capture = PROJECT_ROOT / "stage15r_representation_bottleneck_audit" / "feature_cache" / dataset / "capture_audit.csv"
    write_json(
        output / "cache_audit.json",
        {
            "status": "PASS",
            "dataset": dataset,
            "engine": "tshark field stream using Stage15R-verified Stage12 flow/session state machine",
            "scope": "union of Known Train/Validation rows in frozen low/medium/high Stage12 protocols",
            "target_flows": len(targets),
            "recovered_flows": len(rows),
            "known_test_values_stored": 0,
            "unknown_test_values_stored": 0,
            "non_target_packet_values": "parsed only to maintain session state and discarded immediately",
            "sessionization": {"bidirectional": True, "ipv4_tcp_udp_only": True, "timeout_seconds": timeout, "tcp_syn_fin_rst_boundaries": True},
            "packet_windows": [8, 16, 32, 64],
            "byte_coverage_semantics": "captured frame bytes",
            "time_coverage_semantics": "elapsed time from first through Nth packet divided by full flow duration",
            "split_manifest_sha256": sha256_file(split_path),
            "preprocessing_manifest_sha256": sha256_file(prep_path),
            "stage15r_capture_audit_sha256": sha256_file(stage15r_capture),
            "metrics_sha256": sha256_file(metrics),
        },
    )


if __name__ == "__main__":
    main()
