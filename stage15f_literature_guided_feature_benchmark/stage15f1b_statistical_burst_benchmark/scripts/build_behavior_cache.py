#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from behavior_features import BehaviorAccumulator
from common import (
    PROJECT, ROOT, STAGE12, STAGE14C_INPUT, STAGE15F0, cache_dir,
    feature_names, pilot_specs, protocol_rows, read_csv, read_json,
    sha256_file, write_csv, write_json,
)


@dataclass
class FlowState:
    sequence: int = 0
    session_id: str | None = None
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


def targets_for(dataset: str) -> dict[str, dict]:
    targets: dict[str, dict] = {}
    for spec in pilot_specs():
        if spec["dataset"] != dataset:
            continue
        _, train_rows, val_rows = protocol_rows(dataset, spec["protocol_id"])
        for role, rows in (("known_train", train_rows), ("known_validation", val_rows)):
            for row in rows:
                uid = row["sample_id"]
                current = {**row, "role_seen": role, "protocol_id_seen": spec["protocol_id"]}
                prior = targets.setdefault(uid, current)
                if prior["class_name"] != row["class_name"]:
                    raise RuntimeError(f"inconsistent class for {uid}")
    if not targets:
        raise RuntimeError(f"no preregistered targets for {dataset}")
    return targets


def packet_fields(frame: bytes) -> tuple[int | None, str]:
    if len(frame) < 34 or frame[12:14] != b"\x08\x00":
        return None, ""
    ihl = (frame[14] & 0x0F) * 4
    ip_total = int.from_bytes(frame[16:18], "big")
    protocol = frame[23]
    l4 = 14 + ihl
    if protocol == 6 and len(frame) >= l4 + 20:
        tcp_hlen = ((frame[l4 + 12] >> 4) & 0x0F) * 4
        return max(0, ip_total - ihl - tcp_hlen), "tcp"
    if protocol == 17 and len(frame) >= l4 + 8:
        return max(0, ip_total - ihl - 8), "udp"
    return None, "other"


def allocate(targets: dict[str, dict]) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    ordered = sorted(targets)
    index = {uid: offset for offset, uid in enumerate(ordered)}
    shape = (len(ordered), len(feature_names()))
    return ordered, index, np.full(shape, np.nan, dtype=np.float64), np.full(shape, np.nan, dtype=np.float64)


def build_ustc(targets: dict[str, dict]) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    ordered, index, full, early = allocate(targets)
    seen: set[str] = set()
    audit = []
    for pkl_path in sorted((PROJECT / "data" / "flows").glob("*.pkl")):
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in index:
                continue
            matched += 1
            acc = BehaviorAccumulator(uid)
            for timestamp, caplen, _wirelen, direction, raw_frame in packets:
                payload_length, _protocol = packet_fields(bytes(raw_frame))
                # Frozen Stage-0 tuples use 1 for the initiator direction and
                # 0 for the reverse direction (not a signed +/-1 encoding).
                signed_direction = 1 if int(direction) == 1 else -1
                acc.add(float(timestamp), int(caplen), payload_length, None, None, explicit_direction=signed_direction)
            full_vector = acc.vector()
            full[index[uid]] = full_vector
            early[index[uid]] = full_vector if len(acc.timestamps) <= 16 else acc.vector(16)
            seen.add(uid)
        audit.append({
            "source": str(pkl_path.resolve()), "source_sha256": sha256_file(pkl_path),
            "flows_in_source": len(payload["packets"]), "matched_target_flows": matched,
        })
        print(json.dumps({"event": "pkl_complete", "file": pkl_path.name, "matched": matched}), flush=True)
        del payload
        gc.collect()
    missing = sorted(set(ordered) - seen)
    if missing:
        raise RuntimeError(f"USTC missing {len(missing)} target flows: {missing[:3]}")
    return ordered, full, early, audit


def _value(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def _flag(text: str) -> bool:
    return text.lower() in {"1", "true", "set"}


def _iscx_endpoint(values: list[str]) -> tuple[tuple[str, str], tuple[str, str], str] | None:
    src = _value(values, 3) or _value(values, 4)
    dst = _value(values, 5) or _value(values, 6)
    tcp_s, tcp_d = _value(values, 7), _value(values, 8)
    udp_s, udp_d = _value(values, 9), _value(values, 10)
    if src and dst and tcp_s and tcp_d:
        return (src, tcp_s), (dst, tcp_d), "tcp"
    if src and dst and udp_s and udp_d:
        return (src, udp_s), (dst, udp_d), "udp"
    return None


def build_iscx(dataset: str, targets: dict[str, dict], stderr_root: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    ordered, index, full, early = allocate(targets)
    config = read_json(STAGE12 / "configs" / "stage12_config.json")
    timeout = float(config["preprocessing"]["session_timeout_seconds"])
    prep = read_json(STAGE12 / "protocol" / dataset / "preprocessing_manifest.json")
    target_meta = {}
    for row in read_csv(STAGE12 / "protocol" / dataset / "split_manifest.csv"):
        if row["setting"] == "medium" and row["role"] in {"known_train", "known_validation"}:
            target_meta.setdefault((row["canonical_class"], Path(row["source_file"]).name), row)
    accumulators: dict[str, BehaviorAccumulator] = {}
    audit = []
    fields = (
        "frame.time_epoch", "frame.len", "frame.protocols", "ip.src", "ipv6.src", "ip.dst", "ipv6.dst",
        "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "tcp.flags.syn", "tcp.flags.ack",
        "tcp.flags.fin", "tcp.flags.reset", "tcp.len", "udp.length", "ip.len", "ipv6.plen",
    )
    for class_spec in sorted(prep["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            pcap = Path(raw_path)
            meta = target_meta.get((class_name, pcap.name))
            if meta is None:
                continue
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
            for field_name in fields:
                command.extend(["-e", field_name])
            states: dict[tuple[str, str, str, str, str], FlowState] = {}
            packet_rows = matched_rows = discarded_rows = 0
            stderr_path = stderr_root / f"{class_name}_{capture_index:03d}_{pcap.stem}.log"
            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
                assert process.stdout is not None
                for line in process.stdout:
                    packet_rows += 1
                    values = line.rstrip("\r\n").split("\t")
                    endpoint = _iscx_endpoint(values)
                    if endpoint is None:
                        discarded_rows += 1
                        continue
                    source, destination, protocol = endpoint
                    ordered_endpoint = tuple(sorted((source, destination)))
                    key = (capture_id, protocol, ordered_endpoint[0][0], ordered_endpoint[0][1], ordered_endpoint[1][0] + ":" + ordered_endpoint[1][1])
                    timestamp = float(_value(values, 0))
                    state = states.setdefault(key, FlowState())
                    new_syn = protocol == "tcp" and _flag(_value(values, 11)) and not _flag(_value(values, 12))
                    timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_id is not None and not state.syn_only
                    if state.session_id is None or state.closed or timed_out or syn_starts_new:
                        token = f"{protocol}|{ordered_endpoint[0][0]}:{ordered_endpoint[0][1]}|{ordered_endpoint[1][0]}:{ordered_endpoint[1][1]}"
                        state.session_id = f"{capture_id}|{token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(state.session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                    uid = str(state.session_hash)
                    if uid in targets:
                        acc = accumulators.setdefault(uid, BehaviorAccumulator(uid))
                        frame_length = int(_value(values, 1) or 0)
                        tcp_payload = int(_value(values, 15) or 0)
                        udp_total = int(_value(values, 16) or 0)
                        payload_length = tcp_payload if protocol == "tcp" else max(udp_total - 8, 0)
                        acc.add(timestamp, frame_length, payload_length, source, destination)
                        matched_rows += 1
                    else:
                        discarded_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and (_flag(_value(values, 13)) or _flag(_value(values, 14))):
                        state.closed = True
                returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"tshark failed ({returncode}) for {pcap}; see {stderr_path}")
            audit.append({
                "source": str(pcap.resolve()), "source_sha256": sha256_file(pcap),
                "packet_rows": packet_rows, "matched_target_packet_rows": matched_rows,
                "discarded_non_target_or_non_tcp_udp_rows": discarded_rows,
                "stderr": str(stderr_path),
            })
            print(json.dumps({"event": "capture_complete", "dataset": dataset, "class": class_name, "capture": pcap.name, "matched_flows": len(accumulators), "target_flows": len(targets)}), flush=True)
    missing = sorted(set(targets) - set(accumulators))
    if missing:
        raise RuntimeError(f"{dataset} missing {len(missing)} target flows: {missing[:3]}")
    for uid, acc in accumulators.items():
        full_vector = acc.vector()
        full[index[uid]] = full_vector
        early[index[uid]] = full_vector if len(acc.timestamps) <= 16 else acc.vector(16)
    return ordered, full, early, audit


def _vnat_flow_id(values: list[str]) -> str | None:
    src, dst = values[0] or values[1], values[2] or values[3]
    if not src or not dst:
        return None
    if values[10]:
        return f"tcp:{values[10]}"
    if values[11]:
        return f"udp:{values[11]}"
    first, second = sorted(((src, ""), (dst, "")))
    token = f"ip:{values[4] or values[5] or 'unknown'}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"
    return f"other:{hashlib.sha256(token.encode()).hexdigest()[:24]}"


def build_vnat(targets: dict[str, dict], stderr_root: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    ordered, index, full, early = allocate(targets)
    metadata = {row["flow_uid"]: row for row in read_csv(STAGE14C_INPUT / "flow_image_cache" / "cache_manifest.csv") if row["flow_uid"] in targets}
    if set(metadata) != set(targets):
        raise RuntimeError(f"VNAT missing lineage metadata for {len(set(targets)-set(metadata))} targets")
    by_capture: dict[str, dict[str, BehaviorAccumulator]] = {}
    capture_paths: dict[str, Path] = {}
    for uid, row in metadata.items():
        by_capture.setdefault(row["capture_id"], {})[row["source_flow_id"]] = BehaviorAccumulator(uid)
        capture_paths[row["capture_id"]] = Path(row["source_pcap_path"])
    base_fields = (
        "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt", "tcp.srcport", "tcp.dstport",
        "udp.srcport", "udp.dstport", "tcp.stream", "udp.stream",
    )
    extra_fields = ("frame.time_epoch", "frame.len", "frame.protocols", "ip.len", "ipv6.plen", "tcp.len", "udp.length")
    audit = []
    for capture_index, capture_id in enumerate(sorted(by_capture), start=1):
        pcap = capture_paths[capture_id]
        command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
        for field_name in base_fields + extra_fields:
            command.extend(["-e", field_name])
        packet_rows = matched_rows = discarded_rows = 0
        stderr_path = stderr_root / f"{capture_id}.log"
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
            assert process.stdout is not None
            for line in process.stdout:
                packet_rows += 1
                values = line.rstrip("\r\n").split("\t")
                expected = len(base_fields) + len(extra_fields)
                if len(values) < expected:
                    values += [""] * (expected - len(values))
                flow_id = _vnat_flow_id(values[:len(base_fields)])
                if flow_id is None or flow_id not in by_capture[capture_id]:
                    discarded_rows += 1
                    continue
                timestamp = float(values[12])
                frame_length = int(values[13])
                tcp_payload = int(values[17]) if values[17] else None
                udp_total = int(values[18]) if values[18] else None
                payload_length = tcp_payload if tcp_payload is not None else (max(0, udp_total - 8) if udp_total is not None else None)
                source = (values[0] or values[1], values[6] or values[8])
                destination = (values[2] or values[3], values[7] or values[9])
                by_capture[capture_id][flow_id].add(timestamp, frame_length, payload_length, source, destination)
                matched_rows += 1
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"VNAT tshark failed ({returncode}) for {pcap}; see {stderr_path}")
        audit.append({
            "source": str(pcap.resolve()), "source_sha256": sha256_file(pcap),
            "packet_rows": packet_rows, "matched_target_packet_rows": matched_rows,
            "discarded_non_target_packet_rows": discarded_rows, "stderr": str(stderr_path),
        })
        print(json.dumps({"event": "capture_complete", "dataset": "vnat", "capture_index": capture_index, "capture_total": len(by_capture), "capture": capture_id}), flush=True)
    accumulators = {acc.flow_uid: acc for flows in by_capture.values() for acc in flows.values()}
    missing = sorted(set(targets) - set(accumulators))
    if missing:
        raise RuntimeError(f"VNAT missing {len(missing)} target flows: {missing[:3]}")
    for uid, acc in accumulators.items():
        expected = int(metadata[uid]["packet_count_expected"])
        if len(acc.timestamps) != expected:
            raise RuntimeError(f"VNAT packet-count mismatch for {uid}: {len(acc.timestamps)} != {expected}")
        full_vector = acc.vector()
        full[index[uid]] = full_vector
        early[index[uid]] = full_vector if len(acc.timestamps) <= 16 else acc.vector(16)
    return ordered, full, early, audit


def reference_metrics(dataset: str, targets: set[str]) -> dict[str, dict[str, str]]:
    path = STAGE15F0 / "window_cache" / dataset / "flow_window_metrics.csv.gz"
    selected = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["flow_uid"] in targets:
                selected[row["flow_uid"]] = row
    if set(selected) != targets:
        raise RuntimeError(f"Stage15F0 reference mismatch for {dataset}: missing {len(targets-set(selected))}")
    return selected


def verify_reference(dataset: str, ordered: list[str], full: np.ndarray) -> dict:
    names = feature_names()
    col = {name: index for index, name in enumerate(names)}
    reference = reference_metrics(dataset, set(ordered))
    mismatches = []
    for row_index, uid in enumerate(ordered):
        ref = reference[uid]
        checks = {
            "packet_count": float(ref["packet_count"]),
            "total_bytes": float(ref["total_frame_bytes"]),
            "duration_seconds": float(ref["duration_seconds"]),
            "forward_packet_count": float(ref["forward_packets"]),
            "reverse_packet_count": float(ref["reverse_packets"]),
        }
        for name, expected in checks.items():
            observed = float(full[row_index, col[name]])
            if not np.isclose(observed, expected, rtol=1e-9, atol=1e-6):
                mismatches.append({"flow_uid": uid, "feature": name, "expected": expected, "observed": observed})
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    return {"status": "PASS" if not mismatches else "FAIL", "reference_flows": len(reference), "mismatch_count_preview_limited": len(mismatches), "mismatch_preview": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ustc", "vnat", "iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    output = cache_dir(dataset)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite behavior cache: {output}")
    output.mkdir(parents=True)
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    targets = targets_for(dataset)
    if dataset == "ustc":
        ordered, full, early, audit = build_ustc(targets)
    elif dataset == "vnat":
        ordered, full, early, audit = build_vnat(targets, stderr_root)
    else:
        ordered, full, early, audit = build_iscx(dataset, targets, stderr_root)
    parity = verify_reference(dataset, ordered, full)
    write_json(output / "reference_parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError(f"{dataset} Stage15F0 reference parity failed: {parity['mismatch_preview'][:3]}")
    np.save(output / "flow_uids.npy", np.asarray(ordered, dtype="U80"), allow_pickle=False)
    np.save(output / "full_flow_features.npy", full, allow_pickle=False)
    np.save(output / "early16_features.npy", early, allow_pickle=False)
    write_csv(output / "source_audit.csv", audit)
    write_json(output / "cache_audit.json", {
        "status": "PASS", "dataset": dataset, "target_flows": len(ordered), "recovered_flows": len(ordered),
        "known_test_feature_values_stored": 0, "unknown_test_feature_values_stored": 0,
        "non_target_values": "not retained; ISCX non-target headers/timestamps are used only for exact session state",
        "feature_names": feature_names(), "feature_count": len(feature_names()),
        "full_flow_sha256": sha256_file(output / "full_flow_features.npy"),
        "early16_sha256": sha256_file(output / "early16_features.npy"),
        "flow_uids_sha256": sha256_file(output / "flow_uids.npy"),
        "reference_parity": parity,
    })
    print(json.dumps({"status": "PASS", "dataset": dataset, "flows": len(ordered), "features": len(feature_names())}, sort_keys=True))


if __name__ == "__main__":
    main()
