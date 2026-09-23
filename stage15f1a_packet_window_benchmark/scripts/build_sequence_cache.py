#!/usr/bin/env python3
"""Build first-32 packet sequence caches for frozen Known Train/Validation only."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import PROJECT, ROOT, STAGE12, STAGE14B, STAGE15R, pilot_specs, protocol_rows, read_csv, read_json, sha256_file, write_csv, write_json

MAX_PACKETS = 32
TSHARK_FIELDS = (
    "frame.time_epoch", "frame.len", "ip.src", "ipv6.src", "ip.dst", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
)
VNAT_BASE_FIELDS = (
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "tcp.stream", "udp.stream",
)


@dataclass
class SeqAccumulator:
    uid: str
    expected_count: int | None = None
    count: int = 0
    previous_time: float | None = None
    origin: tuple[str, str] | None = None
    lengths: list[float] = field(default_factory=list)
    iats: list[float] = field(default_factory=list)
    directions: list[int] = field(default_factory=list)

    def add(self, timestamp: float, length: int, source: tuple[str, str], destination: tuple[str, str]) -> None:
        if self.origin is None:
            self.origin = source
        direction = 1 if source == self.origin else (-1 if destination == self.origin else 0)
        iat = 0.0 if self.previous_time is None else max(0.0, timestamp - self.previous_time)
        self.previous_time = timestamp
        self.count += 1
        if len(self.lengths) < MAX_PACKETS:
            self.lengths.append(float(length))
            self.iats.append(float(iat))
            self.directions.append(direction)


@dataclass
class FlowState:
    sequence: int = 0
    session_hash: str | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


def value(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def flag(text: str) -> bool:
    return text.lower() in {"1", "true", "set"}


def endpoint(values: list[str], offset: int = 0):
    src = value(values, offset + 2) or value(values, offset + 3)
    dst = value(values, offset + 4) or value(values, offset + 5)
    tcp_s, tcp_d = value(values, offset + 6), value(values, offset + 7)
    udp_s, udp_d = value(values, offset + 8), value(values, offset + 9)
    if src and dst and tcp_s and tcp_d:
        return (src, tcp_s), (dst, tcp_d), "tcp"
    if src and dst and udp_s and udp_d:
        return (src, udp_s), (dst, udp_d), "udp"
    return None


def target_ids(dataset: str) -> set[str]:
    targets: set[str] = set()
    for spec in pilot_specs():
        if spec["dataset"] != dataset:
            continue
        _, train, val = protocol_rows(dataset, spec["protocol_id"])
        targets.update(row["sample_id"] for row in train + val)
    return targets


def scan_ustc(targets: set[str], output: Path):
    accumulators: dict[str, SeqAccumulator] = {}
    audit = []
    for pkl_path in sorted((PROJECT / "data" / "flows").glob("*.pkl")):
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in targets:
                continue
            matched += 1
            acc = SeqAccumulator(uid, len(packets))
            for timestamp, caplen, _wirelen, direction, _raw_frame in packets:
                # The frozen USTC preprocessing already stores the canonical
                # initiator-relative direction.  Re-inferring it from the first
                # observed packet can invert flows whose first stored packet is
                # responder-to-initiator, so retain the lineage value verbatim.
                current_time = float(timestamp)
                iat = 0.0 if acc.previous_time is None else max(0.0, current_time - acc.previous_time)
                acc.previous_time = current_time
                acc.count += 1
                if len(acc.lengths) < MAX_PACKETS:
                    acc.lengths.append(float(caplen))
                    acc.iats.append(float(iat))
                    acc.directions.append(int(direction))
            accumulators[uid] = acc
        audit.append({"source": str(pkl_path.resolve()), "sha256": sha256_file(pkl_path), "matched_targets": matched})
        print(json.dumps({"event": "pkl_complete", "file": pkl_path.name, "matched": matched}), flush=True)
        del payload
        gc.collect()
    write_csv(output / "source_audit.csv", audit)
    return accumulators


def scan_iscx(dataset: str, targets: set[str], output: Path):
    prep = read_json(STAGE12 / "protocol" / dataset / "preprocessing_manifest.json")
    split_rows = read_csv(STAGE12 / "protocol" / dataset / "split_manifest.csv")
    target_meta = {row["flow_id_sha256"]: row for row in split_rows if row["flow_id_sha256"] in targets}
    capture_keys = {(row["canonical_class"], Path(row["source_file"]).name) for row in target_meta.values()}
    timeout = float(read_json(STAGE12 / "configs" / "stage12_config.json")["preprocessing"]["session_timeout_seconds"])
    accumulators: dict[str, SeqAccumulator] = {}
    audit = []
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    for class_spec in sorted(prep["classes"], key=lambda item: int(item["label"])):
        class_name = str(class_spec["name"])
        for capture_index, raw_path in enumerate(class_spec["captures"]):
            pcap = Path(raw_path)
            if (class_name, pcap.name) not in capture_keys:
                continue
            capture_id = f"class={class_name}|capture={capture_index}|name={pcap.name}"
            command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
            for field_name in TSHARK_FIELDS:
                command += ["-e", field_name]
            states = {}
            packet_rows = matched_rows = 0
            stderr_path = stderr_root / f"{class_name}_{capture_index:03d}_{pcap.stem}.log"
            with stderr_path.open("w", encoding="utf-8") as err:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1024 * 1024)
                assert process.stdout is not None
                for line in process.stdout:
                    packet_rows += 1
                    values = line.rstrip("\r\n").split("\t")
                    ep = endpoint(values)
                    if ep is None:
                        continue
                    source, destination, protocol = ep
                    ordered = tuple(sorted((source, destination)))
                    key = (capture_id, protocol, ordered[0][0], ordered[0][1], ordered[1][0] + ":" + ordered[1][1])
                    timestamp = float(value(values, 0))
                    state = states.setdefault(key, FlowState())
                    new_syn = protocol == "tcp" and flag(value(values, 10)) and not flag(value(values, 11))
                    timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
                    syn_starts_new = new_syn and state.session_hash is not None and not state.syn_only
                    if state.session_hash is None or state.closed or timed_out or syn_starts_new:
                        token = f"{protocol}|{ordered[0][0]}:{ordered[0][1]}|{ordered[1][0]}:{ordered[1][1]}"
                        session_id = f"{capture_id}|{token}|session={state.sequence}"
                        state.session_hash = hashlib.sha256(session_id.encode()).hexdigest()
                        state.sequence += 1
                        state.closed = False
                        state.syn_only = new_syn
                    uid = str(state.session_hash)
                    if uid in targets:
                        accumulators.setdefault(uid, SeqAccumulator(uid)).add(timestamp, int(value(values, 1) or 0), source, destination)
                        matched_rows += 1
                    state.last_timestamp = timestamp
                    if not new_syn:
                        state.syn_only = False
                    if protocol == "tcp" and (flag(value(values, 12)) or flag(value(values, 13))):
                        state.closed = True
                code = process.wait()
            if code:
                raise RuntimeError(f"tshark failed {code}: {pcap}")
            audit.append({"source": str(pcap), "packet_rows": packet_rows, "matched_packet_rows": matched_rows, "stderr": str(stderr_path)})
            print(json.dumps({"event": "capture_complete", "dataset": dataset, "capture": pcap.name, "matched_flows": len(accumulators), "targets": len(targets)}), flush=True)
    write_csv(output / "source_audit.csv", audit)
    return accumulators


def vnat_flow_id(values: list[str]) -> str | None:
    src, dst = values[0] or values[1], values[2] or values[3]
    if not src or not dst:
        return None
    if values[10]:
        return f"tcp:{values[10]}"
    if values[11]:
        return f"udp:{values[11]}"
    first, second = sorted(((src, ""), (dst, "")))
    token = f"ip:{values[4] or values[5] or 'unknown'}|{first[0]}:|{second[0]}:"
    return f"other:{hashlib.sha256(token.encode()).hexdigest()[:24]}"


def scan_vnat(targets: set[str], output: Path):
    meta = {}
    for row in read_csv(STAGE14B / "vnat_split_manifest.csv"):
        if row["flow_uid"] in targets:
            meta.setdefault(row["flow_uid"], row)
    if set(meta) != targets:
        raise RuntimeError(f"missing VNAT metadata: {len(targets-set(meta))}")
    by_capture = defaultdict(dict)
    paths = {}
    for uid, row in meta.items():
        by_capture[row["capture_id"]][row["source_flow_id"]] = SeqAccumulator(uid, int(row["packet_count"]))
        paths[row["capture_id"]] = Path(row["source_pcap_path"])
    stderr_root = output / "tshark_stderr"
    stderr_root.mkdir()
    audit = []
    fields = VNAT_BASE_FIELDS + ("frame.time_epoch", "frame.len")
    for index, capture_id in enumerate(sorted(by_capture), 1):
        pcap = paths[capture_id]
        command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
        for name in fields:
            command += ["-e", name]
        packet_rows = matched_rows = 0
        stderr_path = stderr_root / f"{capture_id}.log"
        with stderr_path.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1024 * 1024)
            assert process.stdout is not None
            for line in process.stdout:
                packet_rows += 1
                values = line.rstrip("\r\n").split("\t")
                values += [""] * (len(fields) - len(values))
                flow_id = vnat_flow_id(values[:12])
                if flow_id not in by_capture[capture_id]:
                    continue
                source = (values[0] or values[1], values[6] or values[8])
                destination = (values[2] or values[3], values[7] or values[9])
                by_capture[capture_id][flow_id].add(float(values[12]), int(values[13]), source, destination)
                matched_rows += 1
            code = process.wait()
        if code:
            raise RuntimeError(f"tshark failed {code}: {pcap}")
        audit.append({"capture_id": capture_id, "source": str(pcap), "packet_rows": packet_rows, "matched_packet_rows": matched_rows, "stderr": str(stderr_path)})
        print(json.dumps({"event": "capture_complete", "capture_index": index, "capture_total": len(by_capture), "capture_id": capture_id}), flush=True)
    write_csv(output / "source_audit.csv", audit)
    return {acc.uid: acc for flows in by_capture.values() for acc in flows.values()}


def save_cache(dataset: str, targets: set[str], accumulators: dict[str, SeqAccumulator], output: Path):
    missing = sorted(targets - set(accumulators))
    if missing:
        write_json(output / "failure.json", {"missing": len(missing), "preview": missing[:100]})
        raise RuntimeError(f"missing {len(missing)} targets")
    ordered = sorted(targets)
    n = len(ordered)
    lengths = np.zeros((n, MAX_PACKETS), np.float32)
    iats = np.zeros_like(lengths)
    directions = np.zeros((n, MAX_PACKETS), np.int8)
    mask = np.zeros((n, MAX_PACKETS), np.uint8)
    counts = np.zeros(n, np.int32)
    mismatch = []
    for index, uid in enumerate(ordered):
        acc = accumulators[uid]
        used = len(acc.lengths)
        lengths[index, :used] = acc.lengths
        iats[index, :used] = acc.iats
        directions[index, :used] = acc.directions
        mask[index, :used] = 1
        counts[index] = acc.count
        if acc.expected_count is not None and acc.count != acc.expected_count:
            mismatch.append((uid, acc.expected_count, acc.count))
    if mismatch:
        raise RuntimeError(f"packet-count mismatch: {mismatch[:5]}")
    np.save(output / "flow_uids.npy", np.asarray(ordered, dtype="U80"), allow_pickle=False)
    np.save(output / "packet_lengths.npy", lengths, allow_pickle=False)
    np.save(output / "packet_iat_seconds.npy", iats, allow_pickle=False)
    np.save(output / "packet_directions.npy", directions, allow_pickle=False)
    np.save(output / "packet_mask.npy", mask, allow_pickle=False)
    np.save(output / "packet_counts.npy", counts, allow_pickle=False)
    legacy_root = STAGE15R / "feature_cache" / dataset if dataset != "vnat" else PROJECT / "stage14c5_feature_representation_audit" / "raw_feature_cache"
    legacy_ids = np.load(legacy_root / "flow_uids.npy", allow_pickle=False)
    legacy_index = {str(uid): i for i, uid in enumerate(legacy_ids)}
    first8_equal = True
    mismatch_counts = {}
    for name, current in (("length", lengths), ("iat", iats), ("direction", directions), ("mask", mask)):
        old_name = {"length": "packet_lengths.npy", "iat": "packet_iat_seconds.npy", "direction": "packet_directions.npy", "mask": "packet_mask.npy"}[name]
        old = np.load(legacy_root / old_name, mmap_mode="r", allow_pickle=False)
        positions = np.asarray([legacy_index[uid] for uid in ordered])
        mismatches = int(np.count_nonzero(old[positions, :8] != current[:, :8]))
        mismatch_counts[name] = mismatches
        first8_equal &= mismatches == 0
    audit = {
        "status": "PASS" if first8_equal else "FAIL",
        "dataset": dataset,
        "scope": "union of frozen pilot Known Train/Validation sample IDs only",
        "target_flows": n,
        "recovered_flows": n,
        "max_packets": MAX_PACKETS,
        "known_test_values_stored": 0,
        "unknown_test_values_stored": 0,
        "legacy_first8_exact_parity": first8_equal,
        "legacy_first8_mismatch_elements": mismatch_counts,
        "flow_uids_sha256": sha256_file(output / "flow_uids.npy"),
        "packet_lengths_sha256": sha256_file(output / "packet_lengths.npy"),
        "packet_iat_sha256": sha256_file(output / "packet_iat_seconds.npy"),
        "packet_directions_sha256": sha256_file(output / "packet_directions.npy"),
        "packet_mask_sha256": sha256_file(output / "packet_mask.npy"),
    }
    write_json(output / "cache_audit.json", audit)
    print(json.dumps(audit, sort_keys=True), flush=True)
    if not first8_equal:
        raise RuntimeError("legacy first8 cache parity failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ustc", "iscx_vpn", "iscx_tor", "vnat"), required=True)
    args = parser.parse_args()
    output = ROOT / "sequence_cache" / args.dataset
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    targets = target_ids(args.dataset)
    if args.dataset == "ustc":
        acc = scan_ustc(targets, output)
    elif args.dataset == "vnat":
        acc = scan_vnat(targets, output)
    else:
        acc = scan_iscx(args.dataset, targets, output)
    save_cache(args.dataset, targets, acc, output)


if __name__ == "__main__":
    main()
