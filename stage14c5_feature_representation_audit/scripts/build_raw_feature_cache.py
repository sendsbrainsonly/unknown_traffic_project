#!/usr/bin/env python3
"""Extract run-independent packet timing/length/direction statistics.

Only flow IDs already present in a Stage 14C Known Train/Validation input
manifest are targets. The cache contains no split role and is never fitted.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import RAW_CACHE, STAGE14C_ROOT, STAT_NAMES, json_dump, load_known_input_rows, protocol_ids, sha256_file, stage14c_common, verify_frozen_inputs


BASE_FIELDS = (
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.stream", "udp.stream",
)
EXTRA_FIELDS = ("frame.time_epoch", "frame.len")


@dataclass
class FlowAccumulator:
    uid: str
    count: int = 0
    total_bytes: int = 0
    first_time: float | None = None
    last_time: float | None = None
    previous_time: float | None = None
    size_mean: float = 0.0
    size_m2: float = 0.0
    iat_count: int = 0
    iat_mean: float = 0.0
    iat_m2: float = 0.0
    origin: tuple[str, str] | None = None
    forward: int = 0
    reverse: int = 0
    seq_iat: list[float] = field(default_factory=list)
    seq_length: list[int] = field(default_factory=list)
    seq_direction: list[int] = field(default_factory=list)

    def add(self, timestamp: float, packet_length: int, endpoint: tuple[str, str], reverse_endpoint: tuple[str, str]) -> None:
        if self.origin is None:
            self.origin = endpoint
        direction = 1 if endpoint == self.origin else (-1 if reverse_endpoint == self.origin else 0)
        if direction == 1:
            self.forward += 1
        elif direction == -1:
            self.reverse += 1
        self.count += 1
        self.total_bytes += packet_length
        delta = packet_length - self.size_mean
        self.size_mean += delta / self.count
        self.size_m2 += delta * (packet_length - self.size_mean)
        if self.first_time is None:
            self.first_time = timestamp
        iat = 0.0 if self.previous_time is None else max(0.0, timestamp - self.previous_time)
        if self.previous_time is not None:
            self.iat_count += 1
            delta_iat = iat - self.iat_mean
            self.iat_mean += delta_iat / self.iat_count
            self.iat_m2 += delta_iat * (iat - self.iat_mean)
        self.previous_time = timestamp
        self.last_time = timestamp
        if len(self.seq_iat) < 8:
            self.seq_iat.append(iat)
            self.seq_length.append(packet_length)
            self.seq_direction.append(direction)

    def statistics(self) -> list[float]:
        duration = 0.0 if self.first_time is None or self.last_time is None else max(0.0, self.last_time - self.first_time)
        size_std = math.sqrt(max(self.size_m2 / self.count, 0.0)) if self.count else 0.0
        iat_std = math.sqrt(max(self.iat_m2 / self.iat_count, 0.0)) if self.iat_count else 0.0
        ratio = (self.forward + 1.0) / (self.reverse + 1.0)
        return [float(self.count), float(self.total_bytes), duration, self.size_mean, size_std, self.iat_mean, iat_std, ratio]


def endpoints(values: list[str]) -> tuple[tuple[str, str], tuple[str, str]]:
    src = values[0] or values[1]
    dst = values[2] or values[3]
    sport = values[6] or values[8]
    dport = values[7] or values[9]
    return (src, sport), (dst, dport)


def main() -> None:
    if RAW_CACHE.exists():
        raise RuntimeError(f"refusing to overwrite {RAW_CACHE}")
    RAW_CACHE.mkdir(parents=True)
    stderr_root = RAW_CACHE / "tshark_stderr"
    stderr_root.mkdir()
    frozen = verify_frozen_inputs()

    target_uids: set[str] = set()
    usage_count = 0
    for protocol_id in protocol_ids():
        rows = load_known_input_rows(protocol_id)
        usage_count += len(rows)
        target_uids.update(row["flow_uid"] for row in rows)

    cache_manifest_path = STAGE14C_ROOT / "flow_image_cache" / "cache_manifest.csv"
    selected_meta: dict[str, dict[str, str]] = {}
    with cache_manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["flow_uid"] in target_uids:
                selected_meta[row["flow_uid"]] = row
    if set(selected_meta) != target_uids:
        raise RuntimeError("Known Train/Validation UID missing from Stage 14C cache manifest")

    by_capture: dict[str, dict[str, FlowAccumulator]] = defaultdict(dict)
    capture_paths: dict[str, Path] = {}
    for uid, row in selected_meta.items():
        by_capture[row["capture_id"]][row["source_flow_id"]] = FlowAccumulator(uid)
        capture_paths[row["capture_id"]] = Path(row["source_pcap_path"])

    capture_rows = []
    for capture_index, capture_id in enumerate(sorted(by_capture), start=1):
        pcap_path = capture_paths[capture_id]
        command = ["tshark", "-n", "-r", str(pcap_path), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
        for name in BASE_FIELDS + EXTRA_FIELDS:
            command.extend(["-e", name])
        packet_rows = 0
        matched_rows = 0
        stderr_path = stderr_root / f"{capture_id}.log"
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
            assert process.stdout is not None
            for line in process.stdout:
                packet_rows += 1
                values = line.rstrip("\r\n").split("\t")
                if len(values) < len(BASE_FIELDS) + len(EXTRA_FIELDS):
                    values += [""] * (len(BASE_FIELDS) + len(EXTRA_FIELDS) - len(values))
                flow_id = stage14c_common.tshark_flow_id(values[: len(BASE_FIELDS)])
                if flow_id is None or flow_id not in by_capture[capture_id]:
                    continue
                try:
                    timestamp = float(values[12])
                    packet_length = int(values[13])
                except ValueError as exc:
                    process.kill()
                    raise RuntimeError(f"{capture_id}: invalid time/length at packet row {packet_rows}") from exc
                ep, rev = endpoints(values)
                by_capture[capture_id][flow_id].add(timestamp, packet_length, ep, rev)
                matched_rows += 1
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"{capture_id}: tshark exited {returncode}")
        capture_rows.append({"capture_id": capture_id, "target_flows": len(by_capture[capture_id]), "pcap_packet_rows": packet_rows, "matched_packet_rows": matched_rows, "source_pcap_path": str(pcap_path), "tshark_stderr": str(stderr_path)})
        print(json.dumps({"event": "capture_complete", "capture_index": capture_index, "capture_total": len(by_capture), "capture_id": capture_id, "target_flows": len(by_capture[capture_id])}, sort_keys=True), flush=True)

    accumulators = {acc.uid: acc for flows in by_capture.values() for acc in flows.values()}
    ordered_uids = sorted(target_uids)
    n = len(ordered_uids)
    iats = np.zeros((n, 8), dtype=np.float32)
    lengths = np.zeros((n, 8), dtype=np.float32)
    directions = np.zeros((n, 8), dtype=np.int8)
    mask = np.zeros((n, 8), dtype=np.uint8)
    statistics = np.zeros((n, len(STAT_NAMES)), dtype=np.float64)
    manifest_rows = []
    mismatches = []
    for index, uid in enumerate(ordered_uids):
        acc = accumulators[uid]
        meta = selected_meta[uid]
        used = min(acc.count, 8)
        iats[index, :used] = acc.seq_iat
        lengths[index, :used] = acc.seq_length
        directions[index, :used] = acc.seq_direction
        mask[index, :used] = 1
        statistics[index] = acc.statistics()
        expected = int(meta["packet_count_expected"])
        if acc.count != expected:
            mismatches.append({"flow_uid": uid, "expected": expected, "observed": acc.count})
        manifest_rows.append({"raw_cache_index": index, "flow_uid": uid, "capture_id": meta["capture_id"], "source_flow_id": meta["source_flow_id"], "packet_count_expected": expected, "packet_count_observed": acc.count, "first8_packets": used})
    if mismatches:
        raise RuntimeError(f"packet-count mismatches: {mismatches[:5]}")

    np.save(RAW_CACHE / "flow_uids.npy", np.asarray(ordered_uids, dtype="U64"), allow_pickle=False)
    np.save(RAW_CACHE / "packet_iat_seconds.npy", iats, allow_pickle=False)
    np.save(RAW_CACHE / "packet_lengths.npy", lengths, allow_pickle=False)
    np.save(RAW_CACHE / "packet_directions.npy", directions, allow_pickle=False)
    np.save(RAW_CACHE / "packet_mask.npy", mask, allow_pickle=False)
    np.save(RAW_CACHE / "flow_statistics.npy", statistics, allow_pickle=False)
    with (RAW_CACHE / "raw_cache_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0])); writer.writeheader(); writer.writerows(manifest_rows)
    with (RAW_CACHE / "capture_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(capture_rows[0])); writer.writeheader(); writer.writerows(capture_rows)
    audit = {
        "status": "PASS",
        "scope": "union of sample IDs appearing in at least one frozen Stage 14C Known Train/Validation input manifest",
        "protocol_count": len(protocol_ids()),
        "known_train_validation_row_uses": usage_count,
        "unique_target_flows": n,
        "packet_count_mismatches": 0,
        "known_test_rows_loaded": 0,
        "unknown_test_rows_loaded": 0,
        "label_values_used_for_extraction": False,
        "statistics": list(STAT_NAMES),
        "direction_rule": "first observed endpoint is forward; exact reverse endpoint is reverse",
        "up_down_ratio": "(forward_packet_count+1)/(reverse_packet_count+1)",
        "stage14b_freeze_hash": frozen["freeze_hash"],
        "stage14c_summary_sha256": frozen["stage14c_summary_sha256"],
        "flow_uids_sha256": sha256_file(RAW_CACHE / "flow_uids.npy"),
        "flow_statistics_sha256": sha256_file(RAW_CACHE / "flow_statistics.npy"),
    }
    json_dump(RAW_CACHE / "cache_audit.json", audit)
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
