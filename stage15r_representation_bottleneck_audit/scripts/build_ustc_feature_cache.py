#!/usr/bin/env python3
"""Recover exact USTC flow statistics from the preserved Stage-0 PKLs."""

from __future__ import annotations

import csv
import gc
import json
import math
import pickle
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, ROOT, USTC_AUDIT_ROOT, sha256_file, write_csv, write_json


MAX_PACKETS = 8


def main() -> None:
    output = ROOT / "feature_cache" / "ustc"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite cache: {output}")
    output.mkdir(parents=True)
    alignment_path = USTC_AUDIT_ROOT / "outputs" / "input_alignment_manifest.csv"
    with alignment_path.open(encoding="utf-8", newline="") as handle:
        alignment = {row["flow_id"]: row for row in csv.DictReader(handle)}
    if not alignment or any(row["input_valid"] != "True" for row in alignment.values()):
        raise RuntimeError("USTC input alignment is absent or contains invalid rows")
    ordered = sorted(alignment)
    index = {uid: offset for offset, uid in enumerate(ordered)}
    n = len(ordered)
    lengths = np.zeros((n, MAX_PACKETS), dtype=np.float32)
    iats = np.zeros_like(lengths)
    directions = np.zeros((n, MAX_PACKETS), dtype=np.int8)
    mask = np.zeros((n, MAX_PACKETS), dtype=np.uint8)
    rows: list[dict[str, object] | None] = [None] * n
    pkl_audit: list[dict[str, object]] = []
    for pkl_path in sorted((PROJECT_ROOT / "data" / "flows").glob("*.pkl")):
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        class_name = str(payload["class_name"])
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in index:
                continue
            matched += 1
            offset = index[uid]
            packet_count = len(packets)
            timestamps = np.asarray([float(packet[0]) for packet in packets], dtype=np.float64)
            frame_lengths = np.asarray([int(packet[1]) for packet in packets], dtype=np.float64)
            packet_directions = np.asarray([int(packet[3]) for packet in packets], dtype=np.int8)
            deltas = np.zeros(packet_count, dtype=np.float64)
            if packet_count > 1:
                deltas[1:] = np.maximum(0.0, np.diff(timestamps))
            used = min(packet_count, MAX_PACKETS)
            lengths[offset, :used] = frame_lengths[:used]
            iats[offset, :used] = deltas[:used]
            directions[offset, :used] = packet_directions[:used]
            mask[offset, :used] = 1
            nonzero_direction = packet_directions[packet_directions != 0]
            direction_changes = int(np.sum(nonzero_direction[1:] != nonzero_direction[:-1])) if len(nonzero_direction) > 1 else 0
            forward = int(np.sum(packet_directions > 0))
            reverse = int(np.sum(packet_directions < 0))
            duration = float(max(0.0, timestamps[-1] - timestamps[0])) if packet_count else 0.0
            # Stage-0 tuples preserve captured raw bytes but not a separately
            # audited transport-payload length.  Keep it unavailable rather
            # than infer padding or protocol offsets from encrypted bytes.
            rows[offset] = {
                "dataset": "ustc",
                "flow_uid": uid,
                "class_name": class_name,
                "source_file": str(payload["source_file"]),
                "domain_state": "Malware" if uid.startswith("Malware__") else "Benign",
                "official_category": class_name,
                "packet_count": packet_count,
                "total_bytes": int(frame_lengths.sum()),
                "payload_bytes": "",
                "duration_seconds": duration,
                "packet_length_mean": float(frame_lengths.mean()) if packet_count else 0.0,
                "packet_length_std": float(frame_lengths.std()) if packet_count else 0.0,
                "iat_mean_seconds": float(deltas[1:].mean()) if packet_count > 1 else 0.0,
                "iat_std_seconds": float(deltas[1:].std()) if packet_count > 1 else 0.0,
                "forward_packets": forward,
                "reverse_packets": reverse,
                "forward_reverse_ratio": (forward + 1.0) / (reverse + 1.0),
                "direction_changes": direction_changes,
                "padding_ratio_packet_slots": max(0, MAX_PACKETS - used) / MAX_PACKETS,
                "truncated_after_packet_8": packet_count > MAX_PACKETS,
                "encryption_regime": "undetermined",
                "protocol_tokens": "",
            }
        pkl_audit.append({
            "pkl_path": str(pkl_path.resolve()), "pkl_sha256": sha256_file(pkl_path),
            "class_name": class_name, "flows_in_pkl": len(payload["packets"]), "matched_frozen_flows": matched,
        })
        del payload
        gc.collect()
        print(json.dumps({"event": "pkl_complete", "file": pkl_path.name, "matched": matched}), flush=True)
    missing = [ordered[i] for i, row in enumerate(rows) if row is None]
    if missing:
        write_json(output / "failure.json", {"status": "FAIL", "missing_count": len(missing), "missing_preview": missing[:100]})
        raise RuntimeError(f"failed to recover {len(missing)} aligned USTC flows")
    write_csv(output / "flow_statistics.csv", [row for row in rows if row is not None])
    write_csv(output / "pkl_audit.csv", pkl_audit)
    np.save(output / "flow_uids.npy", np.asarray(ordered, dtype="U80"), allow_pickle=False)
    np.save(output / "packet_lengths.npy", lengths, allow_pickle=False)
    np.save(output / "packet_iat_seconds.npy", iats, allow_pickle=False)
    np.save(output / "packet_directions.npy", directions, allow_pickle=False)
    np.save(output / "packet_mask.npy", mask, allow_pickle=False)
    write_json(output / "cache_audit.json", {
        "status": "PASS", "target_flows": n, "recovered_flows": n,
        "known_test_or_unknown_test_values_used_for_selection": False,
        "source_alignment_manifest": str(alignment_path.resolve()),
        "source_alignment_sha256": sha256_file(alignment_path),
        "payload_bytes_status": "NOT_AVAILABLE_AS_SEPARATELY_AUDITED_STAGE0_FIELD",
        "sequence_packets": MAX_PACKETS,
        "flow_statistics_sha256": sha256_file(output / "flow_statistics.csv"),
    })


if __name__ == "__main__":
    main()
