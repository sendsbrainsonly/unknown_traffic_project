#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from common import PROJECT, STAGE15R, cache_dir, read_csv, sha256_file, write_json


LEGACY_COLUMNS = [
    "packet_count", "total_bytes", "duration_seconds", "packet_length_mean", "packet_length_std",
    "iat_mean_seconds", "iat_std_seconds", "forward_packets", "reverse_packets",
    "forward_reverse_ratio", "direction_changes", "payload_bytes",
]


def scalar(value) -> float:
    return float("nan") if value in (None, "") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ustc", "vnat", "iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    output = cache_dir(dataset)
    uids = np.load(output / "flow_uids.npy", allow_pickle=False).astype(str)
    if dataset in {"ustc", "iscx_vpn", "iscx_tor"}:
        source = STAGE15R / "feature_cache" / dataset / "flow_statistics.csv"
        mapping = {
            row["flow_uid"]: np.asarray([scalar(row.get(name)) for name in LEGACY_COLUMNS], dtype=np.float32)
            for row in read_csv(source)
        }
        source_hash = sha256_file(source)
    else:
        source = PROJECT / "stage14c5_feature_representation_audit" / "raw_feature_cache"
        source_uids = np.load(source / "flow_uids.npy", allow_pickle=False)
        stats = np.load(source / "flow_statistics.npy", mmap_mode="r", allow_pickle=False)
        directions = np.load(source / "packet_directions.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(source / "packet_mask.npy", mmap_mode="r", allow_pickle=False)
        mapping = {}
        for index, uid in enumerate(source_uids):
            packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std, ratio = map(float, stats[index])
            sequence = directions[index][masks[index].astype(bool)]
            sequence = sequence[sequence != 0]
            changes = float(np.sum(sequence[1:] != sequence[:-1])) if len(sequence) > 1 else 0.0
            mapping[str(uid)] = np.asarray([packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std, np.nan, np.nan, ratio, changes, np.nan], dtype=np.float32)
        source_hash = sha256_file(source / "cache_audit.json")
    missing = [uid for uid in uids if uid not in mapping]
    if missing:
        raise RuntimeError(f"legacy S12 source misses {len(missing)} cache UIDs: {missing[:3]}")
    matrix = np.vstack([mapping[uid] for uid in uids])
    target = output / "legacy_s12_features.npy"
    np.save(target, matrix, allow_pickle=False)
    write_json(output / "legacy_s12_audit.json", {
        "status": "PASS", "dataset": dataset, "flows": len(uids), "features": LEGACY_COLUMNS,
        "source": str(source.resolve()), "source_sha256": source_hash,
        "matrix_sha256": sha256_file(target), "missing_values": int(np.isnan(matrix).sum()),
    })
    print({"status": "PASS", "dataset": dataset, "flows": len(uids)})


if __name__ == "__main__":
    main()
