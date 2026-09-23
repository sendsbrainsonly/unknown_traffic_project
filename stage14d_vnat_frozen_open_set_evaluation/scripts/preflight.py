#!/usr/bin/env python3
"""Freeze Stage 14D inputs before any Test inference."""

from __future__ import annotations

import collections

import numpy as np

from stage14d_common import (
    CACHE_ROOT,
    CHECKPOINT_FREEZE_PATH,
    ROOT,
    checkpoint_records,
    load_manifest_rows,
    load_protocols,
    sha256_file,
    verify_stage14b,
    write_json,
)


def main() -> None:
    if CHECKPOINT_FREEZE_PATH.exists():
        raise RuntimeError(f"refusing to overwrite existing freeze: {CHECKPOINT_FREEZE_PATH}")
    stage14b = verify_stage14b()
    protocols = load_protocols()
    flow_uids = np.load(CACHE_ROOT / "flow_uids.npy", allow_pickle=False)
    images = np.load(CACHE_ROOT / "images.npy", mmap_mode="r", allow_pickle=False)
    cache_uid_set = set(map(str, flow_uids.tolist()))
    if len(cache_uid_set) != 23_449 or images.shape != (23_449, 1024) or images.dtype != np.uint8:
        raise RuntimeError("canonical flow image cache identity/shape mismatch")

    audits = {}
    for protocol_id, protocol in protocols.items():
        rows = load_manifest_rows(protocol_id)
        counts = collections.Counter((row["class_role"], row["split"]) for row in rows)
        expected_roles = {("known", "train"), ("known", "validation"), ("known", "test"), ("unknown", "unknown_test")}
        if set(counts) != expected_roles:
            raise RuntimeError(f"{protocol_id}: unexpected role/split pairs: {counts}")
        if {row["flow_uid"] for row in rows} != cache_uid_set:
            raise RuntimeError(f"{protocol_id}: manifest/cache flow UID set mismatch")
        known = set(map(str, protocol["known_applications"]))
        unknown = set(map(str, protocol["unknown_applications"]))
        if known & unknown:
            raise RuntimeError(f"{protocol_id}: Known/Unknown class overlap")
        train_apps = {row["application"] for row in rows if row["split"] == "train"}
        validation_apps = {row["application"] for row in rows if row["split"] == "validation"}
        unknown_test_apps = {row["application"] for row in rows if row["split"] == "unknown_test"}
        if train_apps != known or validation_apps != known or unknown_test_apps != unknown:
            raise RuntimeError(f"{protocol_id}: frozen class-role membership mismatch")
        audits[protocol_id] = {
            "counts": {f"{role}:{split}": count for (role, split), count in sorted(counts.items())},
            "known_classes": sorted(known),
            "unknown_classes": sorted(unknown),
            "unknown_samples_in_train": 0,
            "unknown_samples_in_validation": 0,
            "test_samples_used_in_support": 0,
            "test_samples_used_in_normalization": 0,
            "test_samples_used_in_threshold_calibration": 0,
        }

    payload = {
        "status": "PASS",
        "stage14b_before": stage14b,
        "checkpoints": checkpoint_records(),
        "protocol_audits": audits,
        "cache": {
            "images_path": str((CACHE_ROOT / "images.npy").resolve()),
            "images_sha256": sha256_file(CACHE_ROOT / "images.npy"),
            "flow_uids_path": str((CACHE_ROOT / "flow_uids.npy").resolve()),
            "flow_uids_sha256": sha256_file(CACHE_ROOT / "flow_uids.npy"),
            "flow_count": 23_449,
        },
        "new_encoder_training": False,
        "test_score_accessed": False,
    }
    write_json(CHECKPOINT_FREEZE_PATH, payload)
    write_json(ROOT / "preflight_verification.json", payload)
    print(f"PASS: froze {len(payload['checkpoints'])} checkpoint hashes and audited {len(audits)} protocols")


if __name__ == "__main__":
    main()

