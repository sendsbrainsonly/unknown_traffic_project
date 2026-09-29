#!/usr/bin/env python3
"""Assemble a verified 23,449-flow cache from disjoint historical role caches."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DEST = ROOT / "input_caches" / "full"
TF_NAMES = ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(path: Path) -> list[str]:
    return np.load(path, allow_pickle=False).astype(str).tolist()


def main() -> None:
    if DEST.exists():
        audit_path = DEST / "cache_audit.json"
        if not audit_path.is_file():
            raise RuntimeError(f"existing full cache has no audit; preserved for inspection: {DEST}")
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS" or existing.get("flows") != 23449:
            raise RuntimeError(f"existing full cache audit is not valid: {existing}")
        print(json.dumps({"status": "PASS_REUSED_STAGE44_CACHE", **existing}, sort_keys=True))
        return
    with (ROOT / "vnat_service_manifest.csv").open(newline="", encoding="utf-8") as handle:
        target = sorted(row["flow_uid"] for row in csv.DictReader(handle))
    if len(target) != 23449 or len(set(target)) != 23449:
        raise RuntimeError("service manifest is not the frozen 23,449-flow pool")
    position = {uid: index for index, uid in enumerate(target)}

    tf_sources = [
        PROJECT / "stage31_four_dataset_three_view_equal/input_caches/vnat/medium_seed2025/tf_fig",
        PROJECT / "stage38_three_view_des_open_set_transfer/vnat/input_caches/unknown_test",
        ROOT / "input_caches/missing_known_test",
    ]
    mfr_sources = [
        (PROJECT / "stage31_four_dataset_three_view_equal/input_caches/vnat/medium_seed2025/yatc_mfr_attempt3/train_flow_ids.npy",
         PROJECT / "stage31_four_dataset_three_view_equal/input_caches/vnat/medium_seed2025/yatc_mfr_attempt3/train_mfr.npy"),
        (PROJECT / "stage31_four_dataset_three_view_equal/input_caches/vnat/medium_seed2025/yatc_mfr_attempt3/validation_flow_ids.npy",
         PROJECT / "stage31_four_dataset_three_view_equal/input_caches/vnat/medium_seed2025/yatc_mfr_attempt3/validation_mfr.npy"),
        (PROJECT / "stage38_three_view_des_open_set_transfer/vnat/input_caches/unknown_test/flow_ids.npy",
         PROJECT / "stage38_three_view_des_open_set_transfer/vnat/input_caches/unknown_test/mfr.npy"),
        (ROOT / "input_caches/missing_known_test/flow_ids.npy",
         ROOT / "input_caches/missing_known_test/mfr.npy"),
    ]
    for source in tf_sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
    for id_path, value_path in mfr_sources:
        if not id_path.is_file() or not value_path.is_file():
            raise FileNotFoundError(f"{id_path} / {value_path}")

    DEST.mkdir(parents=True)
    np.save(DEST / "flow_ids.npy", np.asarray(target, dtype="U80"), allow_pickle=False)
    first = tf_sources[0]
    outputs = {}
    for name in TF_NAMES:
        source = np.load(first / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        outputs[name] = np.lib.format.open_memmap(
            DEST / f"{name}.npy", mode="w+", dtype=source.dtype,
            shape=(len(target),) + source.shape[1:],
        )
    tf_seen: set[str] = set()
    for source in tf_sources:
        source_ids = ids(source / "flow_ids.npy")
        if len(source_ids) != len(set(source_ids)):
            raise RuntimeError(f"duplicate IDs in {source}")
        overlap = tf_seen.intersection(source_ids)
        if overlap:
            raise RuntimeError(f"TF source roles overlap: {len(overlap)}")
        tf_seen.update(source_ids)
        destination = np.asarray([position[uid] for uid in source_ids], dtype=np.int64)
        for name in TF_NAMES:
            values = np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if len(values) != len(source_ids):
                raise RuntimeError(f"TF length mismatch: {source}/{name}")
            outputs[name][destination] = values
    if tf_seen != set(target):
        raise RuntimeError(f"TF coverage mismatch: missing={len(set(target)-tf_seen)} extra={len(tf_seen-set(target))}")
    for output in outputs.values():
        output.flush()

    mfr = np.lib.format.open_memmap(DEST / "mfr.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(target), 40, 40))
    mfr_seen: set[str] = set()
    for id_path, value_path in mfr_sources:
        source_ids = ids(id_path)
        overlap = mfr_seen.intersection(source_ids)
        if overlap:
            raise RuntimeError(f"MFR source roles overlap: {len(overlap)}")
        mfr_seen.update(source_ids)
        values = np.load(value_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(source_ids), 40, 40):
            raise RuntimeError(f"MFR shape mismatch: {value_path}/{values.shape}")
        destination = np.asarray([position[uid] for uid in source_ids], dtype=np.int64)
        mfr[destination] = values
    if mfr_seen != set(target):
        raise RuntimeError(f"MFR coverage mismatch: missing={len(set(target)-mfr_seen)} extra={len(mfr_seen-set(target))}")
    mfr.flush()

    audit = {
        "status": "PASS",
        "flows": len(target),
        "tf_coverage": len(tf_seen),
        "mfr_coverage": len(mfr_seen),
        "duplicate_ids": 0,
        "source_roles_disjoint": True,
        "unknown_or_test_used_for_training": 0,
        "service_manifest_sha256": sha256(ROOT / "vnat_service_manifest.csv"),
        "flow_ids_sha256": sha256(DEST / "flow_ids.npy"),
        "source_hashes": {
            str(path.relative_to(PROJECT)): sha256(path)
            for path in [
                *(source / "flow_ids.npy" for source in tf_sources),
                *(path for pair in mfr_sources for path in pair),
            ]
        },
    }
    (DEST / "cache_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
