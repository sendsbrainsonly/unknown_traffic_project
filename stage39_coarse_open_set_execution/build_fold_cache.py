"""Subset Stage31 Known Train/Val caches by a frozen Stage39 service role manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from freeze_protocols import PROJECT, ROOT, sha, write_json

TF_ARRAYS = ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask")


def run(dataset: str, slug: str):
    fold = ROOT / "settings" / dataset / slug
    protocol = json.loads((fold / "protocol.json").read_text())
    if protocol["status"] != "FROZEN_PRETRAIN" or protocol["checkpoint_reuse"]:
        raise RuntimeError("not a new-training frozen protocol")
    role_path = fold / "role_manifest.csv"
    if sha(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("role manifest hash mismatch")
    rows = list(csv.DictReader(role_path.open(newline="", encoding="utf-8")))
    by_role = {role: sorted(r["sample_id"] for r in rows if r["role"] == role)
               for role in ("known_train", "known_validation")}
    cache_dataset = "vnat" if dataset == "VNAT" else dataset
    base = PROJECT / "stage31_four_dataset_three_view_equal/input_caches" / cache_dataset
    src = base / protocol["protocol_id"] if dataset == "VNAT" else base
    dest = fold / "input_caches" / cache_dataset
    dest = dest / protocol["protocol_id"] if dataset == "VNAT" else dest
    tf_src, tf_dest = src / "tf_fig", dest / "tf_fig"
    mfr_name = "yatc_mfr_attempt3" if dataset == "VNAT" else "yatc_mfr"
    mfr_src, mfr_dest = src / mfr_name, dest / mfr_name
    for folder in (tf_dest, mfr_dest):
        folder.mkdir(parents=True, exist_ok=False)
    tf_audit = json.loads((tf_src / "cache_audit.json").read_text())
    mfr_audit = json.loads((mfr_src / "cache_audit.json").read_text())
    if tf_audit["status"] != "PASS" or mfr_audit["status"] != "PASS":
        raise RuntimeError("source cache not verified")
    tf_ids = np.load(tf_src / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    tf_index = {uid: i for i, uid in enumerate(tf_ids)}
    all_ids = sorted(by_role["known_train"] + by_role["known_validation"])
    if len(tf_index) != len(tf_ids) or any(uid not in tf_index for uid in all_ids):
        raise RuntimeError("new Known ID missing from original TF/FIG cache")
    pos = np.asarray([tf_index[uid] for uid in all_ids], dtype=np.int64)
    np.save(tf_dest / "flow_ids.npy", np.asarray(all_ids, dtype="U80"), allow_pickle=False)
    for name in TF_ARRAYS:
        original = np.load(tf_src / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        subset = np.asarray(original[pos])
        np.save(tf_dest / f"{name}.npy", subset, allow_pickle=False)
        if not np.array_equal(np.load(tf_dest / f"{name}.npy", mmap_mode="r", allow_pickle=False), subset):
            raise RuntimeError(f"cache write parity failed for {name}")
    for role, ids in by_role.items():
        prefix = ({"known_train": "train", "known_validation": "validation"}[role]
                  if dataset == "VNAT" else role)
        source_ids = np.load(mfr_src / f"{prefix}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        source_index = {uid: i for i, uid in enumerate(source_ids)}
        if len(source_index) != len(source_ids) or any(uid not in source_index for uid in ids):
            raise RuntimeError(f"new Known ID missing from original MFR cache: {role}")
        source_images = np.load(mfr_src / f"{prefix}_mfr.npy", mmap_mode="r", allow_pickle=False)
        position = np.asarray([source_index[uid] for uid in ids], dtype=np.int64)
        np.save(mfr_dest / f"{prefix}_flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
        images = np.asarray(source_images[position])
        np.save(mfr_dest / f"{prefix}_mfr.npy", images, allow_pickle=False)
        if not np.array_equal(np.load(mfr_dest / f"{prefix}_mfr.npy", mmap_mode="r", allow_pickle=False), images):
            raise RuntimeError(f"MFR subset parity failed: {role}")
    info = {"status": "PASS", "dataset": dataset, "fold": slug,
            "role_manifest_sha256": sha(role_path),
            "source_cache_audit_sha256": {"tf_fig": sha(tf_src / "cache_audit.json"),
                                          "yatc_mfr": sha(mfr_src / "cache_audit.json")},
            "known_train": len(by_role["known_train"]),
            "known_validation": len(by_role["known_validation"]),
            "original_known_train_val_cache_only": True,
            "unknown_feature_values_loaded": 0, "test_feature_values_loaded": 0}
    write_json(tf_dest / "cache_audit.json", info)
    write_json(mfr_dest / "cache_audit.json", info)
    write_json(fold / "cache_subset_verification.json", {"status": "PASS", **info,
               "unknown_training_samples": 0, "unknown_validation_samples": 0})
    print(json.dumps({"status": "PASS", "dataset": dataset, "fold": slug,
                      "train": len(by_role["known_train"]),
                      "validation": len(by_role["known_validation"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "VNAT"), required=True)
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()
    run(args.dataset, args.fold)
