"""Assemble frozen service-holdout evaluation inputs from existing ISCX caches."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from freeze_protocols import PROJECT, ROOT, sha, write_json

S31 = PROJECT / "stage31_four_dataset_three_view_equal"
S38 = PROJECT / "stage38_three_view_des_open_set_transfer"
TF_ARRAYS = ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask")
EVAL_ROLES = ("known_test", "unknown_eval", "auxiliary_known_service_unseen_app")


def ids(path: Path):
    result = np.load(path, allow_pickle=False).astype(str).tolist()
    if len(result) != len(set(result)):
        raise RuntimeError(f"duplicate frozen cache ID: {path}")
    return result


def load_audit(path: Path):
    audit = json.loads(path.read_text())
    if audit["status"] != "PASS":
        raise RuntimeError(f"source cache audit not PASS: {path}")
    return audit


def build(dataset: str, slug: str):
    fold = ROOT / "settings" / dataset / slug
    p = json.loads((fold / "protocol.json").read_text())
    if p["status"] != "FROZEN_PRETRAIN" or sha(fold / "role_manifest.csv") != p["role_manifest_sha256"]:
        raise RuntimeError("frozen role manifest invalid")
    calibration_path = fold / "detection/calibration.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["status"] != "PASS" or calibration["unknown_used_for_fitting"] or calibration["test_used_for_fitting"]:
        raise RuntimeError("Known-only calibration must be frozen before eval input read")
    with (fold / "role_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted((r for r in csv.DictReader(stream) if r["role"] in EVAL_ROLES),
                      key=lambda r: r["sample_id"])
    wanted = [r["sample_id"] for r in rows]
    if len(wanted) != sum(p["counts"][role] for role in EVAL_ROLES) or len(wanted) != len(set(wanted)):
        raise RuntimeError("evaluation role count/unique ID mismatch")
    base31, base38 = S31 / "input_caches" / dataset, S38 / "input_caches" / dataset
    tf_train, tf_test = base31 / "tf_fig", base38 / "tf_fig_eval"
    mfr_train, mfr_test = base31 / "yatc_mfr", base38 / "yatc_mfr_eval"
    sources = [tf_train, tf_test, mfr_train, mfr_test]
    for source in sources:
        load_audit(source / "cache_audit.json")

    tf_ids_train, tf_ids_test = ids(tf_train / "flow_ids.npy"), ids(tf_test / "flow_ids.npy")
    tf_lookup = {}
    for origin, values in (("trainval", tf_ids_train), ("test", tf_ids_test)):
        for index, uid in enumerate(values):
            if uid in tf_lookup:
                raise RuntimeError(f"TF source role overlap: {uid}")
            tf_lookup[uid] = (origin, index)
    mfr_lookup = {}
    for origin, path in (("train", mfr_train / "known_train_flow_ids.npy"),
                         ("validation", mfr_train / "known_validation_flow_ids.npy"),
                         ("test", mfr_test / "flow_ids.npy")):
        for index, uid in enumerate(ids(path)):
            if uid in mfr_lookup:
                raise RuntimeError(f"MFR source role overlap: {uid}")
            mfr_lookup[uid] = (origin, index)
    if any(uid not in tf_lookup or uid not in mfr_lookup for uid in wanted):
        absent = [uid for uid in wanted if uid not in tf_lookup or uid not in mfr_lookup]
        raise RuntimeError(f"frozen evaluation input missing: {len(absent)}, sample={absent[:5]}")

    target_root = fold / "input_caches" / dataset
    tf_dest, mfr_dest = target_root / "tf_fig_eval", target_root / "yatc_mfr_eval"
    for directory in (tf_dest, mfr_dest):
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "flow_ids.npy", np.asarray(wanted, dtype="U80"), allow_pickle=False)
    for name in TF_ARRAYS:
        tr = np.load(tf_train / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        te = np.load(tf_test / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if tr.shape[1:] != te.shape[1:] or tr.dtype != te.dtype:
            raise RuntimeError(f"source TF array definition changed: {name}")
        dest = np.lib.format.open_memmap(tf_dest / f"{name}.npy", mode="w+", dtype=tr.dtype,
                                         shape=(len(wanted), *tr.shape[1:]))
        for index, uid in enumerate(wanted):
            origin, position = tf_lookup[uid]
            dest[index] = (tr if origin == "trainval" else te)[position]
        dest.flush(); del dest
    mfr_sources = {"train": np.load(mfr_train / "known_train_mfr.npy", mmap_mode="r", allow_pickle=False),
                   "validation": np.load(mfr_train / "known_validation_mfr.npy", mmap_mode="r", allow_pickle=False),
                   "test": np.load(mfr_test / "mfr.npy", mmap_mode="r", allow_pickle=False)}
    mfr = np.lib.format.open_memmap(mfr_dest / "mfr.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(wanted), 40, 40))
    for index, uid in enumerate(wanted):
        origin, position = mfr_lookup[uid]
        mfr[index] = mfr_sources[origin][position]
    mfr.flush(); del mfr
    if ids(tf_dest / "flow_ids.npy") != wanted or ids(mfr_dest / "flow_ids.npy") != wanted:
        raise RuntimeError("saved evaluation cache ID replay mismatch")
    record = {"status": "PASS", "dataset": dataset, "fold": slug,
        "flows": len(wanted), "role_counts": dict(Counter(r["role"] for r in rows)),
        "role_manifest_sha256": p["role_manifest_sha256"],
        "calibration_sha256": sha(calibration_path),
        "source_cache_audit_sha256": {str(source.relative_to(PROJECT)): sha(source / "cache_audit.json")
                                      for source in sources},
        "model_and_threshold_frozen_before_read": True,
        "unknown_used_for_training_or_calibration": 0,
        "test_used_for_training_or_calibration": 0,
        "eval_input_reconstruction": "subset/union of already frozen Stage31 Train/Val and Stage38 Test caches"}
    write_json(tf_dest / "cache_audit.json", record)
    write_json(mfr_dest / "cache_audit.json", record)
    write_json(fold / "detection/eval_cache_verification.json", record)
    print(json.dumps({"status": "PASS", "dataset": dataset, "fold": slug,
                      "evaluation_flows": len(wanted)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()
    build(args.dataset, args.fold)
