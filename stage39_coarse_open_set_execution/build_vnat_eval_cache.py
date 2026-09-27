"""Assemble one VNAT holdout's evaluation cache after Known-only calibration.

This copies only frozen Stage31/32/38 packet representations. It never parses
raw captures or changes the Stage14B/Stage39 role manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from freeze_protocols import PROJECT, ROOT, sha, write_json

TF_ARRAYS = ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask")
EVAL_ROLES = ("known_test", "unknown_eval", "auxiliary_known_service_unseen_app")
PROTOCOL = "medium_seed2025"


def read_ids(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=False).astype(str).tolist()
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate source flow ID: {path}")
    return values


def audited(path: Path) -> str:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError(f"source cache audit did not pass: {path}")
    return sha(path)


def build(slug: str) -> None:
    dataset = "VNAT"
    fold = ROOT / "settings" / dataset / slug
    protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
    role_path = fold / "role_manifest.csv"
    if protocol["status"] != "FROZEN_PRETRAIN" or protocol["protocol_id"] != PROTOCOL or \
       sha(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("Stage39 frozen VNAT role manifest changed")
    calibration_path = fold / "detection/calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration["status"] != "PASS" or calibration["dataset"] != dataset or \
       calibration["fold"] != slug or calibration["role_manifest_sha256"] != protocol["role_manifest_sha256"] or \
       calibration["unknown_used_for_fitting"] or calibration["test_used_for_fitting"]:
        raise RuntimeError("Known-only calibration must precede VNAT evaluation cache access")
    with role_path.open(newline="", encoding="utf-8") as stream:
        selected = sorted((r for r in csv.DictReader(stream) if r["role"] in EVAL_ROLES),
                          key=lambda r: r["sample_id"])
    wanted = [r["sample_id"] for r in selected]
    if len(wanted) != len(set(wanted)) or len(wanted) != sum(protocol["counts"][role] for role in EVAL_ROLES):
        raise RuntimeError("VNAT evaluation ID count/uniqueness mismatch")

    src31 = PROJECT / "stage31_four_dataset_three_view_equal/input_caches/vnat" / PROTOCOL
    src32 = PROJECT / "stage32_three_dataset_coarse_single_seed/input_caches/vnat" / PROTOCOL
    src38 = PROJECT / "stage38_three_view_des_open_set_transfer/vnat/input_caches/unknown_test"
    tf_sources = {"known_train": src31 / "tf_fig", "known_validation": src31 / "tf_fig",
                  "known_test": src32 / "tf_fig_test", "unknown_test": src38}
    mfr_sources = {"known_train": (src31 / "yatc_mfr_attempt3", "train"),
                   "known_validation": (src31 / "yatc_mfr_attempt3", "validation"),
                   "known_test": (src32 / "yatc_mfr_test", "test"),
                   "unknown_test": (src38, "unknown")}
    sources = sorted(set(tf_sources.values()) | {value[0] for value in mfr_sources.values()})
    audits = {str(path.relative_to(PROJECT)): audited(path / "cache_audit.json") for path in sources}
    tf_index, mfr_index = {}, {}
    for role, path in tf_sources.items():
        tf_index[role] = {uid: i for i, uid in enumerate(read_ids(path / "flow_ids.npy"))}
    for role, (path, prefix) in mfr_sources.items():
        name = "flow_ids.npy" if prefix == "unknown" else f"{prefix}_flow_ids.npy"
        mfr_index[role] = {uid: i for i, uid in enumerate(read_ids(path / name))}
    for row in selected:
        role, uid = row["original_role"], row["sample_id"]
        if role not in tf_index or uid not in tf_index[role] or uid not in mfr_index[role]:
            raise RuntimeError(f"frozen VNAT evaluation ID absent from its original role cache: {role}/{uid}")

    target = fold / "input_caches/vnat"
    tf_dest, mfr_dest = target / "tf_fig_eval", target / "yatc_mfr_eval"
    if tf_dest.exists() or mfr_dest.exists():
        raise FileExistsError("VNAT evaluation cache already exists; preserve it for audit")
    tf_dest.mkdir(parents=True)
    mfr_dest.mkdir(parents=True)
    np.save(tf_dest / "flow_ids.npy", np.asarray(wanted, dtype="U80"), allow_pickle=False)
    np.save(mfr_dest / "flow_ids.npy", np.asarray(wanted, dtype="U80"), allow_pickle=False)
    for name in TF_ARRAYS:
        arrays = {role: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                  for role, path in tf_sources.items()}
        exemplar = arrays["known_train"]
        if any(arr.shape[1:] != exemplar.shape[1:] or arr.dtype != exemplar.dtype for arr in arrays.values()):
            raise RuntimeError(f"VNAT source representation mismatch: {name}")
        dest = np.lib.format.open_memmap(tf_dest / f"{name}.npy", mode="w+", dtype=exemplar.dtype,
                                         shape=(len(wanted), *exemplar.shape[1:]))
        for i, row in enumerate(selected):
            role, uid = row["original_role"], row["sample_id"]
            dest[i] = arrays[role][tf_index[role][uid]]
        dest.flush(); del dest
    images = {}
    for role, (path, prefix) in mfr_sources.items():
        name = "mfr.npy" if prefix == "unknown" else f"{prefix}_mfr.npy"
        images[role] = np.load(path / name, mmap_mode="r", allow_pickle=False)
    exemplar = images["known_train"]
    if any(arr.shape[1:] != exemplar.shape[1:] or arr.dtype != exemplar.dtype for arr in images.values()):
        raise RuntimeError("VNAT MFR representation mismatch")
    mfr = np.lib.format.open_memmap(mfr_dest / "mfr.npy", mode="w+", dtype=exemplar.dtype,
                                    shape=(len(wanted), *exemplar.shape[1:]))
    for i, row in enumerate(selected):
        role, uid = row["original_role"], row["sample_id"]
        mfr[i] = images[role][mfr_index[role][uid]]
    mfr.flush(); del mfr
    if read_ids(tf_dest / "flow_ids.npy") != wanted or read_ids(mfr_dest / "flow_ids.npy") != wanted:
        raise RuntimeError("VNAT saved evaluation ID order mismatch")
    evidence = {"status": "PASS", "dataset": dataset, "fold": slug,
                "flows": len(wanted), "role_counts": dict(Counter(r["role"] for r in selected)),
                "original_role_counts": dict(Counter(r["original_role"] for r in selected)),
                "role_manifest_sha256": protocol["role_manifest_sha256"],
                "calibration_sha256": sha(calibration_path), "source_cache_audit_sha256": audits,
                "model_and_threshold_frozen_before_read": True,
                "unknown_used_for_training_or_calibration": 0,
                "test_used_for_training_or_calibration": 0,
                "eval_input_reconstruction": "frozen Stage31 Known Train/Val, Stage32 Known Test, Stage38 Unknown Test"}
    write_json(tf_dest / "cache_audit.json", evidence)
    write_json(mfr_dest / "cache_audit.json", evidence)
    write_json(fold / "detection/eval_cache_verification.json", evidence)
    print(json.dumps({"status": "PASS", "dataset": dataset, "fold": slug,
                      "evaluation_flows": len(wanted)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    build(parser.parse_args().fold)
