#!/usr/bin/env python3
"""Exact-ID subset of Stage34B cached packet views for CIC holdout pilots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from freeze_protocols import ROOT, PROJECT, digest

SOURCE = PROJECT / "stage34b_cic_balanced_retest/cicids2017/input_caches/ustc/A-2"
ROLES = {"trainval": (("train", "known_train"), ("validation", "known_validation")),
         "eval": (("test", "known_test"),)}


def unit_for(key: str) -> dict:
    frozen = json.loads((ROOT / "protocols.json").read_text())
    unit = next(x for x in frozen["units"] if x.get("key") == key)
    if digest(Path(unit["role_manifest"])) != unit["role_manifest_sha256"]:
        raise RuntimeError("CIC role manifest changed")
    return unit


def subset(source: Path, target: Path, ids: list[str], source_ids_file="flow_ids.npy") -> None:
    original = np.load(source / source_ids_file, allow_pickle=False).astype(str).tolist()
    if len(original) != len(set(original)) or not set(ids).issubset(original):
        raise RuntimeError(f"source cache missing selected IDs: {source}")
    pos = {uid: i for i, uid in enumerate(original)}
    positions = np.asarray([pos[uid] for uid in ids], dtype=np.int64)
    target.mkdir(parents=True)
    np.save(target / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    for name in ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask"):
        original_array = np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        result = np.lib.format.open_memmap(target / f"{name}.npy", mode="w+",
                                            dtype=original_array.dtype,
                                            shape=(len(ids), *original_array.shape[1:]))
        for start in range(0, len(ids), 512):
            part = positions[start:start+512]
            result[start:start+len(part)] = original_array[part]
        result.flush()


def subset_mfr(source: Path, target: Path, role: str, ids: list[str],
               *, combined_eval: bool = False) -> None:
    original_ids = np.load(source / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if len(original_ids) != len(set(original_ids)) or not set(ids).issubset(original_ids):
        raise RuntimeError(f"MFR cache missing {role} IDs")
    pos = {uid: i for i, uid in enumerate(original_ids)}
    original = np.load(source / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
    filename = "mfr.npy" if combined_eval else f"{role}_mfr.npy"
    result = np.lib.format.open_memmap(target / filename, mode="w+", dtype=original.dtype,
                                        shape=(len(ids), *original.shape[1:]))
    for start in range(0, len(ids), 512):
        part = ids[start:start+512]
        result[start:start+len(part)] = original[[pos[x] for x in part]]
    result.flush()
    if not combined_eval:
        np.save(target / f"{role}_flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", choices=("unknown_slowhttptest", "unknown_portscan"), required=True)
    parser.add_argument("--phase", choices=("trainval", "eval"), required=True)
    args = parser.parse_args()
    unit = unit_for(args.key)
    with Path(unit["role_manifest"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = ROOT / args.key / "input_caches/ustc/A-2"
    if args.phase == "trainval":
        names = {"known_train", "known_validation"}
        source_tf, source_mfr = SOURCE / "tf_fig", SOURCE / "yatc_mfr"
        tf_path, mfr_path = out / "tf_fig", out / "yatc_mfr"
    else:
        names = {"known_test", "unknown_test"}
        source_tf, source_mfr = SOURCE / "tf_fig_test", SOURCE / "yatc_mfr_test"
        tf_path, mfr_path = out / "tf_fig_eval", out / "yatc_mfr_eval"
        calibration = ROOT / args.key / "detection/calibration.json"
        if not calibration.exists() or json.loads(calibration.read_text())["status"] != "PASS":
            raise RuntimeError("Known-only calibration required before Test cache read")
    if tf_path.exists() or mfr_path.exists():
        raise FileExistsError("refusing to overwrite existing CIC cache")
    for source in (source_tf, source_mfr):
        if json.loads((source / "cache_audit.json").read_text())["status"] != "PASS":
            raise RuntimeError(f"Stage34B source cache failed: {source}")
    ids = sorted(r["flow_id"] for r in rows if r["role"] in names)
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("empty or duplicate selected CIC IDs")
    subset(source_tf, tf_path, ids)
    mfr_path.mkdir(parents=True)
    if args.phase == "trainval":
        for role in names:
            role_ids = sorted(r["flow_id"] for r in rows if r["role"] == role)
            subset_mfr(source_mfr, mfr_path, role, role_ids)
    else:
        # Stage34B MFR Test cache is already ordered by flow ID over all three classes.
        subset_mfr(source_mfr, mfr_path, "known_test", ids, combined_eval=True)
        np.save(mfr_path / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    if np.load(tf_path / "flow_ids.npy", allow_pickle=False).astype(str).tolist() != ids:
        raise RuntimeError("new CIC TF cache ID replay failed")
    audit = {"status": "PASS", "phase": args.phase, "flows": len(ids),
             "role_manifest_sha256": unit["role_manifest_sha256"],
             "source_tf_audit_sha256": digest(source_tf / "cache_audit.json"),
             "source_mfr_audit_sha256": digest(source_mfr / "cache_audit.json"),
             "copy": "exact flow-ID subset of Stage34B packet view; no feature formula changes",
             "test_feature_values_loaded": len(ids) if args.phase == "eval" else 0,
             "unknown_training_samples": 0}
    for path in (tf_path, mfr_path):
        with (path / "cache_audit.json").open("x", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
            f.write("\n")
    print(json.dumps({"status": "PASS", "key": args.key, "phase": args.phase, "flows": len(ids)}))


if __name__ == "__main__":
    main()
