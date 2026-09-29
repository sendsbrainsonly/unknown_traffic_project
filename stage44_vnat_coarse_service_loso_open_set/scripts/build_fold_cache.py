#!/usr/bin/env python3
"""Materialize Known Train/Validation subsets for one frozen Stage44 fold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "input_caches" / "full"
TF_NAMES = ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask")
PROTOCOL_ID = "medium_seed2025"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()
    fold = ROOT / "protocols" / args.fold
    protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
    role_path = fold / "role_manifest.csv"
    if sha256(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("role manifest changed after freeze")
    if json.loads((FULL / "cache_audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("full cache audit is not PASS")
    with role_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_role = {
        role: sorted(row["flow_uid"] for row in rows if row["role"] == role)
        for role in ("known_train", "known_validation")
    }
    if any(len(by_role[role]) != protocol["counts"][role] for role in by_role):
        raise RuntimeError("frozen role count changed")
    full_ids = np.load(FULL / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    index = {uid: position for position, uid in enumerate(full_ids)}
    if len(index) != 23449:
        raise RuntimeError("full cache ID index invalid")

    # Stage31's unchanged trainer only recognizes historical protocol strings.
    # Keep this cache under an isolated Stage44 output namespace; the suffix is
    # only a legacy API alias and does not reuse old sample memberships.
    destination = fold / "stage44_native_runs_attempt3" / "input_caches" / "vnat" / PROTOCOL_ID
    tf_dest = destination / "tf_fig"
    mfr_dest = destination / "yatc_mfr_attempt3"
    tf_dest.mkdir(parents=True, exist_ok=False)
    mfr_dest.mkdir(parents=True, exist_ok=False)
    trainval = sorted(by_role["known_train"] + by_role["known_validation"])
    positions = np.asarray([index[uid] for uid in trainval], dtype=np.int64)
    np.save(tf_dest / "flow_ids.npy", np.asarray(trainval, dtype="U80"), allow_pickle=False)
    for name in TF_NAMES:
        values = np.load(FULL / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        np.save(tf_dest / f"{name}.npy", np.asarray(values[positions]), allow_pickle=False)
    full_mfr = np.load(FULL / "mfr.npy", mmap_mode="r", allow_pickle=False)
    for role, prefix in (("known_train", "train"), ("known_validation", "validation")):
        selected = by_role[role]
        role_positions = np.asarray([index[uid] for uid in selected], dtype=np.int64)
        np.save(mfr_dest / f"{prefix}_flow_ids.npy", np.asarray(selected, dtype="U80"), allow_pickle=False)
        np.save(mfr_dest / f"{prefix}_mfr.npy", np.asarray(full_mfr[role_positions]), allow_pickle=False)
    audit = {
        "status": "PASS",
        "fold": args.fold,
        "known_train": len(by_role["known_train"]),
        "known_validation": len(by_role["known_validation"]),
        "unknown_feature_values_used_for_training": 0,
        "test_feature_values_used_for_training": 0,
        "role_manifest_sha256": sha256(role_path),
        "full_cache_audit_sha256": sha256(FULL / "cache_audit.json"),
    }
    for path in (tf_dest / "cache_audit.json", mfr_dest / "cache_audit.json", fold / "cache_subset_verification.json"):
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
