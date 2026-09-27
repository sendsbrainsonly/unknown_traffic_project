#!/usr/bin/env python3
"""Freeze a 1:1 benign/malicious CIC subset and copy exact cached packet views.

The original Stage34 v2 rows and group assignments are immutable inputs.  This
script never parses Test packets before the original gated Test cache exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "stage34_ustc_cic_closed_set"
SOURCE_MANIFEST = SOURCE / "cicids2017_three_class_manifest_v2.csv"
MANIFEST = ROOT / "balanced_manifest.csv"
AUDIT = ROOT / "balanced_manifest_audit.json"
CLASSES = ("BENIGN", "DoS Slowhttptest", "PortScan")
ROLES = ("train", "validation", "test")
SEED = 2022


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def verified_rows() -> list[dict[str, str]]:
    source_audit = json.loads((SOURCE / "cicids2017_protocol_v2_audit.json").read_text())
    if sha(SOURCE_MANIFEST) != source_audit["v2_manifest_sha256"]:
        raise RuntimeError("Stage34 source manifest hash changed")
    fields, original = rows(SOURCE_MANIFEST)
    _, balanced = rows(MANIFEST)
    audit = json.loads(AUDIT.read_text())
    if sha(MANIFEST) != audit["balanced_manifest_sha256"]:
        raise RuntimeError("balanced manifest hash changed")
    if fields != audit["columns"]:
        raise RuntimeError("manifest columns changed")
    by_id = {r["flow_id"]: r for r in original}
    if len(by_id) != len(original) or len(balanced) != len({r["flow_id"] for r in balanced}):
        raise RuntimeError("duplicate flow ID")
    if any(by_id.get(r["flow_id"]) != r for r in balanced):
        raise RuntimeError("balanced row differs from Stage34 source")
    actual = Counter((r["label"], r["split"]) for r in balanced)
    if {f"{label}|{role}": count for (label, role), count in actual.items()} != audit["balanced_counts"]:
        raise RuntimeError("balanced counts changed")
    for role in ROLES:
        if actual["BENIGN", role] != sum(actual[label, role] for label in CLASSES[1:]):
            raise RuntimeError(f"benign/malicious mismatch in {role}")
    group_role: dict[str, str] = {}
    for row in original:
        gid, role = row["group_id"], row["split"]
        if gid in group_role and group_role[gid] != role:
            raise RuntimeError(f"original group split overlap: {gid}")
        group_role[gid] = role
    return balanced


def make_manifest() -> None:
    source_audit = json.loads((SOURCE / "cicids2017_protocol_v2_audit.json").read_text())
    source_hash = sha(SOURCE_MANIFEST)
    if source_hash != source_audit["v2_manifest_sha256"]:
        raise RuntimeError("Stage34 source manifest hash changed")
    fields, original = rows(SOURCE_MANIFEST)
    if MANIFEST.exists() or AUDIT.exists():
        raise FileExistsError("balanced manifest already frozen")
    if len(original) != len({r["flow_id"] for r in original}):
        raise RuntimeError("duplicate source flow ID")
    if set(r["label"] for r in original) != set(CLASSES):
        raise RuntimeError("unexpected source class set")
    if set(r["split"] for r in original) != set(ROLES):
        raise RuntimeError("unexpected source split")
    selected: list[dict[str, str]] = []
    counts = Counter((r["label"], r["split"]) for r in original)
    for role in ROLES:
        attack = [r for r in original if r["split"] == role and r["label"] != "BENIGN"]
        benign = [r for r in original if r["split"] == role and r["label"] == "BENIGN"]
        if not attack or len(benign) < len(attack):
            raise RuntimeError(f"cannot balance {role}")
        benign.sort(key=lambda r: (hashlib.sha256(
            f"{SEED}|balanced_benign|{role}|{r['flow_id']}".encode()).hexdigest(), r["flow_id"]))
        selected.extend(attack)
        selected.extend(benign[:len(attack)])
    selected.sort(key=lambda r: r["flow_id"])
    with MANIFEST.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    after = Counter((r["label"], r["split"]) for r in selected)
    payload = {
        "status": "PASS", "seed": SEED,
        "selection": "SHA256 rank of BENIGN flow_id within each existing split; retain all attack rows",
        "selection_locked_before_original_test_result": True,
        "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": source_hash,
        "balanced_manifest_sha256": sha(MANIFEST), "columns": fields,
        "source_counts": {f"{label}|{role}": counts[label, role] for label in CLASSES for role in ROLES},
        "balanced_counts": {f"{label}|{role}": after[label, role] for label in CLASSES for role in ROLES},
        "group_rule": "preserve original source_pcap|five-minute group assignment; subset only",
        "unknown_samples_used": 0, "test_feature_values_read": 0,
    }
    AUDIT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    verified_rows()
    print(json.dumps({"status": "PASS", "balanced_counts": payload["balanced_counts"]}))


def copy_cache(phase: str) -> None:
    balanced = verified_rows()
    old = SOURCE / "cicids2017/input_caches/ustc/A-2"
    new = ROOT / "cicids2017/input_caches/ustc/A-2"
    role_specs = (("train", "known_train"), ("validation", "known_validation")) if phase == "trainval" else (("test", "known_test"),)
    old_tf = old / ("tf_fig" if phase == "trainval" else "tf_fig_test")
    old_mfr = old / ("yatc_mfr" if phase == "trainval" else "yatc_mfr_test")
    new_tf = new / old_tf.name
    new_mfr = new / old_mfr.name
    if new_tf.exists() or new_mfr.exists():
        raise FileExistsError(f"balanced {phase} cache already exists")
    for path in (old_tf, old_mfr):
        audit = json.loads((path / "cache_audit.json").read_text())
        if audit["status"] != "PASS" or audit["sample_manifest_sha256"] != sha(SOURCE_MANIFEST):
            raise RuntimeError(f"Stage34 source cache invalid: {path}")
    chosen = sorted(r["flow_id"] for r in balanced if r["split"] in {r for r, _ in role_specs})
    old_ids = np.load(old_tf / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if len(old_ids) != len(set(old_ids)) or not set(chosen).issubset(set(old_ids)):
        raise RuntimeError("TF/FIG source IDs do not cover balanced subset")
    index = {uid: i for i, uid in enumerate(old_ids)}
    positions = np.asarray([index[uid] for uid in chosen], dtype=np.int64)
    new_tf.mkdir(parents=True)
    np.save(new_tf / "flow_ids.npy", np.asarray(chosen, dtype="U80"), allow_pickle=False)
    for name in ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask"):
        src = np.load(old_tf / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        out = np.lib.format.open_memmap(new_tf / f"{name}.npy", mode="w+", dtype=src.dtype,
                                        shape=(len(chosen), *src.shape[1:]))
        for start in range(0, len(chosen), 512):
            end = min(start + 512, len(chosen))
            out[start:end] = src[positions[start:end]]
        out.flush()
    new_mfr.mkdir(parents=True)
    for role, prefix in role_specs:
        ids = sorted(r["flow_id"] for r in balanced if r["split"] == role)
        old_role_ids = np.load(old_mfr / f"{prefix}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if len(old_role_ids) != len(set(old_role_ids)) or not set(ids).issubset(set(old_role_ids)):
            raise RuntimeError(f"MFR source IDs do not cover {role}")
        old_pos = {uid: i for i, uid in enumerate(old_role_ids)}
        src = np.load(old_mfr / f"{prefix}_mfr.npy", mmap_mode="r", allow_pickle=False)
        out = np.lib.format.open_memmap(new_mfr / f"{prefix}_mfr.npy", mode="w+",
                                        dtype=src.dtype, shape=(len(ids), *src.shape[1:]))
        for start in range(0, len(ids), 512):
            batch = ids[start:start + 512]
            out[start:start + len(batch)] = src[[old_pos[uid] for uid in batch]]
        out.flush()
        np.save(new_mfr / f"{prefix}_flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    cache_audit = {
        "status": "PASS", "dataset": "cicids2017", "phase": phase,
        "sample_manifest_sha256": sha(MANIFEST), "flows": len(chosen),
        "source_tf_fig_cache_audit_sha256": sha(old_tf / "cache_audit.json"),
        "source_yatc_cache_audit_sha256": sha(old_mfr / "cache_audit.json"),
        "source_cache": str(old), "copy_method": "exact flow_id indexed subset; packet values unchanged",
        "known_test_features_transformed": len(chosen) if phase == "test" else 0,
        "unknown_samples_used": 0,
    }
    for target in (new_tf, new_mfr):
        (target / "cache_audit.json").write_text(json.dumps(cache_audit, indent=2) + "\n")
    if np.load(new_tf / "flow_ids.npy", allow_pickle=False).astype(str).tolist() != chosen:
        raise RuntimeError("balanced TF/FIG cache readback failed")
    print(json.dumps({"status": "PASS", "phase": phase, "flows": len(chosen)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("manifest", "cache"))
    parser.add_argument("--phase", choices=("trainval", "test"))
    args = parser.parse_args()
    if args.action == "manifest":
        make_manifest()
    elif args.phase:
        copy_cache(args.phase)
    else:
        parser.error("--phase required for cache")


if __name__ == "__main__":
    main()
