#!/usr/bin/env python3
"""Isolated 10% USTC A-2 rerun of the unchanged Stage31 three-view recipe.

The sample is fixed by SHA256 over (seed, frozen role, class, flow ID).  This
module patches only Stage31 path/protocol hooks; architecture and training
functions are imported unchanged.  Test packet values are recovered only after
the three branches and T0 head have been selected on Known Validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent
OLD = PROJECT / "stage31_four_dataset_three_view_equal"
sys.path.insert(0, str(OLD))
import preflight  # noqa: E402
import train_yatc_branch as yatc  # noqa: E402
import train_tf_fig_branch as tf_graph  # noqa: E402
import train_equal_fusion as fusion  # noqa: E402

MANIFEST = ROOT / "ustc_a2_10pct_manifest.csv"
SOURCE = PROJECT / "opendetect_ustc_encoder_audit/outputs/input_alignment_manifest.csv"
CONFIG = PROJECT / "stage3_unknown_utility/outputs/A-2/training_config.json"
SEED = 2022
FRACTION = 0.10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected():
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != len({row["flow_id"] for row in rows}):
        raise RuntimeError("empty or duplicate USTC sample manifest")
    return rows


def protocol_rows(dataset, protocol):
    if (dataset, protocol) != ("ustc", "A-2"):
        raise RuntimeError("unexpected USTC protocol")
    rows = selected()
    trainval = [(r["flow_id"], r["class_name"],
                 "known_train" if r["original_split"] == "train" else "known_validation")
                for r in rows if r["original_split"] in ("train", "val")]
    classes = sorted({r["class_name"] for r in rows})
    for role in ("known_train", "known_validation"):
        if {name for _, name, part in trainval if part == role} != set(classes):
            raise RuntimeError(f"missing class in {role}")
    return trainval, classes, sha(MANIFEST)


def gate():
    run = ROOT / "runs/ustc/A-2"
    for branch in ("trafficformer", "graph", "yatc"):
        sub = run / branch
        meta = json.loads((sub / "metrics.json").read_text())
        if not (sub / "SUCCESS").is_file() or sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"branch not frozen: {branch}")
    head = run / "T0_equal"
    meta = json.loads((head / "known_validation_metrics.json").read_text())
    if not (head / "SUCCESS").is_file():
        raise RuntimeError("T0 head not frozen")
    for name, digest in meta["checkpoint_hashes"].items():
        if sha(head / name) != digest:
            raise RuntimeError(f"T0 checkpoint changed: {name}")
    return meta


def prepare():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    known = set(json.loads(CONFIG.read_text())["known_classes"])
    grouped = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        fields = csv.DictReader(handle)
        header = fields.fieldnames
        for row in fields:
            if row["class_name"] in known and row["original_split"] in ("train", "val", "test"):
                if row["input_valid"] != "True":
                    raise RuntimeError("invalid Known USTC A-2 flow in source")
                grouped[(row["class_name"], row["original_split"])].append(row)
    if set(name for name, _ in grouped) != known or len(grouped) != len(known) * 3:
        raise RuntimeError("incomplete frozen A-2 class/split cross-product")
    sampled = []
    count = []
    for (name, role), rows in sorted(grouped.items()):
        n = max(1, round(FRACTION * len(rows)))
        chosen = sorted(rows, key=lambda r: hashlib.sha256(
            f"{SEED}|{role}|{name}|{r['flow_id']}".encode()).digest())[:n]
        sampled.extend(chosen)
        count.append({"class": name, "split": role, "source": len(rows), "sampled": n})
    ids = [r["flow_id"] for r in sampled]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate flow in stratified sample")
    with MANIFEST.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(sorted(sampled, key=lambda r: r["flow_id"]))
    (ROOT / "ustc_sample_audit.json").write_text(json.dumps({
        "status": "PASS", "seed": SEED, "fraction": FRACTION,
        "source_manifest": str(SOURCE), "source_sha256": sha(SOURCE),
        "a2_config_sha256": sha(CONFIG), "sample_manifest_sha256": sha(MANIFEST),
        "per_class_split": count, "unknown_samples": 0,
        "test_features_read": 0}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "sampled": len(sampled),
                      "train": sum(r["original_split"] == "train" for r in sampled),
                      "val": sum(r["original_split"] == "val" for r in sampled),
                      "test": sum(r["original_split"] == "test" for r in sampled)}), flush=True)


def subset_cache():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    selected_rows = selected()
    for kind in ("tf_fig", "yatc_mfr"):
        src = OLD / "input_caches/ustc/A-2" / kind
        dst = ROOT / "input_caches/ustc/A-2" / kind
        if dst.exists():
            raise FileExistsError(dst)
        dst.mkdir(parents=True)
        source_audit = json.loads((src / "cache_audit.json").read_text())
        if source_audit["status"] != "PASS":
            raise RuntimeError(f"source cache not PASS: {kind}")
        if kind == "tf_fig":
            ids = np.load(src / "flow_ids.npy", mmap_mode="r", allow_pickle=False).astype(str).tolist()
            wanted = sorted(r["flow_id"] for r in selected_rows if r["original_split"] != "test")
            index = {uid: i for i, uid in enumerate(ids)}
            if any(uid not in index for uid in wanted):
                raise RuntimeError("USTC TF cache missing sampled Train/Val ID")
            pos = np.array([index[uid] for uid in wanted], dtype=np.int64)
            np.save(dst / "flow_ids.npy", np.asarray(wanted, dtype="U80"), allow_pickle=False)
            for name in ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask"):
                values = np.load(src / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                np.save(dst / f"{name}.npy", np.asarray(values[pos]), allow_pickle=False)
            role_counts = {"trainval": len(wanted)}
        else:
            role_counts = {}
            for split, role in (("train", "known_train"), ("val", "known_validation")):
                ids = np.load(src / f"{role}_flow_ids.npy", mmap_mode="r", allow_pickle=False).astype(str).tolist()
                wanted = sorted(r["flow_id"] for r in selected_rows if r["original_split"] == split)
                index = {uid: i for i, uid in enumerate(ids)}
                if any(uid not in index for uid in wanted):
                    raise RuntimeError(f"USTC MFR cache missing sampled {role} ID")
                pos = np.array([index[uid] for uid in wanted], dtype=np.int64)
                values = np.load(src / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
                np.save(dst / f"{role}_flow_ids.npy", np.asarray(wanted, dtype="U80"), allow_pickle=False)
                np.save(dst / f"{role}_mfr.npy", np.asarray(values[pos]), allow_pickle=False)
                role_counts[role] = len(wanted)
        (dst / "cache_audit.json").write_text(json.dumps({
            "status": "PASS", "source_cache": str(src),
            "source_cache_audit_sha256": sha(src / "cache_audit.json"),
            "sample_manifest_sha256": sha(MANIFEST), "counts": role_counts,
            "test_feature_values_loaded": 0, "unknown_samples_used": 0}, indent=2) + "\n")
        print(json.dumps({"cache": kind, "status": "PASS", **role_counts}), flush=True)


def patch_training():
    preflight.OUT = ROOT
    yatc.OUT = ROOT
    yatc.protocol_rows = protocol_rows
    yatc.cache_root = lambda dataset, protocol: ROOT / "input_caches/ustc/A-2/yatc_mfr"
    tf_graph.OUT = ROOT
    tf_graph.protocol_rows = protocol_rows
    fusion.OUT = ROOT
    fusion.protocol_rows = protocol_rows


def launch_main(module, arguments):
    old = sys.argv
    sys.argv = [str(module.__file__), *arguments]
    try:
        module.main()
    finally:
        sys.argv = old


def test_inputs():
    gate()
    import build_ustc_tf_fig as tf_builder
    import build_ustc_mfr as mfr_builder
    for builder in (tf_builder, mfr_builder):
        builder.OUT = ROOT
        builder.ALIGN = MANIFEST
        builder.ensure_all_heads_frozen = gate
        launch_main(builder, ["--phase", "test"])


def test_evaluation():
    gate()
    import evaluate_known_test as evaluator
    evaluator.OUT = ROOT
    evaluator.protocol_rows = protocol_rows
    evaluator.ensure_all_heads_frozen = gate
    def sampled_test_rows(dataset, protocol, classes):
        rows = sorted((r["flow_id"], r["class_name"]) for r in selected()
                      if r["original_split"] == "test" and r["class_name"] in classes)
        if {name for _, name in rows} != set(classes):
            raise RuntimeError("sampled Test class missing")
        return rows, sha(MANIFEST)
    evaluator.test_rows = sampled_test_rows
    evaluator.test_cache = lambda dataset, protocol: (
        ROOT / "input_caches/ustc/A-2/tf_fig_test",
        ROOT / "input_caches/ustc/A-2/yatc_mfr_test")
    evaluator.cache_root = yatc.cache_root
    launch_main(evaluator, ["--dataset", "ustc", "--protocol", "A-2"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "subset-cache", "branch", "fusion", "test-inputs", "test-evaluate"))
    parser.add_argument("--branch", choices=("trafficformer", "graph", "yatc"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "subset-cache":
        subset_cache()
    elif args.action == "branch":
        if not args.branch:
            parser.error("--branch is required")
        patch_training()
        if args.branch == "yatc":
            launch_main(yatc, ["--dataset", "ustc", "--protocol", "A-2"])
        else:
            launch_main(tf_graph, ["--dataset", "ustc", "--protocol", "A-2", "--branch", args.branch])
    elif args.action == "fusion":
        patch_training()
        launch_main(fusion, ["--dataset", "ustc", "--protocol", "A-2"])
    elif args.action == "test-inputs":
        test_inputs()
    elif args.action == "test-evaluate":
        test_evaluation()


if __name__ == "__main__":
    main()
