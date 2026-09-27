#!/usr/bin/env python3
"""Use Stage31/32 training recipes with Stage12 Known-only coarse labels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from audit_iscx_known import CATEGORY, PROJECT, ROOT, S12, S31, sha

PROTOCOL = "medium_seed2022"


def lock(dataset: str) -> dict:
    complete = ROOT / "vnat/stage38a_verification.json"
    if not complete.is_file() or json.loads(complete.read_text())["status"] != "PASS":
        raise RuntimeError("Stage38A VNAT frozen evaluation must complete before Stage38B training")
    audit = json.loads((ROOT / "stage38b_known_label_audit.json").read_text())
    config = json.loads((ROOT / "stage38b_training_config_lock.json").read_text())
    if audit["status"] != "PASS" or not config["status"].startswith("FROZEN_BEFORE"):
        raise RuntimeError("Stage38B Known-only audit/config lock absent")
    for label, digest in config["source_code_sha256"].items():
        if sha(PROJECT / label) != digest:
            raise RuntimeError(f"training source changed after lock: {label}")
    unit = next(item for item in audit["datasets"] if item["dataset"] == dataset)
    for label, digest in unit["source_hashes"].items():
        if sha(PROJECT / label) != digest:
            raise RuntimeError(f"frozen Known input changed: {label}")
    return unit


def coarse_protocol_rows(dataset: str, protocol: str):
    if protocol != PROTOCOL:
        raise ValueError("protocol differs from Stage12 medium_seed2022")
    unit = lock(dataset)
    path = S12 / dataset / "split_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        selected = [(row["flow_id_sha256"], CATEGORY[row["official_category"]], row["role"])
                    for row in csv.DictReader(handle) if row["setting"] == "medium" and
                    row["role"] in ("known_train", "known_validation")]
    if len(selected) != unit["role_counts"]["known_train"] + unit["role_counts"]["known_validation"]:
        raise RuntimeError("Stage38B Known Train/Validation count changed")
    if len({uid for uid, _, _ in selected}) != len(selected):
        raise RuntimeError("duplicate Known flow IDs")
    classes = sorted({label for _, label, _ in selected})
    if classes != unit["known_coarse_classes"]:
        raise RuntimeError("coarse classes changed from pre-registered audit")
    by_role = Counter((role, label) for _, label, role in selected)
    if any(by_role[role, label] == 0 for role in ("known_train", "known_validation")
           for label in classes):
        raise RuntimeError("Known Train/Validation missing a coarse class")
    return selected, classes, sha(path)


def preflight(dataset: str) -> None:
    unit = lock(dataset)
    source = S31 / "input_caches" / dataset
    target = ROOT / "input_caches" / dataset
    target.parent.mkdir(exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir()
    for name in ("tf_fig", "yatc_mfr"):
        (target / name).symlink_to(source / name, target_is_directory=True)
    selected, classes, digest = coarse_protocol_rows(dataset, PROTOCOL)
    with (ROOT / f"{dataset}_train_preflight.json").open("x", encoding="utf-8") as handle:
        json.dump({"status": "PASS", "dataset": dataset, "protocol": PROTOCOL,
                   "classes": classes, "known_train": unit["role_counts"]["known_train"],
                   "known_validation": unit["role_counts"]["known_validation"],
                   "split_sha256": digest, "input_reference": str(source),
                   "unknown_training_samples": 0, "known_test_feature_values_loaded": 0,
                   "unknown_test_feature_values_loaded": 0}, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "dataset": dataset, "classes": classes,
                      "known_flows": len(selected)}), flush=True)


def train(dataset: str, component: str) -> None:
    if not (ROOT / f"{dataset}_train_preflight.json").is_file():
        raise RuntimeError("Stage38B dataset preflight missing")
    lock(dataset)
    sys.path.insert(0, str(S31))
    import preflight as stage31_preflight  # noqa: E402
    import train_yatc_branch as yatc  # noqa: E402
    import train_tf_fig_branch as tf_graph  # noqa: E402
    import train_equal_fusion as fusion  # noqa: E402

    stage31_preflight.OUT = ROOT
    yatc.OUT = ROOT
    tf_graph.OUT = ROOT
    fusion.OUT = ROOT
    yatc.protocol_rows = coarse_protocol_rows
    tf_graph.protocol_rows = coarse_protocol_rows
    fusion.protocol_rows = coarse_protocol_rows
    if component in ("trafficformer", "graph"):
        sys.argv = ["train_tf_fig_branch.py", "--dataset", dataset, "--protocol", PROTOCOL,
                    "--branch", component]
        tf_graph.main()
    elif component == "yatc":
        sys.argv = ["train_yatc_branch.py", "--dataset", dataset, "--protocol", PROTOCOL]
        yatc.main()
    elif component == "fusion":
        sys.argv = ["train_equal_fusion.py", "--dataset", dataset, "--protocol", PROTOCOL]
        fusion.main()
    else:
        raise ValueError(component)
    subdir = "T0_equal" if component == "fusion" else component
    output = ROOT / "runs" / dataset / PROTOCOL / subdir
    if not (output / "SUCCESS").is_file():
        raise RuntimeError(f"missing successful Stage38B {component} marker")
    if component == "fusion":
        checkpoint = json.loads((output / "known_validation_metrics.json").read_text())
        for name, digest in checkpoint["checkpoint_hashes"].items():
            if sha(output / name) != digest:
                raise RuntimeError("fusion checkpoint hash mismatch")
    else:
        metrics = json.loads((output / "metrics.json").read_text())
        if sha(output / "model_best.pt") != metrics["checkpoint_sha256"]:
            raise RuntimeError("branch checkpoint hash mismatch")
    lock(dataset)
    print(json.dumps({"status": "PASS", "dataset": dataset, "component": component}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train"))
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--component", choices=("trafficformer", "graph", "yatc", "fusion"))
    args = parser.parse_args()
    if args.phase == "preflight":
        if args.component is not None:
            raise ValueError("component only valid for training")
        preflight(args.dataset)
    else:
        if args.component is None:
            raise ValueError("training component required")
        train(args.dataset, args.component)
