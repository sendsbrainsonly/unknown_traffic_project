#!/usr/bin/env python3
"""Reuse Stage34/31 training and evaluation code on an isolated balanced pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "stage34_ustc_cic_closed_set"
sys.path.insert(0, str(SOURCE))
import cic_runner as stage34  # noqa: E402
from prepare_balanced import AUDIT, MANIFEST, CLASSES, sha, verified_rows  # noqa: E402

CIC = ROOT / "cicids2017"


def sample_rows():
    return verified_rows()


def protocol_rows(dataset, key):
    if (dataset, key) != ("ustc", "A-2"):
        raise RuntimeError("unexpected Stage31 compatibility key")
    selected = [(r["flow_id"], r["label"],
                 "known_train" if r["split"] == "train" else "known_validation")
                for r in sample_rows() if r["split"] in ("train", "validation")]
    for role in ("known_train", "known_validation"):
        if {label for _, label, item_role in selected if item_role == role} != set(CLASSES):
            raise RuntimeError(f"missing class in {role}")
    return selected, sorted(CLASSES), sha(MANIFEST)


def record_alias():
    CIC.mkdir(parents=True, exist_ok=True)
    path = CIC / "COMPATIBILITY_ALIAS.json"
    payload = json.dumps({
        "actual_dataset": "CIC-IDS-2017",
        "actual_protocol": "Stage34 original five-minute group split, Stage34B 1:1 benign/malicious subset",
        "actual_sample_manifest": str(MANIFEST), "actual_sample_sha256": sha(MANIFEST),
        "internal_Stage31_runner_key": "ustc/A-2", "meaning": "CLI/path compatibility only",
        "reused_recipe": "Stage31 TrafficFormer20e, TAGCN50e, YaTC200e, T0 adapter30e/head30e",
        "unknown_samples_used": 0,
    }, indent=2) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError("balanced compatibility metadata drift")
    else:
        with path.open("x") as handle:
            handle.write(payload)


def frozen_head_gate():
    run = CIC / "runs/ustc/A-2"
    for branch in ("trafficformer", "graph", "yatc"):
        sub = run / branch
        meta = json.loads((sub / "metrics.json").read_text())
        if not (sub / "SUCCESS").is_file() or sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"unfrozen balanced {branch}")
    head = run / "T0_equal"
    selected = json.loads((head / "known_validation_metrics.json").read_text())
    for name, expected in selected["checkpoint_hashes"].items():
        if sha(head / name) != expected:
            raise RuntimeError(f"unfrozen balanced head: {name}")
    return selected


def configure():
    stage34.ROOT = ROOT
    stage34.CIC = CIC
    stage34.MANIFEST = MANIFEST
    stage34.sample_rows = sample_rows
    stage34.protocol_rows = protocol_rows
    stage34.record_alias = record_alias
    stage34.inputs.frozen_head_gate = frozen_head_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("branch", "fusion", "evaluate"))
    parser.add_argument("--branch", choices=("trafficformer", "graph", "yatc"))
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text())
    if audit["status"] != "PASS" or sha(MANIFEST) != audit["balanced_manifest_sha256"]:
        raise RuntimeError("balanced sample manifest not frozen")
    configure()
    if args.action == "branch":
        if not args.branch:
            parser.error("--branch required")
        stage34.run_branch(args.branch)
    elif args.action == "fusion":
        stage34.run_fusion()
    else:
        frozen_head_gate()
        stage34.evaluate()
        report_path = CIC / "runs/ustc/A-2/known_test_evaluation/results.json"
        report = json.loads(report_path.read_text())
        # The unchanged Stage34 evaluator uses the old group-split name.  The
        # sample hash already identifies this run; this audit says so explicitly.
        (report_path.parent / "balanced_protocol_audit.json").write_text(json.dumps({
            "actual_protocol": "Stage34B benign/malicious 1:1 subset of Stage34 v2",
            "balanced_manifest_sha256": sha(MANIFEST),
            "evaluator_group_protocol_field": report["protocol"],
            "test_samples": report["known_test_samples"],
            "unknown_samples_used": 0, "test_parameter_selection": 0,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
