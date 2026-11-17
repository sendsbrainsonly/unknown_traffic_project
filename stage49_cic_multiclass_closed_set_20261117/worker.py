#!/usr/bin/env python3
"""Isolated adapter to the unchanged Stage34/31 three-view implementation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import protocol

ROOT = protocol.ROOT
PROJECT = protocol.PROJECT
CIC = ROOT / "cicids2017"
RUN = CIC / "runs/ustc/A-2"


def frozen_head_gate():
    hashes = {}
    for branch in ("trafficformer", "graph", "yatc"):
        folder = RUN / branch
        meta = json.loads((folder / "metrics.json").read_text())
        path = folder / "model_best.pt"
        if not (folder / "SUCCESS").is_file() or protocol.sha(path) != meta["checkpoint_sha256"]:
            raise RuntimeError(f"branch not successfully frozen: {branch}")
        hashes[str(path.relative_to(ROOT))] = meta["checkpoint_sha256"]
    head = RUN / "T0_equal"
    selected = json.loads((head / "known_validation_metrics.json").read_text())
    for name, expected in selected["checkpoint_hashes"].items():
        if protocol.sha(head / name) != expected:
            raise RuntimeError(f"head checkpoint drift: {name}")
        hashes[str((head / name).relative_to(ROOT))] = expected
    return hashes


def configure():
    protocol.verified_rows()
    source = PROJECT / "stage34_ustc_cic_closed_set"
    sys.path.insert(0, str(source))
    import cic_runner as runner

    def protocol_rows(dataset, key):
        if (dataset, key) != ("ustc", "A-2"):
            raise RuntimeError("unexpected compatibility key")
        chosen = [(row["flow_id"], row["label"],
                   "known_train" if row["split"] == "train" else "known_validation")
                  for row in protocol.verified_rows() if row["split"] in ("train", "validation")]
        return chosen, list(protocol.CLASSES), protocol.sha(protocol.MANIFEST)

    def alias():
        CIC.mkdir(parents=True, exist_ok=True)
        path = CIC / "COMPATIBILITY_ALIAS.json"
        value = {"actual_dataset": "CIC-IDS-2017", "actual_task": "Stage49 eleven-class closed-set",
                 "actual_classes": protocol.CLASSES, "seed": protocol.SEED,
                 "sample_manifest": str(protocol.MANIFEST), "sample_manifest_sha256": protocol.sha(protocol.MANIFEST),
                 "internal_path_alias": "ustc/A-2", "meaning": "inherited CLI/path compatibility only"}
        text = json.dumps(value, indent=2) + "\n"
        if path.exists() and path.read_text() != text:
            raise RuntimeError("compatibility provenance changed")
        if not path.exists():
            with path.open("x") as f:
                f.write(text)

    runner.ROOT = ROOT
    runner.CIC = CIC
    runner.MANIFEST = protocol.MANIFEST
    runner.sample_rows = protocol.verified_rows
    runner.protocol_rows = protocol_rows
    runner.record_alias = alias
    runner.protocol.CLASSES = protocol.CLASSES
    runner.inputs.ROOT = ROOT
    runner.inputs.CIC = CIC
    runner.inputs.MANIFEST = protocol.MANIFEST
    runner.inputs.frozen_head_gate = frozen_head_gate
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "inputs", "branch", "fusion", "evaluate"))
    parser.add_argument("--phase", choices=("trainval", "test"))
    parser.add_argument("--branch", choices=("trafficformer", "graph", "yatc"))
    args = parser.parse_args()
    runner = configure()
    if args.action == "preflight":
        runner.patch_modules()
        runner.record_alias()
        sources = protocol.verify_sources()
        tf = runner.tf_graph.source
        yatc = runner.yatc.source
        for path in (tf.PRETRAINED_MODEL, yatc.WEIGHT):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        tf_config = tf.read_json(tf.CONFIG)
        if protocol.sha(tf.PRETRAINED_MODEL) != tf_config["pretrained_model_sha256"]:
            raise RuntimeError("official TrafficFormer pretrained hash mismatch")
        if protocol.sha(yatc.WEIGHT) != yatc.WEIGHT_HASH:
            raise RuntimeError("official YaTC pretrained hash mismatch")
        print(json.dumps({"status": "PASS", "classes": len(protocol.CLASSES),
                          "source_hashes": len(sources), "official_pretraining_hashes": "PASS",
                          "historical_training_modules_importable": True}), flush=True)
    elif args.action == "inputs":
        if args.phase is None:
            parser.error("--phase required")
        sys.argv = [str(runner.inputs.__file__), "--phase", args.phase]
        runner.inputs.main()
    elif args.action == "branch":
        if args.branch is None:
            parser.error("--branch required")
        runner.run_branch(args.branch)
    elif args.action == "fusion":
        runner.run_fusion()
    else:
        before = frozen_head_gate()
        protocol.write_json(ROOT / "checkpoint_hashes_before_test.json", before)
        runner.evaluate()
        output = RUN / "known_test_evaluation"
        # Preserve the inherited implementation's historical description before
        # writing this task's true provenance. Numeric results are untouched.
        inherited = json.loads((output / "results.json").read_text())
        protocol.write_json(output / "inherited_evaluator_metadata.json", inherited)
        inherited.update(protocol="Stage49-eleven-class-group-disjoint", seed=protocol.SEED,
                         caveat="CIC development task; five-minute groups disjoint, class-day and endpoint confounding remains")
        protocol.write_json(output / "results.json", inherited)
        after = frozen_head_gate()
        if before != after:
            raise RuntimeError("checkpoint changed during Test extraction/evaluation")
        protocol.write_json(ROOT / "checkpoint_hashes_after_test.json", after)


if __name__ == "__main__":
    main()
