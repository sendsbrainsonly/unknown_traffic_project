#!/usr/bin/env python3
"""Run unchanged Stage31 three-view recipe on a frozen CIC two-class fold."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from freeze_protocols import ROOT, PROJECT, digest

S31 = PROJECT / "stage31_four_dataset_three_view_equal"
sys.path.insert(0, str(S31))
import preflight  # noqa: E402
import train_yatc_branch as yatc  # noqa: E402
import train_tf_fig_branch as tf_graph  # noqa: E402
import train_equal_fusion as fusion  # noqa: E402


def unit_for(key: str) -> dict:
    frozen = json.loads((ROOT / "protocols.json").read_text())
    unit = next(x for x in frozen["units"] if x.get("key") == key)
    if digest(Path(unit["role_manifest"])) != unit["role_manifest_sha256"]:
        raise RuntimeError("CIC frozen role manifest changed")
    return unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", choices=("unknown_slowhttptest", "unknown_portscan"), required=True)
    parser.add_argument("--component", choices=("trafficformer", "graph", "yatc", "fusion"), required=True)
    args = parser.parse_args()
    unit = unit_for(args.key)
    fold = ROOT / args.key
    alias = fold / "COMPATIBILITY_ALIAS.json"
    if not alias.exists():
        with alias.open("x", encoding="utf-8") as f:
            json.dump({"actual_dataset": "CIC-IDS-2017", "actual_protocol": args.key,
                       "known_classes": unit["known_classes"], "unknown_classes": unit["unknown_classes"],
                       "internal_Stage31_key": "ustc/A-2", "meaning": "CLI compatibility only",
                       "role_manifest": unit["role_manifest"],
                       "role_manifest_sha256": unit["role_manifest_sha256"],
                       "training_recipe": "Stage31 TrafficFormer20e TAGCN50e YaTC200e T0 adapters30e head30e",
                       "unknown_training_samples": 0}, f, indent=2)
            f.write("\n")
    else:
        meta = json.loads(alias.read_text())
        if meta["role_manifest_sha256"] != unit["role_manifest_sha256"]:
            raise RuntimeError("CIC compatibility alias drift")
    audit = json.loads((fold / "input_caches/ustc/A-2/tf_fig/cache_audit.json").read_text())
    if audit["status"] != "PASS" or audit["phase"] != "trainval":
        raise RuntimeError("CIC Known Train/Validation input absent")
    with Path(unit["role_manifest"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [(r["flow_id"], r["class_name"], r["role"]) for r in rows
                if r["role"] in ("known_train", "known_validation")]
    classes = sorted(unit["known_classes"])
    for role in ("known_train", "known_validation"):
        if {name for _, name, r in selected if r == role} != set(classes):
            raise RuntimeError(f"CIC {role} class coverage failed")
    if any(name in unit["unknown_classes"] for _, name, _ in selected):
        raise RuntimeError("Unknown CIC label entered train/validation")

    def protocol_rows(dataset, protocol):
        if (dataset, protocol) != ("ustc", "A-2"):
            raise RuntimeError("unexpected Stage31 CLI key")
        return selected, classes, unit["role_manifest_sha256"]

    preflight.OUT = fold
    yatc.OUT = fold
    tf_graph.OUT = fold
    fusion.OUT = fold
    yatc.protocol_rows = protocol_rows
    tf_graph.protocol_rows = protocol_rows
    fusion.protocol_rows = protocol_rows
    yatc.cache_root = lambda dataset, protocol: fold / "input_caches/ustc/A-2/yatc_mfr"
    run = fold / "runs/ustc/A-2"
    if args.component == "fusion":
        for name in ("trafficformer", "graph", "yatc"):
            sub = run / name
            meta = json.loads((sub / "metrics.json").read_text())
            if not (sub / "SUCCESS").exists() or digest(sub / "model_best.pt") != meta["checkpoint_sha256"]:
                raise RuntimeError(f"CIC frozen branch failed: {name}")
        module, arguments = fusion, ["--dataset", "ustc", "--protocol", "A-2"]
    elif args.component == "yatc":
        module, arguments = yatc, ["--dataset", "ustc", "--protocol", "A-2"]
    else:
        module, arguments = tf_graph, ["--dataset", "ustc", "--protocol", "A-2",
                                       "--branch", args.component]
    previous = sys.argv
    sys.argv = [str(module.__file__), *arguments]
    try:
        module.main()
    finally:
        sys.argv = previous
    target = run / ("T0_equal" if args.component == "fusion" else args.component)
    if not (target / "SUCCESS").exists():
        raise RuntimeError(f"CIC {args.component} missing success marker")
    if args.component == "fusion":
        meta = json.loads((target / "known_validation_metrics.json").read_text())
        if any(digest(target / name) != value for name, value in meta["checkpoint_hashes"].items()):
            raise RuntimeError("CIC fusion hash changed")
    else:
        meta = json.loads((target / "metrics.json").read_text())
        if digest(target / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError("CIC branch checkpoint hash changed")
    print(json.dumps({"status": "PASS", "fold": args.key, "component": args.component}))


if __name__ == "__main__":
    main()
