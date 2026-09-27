#!/usr/bin/env python3
"""Use unchanged Stage31/34 three-view training functions on new A-1/A-3 roles."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from freeze_matched_protocol import PROJECT, ROOT, digest

sys.path[:0] = [str(PROJECT / "stage34_ustc_cic_closed_set"),
                str(PROJECT / "stage31_four_dataset_three_view_equal")]
import ustc_runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("A-1", "A-3"), required=True)
    parser.add_argument("--action", choices=("trafficformer", "graph", "yatc", "fusion"), required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "matched_protocol.json").read_text())
    unit = protocol["units"][args.setting]
    path = Path(unit["role_manifest"])
    if digest(path) != unit["role_manifest_sha256"]:
        raise RuntimeError("role manifest changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    trainval = [(r["flow_id"], r["class_name"], r["role"])
                for r in rows if r["role"] in ("known_train", "known_validation")]
    classes = sorted(unit["known_classes"])
    if any(r[1] not in classes for r in trainval):
        raise RuntimeError("Unknown leaked into training")
    if any({r[1] for r in trainval if r[2] == role} != set(classes)
           for role in ("known_train", "known_validation")):
        raise RuntimeError("known class absent from Train or Validation")
    output = ROOT / args.setting
    for module in (ustc_runner.preflight, ustc_runner.yatc,
                   ustc_runner.tf_graph, ustc_runner.fusion):
        module.OUT = output
    def selected(dataset, setting):
        if (dataset, setting) != ("ustc", "A-2"):
            raise RuntimeError("unexpected Stage31 protocol hook")
        return trainval, classes, unit["role_manifest_sha256"]
    ustc_runner.yatc.protocol_rows = selected
    ustc_runner.tf_graph.protocol_rows = selected
    ustc_runner.fusion.protocol_rows = selected
    ustc_runner.yatc.cache_root = lambda dataset, setting: (
        output / "input_caches/ustc/A-2/yatc_mfr")
    if args.action == "yatc":
        ustc_runner.launch_main(ustc_runner.yatc, ["--dataset", "ustc", "--protocol", "A-2"])
    elif args.action in ("trafficformer", "graph"):
        ustc_runner.launch_main(ustc_runner.tf_graph,
                                ["--dataset", "ustc", "--protocol", "A-2",
                                 "--branch", args.action])
    else:
        ustc_runner.launch_main(ustc_runner.fusion, ["--dataset", "ustc", "--protocol", "A-2"])


if __name__ == "__main__":
    main()
