#!/usr/bin/env python3
"""Apply unchanged Stage40 frozen scorer to matched A-1/A-3 three-view models."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from freeze_matched_protocol import PROJECT, ROOT, digest

sys.path.insert(0, str(PROJECT / "stage40_ustc_cic_open_set"))
import score_frozen  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("A-1", "A-3"), required=True)
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "matched_protocol.json").read_text())
    unit = {**protocol["units"][args.setting], "dataset": "USTC-TFC2016"}
    path = Path(unit["role_manifest"])
    if digest(path) != unit["role_manifest_sha256"]:
        raise RuntimeError("role manifest hash drift")
    with path.open(newline="", encoding="utf-8") as handle:
        roles = list(csv.DictReader(handle))
    if len(roles) != len({r["flow_id"] for r in roles}):
        raise RuntimeError("duplicate flow role")
    if any(r["class_name"] in unit["unknown_classes"] for r in roles
           if r["role"] != "unknown_test"):
        raise RuntimeError("Unknown leakage into fitting roles")
    base = ROOT / args.setting
    run, out = base / "runs/ustc/A-2", base / "detection"
    def setup(key):
        if key != args.setting:
            raise RuntimeError("scenario mismatch")
        return unit, roles, run, out
    def paths(key):
        if key != args.setting:
            raise RuntimeError("scenario mismatch")
        base_cache = base / "input_caches/ustc/A-2"
        return [(base_cache / "tf_fig_eval", base_cache / "yatc_mfr_eval", "combined")]
    score_frozen.setup = setup
    score_frozen.input_paths = paths
    if args.phase == "calibrate":
        score_frozen.calibrate(args.setting)
    else:
        score_frozen.evaluate(args.setting)


if __name__ == "__main__":
    main()
