#!/usr/bin/env python3
"""Run the fixed Stage31 three-view recipe on one Stage44 service fold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PROTOCOL_ID = "service_seed2022"
TRAINER_PROTOCOL_ALIAS = "medium_seed2025"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--component", choices=("trafficformer", "graph", "yatc", "fusion"), required=True)
    args = parser.parse_args()
    fold = ROOT / "protocols" / args.fold
    protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
    role_path = fold / "role_manifest.csv"
    if protocol["status"] != "FROZEN_PRETRAIN" or sha256(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("protocol is not the unchanged frozen Stage44 protocol")
    if json.loads((fold / "cache_subset_verification.json").read_text())["status"] != "PASS":
        raise RuntimeError("Known-only cache subset not verified")
    selected = []
    with role_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["role"] in ("known_train", "known_validation"):
                selected.append((row["flow_uid"], row["service"], row["role"]))
    classes = sorted({label for _, label, role in selected if role == "known_train"})
    if classes != protocol["known_services"]:
        raise RuntimeError(f"known Service labels changed: {classes}")
    counts = Counter(role for _, _, role in selected)
    if any(counts[role] != protocol["counts"][role] for role in counts):
        raise RuntimeError("Known Train/Validation counts changed")

    def frozen_rows(dataset: str, protocol_id: str):
        if dataset != "vnat" or protocol_id != TRAINER_PROTOCOL_ALIAS:
            raise RuntimeError("trainer requested an unfrozen protocol")
        return selected, classes, protocol["role_manifest_sha256"]

    sys.path.insert(0, str(PROJECT / "stage31_four_dataset_three_view_equal"))
    import preflight as preflight31
    import train_yatc_branch as yatc
    import train_tf_fig_branch as tf_graph
    import train_equal_fusion as fusion

    # Contain all legacy trainer writes beneath this Stage44 fold. The accepted
    # protocol string is a path/API alias; rows and labels remain Stage44 frozen.
    native_root = fold / "stage44_native_runs_attempt3"
    preflight31.OUT = native_root
    yatc.OUT = native_root
    tf_graph.OUT = native_root
    fusion.OUT = native_root
    yatc.protocol_rows = frozen_rows
    tf_graph.protocol_rows = frozen_rows
    fusion.protocol_rows = frozen_rows
    if args.component in ("trafficformer", "graph"):
        sys.argv = ["train_tf_fig_branch.py", "--dataset", "vnat", "--protocol", TRAINER_PROTOCOL_ALIAS,
                    "--branch", args.component]
        tf_graph.main()
    elif args.component == "yatc":
        sys.argv = ["train_yatc_branch.py", "--dataset", "vnat", "--protocol", TRAINER_PROTOCOL_ALIAS]
        yatc.main()
    else:
        sys.argv = ["train_equal_fusion.py", "--dataset", "vnat", "--protocol", TRAINER_PROTOCOL_ALIAS]
        fusion.main()
    sub = "T0_equal" if args.component == "fusion" else args.component
    output = native_root / "runs" / "vnat" / TRAINER_PROTOCOL_ALIAS / sub
    if not (output / "SUCCESS").is_file():
        raise RuntimeError("trainer returned without SUCCESS")
    print(json.dumps({"status": "PASS", "fold": args.fold, "component": args.component,
                      "known_train": counts["known_train"],
                      "known_validation": counts["known_validation"],
                      "unknown_usage": 0, "test_usage": 0}), flush=True)


if __name__ == "__main__":
    main()
