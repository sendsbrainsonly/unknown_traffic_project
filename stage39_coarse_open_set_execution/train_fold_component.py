"""Run an unchanged Stage31 branch/fusion recipe on one frozen Stage39 fold."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from freeze_protocols import PROJECT, ROOT, sha


def train(dataset: str, slug: str, component: str):
    fold = ROOT / "settings" / dataset / slug
    protocol = json.loads((fold / "protocol.json").read_text())
    if protocol["status"] != "FROZEN_PRETRAIN" or protocol["checkpoint_reuse"]:
        raise RuntimeError("not a new-training frozen protocol")
    if not (fold / "cache_subset_verification.json").is_file():
        raise RuntimeError("Known-only derived input cache is absent")
    source_lock = PROJECT / "stage38_three_view_des_open_set_transfer/stage38b_training_config_lock.json"
    if sha(source_lock) != protocol["stage38_training_lock_sha256"]:
        raise RuntimeError("training configuration changed since protocol freeze")
    for source, expected in protocol["source_training_code_sha256"].items():
        if sha(PROJECT / source) != expected:
            raise RuntimeError(f"locked training source changed: {source}")
    role_path = fold / "role_manifest.csv"
    if sha(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("role manifest changed")
    selected = []
    with role_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["role"] in ("known_train", "known_validation"):
                selected.append((row["sample_id"], row["classifier_label"], row["role"]))
    classes = sorted({label for _, label, role in selected if role == "known_train"})
    if classes != protocol["known_classifier_labels"]:
        raise RuntimeError("class identity changed")
    by_role = Counter(role for _, _, role in selected)
    if any(by_role[r] != protocol["counts"][r] for r in by_role):
        raise RuntimeError("training sample count changed")

    def frozen_rows(requested_dataset, requested_protocol):
        if requested_dataset != ("vnat" if dataset == "VNAT" else dataset) or requested_protocol != protocol["protocol_id"]:
            raise RuntimeError("trainer requested an unfrozen protocol")
        return selected, classes, protocol["role_manifest_sha256"]

    sys.path.insert(0, str(PROJECT / "stage31_four_dataset_three_view_equal"))
    import preflight as preflight31  # noqa: E402
    import train_yatc_branch as yatc  # noqa: E402
    import train_tf_fig_branch as tf_graph  # noqa: E402
    import train_equal_fusion as fusion  # noqa: E402

    preflight31.OUT = fold
    yatc.OUT = fold
    tf_graph.OUT = fold
    fusion.OUT = fold
    yatc.protocol_rows = frozen_rows
    tf_graph.protocol_rows = frozen_rows
    fusion.protocol_rows = frozen_rows
    name = "vnat" if dataset == "VNAT" else dataset
    if component in ("trafficformer", "graph"):
        sys.argv = ["train_tf_fig_branch.py", "--dataset", name,
                    "--protocol", protocol["protocol_id"], "--branch", component]
        tf_graph.main()
    elif component == "yatc":
        sys.argv = ["train_yatc_branch.py", "--dataset", name, "--protocol", protocol["protocol_id"]]
        yatc.main()
    elif component == "fusion":
        sys.argv = ["train_equal_fusion.py", "--dataset", name, "--protocol", protocol["protocol_id"]]
        fusion.main()
    else:
        raise ValueError(component)
    sub = "T0_equal" if component == "fusion" else component
    output = fold / "runs" / name / protocol["protocol_id"] / sub
    if not (output / "SUCCESS").is_file():
        raise RuntimeError("training source returned without SUCCESS marker")
    report = output / ("known_validation_metrics.json" if component == "fusion" else "metrics.json")
    content = json.loads(report.read_text())
    if component == "fusion":
        if any(sha(output / key) != digest for key, digest in content["checkpoint_hashes"].items()):
            raise RuntimeError("fusion checkpoint hash mismatch")
    elif sha(output / "model_best.pt") != content["checkpoint_sha256"]:
        raise RuntimeError("branch checkpoint hash mismatch")
    print(json.dumps({"status": "PASS", "dataset": dataset, "fold": slug,
                      "component": component}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "VNAT"), required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--component", choices=("trafficformer", "graph", "yatc", "fusion"), required=True)
    args = parser.parse_args()
    train(args.dataset, args.fold, args.component)
