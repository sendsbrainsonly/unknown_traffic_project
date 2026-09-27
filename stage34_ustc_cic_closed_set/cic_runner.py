#!/usr/bin/env python3
"""CIC strict-three-class application of the unchanged Stage31 T0 recipe.

Stage31's CLI accepts only four historical dataset names.  Its internal
``ustc/A-2`` key is used solely inside this CIC-specific, isolated output root;
the actual dataset/protocol and manifest hash are written in CIC provenance.
The imported model classes, optimizers, epoch budgets and checkpoint rules are
not changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import cic_protocol as protocol
import cic_build_inputs as inputs

ROOT = Path(__file__).resolve().parent
CIC = ROOT / "cicids2017"
MANIFEST = ROOT / "cicids2017_three_class_manifest_v2.csv"
OLD = ROOT.parent / "stage31_four_dataset_three_view_equal"
sys.path.insert(0, str(OLD))
import preflight  # noqa: E402
import train_yatc_branch as yatc  # noqa: E402
import train_tf_fig_branch as tf_graph  # noqa: E402
import train_equal_fusion as fusion  # noqa: E402
import evaluate_known_test as evaluator  # noqa: E402


def sample_rows():
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if protocol.digest(MANIFEST) != json.loads((ROOT / "cicids2017_protocol_v2_audit.json").read_text())["v2_manifest_sha256"]:
        raise RuntimeError("CIC sample manifest hash drift")
    return rows


def protocol_rows(dataset, key):
    if (dataset, key) != ("ustc", "A-2"):
        raise RuntimeError("unexpected internal Stage31 compatibility key")
    rows = sample_rows()
    chosen = [(r["flow_id"], r["label"],
               "known_train" if r["split"] == "train" else "known_validation")
              for r in rows if r["split"] in ("train", "validation")]
    classes = sorted(protocol.CLASSES)
    for role in ("known_train", "known_validation"):
        if {c for _, c, split in chosen if split == role} != set(classes):
            raise RuntimeError(f"CIC {role} missing class")
    return chosen, classes, protocol.digest(MANIFEST)


def patch_modules():
    preflight.OUT = CIC
    yatc.OUT = CIC
    yatc.protocol_rows = protocol_rows
    yatc.cache_root = lambda dataset, key: CIC / "input_caches/ustc/A-2/yatc_mfr"
    tf_graph.OUT = CIC
    tf_graph.protocol_rows = protocol_rows
    fusion.OUT = CIC
    fusion.protocol_rows = protocol_rows


def call(module, args):
    old = sys.argv
    sys.argv = [str(module.__file__), *args]
    try:
        module.main()
    finally:
        sys.argv = old


def record_alias():
    path = CIC / "COMPATIBILITY_ALIAS.json"
    CIC.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
            "actual_dataset": "CIC-IDS-2017", "actual_protocol": "strict_three_class_five_minute_group_split_v2",
            "actual_sample_manifest": str(MANIFEST), "actual_sample_sha256": protocol.digest(MANIFEST),
            "internal_Stage31_runner_key": "ustc/A-2", "meaning": "CLI/path compatibility only; not USTC data",
            "reused_recipe": "Stage31 TrafficFormer20e, TAGCN50e, YaTC200e, T0 adapter30e/head30e",
            "unknown_samples_used": 0}, indent=2) + "\n"
    try:
        with path.open("x") as f:
            f.write(payload)
    except FileExistsError:
        if path.read_text() != payload:
            raise RuntimeError("CIC compatibility alias metadata changed")


def run_branch(name):
    patch_modules()
    record_alias()
    if name == "yatc":
        call(yatc, ["--dataset", "ustc", "--protocol", "A-2"])
    else:
        call(tf_graph, ["--dataset", "ustc", "--protocol", "A-2", "--branch", name])


def run_fusion():
    patch_modules()
    record_alias()
    call(fusion, ["--dataset", "ustc", "--protocol", "A-2"])


def evaluate():
    inputs.frozen_head_gate()
    patch_modules()
    evaluator.OUT = CIC
    evaluator.test_cache = lambda dataset, key: (
        CIC / "input_caches/ustc/A-2/tf_fig_test",
        CIC / "input_caches/ustc/A-2/yatc_mfr_test")
    run = CIC / "runs/ustc/A-2"
    output = run / "known_test_evaluation"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _, classes, _ = protocol_rows("ustc", "A-2")
    rows = sorted((r["flow_id"], r["label"]) for r in sample_rows() if r["split"] == "test")
    if {c for _, c in rows} != set(classes):
        raise RuntimeError("CIC Known Test class absent")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for CIC Known Test extraction")
    device = torch.device("cuda:0")
    features, branch_hashes = evaluator.extract_branches("ustc", "A-2", classes, rows, device, run)
    head = run / "T0_equal"
    selected = json.loads((head / "known_validation_metrics.json").read_text())
    paths = {name: evaluator.verify_model(head, name, h)
             for name, h in selected["checkpoint_hashes"].items()}
    scaler = np.load(head / "known_train_view_scalers.npz", allow_pickle=False)
    for name in features:
        features[name] = ((features[name] - scaler[f"{name}_mean"]) /
                          scaler[f"{name}_std"]).astype(np.float32)
        if not np.isfinite(features[name]).all():
            raise RuntimeError(f"nonfinite CIC Test feature: {name}")
    truth = np.asarray([classes.index(c) for _, c in rows], dtype=np.int64)
    adapters = fusion.Adapters(len(classes)).to(device)
    adapters.load_state_dict(torch.load(paths["adapters_best.pt"], map_location="cpu", weights_only=False)["state_dict"])
    encoded = fusion.encode(adapters, {"labels": truth, **features}, device)
    model = fusion.EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
    model.load_state_dict(torch.load(paths["T0_equal_best.pt"], map_location="cpu", weights_only=False)["state_dict"])
    logits = fusion.infer(model, encoded, device)
    chosen = logits.argmax(1)
    score, per = fusion.metrics(truth, chosen, len(classes))
    with (output / "sample_predictions.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("flow_id", "true_class", "predicted_class", "correct"))
        for (uid, label), pred in zip(rows, chosen, strict=True):
            writer.writerow((uid, label, classes[int(pred)], int(classes[int(pred)] == label)))
    np.save(output / "logits.npy", logits.astype(np.float32), allow_pickle=False)
    report = {"status": "PASS", "dataset": "CIC-IDS-2017", "protocol": "strict_three_class_group_disjoint_v2",
              "sample_manifest_sha256": protocol.digest(MANIFEST), "known_test_samples": len(rows),
              "classes": classes, "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
              "weighted_f1": score["weighted_f1"],
              "per_class": [{"class": c, "precision": float(per[0][i]), "recall": float(per[1][i]),
                             "f1": float(per[2][i]), "support": int(per[3][i])}
                            for i, c in enumerate(classes)],
              "branch_checkpoint_sha256": branch_hashes,
              "adapter_checkpoint_sha256": evaluator.sha(paths["adapters_best.pt"]),
              "head_checkpoint_sha256": evaluator.sha(paths["T0_equal_best.pt"]),
              "unknown_samples_loaded": 0, "test_parameter_selection": 0,
              "caveat": "Slowhttptest and PortScan each occur on only one day/one source PCAP; five-minute group disjointness does not remove that confound"}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status": "PASS", "dataset": report["dataset"], "metrics": score}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("branch", "fusion", "evaluate"))
    parser.add_argument("--branch", choices=("trafficformer", "graph", "yatc"))
    args = parser.parse_args()
    if args.action == "branch":
        if not args.branch:
            parser.error("--branch required")
        run_branch(args.branch)
    elif args.action == "fusion":
        run_fusion()
    else:
        evaluate()


if __name__ == "__main__":
    main()
