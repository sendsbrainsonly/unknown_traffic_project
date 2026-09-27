#!/usr/bin/env python3
"""Independently replay a Stage30 Known-Val run and preserve its evidence."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch

from preflight import OUT, known_labels, load_three, sha256, sources
from run_one import Adapters, Fusion, encode, infer, metrics, save_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--encoder-seed", type=int, choices=(2022, 2023), required=True)
    p.add_argument("--training-seed", type=int, choices=range(2022, 2027), required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run / "SUCCESS").is_file() or (run / "independent_replay.json").exists():
        raise RuntimeError("primary run incomplete or replay already exists")
    expected_hashes = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    actual_hashes = {name: sha256(path) for name, path in sources(args.dataset, args.encoder_seed).items()}
    if actual_hashes != expected_hashes[f"{args.dataset}/{args.encoder_seed}"]:
        raise RuntimeError("frozen source hash changed")
    verification = json.loads((run / "verification.json").read_text())
    if verification["known_test_usage"] or verification["unknown_usage"] or verification["encoder_updates"]:
        raise RuntimeError("forbidden data or encoder update")
    for name, digest in verification["checkpoints"].items():
        if sha256(run / name) != digest:
            raise RuntimeError(f"checkpoint hash mismatch: {name}")
    pair = load_three(args.dataset, args.encoder_seed, known_labels()[args.dataset])
    val = pair["roles"]["known_validation"]
    nclasses = len(pair["services"])
    adapter_ckpt = torch.load(run / "adapters_best.pt", map_location="cpu", weights_only=True)
    adapter = Adapters(nclasses)
    adapter.load_state_dict(adapter_ckpt["state_dict"], strict=True)
    encoded = encode(adapter, val, torch.device("cpu"))
    replay = []
    with np.load(run / "known_validation_logits.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["flow_ids"], val["flow_ids"]) or not np.array_equal(stored["labels"], val["labels"]):
            raise RuntimeError("flow ID / label alignment failed")
        for name in ("T0_equal", "T1_three_entropy"):
            ckpt = torch.load(run / f"{name}_best.pt", map_location="cpu", weights_only=True)
            head = Fusion(nclasses, name == "T1_three_entropy",
                          np.asarray(ckpt["known_train_entropy_median"], np.float32),
                          np.asarray(ckpt["known_train_entropy_mad"], np.float32))
            head.load_state_dict(ckpt["state_dict"], strict=True)
            got, weights = infer(head, encoded, torch.device("cpu"), batch_size=23)
            expected = stored[name]
            max_delta = float(np.max(np.abs(got - expected)))
            if max_delta > 2e-4 or not np.array_equal(got.argmax(1), expected.argmax(1)):
                raise RuntimeError(f"checkpoint replay mismatch {name}: {max_delta}")
            if not np.allclose(weights.sum(1), 1., atol=1e-6):
                raise RuntimeError("fusion weights do not sum to one")
            replay.append({"method": name, "max_abs_logit_difference": max_delta,
                           "checkpoint_sha256": sha256(run / f"{name}_best.pt"),
                           "predictions_identical": True})
        table = {row["method"]: row for row in read_csv(run / "run_metrics.csv")}
        for name in ("T0_equal", "T1_three_entropy"):
            recomputed = metrics(val["labels"], stored[name].argmax(1), nclasses)[0]
            if any(abs(recomputed[k] - float(table[name][k])) > 1e-10 for k in recomputed):
                raise RuntimeError(f"metric replay mismatch {name}")
    for file, multiplier in (("known_validation_predictions.csv", 2),
                             ("known_validation_entropy_weights.csv", 2)):
        if len(read_csv(run / file)) != multiplier * len(val["labels"]):
            raise RuntimeError(f"sample evidence incomplete: {file}")
    cf = json.loads((run / "counterfactual.json").read_text())
    if cf["max_batch_abs_logit_difference"] > 1e-5:
        raise RuntimeError("batch invariance failed")
    save_json(run / "independent_replay.json", {
        "status": "PASS", "checkpoint_replays": replay, "metric_replays": 2,
        "flow_ids": len(val["labels"]), "source_hashes_unchanged": True,
        "known_test_usage": 0, "unknown_usage": 0, "encoder_updates": 0})
    lines = [f"# Stage30 {args.dataset}, encoder {args.encoder_seed}, training {args.training_seed}", "",
             "- Status: successful Known-only, three-view entropy-fusion diagnostic.", "",
             "## Data and split", "",
             f"- Frozen Stage20 Known Train: {verification['train_samples']}; Known Validation: {len(val['labels'])}.",
             "- Known Test and Unknown usage: zero. Three frozen input views: TrafficFormer 768, graph 128, YaTC 192.", "",
             "## Configuration and execution", "",
             "- Three 64-D Gaussian adapters, 30 epochs; equal T0 and entropy T1 heads, 30 epochs; batch 256, Adam 1e-3.",
             "- Adapter and head checkpoints selected on Known Validation Macro-F1 only. GPU selection and logs are project-local.", "",
             "## Core results", "", "| Method | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|"]
    for name in ("T0_equal", "T1_three_entropy"):
        row = table[name]
        lines.append(f"| {name} | {float(row['accuracy']):.6f} | {float(row['macro_f1']):.6f} | {float(row['weighted_f1']):.6f} |")
    lines += ["", f"- T1 minus forced-equal Macro-F1: {cf['main']['macro_f1'] - cf['forced_equal']['macro_f1']:+.6f}.",
              "", "## Preserved evidence", "",
              "- All checkpoints, curves, metrics, per-class and per-flow predictions, entropy weights, counterfactuals, hashes, and independent CPU replay.",
              "", "## Limitations", "",
              "- Frozen encoders; this is a controlled development diagnostic, not an official paper reproduction. Head seeds share Known-Val flows.",
              "- No Test or Unknown evaluation; no open-set or generalization claim.", "",
              "## Conclusion and next step", "",
              "- Aggregate the 20 preregistered paired units before deciding whether dynamic fusion helps.", ""]
    (run / "RESULTS.md").write_text("\n".join(lines))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success",
                     "inputs": [{"path": "source_hashes_before.json", "role": "frozen three-view Known-only inputs"}],
                     "execution": {**manifest["execution"], "environment": "fixed TrafficClassifier DGL Conda prefix", "exit_code": 0},
                     "configuration": {"files": ["stage30_three_view_entropy_fusion/EXPERIMENT_PLAN.md",
                                                 "stage30_three_view_entropy_fusion/run_one.py"],
                                       "parameters": {"adapter_epochs": 30, "head_epochs": 30,
                                                      "batch_size": 256, "latent_dim": 64,
                                                      "KL_coefficient": 0.05, "views": 3},
                                       "seeds": [args.encoder_seed, args.training_seed]},
                     "core_results": [{"method": name, "macro_f1": float(table[name]["macro_f1"])}
                                      for name in ("T0_equal", "T1_three_entropy")],
                     "limitations": ["Known-Val development only; frozen encoders; repeated validation flows"],
                     "next_step": "20-unit paired Known-Val aggregation"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "run": str(run), "checkpoint_replays": 2}), flush=True)


if __name__ == "__main__":
    main()
