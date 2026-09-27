#!/usr/bin/env python3
"""Independent CPU checkpoint/metric replay and run-bundle finalization."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch

from preflight import OUT, labels, load_pair, paths, sha
from run_one import Fusion, Pair, encode, infer, metrics, write_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--encoder-seed", type=int, required=True)
    p.add_argument("--training-seed", type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run / "SUCCESS").is_file() or (run / "independent_replay.json").exists():
        raise RuntimeError("primary run incomplete or replay already exists")
    expected_hashes = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    actual_hashes = {name: sha(path) for name, path in paths(args.dataset, args.encoder_seed).items()}
    if actual_hashes != expected_hashes[f"{args.dataset}/{args.encoder_seed}"]:
        raise RuntimeError("frozen source hash changed")
    verification = json.loads((run / "verification.json").read_text())
    if any(verification[k] for k in ("yatc_usage", "known_test_usage", "unknown_usage")):
        raise RuntimeError("forbidden data usage")
    for name, digest in verification["checkpoints"].items():
        if sha(run / name) != digest:
            raise RuntimeError(f"checkpoint hash mismatch: {name}")
    pair = load_pair(args.dataset, args.encoder_seed, labels()[args.dataset])
    val = pair["roles"]["known_validation"]
    classes = len(pair["services"])
    adapter_ckpt = torch.load(run / "adapter_best.pt", map_location="cpu", weights_only=True)
    adapter = Pair(classes)
    adapter.load_state_dict(adapter_ckpt["state_dict"], strict=True)
    encoded = encode(adapter, val, torch.device("cpu"))
    record = []
    with np.load(run / "known_validation_logits.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["flow_ids"], val["flow_ids"]) or \
           not np.array_equal(stored["labels"], val["labels"]) or \
           not np.array_equal(stored["E3_original_pred"], val["e3_original_pred"]):
            raise RuntimeError("flow/label/original-E3 parity failed")
        for name in ("N0_equal", "N1_shared_lambda_entropy"):
            ckpt = torch.load(run / f"{name}_best.pt", map_location="cpu", weights_only=True)
            head = Fusion(classes, name.startswith("N1"),
                          np.asarray(ckpt["entropy_median"], np.float32),
                          np.asarray(ckpt["entropy_mad"], np.float32))
            head.load_state_dict(ckpt["state_dict"], strict=True)
            got, weights = infer(head, encoded, torch.device("cpu"), batch=23)
            expected = stored[name]
            max_delta = float(np.max(np.abs(got - expected)))
            if max_delta > 2e-4 or not np.array_equal(got.argmax(1), expected.argmax(1)):
                raise RuntimeError(f"checkpoint replay mismatch {name}, max={max_delta}")
            if not np.allclose(weights.sum(1), 1, atol=1e-6):
                raise RuntimeError("weights do not sum to one")
            record.append({"method": name, "max_abs_logit_difference": max_delta,
                           "checkpoint_sha256": sha(run / f"{name}_best.pt"),
                           "predictions_identical": True})
        recorded = {r["method"]: r for r in read_csv(run / "run_metrics.csv")}
        for method, pred in (("E3_original", stored["E3_original_pred"]),
                             ("N0_equal", stored["N0_equal"].argmax(1)),
                             ("N1_shared_lambda_entropy", stored["N1_shared_lambda_entropy"].argmax(1))):
            recomputed = metrics(val["labels"], pred, classes)[0]
            if any(abs(recomputed[k] - float(recorded[method][k])) > 1e-10 for k in recomputed):
                raise RuntimeError(f"metric replay mismatch {method}")
    if len(read_csv(run / "known_validation_predictions.csv")) != 3 * len(val["labels"]):
        raise RuntimeError("sample predictions incomplete")
    if len(read_csv(run / "known_validation_entropy_weights.csv")) != 2 * len(val["labels"]):
        raise RuntimeError("entropy weights incomplete")
    cf = json.loads((run / "counterfactual.json").read_text())
    if cf["max_batch_abs_logit_difference"] > 1e-5:
        raise RuntimeError("batch invariance failed")
    write_json(run / "independent_replay.json", {"status": "PASS", "checkpoint_replays": record,
               "metric_replays": 3, "flow_ids": len(val["labels"]),
               "source_hashes_unchanged": True, "yatc_usage": 0,
               "known_test_usage": 0, "unknown_usage": 0})
    table = {r["method"]: r for r in read_csv(run / "run_metrics.csv")}
    lines = [f"# Stage29 {args.dataset} encoder {args.encoder_seed} / training {args.training_seed}", "",
             "- Status: success; native E3 two-branch Known-only diagnostic.", "",
             "## Data and split", "",
             f"- Frozen Stage20 Known Train {verification['train_samples']}; Known Val {len(val['labels'])}. YaTC/Test/Unknown usage 0.", "",
             "## Configuration and execution", "",
             "- Frozen E3 TrafficFormer 768-D and FIG/TAGCN 128-D; 30-epoch Gaussian adapters; N0/N1 30-epoch heads; batch 256, Adam 1e-3.",
             "- Checkpoint selection: Known-Val fine Service Macro-F1. Live GPU selection and tmux logs are project-local.", "",
             "## Core results", "",
             "| Method | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|"]
    for name in ("E3_original", "N0_equal", "N1_shared_lambda_entropy"):
        row = table[name]
        lines.append(f"| {name} | {float(row['accuracy']):.6f} | {float(row['macro_f1']):.6f} | {float(row['weighted_f1']):.6f} |")
    lines += ["", f"- N1 minus forced-equal Macro-F1: {cf['main']['macro_f1']-cf['forced_equal']['macro_f1']:+.6f}.",
              "", "## Preserved evidence", "",
              "- Adapter/N0/N1 checkpoints, complete curves, metrics, class and flow predictions, entropy weights, counterfactuals, input hashes and independent CPU replay.",
              "", "## Limitations", "",
              "- Encoders are frozen; not end-to-end E3 retraining. Head seeds share the same Known-Val flows. Stage20 Test was previously exposed but not read.",
              "", "## Conclusion and next step", "",
              "- Aggregate all 20 paired units against the preregistered Known-Val gate; do not infer Test/open-set benefit.", ""]
    (run / "RESULTS.md").write_text("\n".join(lines))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success",
                     "inputs": [{"path": "source_hashes_before.json", "role": "frozen E3 Known-only"}],
                     "execution": {**manifest["execution"], "environment": "fixed TrafficClassifier DGL conda prefix", "exit_code": 0},
                     "configuration": {"files": ["stage29_e3_native_entropy_fusion/EXPERIMENT_PLAN.md",
                                                 "stage29_e3_native_entropy_fusion/run_one.py"],
                                       "parameters": {"adapter_epochs": 30, "head_epochs": 30,
                                                      "batch_size": 256, "latent_dim": 64,
                                                      "KL_coefficient": 0.05, "shared_lambda": True},
                                       "seeds": [args.encoder_seed, args.training_seed]},
                     "core_results": [{"method": method, "macro_f1": float(table[method]["macro_f1"])}
                                      for method in ("E3_original", "N0_equal", "N1_shared_lambda_entropy")],
                     "limitations": ["Known-Val development only; frozen encoders; repeated validation flows"],
                     "next_step": "20-unit paired Known-Val aggregation"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "run": str(run), "checkpoint_replays": 2}), flush=True)


if __name__ == "__main__":
    main()
