#!/usr/bin/env python3
"""Verify and document one completed Known-only Stage28 unit."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from preflight import OUT, sha256


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--encoder-seed", type=int, required=True)
    p.add_argument("--training-seed", type=int, required=True)
    args = p.parse_args()
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run / "SUCCESS").is_file() or not (run / "baseline_verification.json").is_file():
        raise RuntimeError("primary run or baselines incomplete")
    if (run / "independent_replay.json").exists():
        raise RuntimeError("refusing overwrite of independent replay")
    verification = json.loads((run / "verification.json").read_text())
    baseline = json.loads((run / "baseline_verification.json").read_text())
    if any((verification["test_samples_loaded"], verification["unknown_samples_loaded"],
            baseline["test_values_loaded"], baseline["unknown_values_loaded"])):
        raise RuntimeError("Test or Unknown access detected")
    for category, records in (("adapter", verification["adapter"]), ("fusion", verification["fusion"])):
        for name, record in records.items():
            ckpt = run / (f"{name}_adapter_best.pt" if category == "adapter" else f"{name}_best.pt")
            if sha256(ckpt) != record["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash changed: {ckpt}")
    for method, digest in baseline["checkpoint_sha256"].items():
        if sha256(run / f"baseline_{method}_best.pt") != digest:
            raise RuntimeError(f"baseline checkpoint hash changed: {method}")
    checks = []
    for prefix, csv_file, npz_file, method_key in (
        ("main", "run_metrics.csv", "known_validation_logits.npz", "variant"),
        ("baseline", "baseline_metrics.csv", "baseline_known_validation_logits.npz", "method"),
    ):
        metrics = rows(run / csv_file)
        with np.load(run / npz_file, allow_pickle=False) as arr:
            labels = arr["labels"]
            flow_ids = arr["flow_ids"]
            if len(set(flow_ids.tolist())) != len(flow_ids):
                raise RuntimeError("duplicate validation flow ID")
            for row in metrics:
                method = row[method_key]
                logits = arr[method]
                predicted = logits.argmax(1)
                if len(predicted) != len(labels) or not np.isfinite(logits).all():
                    raise RuntimeError("logits shape/nonfinite")
                _, _, f1, support = precision_recall_fscore_support(
                    labels, predicted, labels=np.arange(logits.shape[1]), zero_division=0,
                )
                recomputed = {"accuracy": float(np.mean(predicted == labels)),
                              "macro_f1": float(f1.mean()),
                              "weighted_f1": float(np.average(f1, weights=support))}
                if any(abs(recomputed[key] - float(row[key])) > 1e-10 for key in recomputed):
                    raise RuntimeError(f"metric replay mismatch: {method}")
                checks.append({"family": prefix, "method": method, **recomputed})
    sample_main = rows(run / "known_validation_predictions.csv")
    sample_base = rows(run / "baseline_predictions.csv")
    expected = verification["validation_samples"]
    if len(sample_main) != 4 * expected or len(sample_base) != 3 * expected:
        raise RuntimeError("sample prediction count mismatch")
    hrows = rows(run / "known_validation_entropy_weights.csv")
    if len(hrows) != 4 * expected:
        raise RuntimeError("entropy weight count mismatch")
    if any(abs(float(r["w_content"]) + float(r["w_behavior"]) - 1) > 1e-5 for r in hrows):
        raise RuntimeError("entropy weights not normalized")
    cf = verification["counterfactual"]
    if cf["batch_invariance_max_abs"] > 1e-5:
        raise RuntimeError("batch-invariance failed")
    (run / "independent_replay.json").write_text(json.dumps({
        "status": "PASS", "logit_metric_replays": len(checks), "methods": checks,
        "main_prediction_rows": len(sample_main), "baseline_prediction_rows": len(sample_base),
        "entropy_weight_rows": len(hrows), "weight_sum": "PASS",
        "batch_invariance_max_abs": cf["batch_invariance_max_abs"],
        "test_values_loaded": 0, "unknown_values_loaded": 0,
    }, indent=2) + "\n")
    main = {r["variant"]: r for r in rows(run / "run_metrics.csv")}
    controls = {r["method"]: r for r in rows(run / "baseline_metrics.csv")}
    lines = [f"# Stage28 {args.dataset} encoder {args.encoder_seed} / training {args.training_seed}", "",
             "- Status: success, Known-only development diagnostic; not ER-CMGI author-equivalent.",
             "", "## Data and split", "",
             f"- Frozen Stage20 flow IDs: Known Train {verification['train_samples']}; Known Validation {expected}.",
             "- Unknown usage 0; Known Test usage 0; no encoder retraining.",
             "", "## Configuration and execution", "",
             "- 30 full epochs for A0/A1 and each P0-P3 head; Known Val Macro-F1 checkpoint selection.",
             "- P0/P1 use A0; P2/P3 use A1. Train-only scaler, PCA, entropy median/MAD.",
             "- Execution: named tmux sessions, see project `.tmux-task/`; checkpoints and curves retained.",
             "", "## Core results", "", "| Method | Val Acc | Val Macro-F1 | Val Weighted-F1 |", "|---|---:|---:|---:|" ]
    for name, r in (*controls.items(), *main.items()):
        lines.append(f"| {name} | {float(r['accuracy']):.6f} | {float(r['macro_f1']):.6f} | {float(r['weighted_f1']):.6f} |")
    lines += ["", f"- P3 forced-equal Macro-F1: {cf['forced_equal']['macro_f1']:.6f}; shuffled entropy: {cf['shuffled_entropy']['macro_f1']:.6f}.",
              "", "## Preserved evidence", "",
              "- Checkpoints, full training curves, per-class and per-sample predictions, logits, entropy weights, counterfactuals, hashes, replay and tmux evidence.",
              "- Independent saved-logit metric replay and hash checks: PASS.",
              "", "## Limitations", "",
              "- Same frozen encoder pair and validation flow pool reused across training seeds; not independent replicates.",
              "- Stage20 Test was previously exposed but not accessed in this unit.",
              "", "## Conclusion and next step", "",
              "- Aggregate paired Known-Val evidence across all 20 planned units, then apply preregistered gate.", ""]
    (run / "RESULTS.md").write_text("\n".join(lines))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                     "status": "success", "experiment_type": "ablation", "claim_scope": "diagnostic",
                     "inputs": [{"path": "Stage20 frozen manifest and Stage22/23/27 source hashes in source_hashes_before.json",
                                 "role": "Known Train/Validation only"}],
                     "execution": {**manifest["execution"], "environment": "fixed TrafficClassifier DGL conda prefix",
                                   "exit_code": 0},
                     "configuration": {"files": ["stage28_er_cmgi_entropy_fusion/run_one.py",
                                                 "stage28_er_cmgi_entropy_fusion/baselines.py"],
                                       "parameters": {"adapter_epochs": 30, "head_epochs": 30,
                                                      "batch_size": 256, "latent_dim": 64,
                                                      "adapter_kl_coefficient": 0.05,
                                                      "cross_view_mse_coefficient": 0.1},
                                       "seeds": [args.encoder_seed, args.training_seed]},
                     "core_results": checks,
                     "limitations": ["Stage20 Test was previously exposed but not read in this run",
                                     "paper-inspired mechanism adaptation, not original ER-CMGI reproduction"],
                     "next_step": "preregistered across-unit Known-Val aggregation"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "run": str(run), "methods": len(checks)}), flush=True)


if __name__ == "__main__":
    main()
