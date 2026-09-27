#!/usr/bin/env python3
"""Preregistered paired Known-Val summary for true three-view entropy fusion."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

from preflight import OUT, PROJECT, sha256, sources
from run_one import VIEWS, save_csv, save_json

DATASETS = ("iscx_vpn", "iscx_tor")
METHODS = ("T0_equal", "T1_three_entropy")
MEASURES = ("accuracy", "macro_f1", "weighted_f1")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    if (OUT / "stage30_report.md").exists():
        raise RuntimeError("aggregate already exists")
    progress = json.loads((OUT / "queue_progress.json").read_text())
    if progress["status"] != "success":
        raise RuntimeError("queue is incomplete")
    all_metrics, all_classes, paired, entropy, counterfactual, rescue = [], [], [], [], [], []
    for dataset in DATASETS:
        for encoder in (2022, 2023):
            for train in range(2022, 2027):
                run = OUT / "runs" / dataset / f"encoder{encoder}_train{train}"
                if not (run / "SUCCESS").is_file() or not (run / "independent_replay.json").is_file():
                    raise RuntimeError(f"incomplete run: {run}")
                verification = json.loads((run / "independent_replay.json").read_text())
                if verification["status"] != "PASS" or len(verification["checkpoint_replays"]) != 2 or any(
                    verification[k] for k in ("known_test_usage", "unknown_usage", "encoder_updates")):
                    raise RuntimeError(f"forbidden use or failed replay: {run}")
                if json.loads((run / "manifest.json").read_text())["status"] != "success":
                    raise RuntimeError(f"unvalidated run bundle: {run}")
                table = {}
                for row in read_csv(run / "run_metrics.csv"):
                    item = {"dataset": dataset, "encoder_seed": encoder, "training_seed": train,
                            "method": row["method"], "best_epoch": int(row["best_epoch"]),
                            "samples": int(row["samples"]),
                            **{m: float(row[m]) for m in MEASURES}}
                    all_metrics.append(item)
                    table[item["method"]] = item
                if set(table) != set(METHODS):
                    raise RuntimeError(f"missing method: {run}")
                paired.append({"dataset": dataset, "encoder_seed": encoder, "training_seed": train,
                               **{f"delta_{m}": table[METHODS[1]][m] - table[METHODS[0]][m]
                                  for m in MEASURES}})
                for row in read_csv(run / "per_class.csv"):
                    all_classes.append({"dataset": dataset, "encoder_seed": encoder,
                                        "training_seed": train, "method": row["method"],
                                        "class": row["class"], "support": int(row["support"]),
                                        "precision": float(row["precision"]),
                                        "recall": float(row["recall"]), "f1": float(row["f1"])})
                hrows = [r for r in read_csv(run / "known_validation_entropy_weights.csv")
                         if r["method"] == METHODS[1]]
                w = np.asarray([[float(r[f"w_{v}"]) for v in VIEWS] for r in hrows])
                h = np.asarray([[float(r[f"H_{v}"]) for v in VIEWS] for r in hrows])
                if len(w) != table[METHODS[1]]["samples"] or not np.allclose(w.sum(1), 1, atol=1e-6):
                    raise RuntimeError(f"weight/sample mismatch: {run}")
                curves = [r for r in read_csv(run / "training_history.csv") if r["phase"] == METHODS[1]]
                epoch = table[METHODS[1]]["best_epoch"]
                selected = next(r for r in curves if int(r["epoch"]) == epoch)
                entropy.append({"dataset": dataset, "encoder_seed": encoder,
                                "training_seed": train, "samples": len(hrows),
                                **{f"lambda_{v}": float(selected[f"lambda_{v}"]) for v in VIEWS},
                                **{f"w_{v}_mean": float(w[:,i].mean()) for i,v in enumerate(VIEWS)},
                                **{f"w_{v}_std": float(w[:,i].std()) for i,v in enumerate(VIEWS)},
                                **{f"H_{v}_std": float(h[:,i].std()) for i,v in enumerate(VIEWS)},
                                "weight_collapse_ratio": float((w.max(1) > .95).mean())})
                cf = json.loads((run / "counterfactual.json").read_text())
                counterfactual.append({"dataset": dataset, "encoder_seed": encoder,
                                       "training_seed": train,
                                       "T1_macro_f1": cf["main"]["macro_f1"],
                                       "forced_equal_macro_f1": cf["forced_equal"]["macro_f1"],
                                       "shuffled_entropy_macro_f1": cf["shuffled_entropy"]["macro_f1"],
                                       "delta_vs_forced_equal": cf["main"]["macro_f1"] - cf["forced_equal"]["macro_f1"],
                                       "delta_vs_shuffled_entropy": cf["main"]["macro_f1"] - cf["shuffled_entropy"]["macro_f1"],
                                       "batch_invariance_max_abs": cf["max_batch_abs_logit_difference"]})
                pred = read_csv(run / "known_validation_predictions.csv")
                by_method = {method: {r["flow_id"]: int(r["correct"]) for r in pred if r["method"] == method}
                             for method in METHODS}
                if set(by_method[METHODS[0]]) != set(by_method[METHODS[1]]):
                    raise RuntimeError(f"prediction flow mismatch: {run}")
                a, b = by_method[METHODS[0]], by_method[METHODS[1]]
                rescue.append({"dataset": dataset, "encoder_seed": encoder, "training_seed": train,
                               "both_correct": sum(a[f] == b[f] == 1 for f in a),
                               "both_wrong": sum(a[f] == b[f] == 0 for f in a),
                               "T1_rescues_T0": sum(a[f] == 0 and b[f] == 1 for f in a),
                               "T1_hurts_T0": sum(a[f] == 1 and b[f] == 0 for f in a)})
    if len(all_metrics) != 40 or len(paired) != 20 or len(counterfactual) != 20:
        raise RuntimeError("aggregate count mismatch")
    before = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    after = {}
    for dataset in DATASETS:
        for encoder in (2022, 2023):
            key = f"{dataset}/{encoder}"
            after[key] = {name: sha256(path) for name, path in sources(dataset, encoder).items()}
            if after[key] != before[key]:
                raise RuntimeError(f"frozen source changed: {key}")
    save_json(OUT / "frozen_input_hashes_after.json", after)
    save_csv(OUT / "stage30_run_metrics.csv", all_metrics)
    save_csv(OUT / "stage30_paired_comparison.csv", paired)
    save_csv(OUT / "stage30_per_class.csv", all_classes)
    save_csv(OUT / "stage30_entropy_diagnostics.csv", entropy)
    save_csv(OUT / "stage30_counterfactual.csv", counterfactual)
    save_csv(OUT / "stage30_prediction_rescue.csv", rescue)
    summary, gates = [], {}
    for dataset in DATASETS:
        for method in METHODS:
            values = [r for r in all_metrics if r["dataset"] == dataset and r["method"] == method]
            summary.append({"dataset": dataset, "method": method, "units": len(values),
                            **{f"{m}_{s}": float(np.mean([r[m] for r in values]) if s == "mean" else
                               np.std([r[m] for r in values], ddof=1))
                               for m in MEASURES for s in ("mean", "std")}})
        comparison = [r for r in paired if r["dataset"] == dataset]
        deltas = np.asarray([r["delta_macro_f1"] for r in comparison])
        gate = {"mean_delta_macro_f1": float(deltas.mean()),
                "positive_units": int((deltas > 0).sum()),
                "worst_delta_macro_f1": float(deltas.min()),
                "mean_delta_accuracy": float(np.mean([r["delta_accuracy"] for r in comparison])),
                "mean_delta_weighted_f1": float(np.mean([r["delta_weighted_f1"] for r in comparison]))}
        gate["pass"] = bool(gate["mean_delta_macro_f1"] >= .005 and gate["positive_units"] >= 7 and
                            gate["worst_delta_macro_f1"] > -.02 and gate["mean_delta_accuracy"] >= 0 and
                            gate["mean_delta_weighted_f1"] >= 0)
        gates[dataset] = gate
    save_csv(OUT / "stage30_dataset_summary.csv", summary)
    historical = []
    for row in read_csv(PROJECT / "stage27_yatc_feature_level_fusion" / "run_metrics.csv"):
        if (row["role"] == "known_validation" and row["level"] == "fine" and
                row["method"] in ("F1_LinearConcat", "F2_EqualProjected", "F3_FeatureGate")):
            historical.append({"dataset": row["dataset"], "encoder_seed": int(row["seed"]),
                               "method": row["method"], "samples": int(row["samples"]),
                               **{m: float(row[m]) for m in MEASURES}})
    if len(historical) != 12:
        raise RuntimeError(f"Stage27 historical rows expected 12, got {len(historical)}")
    save_csv(OUT / "stage30_historical_stage27_reference.csv", historical)
    cf_mean = float(np.mean([r["delta_vs_forced_equal"] for r in counterfactual]))
    cf_positive = sum(r["delta_vs_forced_equal"] > 0 for r in counterfactual)
    final_gate = "PASS" if all(g["pass"] for g in gates.values()) and cf_mean > 0 and cf_positive >= 14 else "FAIL"
    check = {"status": "PASS", "run_bundles": 20, "checkpoint_replays": 40,
             "frozen_source_hashes_unchanged": True, "independent_views": 3,
             "known_test_usage": 0, "unknown_usage": 0, "encoder_updates": 0,
             "gate": final_gate, "gate_details": gates,
             "counterfactual_mean_delta_vs_equal": cf_mean,
             "counterfactual_positive_units": cf_positive,
             "batch_invariance_max_abs": max(r["batch_invariance_max_abs"] for r in counterfactual)}
    save_json(OUT / "completion_verification.json", check)
    lines = ["# Stage 30 — True three-view entropy fusion", "",
             "Known-only development experiment: frozen TrafficFormer 768-D, FIG/TAGCN 128-D, YaTC 192-D as **three independent views**. No encoder updates, Known Test or Unknown values.",
             "", "## Known Validation: matched new heads", "",
             "| Dataset | Method | Macro-F1 mean ± SD | Accuracy mean | Weighted-F1 mean |",
             "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['macro_f1_mean']:.6f} ± {row['macro_f1_std']:.6f} | "
                     f"{row['accuracy_mean']:.6f} | {row['weighted_f1_mean']:.6f} |")
    lines += ["", "## Preregistered paired gate", "",
              "| Dataset | T1−T0 mean ΔMacro-F1 | Positive / 10 | Worst Δ | ΔAccuracy | ΔWeighted-F1 | Pass |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for dataset, row in gates.items():
        lines.append(f"| {dataset} | {row['mean_delta_macro_f1']:+.6f} | {row['positive_units']}/10 | "
                     f"{row['worst_delta_macro_f1']:+.6f} | {row['mean_delta_accuracy']:+.6f} | "
                     f"{row['mean_delta_weighted_f1']:+.6f} | {row['pass']} |")
    lines += ["", "## Frozen Stage27 historical reference (not the causal control)", "",
              "| Dataset | Method | Known-Val Macro-F1 mean across 2 encoder seeds |",
              "|---|---|---:|"]
    for dataset in DATASETS:
        for method in ("F1_LinearConcat", "F2_EqualProjected", "F3_FeatureGate"):
            value = np.mean([r["macro_f1"] for r in historical if r["dataset"] == dataset and r["method"] == method])
            lines.append(f"| {dataset} | {method} | {value:.6f} |")
    lines += ["", "## Mechanism diagnostics", "",
              f"- T1−its own forced-equal head: mean ΔMacro-F1 {cf_mean:+.6f}; positive {cf_positive}/20.",
              f"- Maximum batch-size logit difference: {check['batch_invariance_max_abs']:.8g}.",
              f"- Mean maximum-view weight-collapse ratio: {np.mean([r['weight_collapse_ratio'] for r in entropy]):.6f}.",
              f"- T1 rescues/hurts T0: {sum(r['T1_rescues_T0'] for r in rescue)}/{sum(r['T1_hurts_T0'] for r in rescue)} decisions; repeated flows across head seeds.",
              "", "## Interpretation and limits", "",
              "- Three-view Gaussian adapter and λ rule are explicitly inspired by ER-CMGI, not an exact paper reproduction. Stage27 F3 uses a different softmax gate.",
              "- Five head seeds per encoder pair reuse Known-Val flows; units are not independent captures. Stage20 Test was previously exposed in earlier work, but was not opened here.",
              "- Closed-set Known-Val results do not establish open-set or external Test benefit.",
              f"- Preregistered gate: **{final_gate}**. " + ("The branch merits a separately frozen follow-up." if final_gate == "PASS" else "Do not promote or retune this entropy branch using Test/Unknown."), ""]
    (OUT / "stage30_report.md").write_text("\n".join(lines))
    root_results = ["# Stage 30 — True three-view entropy fusion", "",
                    "- Status: success; 20/20 Known-only development units complete.", "",
                    "## Data and split", "",
                    "- Frozen Stage20 Known Train/Validation, 3 independent Stage22/27 feature views. Known Test/Unknown usage 0.", "",
                    "## Configuration and execution", "",
                    "- Shared 30-epoch three-view Gaussian adapters; equal T0 versus per-flow entropy T1; at most three physical GPUs.", "",
                    "## Core results", "", f"- Preregistered gate: **{final_gate}**.",
                    f"- T1−forced-equal mean Macro-F1: {cf_mean:+.6f}; positive {cf_positive}/20.",
                    "- See `stage30_report.md` for both dataset tables and paired results.", "",
                    "## Preserved evidence", "",
                    "- 20 validated run bundles, 40 independent CPU checkpoint replays, curves, predictions, weights, counterfactuals and input hashes.", "",
                    "## Limitations", "",
                    "- Frozen encoders, not end-to-end retraining; reused validation flows; no Test/open-set claim.", "",
                    "## Conclusion and next step", "",
                    "- Stop at the preregistered Known-Val gate; do not tune on Test/Unknown.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(root_results))
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success",
                     "inputs": [{"path": "frozen_input_hashes_before.json", "role": "Stage20/22/27 Known-only sources"}],
                     "execution": {**manifest["execution"], "tmux_session": "codex_stage30_queue_20260925",
                                   "command": "run_queue.py; aggregate.py", "exit_code": 0,
                                   "environment": "fixed TrafficClassifier DGL Conda prefix",
                                   "physical_gpu_ids": [0, 1, 2]},
                     "configuration": {"files": ["stage30_three_view_entropy_fusion/EXPERIMENT_PLAN.md",
                                                 "stage30_three_view_entropy_fusion/run_one.py"],
                                       "parameters": {"encoder_seeds": [2022, 2023],
                                                      "training_seeds": list(range(2022, 2027)),
                                                      "adapter_epochs": 30, "head_epochs": 30,
                                                      "batch_size": 256, "views": 3},
                                       "seeds": list(range(2022, 2027))},
                     "core_results": [{"gate": final_gate, "units": 20,
                                       "counterfactual_mean_delta": cf_mean}],
                     "limitations": ["Known-Val development only; frozen encoders; repeated validation flows"],
                     "next_step": "stop; no Test-guided tuning"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "units": 20, "gate": final_gate,
                      "counterfactual_mean_delta": cf_mean}), flush=True)


if __name__ == "__main__":
    main()
