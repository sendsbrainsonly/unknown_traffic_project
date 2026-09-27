#!/usr/bin/env python3
"""Preregistered paired Known-Val summary for the E3-only Stage29 experiment."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

from preflight import OUT, DATASETS, ENCODER_SEEDS, paths, sha
from run_one import write_csv, write_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    if (OUT / "stage29_report.md").exists():
        raise RuntimeError("aggregate already exists")
    progress = json.loads((OUT / "queue_progress.json").read_text())
    if progress["status"] != "success":
        raise RuntimeError("queue is not complete")
    all_metrics, all_classes, paired, entropy, counterfactual, rescue = [], [], [], [], [], []
    runs = []
    for dataset in DATASETS:
        for encoder in ENCODER_SEEDS:
            for train in range(2022, 2027):
                run = OUT / "runs" / dataset / f"encoder{encoder}_train{train}"
                if not (run / "SUCCESS").is_file() or not (run / "independent_replay.json").is_file():
                    raise RuntimeError(f"incomplete run: {run}")
                verification = json.loads((run / "independent_replay.json").read_text())
                if verification["status"] != "PASS" or len(verification["checkpoint_replays"]) != 2 or \
                   any(verification[k] for k in ("yatc_usage", "known_test_usage", "unknown_usage")):
                    raise RuntimeError(f"failed replay or forbidden use: {run}")
                if json.loads((run / "manifest.json").read_text())["status"] != "success":
                    raise RuntimeError(f"run bundle not validated: {run}")
                runs.append(f"{dataset}/encoder{encoder}_train{train}")
                table = {}
                for row in read_csv(run / "run_metrics.csv"):
                    value = {"dataset": dataset, "encoder_seed": encoder, "training_seed": train,
                             "method": row["method"], "best_epoch": int(row["best_epoch"]),
                             "samples": int(row["samples"]),
                             **{m: float(row[m]) for m in ("accuracy", "macro_f1", "weighted_f1")}}
                    all_metrics.append(value)
                    table[value["method"]] = value
                if set(table) != {"E3_original", "N0_equal", "N1_shared_lambda_entropy"}:
                    raise RuntimeError("missing method")
                for older in ("N0_equal", "E3_original"):
                    a, b = table["N1_shared_lambda_entropy"], table[older]
                    paired.append({"dataset": dataset, "encoder_seed": encoder,
                                   "training_seed": train, "control": older,
                                   **{f"delta_{m}": a[m] - b[m]
                                      for m in ("accuracy", "macro_f1", "weighted_f1")}})
                for row in read_csv(run / "per_class.csv"):
                    all_classes.append({"dataset": dataset, "encoder_seed": encoder,
                                        "training_seed": train, "method": row["method"],
                                        "class": row["class"], "support": int(row["support"]),
                                        "precision": float(row["precision"]),
                                        "recall": float(row["recall"]), "f1": float(row["f1"])})
                hrows = [r for r in read_csv(run / "known_validation_entropy_weights.csv")
                         if r["method"] == "N1_shared_lambda_entropy"]
                w = np.asarray([float(r["w_trafficformer"]) for r in hrows])
                h = np.asarray([[float(r["H_trafficformer"]), float(r["H_graph"])] for r in hrows])
                curves = [r for r in read_csv(run / "training_history.csv")
                          if r["phase"] == "N1_shared_lambda_entropy"]
                epoch = table["N1_shared_lambda_entropy"]["best_epoch"]
                lam = float(next(r for r in curves if int(r["epoch"]) == epoch)["shared_lambda"])
                entropy.append({"dataset": dataset, "encoder_seed": encoder,
                                "training_seed": train, "samples": len(hrows), "best_lambda": lam,
                                "w_trafficformer_mean": float(w.mean()),
                                "w_trafficformer_std": float(w.std()),
                                "w_trafficformer_p01": float(np.quantile(w, .01)),
                                "w_trafficformer_p99": float(np.quantile(w, .99)),
                                "weight_collapse_ratio": float(((w > .95) | (w < .05)).mean()),
                                "H_trafficformer_std": float(h[:, 0].std()),
                                "H_graph_std": float(h[:, 1].std())})
                cf = json.loads((run / "counterfactual.json").read_text())
                counterfactual.append({"dataset": dataset, "encoder_seed": encoder,
                                        "training_seed": train,
                                        "N1_macro_f1": cf["main"]["macro_f1"],
                                        "forced_equal_macro_f1": cf["forced_equal"]["macro_f1"],
                                        "shuffled_entropy_macro_f1": cf["shuffled_entropy"]["macro_f1"],
                                        "delta_vs_forced_equal": cf["main"]["macro_f1"]-cf["forced_equal"]["macro_f1"],
                                        "delta_vs_shuffled_entropy": cf["main"]["macro_f1"]-cf["shuffled_entropy"]["macro_f1"],
                                        "batch_invariance_max_abs": cf["max_batch_abs_logit_difference"]})
                pred = read_csv(run / "known_validation_predictions.csv")
                by_method = {method: {r["flow_id"]: int(r["correct"]) for r in pred if r["method"] == method}
                             for method in ("N0_equal", "N1_shared_lambda_entropy")}
                if set(by_method["N0_equal"]) != set(by_method["N1_shared_lambda_entropy"]):
                    raise RuntimeError("prediction flow mismatch")
                rescue.append({"dataset": dataset, "encoder_seed": encoder, "training_seed": train,
                               "both_correct": sum(a == b == 1 for a, b in zip(*[list(by_method[m].values())
                                                for m in ("N0_equal", "N1_shared_lambda_entropy")], strict=True)),
                               "both_wrong": sum(a == b == 0 for a, b in zip(*[list(by_method[m].values())
                                              for m in ("N0_equal", "N1_shared_lambda_entropy")], strict=True)),
                               "N1_rescues_N0": sum(by_method["N0_equal"][fid] == 0 and
                                                    by_method["N1_shared_lambda_entropy"][fid] == 1
                                                    for fid in by_method["N0_equal"]),
                               "N1_hurts_N0": sum(by_method["N0_equal"][fid] == 1 and
                                                  by_method["N1_shared_lambda_entropy"][fid] == 0
                                                  for fid in by_method["N0_equal"])})
    if len(runs) != 20 or len(all_metrics) != 60 or len(paired) != 40:
        raise RuntimeError("aggregate count mismatch")
    before = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    after = {}
    for dataset in DATASETS:
        for encoder in ENCODER_SEEDS:
            key = f"{dataset}/{encoder}"
            after[key] = {name: sha(path) for name, path in paths(dataset, encoder).items()}
            if before[key] != after[key]:
                raise RuntimeError(f"frozen source changed: {key}")
    write_json(OUT / "frozen_input_hashes_after.json", after)
    write_csv(OUT / "stage29_run_metrics.csv", all_metrics)
    write_csv(OUT / "stage29_paired_comparison.csv", paired)
    write_csv(OUT / "stage29_per_class.csv", all_classes)
    write_csv(OUT / "stage29_entropy_diagnostics.csv", entropy)
    write_csv(OUT / "stage29_counterfactual.csv", counterfactual)
    write_csv(OUT / "stage29_prediction_rescue.csv", rescue)
    summary = []
    gate_details = {}
    for dataset in DATASETS:
        for method in ("E3_original", "N0_equal", "N1_shared_lambda_entropy"):
            values = [r for r in all_metrics if r["dataset"] == dataset and r["method"] == method]
            summary.append({"dataset": dataset, "method": method, "units": len(values),
                            **{f"{m}_{s}": float(np.mean([r[m] for r in values]) if s == "mean" else
                               np.std([r[m] for r in values], ddof=1))
                               for m in ("accuracy", "macro_f1", "weighted_f1") for s in ("mean", "std")}})
        comparison = [r for r in paired if r["dataset"] == dataset and r["control"] == "N0_equal"]
        deltas = np.asarray([r["delta_macro_f1"] for r in comparison])
        vs_e3 = [r["delta_macro_f1"] for r in paired if r["dataset"] == dataset and r["control"] == "E3_original"]
        result = {"mean_delta_N1_minus_N0": float(deltas.mean()),
                  "positive_units": int((deltas > 0).sum()),
                  "worst_delta": float(deltas.min()),
                  "mean_delta_accuracy": float(np.mean([r["delta_accuracy"] for r in comparison])),
                  "mean_delta_weighted_f1": float(np.mean([r["delta_weighted_f1"] for r in comparison])),
                  "mean_delta_N1_minus_E3": float(np.mean(vs_e3))}
        result["pass"] = bool(result["mean_delta_N1_minus_N0"] >= .005 and
                              result["positive_units"] >= 7 and result["worst_delta"] > -.02 and
                              result["mean_delta_accuracy"] >= 0 and result["mean_delta_weighted_f1"] >= 0 and
                              result["mean_delta_N1_minus_E3"] >= 0)
        gate_details[dataset] = result
    write_csv(OUT / "stage29_dataset_summary.csv", summary)
    cf_mean = float(np.mean([r["delta_vs_forced_equal"] for r in counterfactual]))
    cf_positive = sum(r["delta_vs_forced_equal"] > 0 for r in counterfactual)
    gate = "PASS" if all(x["pass"] for x in gate_details.values()) and cf_mean > 0 and cf_positive >= 14 else "FAIL"
    check = {"status": "PASS", "run_bundles": 20, "checkpoint_replays": 40,
             "frozen_source_hashes_unchanged": True, "native_branches": 2,
             "yatc_usage": 0, "known_test_usage": 0, "unknown_usage": 0,
             "gate": gate, "gate_details": gate_details,
             "counterfactual_mean_delta_vs_equal": cf_mean,
             "counterfactual_positive_units": cf_positive,
             "batch_invariance_max_abs": max(r["batch_invariance_max_abs"] for r in counterfactual)}
    write_json(OUT / "completion_verification.json", check)
    lines = ["# Stage 29 — Native E3-only entropy fusion", "",
             "This is a Known-only development diagnostic on the original two E3 encoders, not YaTC or end-to-end E3 retraining.",
             "", "## Scope and provenance", "",
             "- Stage22 TrafficFormer 768-D + FIG/TAGCN 128-D, exact Stage20 Known Train/Validation Service flows.",
             "- Frozen encoders and original E3 head; 20 paired new-head units. No YaTC, Known Test or Unknown values were loaded.",
             "- Entropy fusion has a single shared λ so its two-view high/low formula can move off the Stage28 cancellation line.",
             "- All 40 N0/N1 checkpoints independently replayed on CPU; frozen source hashes unchanged.",
             "", "## Known Validation results", "",
             "| Dataset | Method | Macro-F1 mean ± SD | Accuracy mean | Weighted-F1 mean |",
             "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['macro_f1_mean']:.6f} ± {row['macro_f1_std']:.6f} | "
                     f"{row['accuracy_mean']:.6f} | {row['weighted_f1_mean']:.6f} |")
    lines += ["", "## Paired dynamic-fusion gate", "",
              "| Dataset | N1−N0 mean ΔMacro-F1 | Positive / 10 | Worst Δ | N1−original E3 | Pass |",
              "|---|---:|---:|---:|---:|---|"]
    for dataset, row in gate_details.items():
        lines.append(f"| {dataset} | {row['mean_delta_N1_minus_N0']:+.6f} | {row['positive_units']}/10 | "
                     f"{row['worst_delta']:+.6f} | {row['mean_delta_N1_minus_E3']:+.6f} | {row['pass']} |")
    lines += ["", "## Mechanism checks", "",
              f"- N1−forced-equal mean Macro-F1: {cf_mean:+.6f}; positive in {cf_positive}/20 paired units.",
              f"- Batch-invariance maximum absolute logit difference: {check['batch_invariance_max_abs']:.8g}.",
              f"- Learned λ mean across 20 units: {np.mean([r['best_lambda'] for r in entropy]):.6f}; mean TrafficFormer weight: {np.mean([r['w_trafficformer_mean'] for r in entropy]):.6f}.",
              f"- Sample-level N1 rescues/hurts N0: {sum(r['N1_rescues_N0'] for r in rescue)}/{sum(r['N1_hurts_N0'] for r in rescue)} decisions (head seeds reuse the same flows).",
              "", "## Limitations and conclusion", "",
              "- The Stage20 Test was previously exposed in unrelated earlier work but was not opened here; these numbers are Known-Val development evidence only.",
              "- Two encoder seeds and repeated head seeds share validation flows; do not treat 10 units per dataset as 10 independent captures.",
              "- Shared-λ is a documented correction to the two-view algebra, not the paper's exact dynamic-fusion implementation.",
              f"- Preregistered gate: **{gate}**. " + ("N1 merits a separately frozen follow-up." if gate == "PASS" else
                                                   "Do not promote N1 or tune using Test; stop the entropy-fusion route under this protocol."), ""]
    (OUT / "stage29_report.md").write_text("\n".join(lines))
    root_results = ["# Stage 29 native E3-only entropy fusion", "", "- Status: success; 20/20 Known-only development units complete.",
                    "", "## Data and split", "", "- Frozen Stage20 Known Train/Validation; E3 TrafficFormer and FIG/TAGCN only. YaTC/Test/Unknown usage 0.",
                    "", "## Configuration and execution", "", "- Same 30-epoch Gaussian adapters for N0/N1; shared-λ dynamic versus equal-weight control; at most three physical GPUs.",
                    "", "## Core results", "", f"- Preregistered gate: **{gate}**.",
                    f"- N1−forced-equal average Macro-F1: {cf_mean:+.6f}.",
                    "- See `stage29_report.md` for both dataset tables and paired counts.",
                    "", "## Preserved evidence", "", "- 20 validated unit bundles, 40 CPU checkpoint replays, curves, predictions, weights, counterfactuals and source hashes.",
                    "", "## Limitations", "", "- Frozen encoders, not end-to-end retraining; repeated validation flows; no untouched Test or Unknown claim.",
                    "", "## Conclusion and next step", "", "- Stop at the preregistered Known-Val gate; no next stage automatically launched.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(root_results))
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success",
                     "inputs": [{"path": "frozen_input_hashes_before.json", "role": "Stage20/22 Known-only sources"}],
                     "execution": {**manifest["execution"], "tmux_session": "codex_stage29_queue_20260925",
                                   "command": "run_queue.py; aggregate.py", "exit_code": 0,
                                   "environment": "fixed TrafficClassifier DGL conda prefix",
                                   "physical_gpu_ids": [0, 1, 2]},
                     "configuration": {"files": ["stage29_e3_native_entropy_fusion/EXPERIMENT_PLAN.md",
                                                 "stage29_e3_native_entropy_fusion/run_one.py"],
                                       "parameters": {"encoder_seeds": [2022, 2023],
                                                      "training_seeds": list(range(2022, 2027)),
                                                      "adapter_epochs": 30, "head_epochs": 30,
                                                      "batch_size": 256}, "seeds": list(range(2022, 2027))},
                     "core_results": [{"gate": gate, "units": 20,
                                       "counterfactual_mean_delta": cf_mean}],
                     "limitations": ["Known-Val development only; frozen encoders and repeated validation flows"],
                     "next_step": "stop; no Test-guided tuning"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "units": 20, "gate": gate,
                      "counterfactual_mean_delta": cf_mean}), flush=True)


if __name__ == "__main__":
    main()
