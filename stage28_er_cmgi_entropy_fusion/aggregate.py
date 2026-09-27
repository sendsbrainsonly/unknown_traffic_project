#!/usr/bin/env python3
"""Preregistered Known-Val aggregation and gate for all 20 Stage28 units."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from preflight import OUT, DATASETS, ENCODER_SEEDS, sha256, sources
from run_one import save_csv, save_json

TRAIN_SEEDS = tuple(range(2022, 2027))
PAIRS = (("P1_A0_entropy", "P0_A0_equal"), ("P2_A1_equal", "P0_A0_equal"),
         ("P3_A1_entropy", "P2_A1_equal"), ("P3_A1_entropy", "P1_A0_entropy"),
         ("P3_A1_entropy", "YaTC"), ("P3_A1_entropy", "F1_LinearConcat"),
         ("P3_A1_entropy", "F2_EqualProjected"))
CONTROLS = ("YaTC", "F1_LinearConcat", "F2_EqualProjected", "P2_A1_equal")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    if (OUT / "stage28_report.md").exists():
        raise RuntimeError("refusing to overwrite Stage28 aggregate")
    progress = json.loads((OUT / "queue_progress.json").read_text())
    if progress["status"] != "success":
        raise RuntimeError("queue not successful; aggregate would be incomplete")
    metrics, class_rows, paired, entropy_rows, entropy_class_rows, counterfactual_rows = [], [], [], [], [], []
    run_ids = []
    for dataset in DATASETS:
        for encoder in ENCODER_SEEDS:
            for train in TRAIN_SEEDS:
                run = OUT / "runs" / dataset / f"encoder{encoder}_train{train}"
                if not (run / "SUCCESS").exists() or not (run / "independent_replay.json").is_file():
                    raise RuntimeError(f"incomplete unit: {run}")
                replay = json.loads((run / "independent_replay.json").read_text())
                if replay["status"] != "PASS" or replay["logit_metric_replays"] != 7:
                    raise RuntimeError(f"replay failed: {run}")
                manifest = json.loads((run / "manifest.json").read_text())
                if manifest["status"] != "success":
                    raise RuntimeError(f"run bundle not success: {run}")
                run_ids.append(f"{dataset}/encoder{encoder}_train{train}")
                table = {}
                for file, method_col in (("run_metrics.csv", "variant"), ("baseline_metrics.csv", "method")):
                    for row in read_csv(run / file):
                        method = row[method_col]
                        if method in table:
                            raise RuntimeError("duplicate method")
                        normalized = {"dataset": dataset, "encoder_seed": encoder,
                                      "training_seed": train, "method": method,
                                      "samples": int(row["samples"]),
                                      "best_epoch": int(row["best_epoch"]),
                                      **{m: float(row[m]) for m in ("accuracy", "macro_f1", "weighted_f1")}}
                        metrics.append(normalized)
                        table[method] = normalized
                for newer, older in PAIRS:
                    a, b = table[newer], table[older]
                    paired.append({"dataset": dataset, "encoder_seed": encoder, "training_seed": train,
                                   "new_method": newer, "control": older,
                                   **{f"delta_{m}": a[m] - b[m]
                                      for m in ("accuracy", "macro_f1", "weighted_f1")}})
                for file, key in (("per_class.csv", "variant"), ("baseline_per_class.csv", "method")):
                    for row in read_csv(run / file):
                        class_rows.append({"dataset": dataset, "encoder_seed": encoder,
                                           "training_seed": train, "method": row[key],
                                           "class": row["class"], "support": int(row["support"]),
                                           "precision": float(row["precision"]),
                                           "recall": float(row["recall"]), "f1": float(row["f1"])})
                hrows = read_csv(run / "known_validation_entropy_weights.csv")
                for method in ("P1_A0_entropy", "P3_A1_entropy"):
                    selected = [r for r in hrows if r["variant"] == method]
                    w = np.array([[float(r["w_content"]), float(r["w_behavior"])] for r in selected])
                    h = np.array([[float(r["H_content"]), float(r["H_behavior"])] for r in selected])
                    entropy_rows.append({"dataset": dataset, "encoder_seed": encoder,
                                         "training_seed": train, "method": method,
                                         "samples": len(selected), "w_content_mean": float(w[:, 0].mean()),
                                         "w_content_std": float(w[:, 0].std()),
                                         "w_content_p01": float(np.quantile(w[:, 0], 0.01)),
                                         "w_content_p99": float(np.quantile(w[:, 0], 0.99)),
                                         "collapse_ratio": float((w.max(1) > 0.95).mean()),
                                         "H_content_std": float(h[:, 0].std()),
                                         "H_behavior_std": float(h[:, 1].std())})
                    if method == "P3_A1_entropy":
                        for service in sorted({r["true_service"] for r in selected}):
                            for correct in (0, 1):
                                group = [r for r in selected if r["true_service"] == service and
                                         int(r["correct"]) == correct]
                                if not group:
                                    continue
                                entropy_class_rows.append({"dataset": dataset, "encoder_seed": encoder,
                                                           "training_seed": train, "class": service,
                                                           "correct": correct, "samples": len(group),
                                                           "H_content_mean": float(np.mean([float(r["H_content"]) for r in group])),
                                                           "H_behavior_mean": float(np.mean([float(r["H_behavior"]) for r in group])),
                                                           "w_content_mean": float(np.mean([float(r["w_content"]) for r in group]))})
                cf = json.loads((run / "counterfactual.json").read_text())
                counterfactual_rows.append({"dataset": dataset, "encoder_seed": encoder,
                                            "training_seed": train,
                                            "P3_macro_f1": cf["main"]["macro_f1"],
                                            "forced_equal_macro_f1": cf["forced_equal"]["macro_f1"],
                                            "shuffled_entropy_macro_f1": cf["shuffled_entropy"]["macro_f1"],
                                            "delta_vs_equal": cf["main"]["macro_f1"] - cf["forced_equal"]["macro_f1"],
                                            "delta_vs_shuffle": cf["main"]["macro_f1"] - cf["shuffled_entropy"]["macro_f1"],
                                            "batch_invariance_max_abs": cf["batch_invariance_max_abs"]})
    if len(metrics) != 140 or len(paired) != 140 or len(counterfactual_rows) != 20:
        raise RuntimeError("unexpected aggregate row count")
    after = {}
    before = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    for dataset in DATASETS:
        for encoder in ENCODER_SEEDS:
            key = f"{dataset}/{encoder}"
            after[key] = {name: sha256(path) for name, path in sources(dataset, encoder).items()}
            if after[key] != before[key]:
                raise RuntimeError(f"frozen input hash changed: {key}")
    save_json(OUT / "frozen_input_hashes_after.json", after)
    save_csv(OUT / "stage28_run_metrics.csv", metrics)
    save_csv(OUT / "stage28_paired_comparison.csv", paired)
    save_csv(OUT / "stage28_per_class.csv", class_rows)
    save_csv(OUT / "stage28_entropy_diagnostics.csv", entropy_rows)
    save_csv(OUT / "stage28_entropy_by_class.csv", entropy_class_rows)
    save_csv(OUT / "stage28_counterfactual.csv", counterfactual_rows)
    summary = []
    gates = {}
    for dataset in DATASETS:
        subset = [r for r in metrics if r["dataset"] == dataset]
        for method in ("YaTC", "F1_LinearConcat", "F2_EqualProjected", "P0_A0_equal",
                       "P1_A0_entropy", "P2_A1_equal", "P3_A1_entropy"):
            records = [r for r in subset if r["method"] == method]
            summary.append({"dataset": dataset, "method": method, "units": len(records),
                            **{f"{m}_{stat}": float(getattr(np, stat)([r[m] for r in records], ddof=1)
                               if stat == "std" else np.mean([r[m] for r in records]))
                               for m in ("accuracy", "macro_f1", "weighted_f1") for stat in ("mean", "std")}})
        gates[dataset] = {}
        for control in CONTROLS:
            values = [r for r in paired if r["dataset"] == dataset and
                      r["new_method"] == "P3_A1_entropy" and r["control"] == control]
            delta = np.asarray([r["delta_macro_f1"] for r in values])
            gates[dataset][control] = {"mean_delta_macro_f1": float(delta.mean()),
                                       "positive_units": int((delta > 0).sum()),
                                       "worst_delta_macro_f1": float(delta.min()),
                                       "mean_delta_accuracy": float(np.mean([r["delta_accuracy"] for r in values])),
                                       "mean_delta_weighted_f1": float(np.mean([r["delta_weighted_f1"] for r in values])),
                                       "pass": bool(delta.mean() >= 0.005 and (delta > 0).sum() >= 7 and
                                                    delta.min() >= -0.02 and
                                                    np.mean([r["delta_accuracy"] for r in values]) >= 0 and
                                                    np.mean([r["delta_weighted_f1"] for r in values]) >= 0)}
    save_csv(OUT / "stage28_dataset_summary.csv", summary)
    class_summary = []
    for dataset in DATASETS:
        names = sorted({r["class"] for r in class_rows if r["dataset"] == dataset})
        for name in names:
            a = [r for r in class_rows if r["dataset"] == dataset and r["class"] == name and
                 r["method"] == "P3_A1_entropy"]
            b = [r for r in class_rows if r["dataset"] == dataset and r["class"] == name and
                 r["method"] == "YaTC"]
            if len(a) != 10 or len(b) != 10:
                raise RuntimeError("per-class aggregation missing units")
            class_summary.append({"dataset": dataset, "class": name,
                                  "P3_mean_f1": float(np.mean([r["f1"] for r in a])),
                                  "YaTC_mean_f1": float(np.mean([r["f1"] for r in b])),
                                  "P3_minus_YaTC_mean_f1": float(np.mean([r["f1"] - s["f1"]
                                                                         for r, s in zip(a, b, strict=True)])),
                                  "P3_mean_recall": float(np.mean([r["recall"] for r in a])),
                                  "YaTC_mean_recall": float(np.mean([r["recall"] for r in b]))})
    save_csv(OUT / "stage28_class_summary.csv", class_summary)
    cf_mean = float(np.mean([r["delta_vs_equal"] for r in counterfactual_rows]))
    cf_positive = sum(r["delta_vs_equal"] > 0 for r in counterfactual_rows)
    gate = "PASS" if all(x["pass"] for by_control in gates.values() for x in by_control.values()) and \
           cf_mean > 0 and cf_positive >= 14 else "FAIL"
    if gate == "PASS":
        conclusion = "P3 clears the preregistered Known-Val gate; one candidate may be frozen for retrospective Test review."
    else:
        conclusion = "P3 does not clear the preregistered Known-Val gate; no Stage28B diffusion or Test-based tuning."
    save_json(OUT / "completion_verification.json", {
        "status": "PASS", "run_bundles": len(run_ids), "run_ids": run_ids,
        "all_replays_pass": True, "all_source_hashes_unchanged": True,
        "test_values_loaded": 0, "unknown_values_loaded": 0,
        "paired_units_per_dataset": 10, "gate": gate,
        "gate_details": gates, "counterfactual_mean_delta_vs_equal": cf_mean,
        "counterfactual_positive_units": cf_positive,
        "max_batch_invariance_abs": max(r["batch_invariance_max_abs"] for r in counterfactual_rows),
    })
    lines = ["# Stage 28 — ER-CMGI-inspired entropy fusion: Known-only development results", "",
             "This is a mechanism adaptation, not an author-equivalent ER-CMGI reproduction.",
             "The Stage20 Test was previously exposed and was not read in this experiment; Unknown usage was 0.",
             "", "## Frozen inputs and execution", "",
             "- Exact Stage20 Known Train/Validation flow IDs; frozen E3 and YaTC representations; 4 encoder pairs × 5 new training seeds = 20 units.",
             "- Two 30-epoch Gaussian adapters (A0/A1); four 30-epoch fusion heads P0–P3. Matched F1/F2 heads were retrained at each new seed.",
             "- All scaler/PCA/entropy calibration fit Known Train; checkpoints selected only by Known Validation Macro-F1.",
             "- All 20 unit bundles passed saved-logit independent replay; frozen input hashes unchanged.",
             "", "## Closed-set Known Validation results", "",
             "| Dataset | Method | Macro-F1 mean ± SD | Accuracy mean | Weighted-F1 mean |",
             "|---|---|---:|---:|---:|" ]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['macro_f1_mean']:.6f} ± {row['macro_f1_std']:.6f} | "
                     f"{row['accuracy_mean']:.6f} | {row['weighted_f1_mean']:.6f} |")
    lines += ["", "## Preregistered paired P3 comparisons", "",
              "| Dataset | Control | Mean ΔMacro-F1 | Positive / 10 | Worst Δ | Pass |",
              "|---|---|---:|---:|---:|---|" ]
    for dataset, controls in gates.items():
        for control, row in controls.items():
            lines.append(f"| {dataset} | {control} | {row['mean_delta_macro_f1']:+.6f} | "
                         f"{row['positive_units']}/10 | {row['worst_delta_macro_f1']:+.6f} | {row['pass']} |")
    lines += ["", "## Mechanism contrasts", "",
              "| Dataset | Contrast | Mean ΔMacro-F1 | Positive / 10 |",
              "|---|---|---:|---:|"]
    for dataset in DATASETS:
        for newer, older in PAIRS[:4]:
            values = [r["delta_macro_f1"] for r in paired if r["dataset"] == dataset and
                      r["new_method"] == newer and r["control"] == older]
            lines.append(f"| {dataset} | {newer} − {older} | {np.mean(values):+.6f} | "
                         f"{sum(v > 0 for v in values)}/10 |")
    lines += ["", "## Class-conditional P3 versus YaTC", "",
              "| Dataset | Class | Δ mean F1 | P3 mean recall | YaTC mean recall |",
              "|---|---|---:|---:|---:|"]
    for row in class_summary:
        lines.append(f"| {row['dataset']} | {row['class']} | {row['P3_minus_YaTC_mean_f1']:+.6f} | "
                     f"{row['P3_mean_recall']:.6f} | {row['YaTC_mean_recall']:.6f} |")
    lines += ["", "## Entropy mechanism and counterfactual", "",
              f"- P3 minus forced-equal Macro-F1 mean: {cf_mean:+.6f}; positive units: {cf_positive}/20.",
              f"- Batch-invariance maximum logit absolute difference: {max(r['batch_invariance_max_abs'] for r in counterfactual_rows):.8g}.",
              "- With exactly two views, if λ_b+λ_s=1 (including the 0.5/0.5 initialization), the proposed high/low blend is algebraically 0.5/0.5 regardless of per-sample entropy. Learned coefficients must jointly leave this cancellation line before dynamic weights can matter. The mechanism test and equal/shuffle controls assess this explicitly.",
              "- This is a structural limitation of the preregistered two-view formula, not a post-hoc hyperparameter invitation.",
              "", "## Limitations", "",
              "- Only two datasets and two frozen encoder seeds; five head seeds share the same validation samples and are not independent dataset replicates.",
              "- Content view mixes TrafficFormer and YaTC; behavior view is FIG, unlike the paper's raw byte/packet-length pair.",
              "- No Known Test or Unknown outcome is included. No open-set or external-generalization claim follows.",
              "", "## Gate and conclusion", "",
              f"- Gate: **{gate}**. {conclusion}", ""]
    (OUT / "stage28_report.md").write_text("\n".join(lines))
    results = ["# Stage28 ER-CMGI-inspired entropy fusion", "", "- Status: success (all 20 Known-only diagnostic units complete).",
               "- Claim scope: diagnostic mechanism adaptation; not original ER-CMGI reproduction.",
               "", "## Data and split", "", "- Stage20 frozen ISCX-VPN and ISCXTor Known Train/Validation only; no Test/Unknown usage.",
               "", "## Configuration and execution", "", "- 4 encoder pairs × 5 training seeds; A0/A1 adapters and P0–P3 heads, 30 epochs each; ≤3 physical GPUs in queue.",
               "- Full configuration, hashes, logs and checkpoints retained in each run bundle.",
               "", "## Core results", "", f"- Preregistered P3 gate: **{gate}**.",
               f"- P3 minus forced-equal mean Macro-F1: {cf_mean:+.6f}.",
               "- See `stage28_report.md` and the CSV tables for all dataset, method, seed, class and paired values.",
               "", "## Preserved evidence", "", "- 20 validated run bundles, source hashes, training histories, checkpoints, predictions, logits, counterfactuals and replay files.",
               "", "## Limitations", "", "- Previously exposed Test not touched; Known-Val development evidence only.",
               "- Repeated head seeds share flow pools; not independent external trials.",
               "", "## Conclusion and next step", "", f"- {conclusion}", ""]
    (OUT / "RESULTS.md").write_text("\n".join(results))
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success",
                     "inputs": [{"path": "frozen_input_hashes_before.json", "role": "immutable Stage20/22/23/27 Known-only source audit"}],
                     "execution": {**manifest["execution"], "tmux_session": "codex_stage28_queue2_20260925",
                                   "command": "run_queue.py; aggregate.py", "exit_code": 0,
                                   "environment": "fixed TrafficClassifier DGL conda prefix",
                                   "physical_gpu_ids": [0, 1, 2]},
                     "configuration": {"files": ["STAGE28_ER_CMGI_INSPIRED_DYNAMIC_FUSION_PLAN.md",
                                                 "stage28_er_cmgi_entropy_fusion/run_one.py"],
                                       "parameters": {"datasets": list(DATASETS), "encoder_seeds": list(ENCODER_SEEDS),
                                                      "training_seeds": list(TRAIN_SEEDS), "epochs": 30,
                                                      "batch_size": 256}, "seeds": list(TRAIN_SEEDS)},
                     "core_results": [{"gate": gate, "counterfactual_mean_delta_vs_equal": cf_mean,
                                       "completed_units": len(run_ids)}],
                     "limitations": ["Known Validation development only", "Stage20 Test previously exposed but not read",
                                     "two encoder seeds and shared validation flows"],
                     "next_step": "stop Stage28; no subsequent stage automatically started"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "units": len(run_ids), "gate": gate,
                      "counterfactual_mean_delta_vs_equal": cf_mean}), flush=True)


if __name__ == "__main__":
    main()
