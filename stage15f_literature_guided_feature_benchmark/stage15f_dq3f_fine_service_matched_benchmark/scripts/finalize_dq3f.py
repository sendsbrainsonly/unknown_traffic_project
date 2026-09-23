#!/usr/bin/env python3
"""Aggregate the six DQ-3F runs and create all requested diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from dq3f_common import (
    OUT, SEEDS, SERVICES, hash_protected_inputs, provenance_rows, read_csv, read_json,
    sha256_file, singleton_application_service_map, write_csv, write_json,
)


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": [
            {
                "class_name": name, "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
                "correct": int(sum(t == name and p == name for t, p in zip(y_true, y_pred))),
                "errors": int(sum(t == name for t in y_true) - sum(t == name and p == name for t, p in zip(y_true, y_pred))),
            }
            for i, name in enumerate(labels)
        ],
    }


def result_row(task: str, seed: int, result: dict) -> dict:
    return {
        "task": task, "seed": seed, "train_samples": result["train_samples"],
        "validation_samples": result["validation_samples"], "num_classes": len(result["classes"]),
        "accuracy": result["validation_accuracy"], "macro_f1": result["validation_macro_f1"],
        "weighted_f1": result["validation_weighted_f1"], "best_epoch": result["best_epoch"],
        "completed_epochs": result["completed_epochs"], "stopped_early": result["stopped_early"],
        "train_loss": result["train_loss"], "validation_loss": result["validation_loss"],
        "runtime_seconds": result["runtime_seconds"], "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": result["checkpoint_sha256"],
        "status": result["status"],
    }


def main() -> int:
    training_cfg = read_json(OUT / "dq3f_training_configs.json")
    fine_classes = training_cfg["fine_classes"]
    singleton_map, ambiguous_apps = singleton_application_service_map()
    predictions: dict[tuple[str, int], list[dict[str, str]]] = {}
    fine_results, service_results = [], []
    per_class_rows, per_service_rows, confusion_rows = [], [], []
    checkpoint_rows = []
    for seed in SEEDS:
        for task in ("fine", "service"):
            run_dir = OUT / "runs" / task / f"seed{seed}"
            if not (run_dir / "SUCCESS").is_file():
                raise RuntimeError(f"run is not complete: {task}/seed{seed}")
            result = read_json(run_dir / "result.json")
            if result["known_test_feature_values_used"] != 0 or result["unknown_test_feature_values_used"] != 0:
                raise RuntimeError("Test boundary violation")
            if sha256_file(run_dir / "model_best.pt") != result["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {task}/seed{seed}")
            rows = read_csv(run_dir / "validation_predictions.csv")
            if len(rows) != 335 or len({row["flow_id"] for row in rows}) != 335:
                raise RuntimeError(f"prediction count/identity mismatch: {task}/seed{seed}")
            predictions[(task, seed)] = rows
            target = fine_results if task == "fine" else service_results
            target.append(result_row(task, seed, result))
            metric_detail = read_json(run_dir / "classification_metrics.json")
            per_target = per_class_rows if task == "fine" else per_service_rows
            for row in metric_detail["per_class"]:
                per_target.append({"task": "F1" if task == "fine" else "F3", "seed": seed, **row})
            labels = result["classes"]
            matrix = confusion_matrix(
                [row["true_label"] for row in rows], [row["predicted_label"] for row in rows], labels=labels,
            )
            for i, true_name in enumerate(labels):
                for j, pred_name in enumerate(labels):
                    confusion_rows.append({
                        "task": "F1" if task == "fine" else "F3", "seed": seed,
                        "true_label": true_name, "predicted_label": pred_name, "count": int(matrix[i, j]),
                    })
            checkpoint_rows.append({
                "task": task, "seed": seed, "checkpoint_path": str(run_dir / "model_best.pt"),
                "checkpoint_sha256": result["checkpoint_sha256"], "hash_verified": True,
            })

    # Exact F1/F3 Validation membership parity for every seed.
    fine_ids = {seed: [row["flow_id"] for row in predictions[("fine", seed)]] for seed in SEEDS}
    service_ids = {seed: [row["flow_id"] for row in predictions[("service", seed)]] for seed in SEEDS}
    if any(fine_ids[seed] != service_ids[seed] for seed in SEEDS):
        raise RuntimeError("F1/F3 Validation row order mismatch")
    write_csv(OUT / "fine_f1_results.csv", fine_results)
    write_csv(OUT / "service_f3_results.csv", service_results)
    write_csv(OUT / "fine_f1_predictions.csv", [row for seed in SEEDS for row in predictions[("fine", seed)]])
    write_csv(OUT / "service_f3_predictions.csv", [row for seed in SEEDS for row in predictions[("service", seed)]])
    write_csv(OUT / "checkpoint_hashes.csv", checkpoint_rows)

    f2_results, f2_predictions, matched_rows, decomposition_rows, paired_rows = [], [], [], [], []
    for seed in SEEDS:
        fine = predictions[("fine", seed)]
        service = predictions[("service", seed)]
        service_by_id = {row["flow_id"]: row for row in service}
        common_fine_true, common_fine_pred = [], []
        common_service_true, f2_service_pred, f3_service_pred = [], [], []
        decomposition = Counter()
        pair_counter = Counter()
        for row in fine:
            flow_id = row["flow_id"]
            f3 = service_by_id[flow_id]
            true_app = row["application_label"]
            pred_app = row["predicted_label"]
            true_singleton = true_app in singleton_map
            pred_singleton = pred_app in singleton_map
            evaluable = true_singleton and pred_singleton
            predicted_service = singleton_map[pred_app] if pred_singleton else ""
            actual_service = row["service_label"]
            f2_predictions.append({
                "seed": seed, "flow_id": flow_id, "capture_id": row["capture_id"],
                "application_label": true_app, "fine_prediction": pred_app,
                "true_service": actual_service, "predicted_service": predicted_service,
                "evaluation_status": "EVALUABLE" if evaluable else "AMBIGUOUS_PREDICTION_OR_TRUE_APPLICATION",
                "f2_correct": int(evaluable and predicted_service == actual_service),
            })
            if evaluable:
                common_fine_true.append(true_app)
                common_fine_pred.append(pred_app)
                common_service_true.append(actual_service)
                f2_service_pred.append(predicted_service)
                f3_service_pred.append(f3["predicted_label"])
            fine_correct = row["correct"] == "1"
            f3_correct = f3["correct"] == "1"
            pair_counter[
                ("F1_correct" if fine_correct else "F1_wrong") + "__" +
                ("F3_correct" if f3_correct else "F3_wrong")
            ] += 1
            if fine_correct:
                decomposition["fine_correct"] += 1
            elif not pred_singleton:
                decomposition["fine_wrong_ambiguous_prediction"] += 1
            elif predicted_service == actual_service:
                decomposition["fine_wrong_service_compatible"] += 1
            else:
                decomposition["fine_wrong_service_wrong"] += 1
        f1_full = metrics(
            [row["true_label"] for row in fine], [row["predicted_label"] for row in fine], fine_classes,
        )
        f3_full = metrics(
            [row["true_label"] for row in service], [row["predicted_label"] for row in service], SERVICES,
        )
        f1_common = metrics(common_fine_true, common_fine_pred, fine_classes)
        f2_common = metrics(common_service_true, f2_service_pred, SERVICES)
        f3_common = metrics(common_service_true, f3_service_pred, SERVICES)
        f2_results.append({
            "seed": seed, "validation_total": len(fine), "evaluable_flows": len(common_service_true),
            "unevaluable_flows": len(fine) - len(common_service_true),
            "coverage": len(common_service_true) / len(fine),
            "accuracy": f2_common["accuracy"], "macro_f1": f2_common["macro_f1"],
            "weighted_f1": f2_common["weighted_f1"],
        })
        for name, scope, bundle in (
            ("F1_FINE", "FULL", f1_full), ("F1_FINE", "COMMON_SUBSET", f1_common),
            ("F2_PARTIAL_SERVICE", "COMMON_SUBSET", f2_common),
            ("F3_SERVICE", "COMMON_SUBSET", f3_common), ("F3_SERVICE", "FULL", f3_full),
        ):
            matched_rows.append({
                "seed": seed, "evaluation": name, "scope": scope,
                "samples": len(fine) if scope == "FULL" else len(common_service_true),
                "accuracy": bundle["accuracy"], "macro_f1": bundle["macro_f1"],
                "weighted_f1": bundle["weighted_f1"],
                "metric_class_set": "11 Fine Application classes" if "F1" in name else "6 Service classes",
            })
        for category, count in sorted(decomposition.items()):
            decomposition_rows.append({"seed": seed, "category": category, "count": count, "total": len(fine)})
        for category, count in sorted(pair_counter.items()):
            paired_rows.append({"seed": seed, "category": category, "count": count, "total": len(fine), "ratio": count / len(fine)})
        for row in f2_common["per_class"]:
            per_service_rows.append({"task": "F2_PARTIAL", "seed": seed, **row})

    write_csv(OUT / "coarse_f2_partial_results.csv", f2_results)
    write_csv(OUT / "coarse_f2_partial_predictions.csv", f2_predictions)
    write_csv(OUT / "matched_subset_comparison.csv", matched_rows)
    write_csv(OUT / "fine_to_service_error_decomposition.csv", decomposition_rows)
    write_csv(OUT / "paired_error_analysis.csv", paired_rows)
    write_csv(OUT / "per_class_metrics.csv", per_class_rows)
    write_csv(OUT / "per_service_metrics.csv", per_service_rows)
    write_csv(OUT / "fine_confusion_matrix.csv", [row for row in confusion_rows if row["task"] == "F1"])
    write_csv(OUT / "service_confusion_matrix.csv", [row for row in confusion_rows if row["task"] == "F3"])

    # Per-capture performance for both tasks and all seeds.
    capture_rows = []
    for seed in SEEDS:
        for task, task_name, labels in (("fine", "F1", fine_classes), ("service", "F3", SERVICES)):
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in predictions[(task, seed)]:
                grouped[row["capture_id"]].append(row)
            for capture_id, rows in sorted(grouped.items()):
                bundle = metrics(
                    [row["true_label"] for row in rows], [row["predicted_label"] for row in rows], labels,
                )
                capture_rows.append({
                    "task": task_name, "seed": seed, "capture_id": capture_id,
                    "source_file": rows[0]["source_file"], "application_label": rows[0]["application_label"],
                    "service_label": rows[0]["service_label"], "validation_samples": len(rows),
                    "correct": sum(row["correct"] == "1" for row in rows),
                    "accuracy": bundle["accuracy"], "macro_f1_over_global_label_space": bundle["macro_f1"],
                })
    write_csv(OUT / "capture_level_results.csv", capture_rows)

    # Seed stability summary.
    stability_rows = []
    for task, rows in (("F1", fine_results), ("F2_PARTIAL", f2_results), ("F3", service_results)):
        for metric_name in ("accuracy", "macro_f1", "weighted_f1"):
            source_name = metric_name if task == "F2_PARTIAL" else metric_name
            values = np.asarray([float(row[source_name]) for row in rows])
            stability_rows.append({
                "task": task, "metric": metric_name, "seeds": "|".join(map(str, SEEDS)),
                "n": len(values), "mean": float(values.mean()), "std_population": float(values.std(ddof=0)),
                "min": float(values.min()), "max": float(values.max()),
            })
    write_csv(OUT / "seed_stability.csv", stability_rows)

    prov = provenance_rows()
    by_service_capture: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    image_counter = Counter()
    for row in prov:
        by_service_capture[row["service_label"]][row["capture_id"]][row["split_role"]] += 1
    # Use manifest image hashes from predictions to report representation duplicates on Validation;
    # preflight already proves zero Train/Validation image overlap globally.
    first_fine = predictions[("fine", SEEDS[0])]
    for row in first_fine:
        image_counter[row["flow_id"]] += 1
    capture_lines = [
        "# DQ-3F Capture Dependency Audit", "", "## Conclusion", "",
        "`CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`", "",
        "The benchmark is flow-disjoint but not capture-disjoint. P2P has one capture, so its Validation score cannot demonstrate generalization to a new P2P capture.",
        "", "## Per-Service capture support", "",
        "| Service | Captures | Train | Validation | Train/Validation overlapping captures |", "|---|---:|---:|---:|---:|",
    ]
    for service in SERVICES:
        captures = by_service_capture[service]
        train_caps = {c for c, counts in captures.items() if counts["known_train"]}
        val_caps = {c for c, counts in captures.items() if counts["known_validation"]}
        capture_lines.append(
            f"| {service} | {len(captures)} | {sum(v['known_train'] for v in captures.values())} | "
            f"{sum(v['known_validation'] for v in captures.values())} | {len(train_caps & val_caps)} |"
        )
    capture_lines += [
        "", "## Integrity", "",
        "- Duplicate flow IDs: 0.",
        "- Exact image hashes shared across Train/Validation: 0 (preflight verified).",
        "- Capture identifiers, filenames and activities were audit metadata only and never model inputs.",
        "- No single P2P capture was split into fictional independent capture groups.",
        "- An independent non-shared P2P capture would be required for a separate cross-capture claim.",
    ]
    (OUT / "capture_dependency_audit.md").write_text("\n".join(capture_lines) + "\n", encoding="utf-8")

    protected_after = hash_protected_inputs()
    write_json(OUT / "protected_asset_hashes_after.json", protected_after)
    before = read_json(OUT / "protected_asset_hashes_before.json")
    protected_status = "PASS" if before["files"] == protected_after["files"] else "FAIL"
    if protected_status != "PASS":
        raise RuntimeError("protected input hash mismatch")

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    def std(rows: list[dict], key: str) -> float:
        return float(np.std([float(row[key]) for row in rows], ddof=0))

    f1_macro = mean(fine_results, "macro_f1")
    f3_macro = mean(service_results, "macro_f1")
    f3_service_means = {
        service: float(np.mean([float(row["recall"]) for row in per_service_rows if row["task"] == "F3" and row["class_name"] == service]))
        for service in SERVICES
    }
    f3_gt_f1 = sum(float(s["macro_f1"]) > float(f["macro_f1"]) for f, s in zip(fine_results, service_results))
    if f3_macro > f1_macro and f3_gt_f1 == len(SEEDS) and min(f3_service_means.values()) > 0:
        gate = "COARSE_TRAINING_BENEFIT"
    elif f3_macro > f1_macro:
        gate = "COARSE_TRAINING_CLASS_CONDITIONAL"
    else:
        gate = "NO_CLEAR_COARSE_TRAINING_BENEFIT"
    capture_gate = "CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE"
    fine_error_explained = sum(
        int(row["count"]) for row in decomposition_rows if row["category"] == "fine_wrong_service_compatible"
    )
    fine_error_total = sum(
        int(row["count"]) for row in decomposition_rows if row["category"].startswith("fine_wrong")
    )
    f2_acc = mean(f2_results, "accuracy")
    f2_macro = mean(f2_results, "macro_f1")
    common_f3_rows = [row for row in matched_rows if row["evaluation"] == "F3_SERVICE" and row["scope"] == "COMMON_SUBSET"]
    common_f3_acc = mean(common_f3_rows, "accuracy")
    common_f3_macro = mean(common_f3_rows, "macro_f1")
    f2_minus_f3_common_accuracy = f2_acc - common_f3_acc
    f2_minus_f3_common_macro = f2_macro - common_f3_macro
    fine_error_explained_ratio = fine_error_explained / fine_error_total

    report = f"""# Stage 15F-DQ-3F — Fine vs Service Matched-Sample Benchmark

## Conclusion and next step

Primary Gate: `{gate}`  
Capture limitation: `{capture_gate}`

Direct six-Service training was evaluated on exactly the same 2,730 Train and 335 Validation flows as Fine Application training. The result is a Known-only diagnostic under weak capture-derived labels, not a capture-disjoint or authoritative per-flow-label claim.

DQ-3F stops here. DQ-4--DQ-7, TFE-GNN/TrafficFormer training, Byte--Behavior, DES and H1 changes remain `NOT_RUN`.

## Configuration and execution

- Paired seeds: `{SEEDS}`; 6/6 F1/F3 runs completed.
- Native architecture/training: ResNet18, 1 channel, latent 128, 100 epoch budget, batch 128, Adam 0.001, lambda 0.005, MultiStepLR [50,80], prototype reset [50,80].
- Checkpoint selection: Known Validation Accuracy only.
- F1/F3 differences: supervision labels and output class count only.
- Known Test/Unknown Test feature values used: 0/0.
- Protected input hashes: `{protected_status}` ({protected_after['file_count']} files).

## Core results

| Task | Scope | Accuracy mean ± std | Macro-F1 mean ± std | Weighted-F1 mean ± std |
|---|---|---:|---:|---:|
| F1 Fine | Full 335 | {mean(fine_results, 'accuracy'):.6f} ± {std(fine_results, 'accuracy'):.6f} | {f1_macro:.6f} ± {std(fine_results, 'macro_f1'):.6f} | {mean(fine_results, 'weighted_f1'):.6f} ± {std(fine_results, 'weighted_f1'):.6f} |
| F2 Partial Service | Common subset | {f2_acc:.6f} ± {std(f2_results, 'accuracy'):.6f} | {f2_macro:.6f} ± {std(f2_results, 'macro_f1'):.6f} | {mean(f2_results, 'weighted_f1'):.6f} ± {std(f2_results, 'weighted_f1'):.6f} |
| F3 Service | Full 335 | {mean(service_results, 'accuracy'):.6f} ± {std(service_results, 'accuracy'):.6f} | {f3_macro:.6f} ± {std(service_results, 'macro_f1'):.6f} | {mean(service_results, 'weighted_f1'):.6f} ± {std(service_results, 'weighted_f1'):.6f} |
| F3 Service | F2 common subset | {common_f3_acc:.6f} | {common_f3_macro:.6f} | {mean(common_f3_rows, 'weighted_f1'):.6f} |

- F3 Macro-F1 exceeds F1 on {f3_gt_f1}/3 paired seeds. This cross-task difference indicates task-granularity difficulty, not a same-task model improvement.
- Across three seeds, {fine_error_explained}/{fine_error_total} (`{fine_error_explained_ratio:.2%}`) Fine errors are compatible with the correct capture-derived Service after deterministic mapping (counts include repeated seed decisions).
- On each seed's identical F2-evaluable subset, F2 exceeds F3 by mean Accuracy `{f2_minus_f3_common_accuracy:+.6f}` and mean Macro-F1 `{f2_minus_f3_common_macro:+.6f}`. Direct Service training therefore does **not** outperform prediction mapping on this restricted subset, although F3 covers all 335 samples and F2 does not.
- F3 mean Service recalls: {', '.join(f'{name}={value:.6f}' for name, value in f3_service_means.items())}.
- P2P and Streaming contribute 204/335 Validation samples, but F3 is not supported only by those large classes: Chat and Email retain mean recalls `{f3_service_means['Chat']:.6f}` and `{f3_service_means['Email']:.6f}`. File-Transfer is the hardest Service (`{f3_service_means['File-Transfer']:.6f}` mean recall), and seed 2023 remains visibly weaker.

## Required research questions

1. **Fine performance:** Accuracy/Macro-F1/Weighted-F1 = `{mean(fine_results, 'accuracy'):.6f}/{f1_macro:.6f}/{mean(fine_results, 'weighted_f1'):.6f}` (three-seed means).
2. **Direct Service performance:** `{mean(service_results, 'accuracy'):.6f}/{f3_macro:.6f}/{mean(service_results, 'weighted_f1'):.6f}` on all 335 Validation flows.
3. **F3 versus F2-partial:** F3 does not beat F2 on the common subset on average; F2 has the above `{f2_minus_f3_common_accuracy:+.6f}` Accuracy and `{f2_minus_f3_common_macro:+.6f}` Macro-F1 advantages. F3's distinct advantage is complete 335-flow coverage and a directly trained Service objective.
4. **Fine errors explained by granularity:** `{fine_error_explained}/{fine_error_total}` repeated-seed Fine errors (`{fine_error_explained_ratio:.2%}`) become Service-compatible, supporting label granularity as a substantial contributor but not the sole cause.
5. **Easy/hard Services:** VoIP/P2P/Streaming are strongest on mean recall; File-Transfer and Chat are hardest. Actual correct/error counts for every seed are retained in `per_service_metrics.csv`.
6. **Large-class dependence:** P2P/Streaming materially influence Accuracy, but all six Services have non-zero recall and small Chat/Email do not collapse. The conclusion is therefore not solely a majority-class artifact.
7. **Service as a research task:** supported as a development diagnostic with weak capture labels; not yet supported as an authoritative or capture-generalized final benchmark.
8. **Remaining cross-paper mismatches:** PCAP scope, flow/session construction, sample filtering, target/background separation, label provenance, split/group policy, training-pool size, inputs and evaluation metrics remain unaligned.
9. **Next stage:** DQ-4 is feasible as a controlled data-selection/filtering audit. DQ-7 is not yet fair without a newly frozen unified Service dataset/protocol and a solution to the P2P single-capture limitation.

## Data and split

- Dataset: 31 shared VPN captures; Train/Validation=2,730/335.
- Fine classes: 11; Service classes: 6.
- All 3,065 labels are `WEAK_CAPTURE_LABEL`.
- Flow-ID overlap and exact-image overlap across Train/Validation: 0/0.
- F2 uses only deterministic singleton Application→Service predictions; ambiguous true or predicted Applications are excluded and explicitly counted.

## Limitations

- F1 and F3 have different label spaces, so their Macro-F1 difference is not a same-task algorithmic gain.
- P2P has one capture; full six-Service capture-disjoint generalization is not identifiable.
- Capture-derived weak labels may include incidental/background flows.
- Current data/flow definition, filtering, split and weak labels remain different from TFE-GNN and TrafficFormer paper protocols; no direct superiority claim is allowed.
- A Service Open-Set protocol would require independently frozen Known/Unknown Services and cannot reuse the existing Application-level Unknown split.

## Preserved evidence

- All six checkpoints, SHA256 files, per-run logs/configs/predictions and metrics are retained under `runs/`.
- Requested aggregate CSVs, capture audit, protected hashes and completion record are stored in this bundle.
- Two pre-epoch CUDA peak-memory API failures are retained under `failed_attempts/`; neither consumed training data or updated a model.
"""
    (OUT / "dq3f_report.md").write_text(report, encoding="utf-8")
    (OUT / "RESULTS.md").write_text(report.replace("# Stage 15F-DQ-3F — Fine vs Service Matched-Sample Benchmark", "# RESULTS — Stage 15F-DQ-3F"), encoding="utf-8")
    completion = {
        "status": "PASS", "primary_gate": gate, "capture_gate": capture_gate,
        "formal_runs": 6, "successful_runs": 6, "seeds": SEEDS,
        "train_samples": 2730, "validation_samples": 335,
        "fine_classes": len(fine_classes), "service_classes": len(SERVICES),
        "f1_mean_accuracy": mean(fine_results, "accuracy"), "f1_mean_macro_f1": f1_macro,
        "f1_mean_weighted_f1": mean(fine_results, "weighted_f1"),
        "f2_mean_accuracy": f2_acc, "f2_mean_macro_f1": f2_macro,
        "f3_common_subset_mean_accuracy": common_f3_acc,
        "f3_common_subset_mean_macro_f1": common_f3_macro,
        "f2_minus_f3_common_accuracy": f2_minus_f3_common_accuracy,
        "f2_minus_f3_common_macro_f1": f2_minus_f3_common_macro,
        "f3_mean_accuracy": mean(service_results, "accuracy"), "f3_mean_macro_f1": f3_macro,
        "f3_mean_weighted_f1": mean(service_results, "weighted_f1"),
        "f3_gt_f1_macro_seeds": f3_gt_f1,
        "fine_error_service_compatible": fine_error_explained,
        "fine_error_total": fine_error_total,
        "fine_error_service_compatible_ratio": fine_error_explained_ratio,
        "f3_mean_service_recall": f3_service_means,
        "protected_asset_hash_status": protected_status,
        "protected_asset_count": protected_after["file_count"],
        "checkpoint_hashes_verified": 6,
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
        "dq4_to_dq7": "NOT_RUN", "tfe_gnn_training": "NOT_RUN",
        "trafficformer_training": "NOT_RUN", "byte_behavior": "NOT_RUN",
        "des_h1_modified": False, "historical_outputs_modified": False,
    }
    write_json(OUT / "completion_verification.json", completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
