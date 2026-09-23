#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support

from stage22_common import CONFIG, OUT, PROJECT, RUN_ROOT, read_json, write_csv, write_json


DATASETS = ("iscx_vpn", "iscx_tor")
ENCODERS = ("E1", "E2", "E3")


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray):
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "worst_class_f1": float(per_class[2].min()),
        "best_class_f1": float(per_class[2].max()),
    }, per_class


def main() -> None:
    config = read_json(CONFIG)
    seeds = [int(seed) for seed in config["seeds"]]
    run_rows, class_rows, confusion_rows, gap_rows = [], [], [], []
    for dataset in DATASETS:
        services = config["datasets"][dataset]["services"]
        service_to_id = {name: index for index, name in enumerate(services)}
        label_ids = np.arange(len(services))
        for seed in seeds:
            run = RUN_ROOT / dataset / f"seed{seed}"
            predictions = load_csv(run / "predictions.csv")
            base_results = {row["encoder"]: row for row in load_csv(run / "results.csv")}
            for encoder in ENCODERS:
                rows = [row for row in predictions if row["encoder"] == encoder and row["role"] == "known_test"]
                y_true = np.asarray([service_to_id[row["true_service"]] for row in rows])
                y_pred = np.asarray([service_to_id[row["predicted_service"]] for row in rows])
                summary, per_class = metrics(y_true, y_pred, label_ids)
                run_rows.append({"dataset": dataset, "seed": seed, "encoder": encoder, **summary})
                for class_id, service in enumerate(services):
                    class_rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "encoder": encoder,
                        "service": service,
                        "precision": float(per_class[0][class_id]),
                        "recall": float(per_class[1][class_id]),
                        "f1": float(per_class[2][class_id]),
                        "support": int(per_class[3][class_id]),
                    })
                matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
                for true_id, true_service in enumerate(services):
                    denom = max(1, int(matrix[true_id].sum()))
                    for pred_id, predicted_service in enumerate(services):
                        confusion_rows.append({
                            "dataset": dataset,
                            "seed": seed,
                            "encoder": encoder,
                            "true_service": true_service,
                            "predicted_service": predicted_service,
                            "count": int(matrix[true_id, pred_id]),
                            "row_normalized": float(matrix[true_id, pred_id] / denom),
                        })
                gap_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "encoder": encoder,
                    "validation_macro_f1": float(base_results[encoder]["validation_macro_f1"]),
                    "test_macro_f1": summary["macro_f1"],
                    "test_minus_validation_macro_f1": summary["macro_f1"] - float(base_results[encoder]["validation_macro_f1"]),
                })

    metric_names = [key for key in run_rows[0] if key not in {"dataset", "seed", "encoder"}]
    aggregate_rows = []
    for dataset in DATASETS:
        for encoder in ENCODERS:
            values = [row for row in run_rows if row["dataset"] == dataset and row["encoder"] == encoder]
            item = {"dataset": dataset, "encoder": encoder, "seeds": len(values)}
            for name in metric_names:
                scores = np.asarray([row[name] for row in values], dtype=float)
                item[f"mean_{name}"] = float(scores.mean())
                item[f"std_{name}"] = float(scores.std(ddof=0))
                item[f"min_{name}"] = float(scores.min())
                item[f"max_{name}"] = float(scores.max())
            aggregate_rows.append(item)

    class_aggregate_rows = []
    grouped_classes = defaultdict(list)
    for row in class_rows:
        grouped_classes[(row["dataset"], row["encoder"], row["service"])].append(row)
    for (dataset, encoder, service), values in sorted(grouped_classes.items()):
        item = {"dataset": dataset, "encoder": encoder, "service": service, "seeds": len(values), "support_per_seed": values[0]["support"]}
        for name in ("precision", "recall", "f1"):
            scores = np.asarray([row[name] for row in values], dtype=float)
            item[f"mean_{name}"] = float(scores.mean())
            item[f"std_{name}"] = float(scores.std(ddof=0))
        class_aggregate_rows.append(item)

    lookup = {(row["dataset"], int(row["seed"]), row["encoder"]): row for row in run_rows}
    delta_rows = []
    for dataset in DATASETS:
        for seed in seeds:
            for baseline in ("E1", "E2"):
                delta_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "comparison": f"E3_minus_{baseline}",
                    **{f"delta_{name}": lookup[(dataset, seed, "E3")][name] - lookup[(dataset, seed, baseline)][name] for name in metric_names},
                })

    previous = load_csv(PROJECT / "stage21_coarse_service_ours_e3_benchmark" / "multiview_run_metrics.csv")
    previous_lookup = {(row["dataset"], int(row["seed"]), row["encoder"]): row for row in previous}
    stage21_rows = []
    for key, current in sorted(lookup.items()):
        if key not in previous_lookup:
            continue
        old = previous_lookup[key]
        stage21_rows.append({
            "dataset": key[0],
            "seed": key[1],
            "encoder": key[2],
            "stage21_random3_accuracy": float(old["accuracy"]),
            "stage22_pretrained20_accuracy": current["accuracy"],
            "delta_accuracy": current["accuracy"] - float(old["accuracy"]),
            "stage21_random3_macro_f1": float(old["macro_f1"]),
            "stage22_pretrained20_macro_f1": current["macro_f1"],
            "delta_macro_f1": current["macro_f1"] - float(old["macro_f1"]),
            "stage21_random3_weighted_f1": float(old["weighted_f1"]),
            "stage22_pretrained20_weighted_f1": current["weighted_f1"],
            "delta_weighted_f1": current["weighted_f1"] - float(old["weighted_f1"]),
            "interpretation": "combined official-pretraining plus 20-epoch budget effect; not pure pretraining causality",
        })

    write_csv(OUT / "run_metrics.csv", run_rows)
    write_csv(OUT / "aggregate_metrics.csv", aggregate_rows)
    write_csv(OUT / "per_class_metrics.csv", class_rows)
    write_csv(OUT / "per_class_aggregate.csv", class_aggregate_rows)
    write_csv(OUT / "confusion_matrices.csv", confusion_rows)
    write_csv(OUT / "validation_test_gap.csv", gap_rows)
    write_csv(OUT / "e3_vs_branch_deltas.csv", delta_rows)
    write_csv(OUT / "paired_vs_stage21_random3.csv", stage21_rows)
    primary = [row for row in aggregate_rows if row["encoder"] in {"E1", "E3"}]
    write_json(OUT / "aggregate_summary.json", {
        "status": "SUCCESS",
        "datasets": list(DATASETS),
        "seeds": seeds,
        "primary_comparison": "E3 minus E1",
        "primary_aggregate": primary,
    })
    print(json.dumps({"status": "SUCCESS", "primary_aggregate": primary}, indent=2))


if __name__ == "__main__":
    main()
