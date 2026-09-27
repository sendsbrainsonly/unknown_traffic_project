#!/usr/bin/env python3
"""Independent Stage23 exact-flow replay and matched closed-set table."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

from finalize_opendetect import (
    DATASETS, EXPECTED_HASH, MANIFEST, METRICS, OUT, SEEDS,
    load_csv, load_json, replay_metrics, require, save_csv, save_json, sha256,
)

METHODS = (
    ("RoNeTC-runnable-reconstruction", "ronetc_stage20"),
    ("YaTC-official-pretrained-Stage20", "yatc_stage20"),
)
ROLES = ("known_validation", "known_test")


def main() -> None:
    checks: list[str] = []
    require(sha256(MANIFEST) == EXPECTED_HASH, "Stage20 hash unchanged", checks)
    stage20 = load_csv(MANIFEST)
    expected: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    services: dict[str, set[str]] = defaultdict(set)
    for row in stage20:
        key = row["dataset"], row["closed_role"]
        require(row["flow_id"] not in expected[key], "unique Stage20 flow ID", checks)
        expected[key][row["flow_id"]] = row["service_label"]
        services[row["dataset"]].add(row["service_label"])
    require(len(stage20) == 22136, "Stage20 exact flow count", checks)
    baseline = load_csv(OUT / "closed_set_run_results_verified.csv")
    require(len(baseline) == 12, "12 independently verified E1/E3/OD rows", checks)
    require(len({(x["dataset"], x["seed"], x["method"]) for x in baseline}) == 12,
            "baseline unique rows", checks)
    new_rows, class_rows, confusion_rows, audits = [], [], [], []
    for method, folder in METHODS:
        for dataset in DATASETS:
            ordered_services = sorted(services[dataset])
            for seed in SEEDS:
                run = OUT / "runs" / folder / dataset / f"seed{seed}_formal"
                require((run / "SUCCESS").is_file() and not (run / "FAILURE.json").exists(),
                        f"{method}/{dataset}/{seed} successful", checks)
                config, metrics = load_json(run / "config.json"), load_json(run / "metrics.json")
                require(config["method"] == metrics["method"] == method, "method identity", checks)
                require(config["dataset"] == metrics["dataset"] == dataset and
                        int(config["seed"]) == int(metrics["seed"]) == seed, "run identity", checks)
                require(config["stage20_manifest_sha256"] == metrics["stage20_manifest_sha256"] == EXPECTED_HASH,
                        "frozen hash in run", checks)
                require(set(config["services"]) == services[dataset], "exact service set", checks)
                require(all(int(config[k]) == 0 for k in
                            ("test_selection_samples", "unknown_training_samples", "unknown_validation_samples")),
                        "no Test/Unknown selection", checks)
                require(sha256(run / "model_best.pt") == metrics["checkpoint_sha256"],
                        "checkpoint SHA256", checks)
                history = [json.loads(line) for line in
                           (run / "training_history.jsonl").read_text(encoding="utf-8").splitlines()]
                require(len(history) == int(metrics["completed_epochs"]), "full training history", checks)
                if folder == "ronetc_stage20":
                    require(len(history) == 100, "RoNeTC full 100 epochs", checks)
                    criterion = [float(h["validation_accuracy"]) for h in history]
                    key = "accuracy"
                    require(config["checkpoint_selection"] == "Known Validation accuracy only",
                            "RoNeTC Known Validation checkpoint rule", checks)
                    require(sha256(run / "predictions.csv") == metrics["predictions_sha256"],
                            "RoNeTC prediction SHA256", checks)
                    predictions = load_csv(run / "predictions.csv")
                    by_role = {role: [x for x in predictions if x["role"] == role] for role in ROLES}
                    require(len(predictions) == sum(map(len, by_role.values())), "RoNeTC roles only", checks)
                else:
                    require(len(history) == 200, "YaTC full 200 epochs", checks)
                    criterion = [float(h["validation"]["weighted_f1"]) for h in history]
                    key = "weighted_f1"
                    require(config["checkpoint_selection"] == "Known Validation weighted F1 (author macro_f1 field)",
                            "YaTC author checkpoint rule", checks)
                    eval_result = load_json(run / "test_evaluation_cuda.json")
                    require(eval_result["status"] == "PASS", "YaTC post-selection evaluation", checks)
                    require(eval_result["device"] == "cuda:0" and eval_result["evaluation_tag"] == "cuda",
                            "YaTC CUDA mixed-precision replay path", checks)
                    require(eval_result["checkpoint_sha256_before"] == eval_result["checkpoint_sha256_after"] ==
                            metrics["checkpoint_sha256"], "YaTC checkpoint frozen through Test", checks)
                    require(eval_result["stage20_manifest_sha256"] == EXPECTED_HASH and
                            int(eval_result["test_selection_samples"]) == 0, "YaTC Test not selected", checks)
                    val_file, test_file = run / "validation_predictions.csv", run / "test_predictions_cuda.csv"
                    require(sha256(val_file) == metrics["validation_predictions_sha256"] and
                            sha256(test_file) == eval_result["predictions_sha256"],
                            "YaTC prediction hashes", checks)
                    by_role = {"known_validation": load_csv(val_file), "known_test": load_csv(test_file)}
                maximum = max(criterion)
                best_epoch = criterion.index(maximum) + 1
                require(best_epoch == int(metrics["best_epoch"]), "best epoch from Known Validation", checks)
                require(math.isclose(maximum, float(metrics["validation"][key]), abs_tol=1e-12),
                        "checkpoint best validation metric", checks)
                computed = {}
                for role in ROLES:
                    rows = by_role[role]
                    wanted = expected[dataset, role]
                    ids = [row["flow_id"] for row in rows]
                    require(len(ids) == len(wanted) and len(ids) == len(set(ids)) and set(ids) == set(wanted),
                            "exact per-role flow membership", checks)
                    require(all(row["dataset"] == dataset and int(row["seed"]) == seed and
                                row["role"] == role and row["method"] == method and
                                row["true_service"] == wanted[row["flow_id"]] and
                                row["predicted_service"] in services[dataset] and
                                int(row["correct"]) == int(row["true_service"] == row["predicted_service"])
                                for row in rows), "prediction identity/label/correctness", checks)
                    values, details, matrix = replay_metrics(rows, ordered_services)
                    computed[role] = values
                    section = "validation" if role == "known_validation" else "test"
                    claimed = metrics[section] if folder == "ronetc_stage20" or role == "known_validation" else eval_result["test"]
                    for metric in METRICS:
                        require(math.isclose(values[metric], float(claimed[metric]), abs_tol=1e-12),
                                "independent metric replay", checks)
                    for detail in details:
                        class_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                           "role": role, **detail})
                    for item in matrix:
                        confusion_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                               "role": role, **item})
                new_rows.append({
                    "dataset": dataset, "seed": seed, "method": method, "source_stage": "Stage23",
                    "classes": len(services[dataset]),
                    "train_flows": len(expected[dataset, "known_train"]),
                    "validation_flows": len(expected[dataset, "known_validation"]),
                    "test_flows": len(expected[dataset, "known_test"]),
                    **computed["known_test"], "manifest_sha256": EXPECTED_HASH,
                    "sample_parity": "PASS", "checkpoint_selection": config["checkpoint_selection"],
                    "status": "NEW_VERIFIED",
                })
                audits.append({"dataset": dataset, "seed": seed, "method": method,
                               "best_epoch": best_epoch, "completed_epochs": len(history),
                               "checkpoint_sha256": metrics["checkpoint_sha256"],
                               "validation": computed["known_validation"],
                               "test": computed["known_test"]})
    combined = baseline + new_rows
    require(len(combined) == 20, "20 complete method rows", checks)
    keyed = {(r["dataset"], int(r["seed"]), r["method"]): r for r in combined}
    require(len(keyed) == 20, "20 unique method rows", checks)
    all_class_rows = list(class_rows)
    all_confusion_rows = list(confusion_rows)
    for dataset in DATASETS:
        ordered_services = sorted(services[dataset])
        for seed in SEEDS:
            stage22 = (OUT.parent / "stage22_pretrained_trafficformer_e3_closed_set_comparison" /
                       "runs" / dataset / f"seed{seed}" / "predictions.csv")
            source_sets = [
                (load_csv(stage22), "encoder", "E1", "TrafficFormer-pretrained"),
                (load_csv(stage22), "encoder", "E3", "OURS-E3-T8-pretrained"),
                (load_csv(OUT / "runs" / "opendetect_corrected_paper" / dataset /
                          f"seed{seed}" / "predictions.csv"),
                 "method", "Open-Detect corrected-paper", "Open-Detect corrected-paper"),
            ]
            for source, column, selector, method in source_sets:
                for role in ROLES:
                    rows = [r for r in source if r[column] == selector and r["role"] == role]
                    wanted = expected[dataset, role]
                    ids = [r["flow_id"] for r in rows]
                    require(len(ids) == len(wanted) and len(ids) == len(set(ids)) and set(ids) == set(wanted),
                            "baseline per-class exact membership", checks)
                    require(all(r["true_service"] == wanted[r["flow_id"]] and
                                r["predicted_service"] in services[dataset]
                                for r in rows), "baseline per-class labels", checks)
                    values, details, matrix = replay_metrics(rows, ordered_services)
                    if role == "known_test":
                        for metric in METRICS:
                            require(math.isclose(values[metric], float(keyed[dataset, seed, method][metric]),
                                                 abs_tol=1e-12),
                                    "baseline per-class metric replay", checks)
                    all_class_rows.extend({"dataset": dataset, "seed": seed, "method": method,
                                           "role": role, **detail} for detail in details)
                    all_confusion_rows.extend({"dataset": dataset, "seed": seed, "method": method,
                                               "role": role, **item} for item in matrix)
    summary = []
    for dataset in DATASETS:
        for method in sorted({r["method"] for r in combined}):
            pair = [r for r in combined if r["dataset"] == dataset and r["method"] == method]
            require(len(pair) == 2, "two matched seeds per dataset/method", checks)
            line = {"dataset": dataset, "method": method, "seeds": "2022;2023"}
            for metric in METRICS:
                values = [float(r[metric]) for r in pair]
                line[f"{metric}_mean"] = mean(values)
                line[f"{metric}_std_population"] = pstdev(values)
                line[f"{metric}_std_sample"] = stdev(values)
            summary.append(line)
    comparisons = []
    for dataset in DATASETS:
        for challenger in sorted({r["method"] for r in combined}):
            for reference in ("TrafficFormer-pretrained", "OURS-E3-T8-pretrained"):
                if challenger == reference:
                    continue
                line = {"dataset": dataset, "challenger": challenger, "reference": reference}
                for metric in METRICS:
                    delta = [float(keyed[dataset, seed, challenger][metric]) -
                             float(keyed[dataset, seed, reference][metric]) for seed in SEEDS]
                    line[f"delta_{metric}_mean"] = mean(delta)
                    line[f"delta_{metric}_seed2022"] = delta[0]
                    line[f"delta_{metric}_seed2023"] = delta[1]
                    line[f"positive_{metric}_seeds"] = sum(d > 0 for d in delta)
                comparisons.append(line)
    save_csv(OUT / "stage23_full_run_results.csv", combined, list(combined[0]))
    save_csv(OUT / "stage23_new_method_per_class.csv", class_rows, list(class_rows[0]))
    save_csv(OUT / "stage23_new_method_confusion.csv", confusion_rows, list(confusion_rows[0]))
    save_csv(OUT / "stage23_all_method_per_class.csv", all_class_rows, list(all_class_rows[0]))
    save_csv(OUT / "stage23_all_method_confusion.csv", all_confusion_rows, list(all_confusion_rows[0]))
    save_csv(OUT / "stage23_full_summary.csv", summary, list(summary[0]))
    save_csv(OUT / "stage23_paired_vs_e1_e3.csv", comparisons, list(comparisons[0]))
    require(sha256(MANIFEST) == EXPECTED_HASH, "Stage20 hash still unchanged", checks)
    save_json(OUT / "completion_verification.json", {
        "status": "PASS_FIVE_METHOD_MATCHED_TABLE", "checks": len(checks), "failures": [],
        "stage20_manifest_sha256": EXPECTED_HASH, "verified_runs": 20,
        "new_run_audit": audits, "other_methods": "subject to method cards and non-comparability audit",
    })
    print(json.dumps({"status": "PASS_FIVE_METHOD_MATCHED_TABLE", "checks": len(checks),
                      "verified_runs": len(combined)}))


if __name__ == "__main__":
    main()
