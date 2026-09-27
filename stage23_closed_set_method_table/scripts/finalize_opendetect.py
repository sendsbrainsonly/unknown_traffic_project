#!/usr/bin/env python3
"""Independently replay four frozen Open-Detect closed-set runs.

This script never imports the training implementation or opens image arrays.
It audits hashes, membership, labels and metrics from persisted predictions.
Existing Stage22 baseline rows are preserved in their original file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
BASELINES = OUT / "closed_set_run_results.csv"
RUNS = OUT / "runs" / "opendetect_corrected_paper"
EXPECTED_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)
ROLES = ("known_validation", "known_test")
METRICS = ("accuracy", "macro_f1", "weighted_f1")
METHOD = "Open-Detect corrected-paper"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def replay_metrics(rows: list[dict[str, str]], services: list[str]) -> tuple[dict[str, float], list[dict], list[dict]]:
    count = Counter((row["true_service"], row["predicted_service"]) for row in rows)
    per_class = []
    matrix = []
    for true in services:
        tp = count[true, true]
        support = sum(count[true, predicted] for predicted in services)
        predicted_total = sum(count[actual, true] for actual in services)
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted_total) if support + predicted_total else 0.0
        per_class.append({
            "service": true, "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        })
        for predicted in services:
            matrix.append({"true_service": true, "predicted_service": predicted, "count": count[true, predicted]})
    total = len(rows)
    return {
        "accuracy": sum(count[name, name] for name in services) / total,
        "macro_f1": mean(row["f1"] for row in per_class),
        "weighted_f1": sum(row["support"] * row["f1"] for row in per_class) / total,
    }, per_class, matrix


def main() -> None:
    checks: list[str] = []
    require(sha256(MANIFEST) == EXPECTED_HASH, "Stage20 manifest SHA256 unchanged", checks)
    manifest = load_csv(MANIFEST)
    expected: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    services: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        key = (row["dataset"], row["closed_role"])
        flow_id = row["flow_id"]
        require(flow_id not in expected[key], f"unique Stage20 {key}/{flow_id}", checks)
        expected[key][flow_id] = row["service_label"]
        services[row["dataset"]].add(row["service_label"])
    require(len(manifest) == 22136, "Stage20 total 22136", checks)
    baseline_rows = load_csv(BASELINES)
    require(len(baseline_rows) == 8, "eight reused Stage22 baseline rows", checks)
    require(all(row["status"] == "REUSED_VERIFIED" for row in baseline_rows), "Stage22 baseline status", checks)
    require(all(row["manifest_sha256"] == EXPECTED_HASH for row in baseline_rows), "Stage22 baseline hashes", checks)

    od_rows = []
    class_rows = []
    matrix_rows = []
    run_audit = []
    for dataset in DATASETS:
        ordered_services = sorted(services[dataset])
        for seed in SEEDS:
            run = RUNS / dataset / f"seed{seed}"
            require((run / "SUCCESS").is_file(), f"{dataset}/{seed} SUCCESS marker", checks)
            require(not (run / "FAILURE.json").exists(), f"{dataset}/{seed} no failure marker", checks)
            config, metrics = load_json(run / "config.json"), load_json(run / "metrics.json")
            require(config["dataset"] == dataset and int(config["seed"]) == seed, f"{dataset}/{seed} config identity", checks)
            require(metrics["dataset"] == dataset and int(metrics["seed"]) == seed, f"{dataset}/{seed} metrics identity", checks)
            require(config["stage20_manifest_sha256"] == EXPECTED_HASH == metrics["stage20_manifest_sha256"], f"{dataset}/{seed} frozen hash", checks)
            require(config["selection"] == "Known Validation accuracy only", f"{dataset}/{seed} validation-only selection", checks)
            require(all(int(config[key]) == int(metrics[key]) == 0 for key in
                        ("test_selection_samples", "unknown_training_samples", "unknown_validation_samples")),
                    f"{dataset}/{seed} no Test/Unknown selection", checks)
            require(set(config["services"]) == services[dataset], f"{dataset}/{seed} service set", checks)
            require(sha256(run / "model_best.pt") == metrics["checkpoint_sha256"], f"{dataset}/{seed} checkpoint SHA", checks)
            require(sha256(run / "predictions.csv") == metrics["predictions_sha256"], f"{dataset}/{seed} predictions SHA", checks)
            history = [json.loads(line) for line in (run / "training_history.jsonl").read_text(encoding="utf-8").splitlines()]
            require(len(history) == int(metrics["completed_epochs"]), f"{dataset}/{seed} history epoch count", checks)
            best_acc = max(float(entry["validation"]["accuracy"]) for entry in history)
            best_epoch = next(int(entry["epoch"]) for entry in history if float(entry["validation"]["accuracy"]) == best_acc)
            require(best_epoch == int(metrics["best_epoch"]), f"{dataset}/{seed} best epoch rule", checks)
            require(math.isclose(best_acc, float(metrics["validation"]["accuracy"]), abs_tol=1e-12), f"{dataset}/{seed} best validation accuracy", checks)

            predictions = load_csv(run / "predictions.csv")
            require(len(predictions) == sum(len(expected[dataset, role]) for role in ROLES), f"{dataset}/{seed} prediction total", checks)
            by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in predictions:
                require(row["dataset"] == dataset and int(row["seed"]) == seed and row["method"] == METHOD,
                        f"{dataset}/{seed} prediction identity", checks)
                require(row["role"] in ROLES, f"{dataset}/{seed} prediction role", checks)
                by_role[row["role"]].append(row)
            computed = {}
            for role in ROLES:
                rows = by_role[role]
                wanted = expected[dataset, role]
                ids = [row["flow_id"] for row in rows]
                require(len(ids) == len(wanted) and set(ids) == set(wanted), f"{dataset}/{seed}/{role} exact IDs", checks)
                require(len(ids) == len(set(ids)), f"{dataset}/{seed}/{role} no duplicate IDs", checks)
                require(all(row["true_service"] == wanted[row["flow_id"]] for row in rows),
                        f"{dataset}/{seed}/{role} exact labels", checks)
                require(all(row["predicted_service"] in services[dataset] for row in rows),
                        f"{dataset}/{seed}/{role} valid predictions", checks)
                require(all(int(row["correct"]) == int(row["true_service"] == row["predicted_service"]) for row in rows),
                        f"{dataset}/{seed}/{role} correct field", checks)
                values, class_detail, matrix = replay_metrics(rows, ordered_services)
                computed[role] = values
                section = "validation" if role == "known_validation" else "test"
                for metric in METRICS:
                    require(math.isclose(values[metric], float(metrics[section][metric]), abs_tol=1e-12),
                            f"{dataset}/{seed}/{role}/{metric} replay", checks)
                reported = {row["service"]: row for row in metrics[f"{section}_per_class"]}
                for detail in class_detail:
                    original = reported[detail["service"]]
                    require(int(detail["support"]) == int(original["support"]) and all(
                        math.isclose(detail[name], float(original[name]), abs_tol=1e-12)
                        for name in ("precision", "recall", "f1")),
                        f"{dataset}/{seed}/{role}/{detail['service']} per-class replay", checks)
                    class_rows.append({"dataset": dataset, "seed": seed, "method": METHOD, "role": role, **detail})
                matrix_rows.extend({"dataset": dataset, "seed": seed, "method": METHOD, "role": role, **item} for item in matrix)
            od_rows.append({
                "dataset": dataset, "seed": seed, "method": METHOD, "source_stage": "Stage23",
                "classes": len(services[dataset]), "train_flows": len(expected[dataset, "known_train"]),
                "validation_flows": len(expected[dataset, "known_validation"]),
                "test_flows": len(expected[dataset, "known_test"]),
                **computed["known_test"], "manifest_sha256": EXPECTED_HASH,
                "sample_parity": "PASS", "checkpoint_selection": config["selection"],
                "status": "NEW_VERIFIED",
            })
            run_audit.append({
                "dataset": dataset, "seed": seed, "best_epoch": best_epoch,
                "completed_epochs": len(history), "validation": computed["known_validation"],
                "test": computed["known_test"], "checkpoint_sha256": metrics["checkpoint_sha256"],
                "predictions_sha256": metrics["predictions_sha256"],
                "runtime_seconds": metrics["runtime_seconds"],
                "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
            })

    combined = baseline_rows + od_rows
    by_key = {(row["dataset"], int(row["seed"]), row["method"]): row for row in combined}
    require(len(by_key) == len(combined) == 12, "twelve unique verified method rows", checks)
    summary_rows = []
    for dataset in DATASETS:
        for method in sorted({row["method"] for row in combined}):
            selected = [row for row in combined if row["dataset"] == dataset and row["method"] == method]
            require(len(selected) == 2, f"{dataset}/{method} two seeds", checks)
            line = {"dataset": dataset, "method": method, "seeds": "2022;2023"}
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                line[f"{metric}_mean"] = mean(values)
                line[f"{metric}_std_population"] = pstdev(values)
                line[f"{metric}_std_sample"] = stdev(values)
            summary_rows.append(line)
    pair_rows = []
    methods = sorted({row["method"] for row in combined})
    for dataset in DATASETS:
        for left in methods:
            for right in methods:
                if left == right:
                    continue
                line = {"dataset": dataset, "left_method": left, "right_method": right, "seeds": "2022;2023"}
                for metric in METRICS:
                    deltas = [
                        float(by_key[dataset, seed, left][metric]) - float(by_key[dataset, seed, right][metric])
                        for seed in SEEDS
                    ]
                    line[f"delta_{metric}_mean"] = mean(deltas)
                    line[f"delta_{metric}_seed2022"] = deltas[0]
                    line[f"delta_{metric}_seed2023"] = deltas[1]
                    line[f"positive_{metric}_seeds"] = sum(value > 0 for value in deltas)
                pair_rows.append(line)

    save_csv(OUT / "opendetect_run_results.csv", od_rows, list(od_rows[0]))
    save_csv(OUT / "closed_set_run_results_verified.csv", combined, list(combined[0]))
    save_csv(OUT / "opendetect_per_class_results.csv", class_rows, list(class_rows[0]))
    save_csv(OUT / "opendetect_confusion_matrices.csv", matrix_rows, list(matrix_rows[0]))
    save_csv(OUT / "aggregate_summary.csv", summary_rows, list(summary_rows[0]))
    save_csv(OUT / "paired_comparison.csv", pair_rows, list(pair_rows[0]))
    require(sha256(MANIFEST) == EXPECTED_HASH, "Stage20 manifest SHA256 after replay", checks)
    save_json(OUT / "phase1_completion_verification.json", {
        "status": "PASS_OPENDETECT_PHASE_ONLY", "checks": len(checks), "failures": [],
        "stage20_manifest_sha256": EXPECTED_HASH, "verified_baseline_rows": 8,
        "verified_opendetect_runs": 4, "total_verified_method_rows": len(combined),
        "run_audit": run_audit, "other_methods_pending": True,
    })
    print(json.dumps({"status": "PASS_OPENDETECT_PHASE_ONLY", "checks": len(checks),
                      "verified_opendetect_runs": 4, "total_verified_method_rows": len(combined)}))


if __name__ == "__main__":
    main()
