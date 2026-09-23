#!/usr/bin/env python3
"""Collect validation-only E0 metrics from frozen native Open-Detect runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from common import ROOT, STAGE12_ROOT, STAGE14C_NATIVE_ROOT, STAGE3_ROOT, read_json, sha256_file, write_csv, write_json


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_arrays(y_true: np.ndarray, y_pred: np.ndarray, n: int):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(n), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)), "precision": precision,
        "recall": recall, "f1": f1, "support": support,
        "confusion": confusion_matrix(y_true, y_pred, labels=np.arange(n)),
    }


def save(dataset: str, protocol_id: str, class_names: list[str], metrics: dict, extra: dict, source: Path, per_class_override=None, confusion_override=None) -> None:
    output = ROOT / "pilot_runs" / "e0" / dataset / protocol_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite E0 collection: {output}")
    output.mkdir(parents=True)
    confusion = metrics.get("confusion") if confusion_override is None else confusion_override
    np.save(output / "validation_confusion_matrix.npy", confusion, allow_pickle=False)
    if per_class_override is None:
        per_class = [{"experiment": "E0", "dataset": dataset, "protocol_id": protocol_id, "class_name": name, "precision": float(metrics["precision"][i]), "recall": float(metrics["recall"][i]), "f1": float(metrics["f1"][i]), "support": int(metrics["support"][i])} for i, name in enumerate(class_names)]
    else:
        per_class = [{"experiment": "E0", "dataset": dataset, "protocol_id": protocol_id, "class_name": str(row.get("application", row.get("class_name"))), "precision": float(row["precision"]), "recall": float(row["recall"]), "f1": float(row["f1"]), "support": int(row["support"])} for row in per_class_override]
    write_csv(output / "per_class_results.csv", per_class)
    result = {
        "status": "PASS_REUSED_FROZEN", "experiment": "E0", "dataset": dataset, "protocol_id": protocol_id,
        "known_test_samples_used": 0, "unknown_test_samples_used": 0, "class_names": class_names,
        "validation_accuracy": float(metrics["accuracy"]), "validation_macro_f1": float(metrics["macro_f1"]),
        "validation_weighted_f1": float(metrics["weighted_f1"]), "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source), **extra,
    }
    write_json(output / "result.json", result)


def collect_iscx(dataset: str) -> None:
    protocol_id = "medium_seed2022"
    run = STAGE12_ROOT / "runs" / dataset / "medium" / "seed2022"
    bundle_path = run / "known_train_validation_features.npz"
    bundle = np.load(bundle_path, allow_pickle=False)
    y_true = bundle["validation_labels_reindexed"]
    y_pred = bundle["validation_m0_predictions"]
    class_names = read_json(STAGE12_ROOT / "artifacts" / dataset / "protocol" / "medium" / "protocol.json")["known_classes"]
    metrics = metric_arrays(y_true, y_pred, len(class_names))
    freeze = read_json(run / "pretest_freeze.json")
    history = rows(run / "training_log.csv")
    best = next(row for row in history if int(row["epoch"]) == int(freeze["best_epoch"]))
    save(dataset, protocol_id, class_names, metrics, {
        "known_train_samples": len(bundle["train_labels_reindexed"]), "known_validation_samples": len(y_true),
        "best_epoch": int(freeze["best_epoch"]), "epochs_completed": int(freeze["completed_epochs"]),
        "train_loss": float(best["train_total"]), "validation_loss": float(best["validation_total"]),
        "runtime_seconds": float(sum(float(row["elapsed_seconds"]) for row in history)), "peak_gpu_memory_bytes": "NOT_RECORDED",
        "checkpoint_sha256": freeze["checkpoint_sha256"], "objective": "native Open-Detect joint loss",
    }, bundle_path)


def collect_vnat(protocol_id: str) -> None:
    run = STAGE14C_NATIVE_ROOT / "runs" / protocol_id
    result = read_json(run / "result.json")
    per_class = read_json(run / "per_class_metrics.json")
    history = rows(run / "history.csv")
    best = next(row for row in history if int(row["epoch"]) == int(result["best_epoch"]))
    confusion = np.load(run / "validation_confusion_matrix.npy", allow_pickle=False)
    metrics = {"accuracy": result["validation_accuracy"], "macro_f1": result["validation_macro_f1"], "weighted_f1": result["validation_weighted_f1"]}
    save("vnat", protocol_id, result["known_classes"], metrics, {
        "known_train_samples": result["train_samples"], "known_validation_samples": result["validation_samples"],
        "best_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"],
        "train_loss": float(best["train_total"]), "validation_loss": float(best["val_total"]),
        "runtime_seconds": result["duration_seconds"], "peak_gpu_memory_bytes": "NOT_RECORDED",
        "checkpoint_sha256": result["checkpoint_sha256"], "objective": "native Open-Detect joint loss",
    }, run / "result.json", per_class, confusion)


def collect_ustc() -> None:
    output = STAGE3_ROOT / "outputs" / "A-2"
    artifact = STAGE3_ROOT / "artifacts" / "A-2"
    config = read_json(output / "training_config.json")
    selection = read_json(output / "checkpoint_selection.json")
    frame = pd.read_parquet(artifact / "val_known_mu.parquet", columns=["class_name", "native_predicted_known_class"])
    class_names = list(config["known_classes"])
    local = {name: index for index, name in enumerate(class_names)}
    y_true = frame["class_name"].map(local).to_numpy(dtype=np.int64)
    y_pred = frame["native_predicted_known_class"].map(local).to_numpy(dtype=np.int64)
    if np.any(pd.isna(y_true)) or np.any(pd.isna(y_pred)):
        raise RuntimeError("USTC validation prediction has an unmapped class")
    metrics = metric_arrays(y_true, y_pred, len(class_names))
    history = rows(output / "training_metrics.csv")
    best = next(row for row in history if int(row["epoch"]) == int(selection["best_epoch"]))
    save("ustc", "A-2", class_names, metrics, {
        "known_train_samples": config["train_samples"], "known_validation_samples": config["validation_samples"],
        "best_epoch": selection["best_epoch"], "epochs_completed": selection["stop_epoch"],
        "train_loss": float(best["train_total"]), "validation_loss": float(best["val_total"]),
        "runtime_seconds": float(history[-1]["elapsed_seconds"]), "peak_gpu_memory_bytes": "NOT_RECORDED",
        "checkpoint_sha256": selection["checkpoint_sha256"], "objective": "native Open-Detect joint loss",
    }, artifact / "val_known_mu.parquet")


def main() -> None:
    collect_iscx("iscx_vpn")
    collect_iscx("iscx_tor")
    collect_vnat("medium_seed2025")
    collect_vnat("medium_seed2026")
    collect_ustc()
    write_json(ROOT / "pilot_runs" / "e0" / "collection_status.json", {"status": "PASS", "runs": 5, "known_test_samples_used": 0, "unknown_test_samples_used": 0})


if __name__ == "__main__":
    main()
