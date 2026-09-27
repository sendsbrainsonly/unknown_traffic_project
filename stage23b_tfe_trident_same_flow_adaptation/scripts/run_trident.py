#!/usr/bin/env python3
"""Matched Stage20 Trident-early8-86D classifier, with Known-only fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
SOURCE = PROJECT.parent / "Trident/code/reproduction"
sys.path.insert(0, str(SOURCE))
from run_ustc_ae_v3 import log_losses, model_loss_matrix  # noqa: E402
from run_ustc_ae_corrected import train_known_learners  # noqa: E402

MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
FROZEN = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
CONFIG = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison/config.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_input(dataset: str, phase: str, services: list[str]):
    root = OUT / "inputs" / dataset / phase
    audit = json.loads((root / "input_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["stage20_manifest_sha256"] != FROZEN:
        raise RuntimeError("input audit failed")
    for filename, expected in audit["file_hashes"].items():
        if digest(root / filename) != expected:
            raise RuntimeError(f"input hash mismatch: {filename}")
    with (root / "membership.csv").open(newline="", encoding="utf-8") as f:
        members = list(csv.DictReader(f))
    ids = [str(x) for x in np.load(root / "flow_ids.npy", allow_pickle=False)]
    if ids != [x["flow_id"] for x in members]:
        raise RuntimeError("flow ID alignment failed")
    y = np.asarray([services.index(x["service_label"]) for x in members], dtype=np.int64)
    x = np.load(root / "trident86.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (len(members), 86) or not np.isfinite(x).all():
        raise RuntimeError("Trident input shape/finite check failed")
    return x, y, members, audit


def metrics(y: np.ndarray, pred: np.ndarray, services: list[str]) -> dict:
    labels = np.arange(len(services))
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(support[i])}
                      for i, name in enumerate(services)},
    }


def save_predictions(path: Path, dataset: str, seed: int, role: str,
                     members: list[dict], y: np.ndarray, pred: np.ndarray,
                     probability: np.ndarray, services: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8") as f:
        names = ["dataset", "seed", "role", "method", "flow_id", "true_service",
                 "predicted_service", "correct"] + [f"prob_{name}" for name in services]
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row, truth, guess, probs in zip(members, y, pred, probability):
            result = {"dataset": dataset, "seed": seed, "role": role,
                      "method": "Trident-early8-86D", "flow_id": row["flow_id"],
                      "true_service": services[int(truth)],
                      "predicted_service": services[int(guess)],
                      "correct": int(truth == guess)}
            result.update({f"prob_{name}": float(probs[i]) for i, name in enumerate(services)})
            writer.writerow(result)


def main(dataset: str, seed: int, phase: str) -> None:
    if digest(MANIFEST) != FROZEN:
        raise RuntimeError("Stage20 hash mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if seed not in config["seeds"]:
        raise ValueError("seed outside Stage22")
    services = config["datasets"][dataset]["services"]
    run = OUT / "runs" / "trident" / dataset / f"seed{seed}"
    if phase == "train":
        if run.exists():
            raise FileExistsError(f"preserve previous run: {run}")
        run.mkdir(parents=True)
        x, y, rows, audit = load_input(dataset, "trainval", services)
        train = np.flatnonzero([r["role"] == "known_train" for r in rows])
        val = np.flatnonzero([r["role"] == "known_validation" for r in rows])
        if (len(train), len(val)) != (config["datasets"][dataset]["expected_train"],
                                     config["datasets"][dataset]["expected_validation"]):
            raise RuntimeError("frozen role counts changed")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        ae_rel, cal_rel = next(splitter.split(np.zeros(len(train)), y[train]))
        ae_indices, cal_indices = train[ae_rel], train[cal_rel]
        scaler = StandardScaler().fit(x[train])
        ae_x = scaler.transform(x[ae_indices]).astype(np.float32)
        cal_x = scaler.transform(x[cal_indices]).astype(np.float32)
        val_x = scaler.transform(x[val]).astype(np.float32)
        torch.set_num_threads(4)
        known_labels = np.arange(len(services), dtype=np.int64)
        models, thresholds, methods, records = train_known_learners(
            ae_x, y[ae_indices], known_labels, 10, 10, 0.001, 5e-4, seed)
        cal_losses = model_loss_matrix(models, cal_x, 1024)
        forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        class_weight="balanced", n_jobs=4, random_state=seed)
        forest.fit(log_losses(cal_losses), y[cal_indices])
        val_losses = model_loss_matrix(models, val_x, 1024)
        proba = forest.predict_proba(log_losses(val_losses))
        pred = forest.classes_[np.argmax(proba, axis=1)].astype(np.int64)
        if not np.array_equal(forest.classes_, known_labels):
            raise RuntimeError("Trident calibrator missing service class")
        result = metrics(y[val], pred, services)
        checkpoint = run / "selected_model.joblib"
        joblib.dump({"scaler": scaler, "models": models, "calibrator": forest,
                     "services": services, "ae_indices": ae_indices.tolist(),
                     "cal_indices": cal_indices.tolist(), "train_records": records,
                     "evt_thresholds_diagnostic_only": thresholds.tolist(),
                     "evt_methods_diagnostic_only": methods}, checkpoint)
        save_predictions(run / "known_val_predictions.csv", dataset, seed, "known_validation",
                         [rows[i] for i in val], y[val], pred, proba, services)
        report = {
            "status": "SELECTION_COMPLETE", "method": "Trident-early8-86D",
            "dataset": dataset, "seed": seed, "train_flows": len(train),
            "ae_train_flows": len(ae_indices), "calibration_train_flows": len(cal_indices),
            "validation_flows": len(val), "known_test_loaded": 0, "unknown_loaded": 0,
            "validation": result, "stage20_manifest_sha256": FROZEN,
            "input_hashes": audit["file_hashes"], "checkpoint_sha256": digest(checkpoint),
            "selection_rule": "fixed ten-epoch Trident pipeline; no Test selection",
        }
        (run / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (run / "SELECTION_COMPLETE").write_text("PASS\n", encoding="utf-8")
        print(json.dumps({"event": "selection_complete", "dataset": dataset,
                          "seed": seed, "val_macro_f1": result["macro_f1"]}), flush=True)
    else:
        if not (run / "SELECTION_COMPLETE").is_file():
            raise RuntimeError("Trident selection not complete")
        record = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        if digest(run / "selected_model.joblib") != record["checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint changed")
        if (run / "known_test_predictions.csv").exists():
            raise FileExistsError("preserve prior Test evaluation")
        x, y, rows, audit = load_input(dataset, "test", services)
        bundle = joblib.load(run / "selected_model.joblib")
        torch.set_num_threads(4)
        scaled = bundle["scaler"].transform(x).astype(np.float32)
        losses = model_loss_matrix(bundle["models"], scaled, 1024)
        proba = bundle["calibrator"].predict_proba(log_losses(losses))
        pred = bundle["calibrator"].classes_[np.argmax(proba, axis=1)].astype(np.int64)
        result = metrics(y, pred, services)
        save_predictions(run / "known_test_predictions.csv", dataset, seed, "known_test",
                         rows, y, pred, proba, services)
        report = {"status": "COMPLETE", "method": "Trident-early8-86D",
                  "dataset": dataset, "seed": seed, "known_test_flows": len(rows),
                  "test": result, "checkpoint_sha256": record["checkpoint_sha256"],
                  "stage20_manifest_sha256": FROZEN, "test_input_hashes": audit["file_hashes"]}
        (run / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (run / "SUCCESS").write_text("PASS\n", encoding="utf-8")
        print(json.dumps({"event": "test_complete", "dataset": dataset,
                          "seed": seed, "test_macro_f1": result["macro_f1"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    p.add_argument("--phase", choices=("train", "test"), required=True)
    a = p.parse_args()
    main(a.dataset, a.seed, a.phase)
