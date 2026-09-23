#!/usr/bin/env python3
"""Shared, fail-closed utilities for frozen VNAT Stage 14D."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STAGE14B_ROOT = PROJECT_ROOT / "stage14b_vnat_protocol_freeze"
STAGE14C_NATIVE_ROOT = PROJECT_ROOT / "stage14c_native_opendetect_vnat"
CACHE_ROOT = PROJECT_ROOT / "stage14c_vnat_encoder_training" / "flow_image_cache"
OPENDETECT_ROOT = PROJECT_ROOT.parent / "Open-Detect"
PROTOCOL_PATH = STAGE14B_ROOT / "vnat_open_set_protocol.json"
SPLIT_MANIFEST_PATH = STAGE14B_ROOT / "vnat_split_manifest.csv"
FREEZE_HASH_PATH = STAGE14B_ROOT / "outputs" / "freeze_hashes.json"
CONFIG_PATH = ROOT / "config.json"
CHECKPOINT_FREEZE_PATH = ROOT / "checkpoint_freeze.json"
ARTIFACT_ROOT = ROOT / "artifacts"

EXPECTED_FREEZE_HASH = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_PROTOCOL_SHA256 = "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced"
EXPECTED_SPLIT_SHA256 = "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e"
SETTINGS = ("Low", "Medium", "High")
SEEDS = (2022, 2023, 2024, 2025, 2026)
METHODS = ("M0", "M1", "M2")
METRICS = ("auroc", "auprc", "ufar", "known_frr", "known_macro_f1")
HARD_CLASSES = {"rsync", "scp", "sftp"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def verify_stage14b() -> dict[str, str]:
    recorded = read_json(FREEZE_HASH_PATH)
    actual = {
        "freeze_hash": str(recorded["freeze_hash"]),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
    }
    expected = {
        "freeze_hash": EXPECTED_FREEZE_HASH,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
    }
    if actual != expected:
        raise RuntimeError(f"Stage 14B freeze mismatch: expected={expected}, actual={actual}")
    return actual


def load_protocols() -> dict[str, dict[str, object]]:
    verify_stage14b()
    document = read_json(PROTOCOL_PATH)
    if document.get("freeze_hash") != EXPECTED_FREEZE_HASH:
        raise RuntimeError("protocol document freeze hash mismatch")
    protocols = {str(item["protocol_id"]): item for item in document["protocols"]}
    expected = {f"{setting.lower()}_seed{seed}" for setting in SETTINGS for seed in SEEDS}
    if set(protocols) != expected:
        raise RuntimeError(f"unexpected protocol grid: {sorted(protocols)}")
    return protocols


def load_manifest_rows(protocol_id: str) -> list[dict[str, str]]:
    with SPLIT_MANIFEST_PATH.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["protocol_id"] == protocol_id]
    if len(rows) != 23_449:
        raise RuntimeError(f"{protocol_id}: expected 23449 manifest rows, got {len(rows)}")
    if len({row["flow_uid"] for row in rows}) != len(rows):
        raise RuntimeError(f"{protocol_id}: duplicate flow_uid in frozen manifest")
    return rows


def checkpoint_records() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for protocol_id in load_protocols():
        result_path = STAGE14C_NATIVE_ROOT / "runs" / protocol_id / "result.json"
        completed_path = result_path.parent / "COMPLETED"
        if not result_path.is_file() or not completed_path.is_file():
            raise RuntimeError(f"missing successful Native Stage 14C run: {protocol_id}")
        result = read_json(result_path)
        if result.get("status") != "success":
            raise RuntimeError(f"Native Stage 14C run not successful: {protocol_id}")
        checkpoint = Path(str(result["checkpoint_path"]))
        actual = sha256_file(checkpoint)
        expected = str(result["checkpoint_sha256"])
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {protocol_id}")
        records[protocol_id] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": actual,
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "validation_accuracy": float(result["validation_accuracy"]),
            "validation_macro_f1": float(result["validation_macro_f1"]),
            "known_classes": list(result["known_classes"]),
            "unknown_classes": list(result["unknown_classes"]),
        }
    return records


def verify_checkpoint_freeze() -> dict[str, dict[str, object]]:
    frozen = read_json(CHECKPOINT_FREEZE_PATH)
    records = frozen["checkpoints"]
    expected_ids = set(load_protocols())
    if set(records) != expected_ids:
        raise RuntimeError("checkpoint freeze protocol grid mismatch")
    for protocol_id, record in records.items():
        if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint changed: {protocol_id}")
        if sha256_file(Path(record["result_path"])) != record["result_sha256"]:
            raise RuntimeError(f"Native result metadata changed: {protocol_id}")
    return records


def robust_parameters(scores: np.ndarray, epsilon: float) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("validation scores must be a finite non-empty vector")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0:
        raise ValueError("Known Validation MAD is non-positive")
    return {"median": median, "mad": mad, "epsilon": epsilon, "denominator": mad + epsilon}


def robust_normalize(scores: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    values = (np.asarray(scores, dtype=np.float64) - parameters["median"]) / parameters["denominator"]
    if np.any(~np.isfinite(values)):
        raise RuntimeError("non-finite robust-normalized score")
    return values


def empirical_centroids(train_mu: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(train_labels, dtype=np.int64)
    classes = sorted(np.unique(labels).tolist())
    if classes != list(range(len(classes))):
        raise RuntimeError("Known Train labels are not a contiguous local class range")
    return np.vstack([np.asarray(train_mu, dtype=np.float64)[labels == label].mean(axis=0) for label in classes])


def centroid_scores(query_mu: np.ndarray, centroids: np.ndarray, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_mu, dtype=np.float64)
    scores = np.empty(len(query), dtype=np.float64)
    predictions = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        distances = cdist(query[start:stop], centroids, metric="sqeuclidean")
        scores[start:stop] = distances.min(axis=1)
        predictions[start:stop] = distances.argmin(axis=1)
    return scores, predictions


def local_knn10_scores(
    query_mu: np.ndarray,
    predicted_classes: np.ndarray,
    train_mu: np.ndarray,
    train_labels: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    query = np.asarray(query_mu, dtype=np.float64)
    train = np.asarray(train_mu, dtype=np.float64)
    predictions = np.asarray(predicted_classes, dtype=np.int64)
    labels = np.asarray(train_labels, dtype=np.int64)
    scores = np.full(len(query), np.nan, dtype=np.float64)
    for class_index in sorted(np.unique(predictions).tolist()):
        reference = train[labels == class_index]
        selected = np.flatnonzero(predictions == class_index)
        if len(reference) < 10:
            raise RuntimeError(f"class {class_index} has only {len(reference)} Known-Train support samples")
        for start in range(0, len(selected), chunk_size):
            indices = selected[start : start + chunk_size]
            distances = cdist(query[indices], reference, metric="euclidean")
            nearest = np.partition(distances, 9, axis=1)[:, :10]
            scores[indices] = nearest.mean(axis=1)
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("non-finite kNN-10 scores")
    return scores


def threshold_p95(validation_scores: np.ndarray) -> float:
    return float(np.quantile(np.asarray(validation_scores), 0.95, method="higher"))


def detection_metrics(
    validation_scores: np.ndarray,
    known_test_scores: np.ndarray,
    unknown_test_scores: np.ndarray,
) -> dict[str, float | int | str]:
    threshold = threshold_p95(validation_scores)
    known = np.asarray(known_test_scores, dtype=np.float64)
    unknown = np.asarray(unknown_test_scores, dtype=np.float64)
    y_true = np.concatenate([np.zeros(len(known), dtype=np.int64), np.ones(len(unknown), dtype=np.int64)])
    y_score = np.concatenate([known, unknown])
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        "threshold_source": "Known Validation P95 only; numpy method=higher",
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "ufar": float(np.mean(unknown < threshold)),
        "known_frr": float(np.mean(known >= threshold)),
        "binary_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "known_test_samples": int(len(known)),
        "unknown_test_samples": int(len(unknown)),
        "validation_samples": int(len(validation_scores)),
        "validation_known_frr": float(np.mean(np.asarray(validation_scores) >= threshold)),
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "known_accuracy": float(accuracy_score(labels, predictions)),
        "known_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "known_weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def per_class_classification(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> list[dict[str, object]]:
    class_ids = np.arange(len(class_names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, zero_division=0
    )
    return [
        {
            "application": class_names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in class_ids
    ]


def summary_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_bootstrap_ci(values: Iterable[float], repetitions: int, seed: int) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

