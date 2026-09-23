#!/usr/bin/env python3
"""Shared helpers for the frozen Stage 10B post-hoc diagnosis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STAGE7_ROOT = PROJECT_ROOT / "stage7_cipherspectrum_known_training"
STAGE8A_ROOT = PROJECT_ROOT / "stage8a_cipherspectrum_known_density"
STAGE9_ROOT = PROJECT_ROOT / "stage9_cipherspectrum_final_test"
CONFIG_PATH = ROOT / "configs/diagnosis_config.json"
SETTINGS = ("low", "medium", "high")
SCORE_ORDER = tuple(f"S{i}" for i in range(7))
SCORE_NAMES = {
    "S0": "Native KL",
    "S1": "ProtoMeanOnly",
    "S2": "EmpiricalCentroidRaw",
    "S3": "EmpiricalCentroidScaled",
    "S4": "EmpiricalCentroidPCA64",
    "S5": "FullK1",
    "S6": "FullK2",
}
SCOPE = "POST_HOC_DIAGNOSTIC_ONLY"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows) or fields is not None, f"cannot infer fields for empty CSV: {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def squared_distance_matrix(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Euclidean squared distances, clipped only for round-off below zero."""
    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    result = np.sum(x * x, axis=1, keepdims=True) + np.sum(c * c, axis=1)[None, :] - 2.0 * (x @ c.T)
    return np.maximum(result, 0.0)


def anomaly_metrics(known: np.ndarray, unknown: np.ndarray) -> dict[str, float]:
    k = np.asarray(known, dtype=np.float64)
    u = np.asarray(unknown, dtype=np.float64)
    require(k.ndim == u.ndim == 1 and len(k) and len(u), "scores must be non-empty vectors")
    require(np.isfinite(k).all() and np.isfinite(u).all(), "non-finite diagnostic scores")
    labels = np.concatenate([np.zeros(len(k), dtype=np.int8), np.ones(len(u), dtype=np.int8)])
    scores = np.concatenate([k, u])
    u_stat = float(mannwhitneyu(u, k, alternative="two-sided").statistic)
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "cliffs_delta_unknown_minus_known": float(2.0 * u_stat / (len(u) * len(k)) - 1.0),
    }


def score_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p05": float(np.quantile(x, 0.05)),
        f"{prefix}_p95": float(np.quantile(x, 0.95)),
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    from scipy.stats import spearmanr

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan, math.nan, "UNDEFINED_CONSTANT_OR_INSUFFICIENT"
    result = spearmanr(a, b)
    return float(result.statistic), float(result.pvalue), "OK"
