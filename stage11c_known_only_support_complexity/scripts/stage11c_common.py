#!/usr/bin/env python3
"""Shared Known-only computations for Stage 11C."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import ks_2samp, wasserstein_distance


STAGE_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_ROOT = STAGE_ROOT.parent
CONFIG_PATH = STAGE_ROOT / "configs" / "stage11c_config.json"
STAGE11B_ROOT = UNKNOWN_ROOT / "stage11b_decoupled_support_readout"
SUMMARY_ROOT = STAGE_ROOT / "outputs" / "summary"
SCENARIOS = ("a1", "a2", "a3")
SEEDS = (2022, 2023, 2024, 2025, 2026)
EPS = 1e-12


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_name(fold: int, seed: int) -> str:
    return f"fold{fold}_seed{seed}"


def source_run_dir(scenario: str, fold: int, seed: int) -> Path:
    return STAGE11B_ROOT / "artifacts" / scenario / run_name(fold, seed)


def output_run_dir(scenario: str, fold: int, seed: int) -> Path:
    # The first formal launch failed its deliberately strict centroid parity
    # check before producing any statistics. Preserve that terminal attempt in
    # the parent bundle and write the corrected execution to attempt2.
    return STAGE_ROOT / "artifacts" / scenario / run_name(fold, seed) / "attempt2"


def load_known_inputs(scenario: str, fold: int, seed: int) -> dict:
    """Load only explicitly allowed Known Train/Validation arrays.

    np.load exposes an archive directory, but only the six named Known arrays and
    four validation score/prediction arrays below are materialized. No Test key
    is indexed anywhere in Stage 11C.
    """

    source = source_run_dir(scenario, fold, seed)
    result = read_json(source / "results.json")
    if result["representation_parity"]["status"] != "PASS":
        raise RuntimeError(f"Stage11B parity is not PASS: {source}")
    if result["frozen_source_modified"] or result["next_stage_started"]:
        raise RuntimeError(f"Stage11B frozen contract failed: {source}")
    with np.load(source / "sample_outputs.npz", allow_pickle=False) as archive:
        allowed = {
            "train_mu",
            "train_labels_reindexed",
            "validation_mu",
            "validation_labels_reindexed",
            "validation_r1_scores",
            "validation_r1_predictions_reindexed",
            "validation_r2_scores",
            "validation_r2_predictions_reindexed",
        }
        arrays = {key: np.asarray(archive[key]).copy() for key in allowed}
    support = joblib.load(source / "support_models.joblib")
    thresholds = read_json(source / "thresholds.json")
    return {"source": source, "result": result, "arrays": arrays, "support": support, "thresholds": thresholds}


def mle_covariance(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    centered = x - mean
    covariance = centered.T @ centered / float(len(x))
    covariance = (covariance + covariance.T) * 0.5
    return mean, covariance


def spectrum_statistics(x: np.ndarray, space: str) -> dict[str, float | str | int]:
    _, covariance = mle_covariance(x)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    safe = np.maximum(eigenvalues, EPS)
    total = float(np.sum(safe))
    probabilities = safe / total
    dimension = int(x.shape[1])
    return {
        "space": space,
        "dimension": dimension,
        "condition_number": float(safe[0] / safe[-1]),
        "anisotropy_ratio": float(safe[0] / np.mean(safe)),
        "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "top1_ratio": float(np.sum(safe[:1]) / total),
        "top5_ratio": float(np.sum(safe[: min(5, dimension)]) / total),
        "top10_ratio": float(np.sum(safe[: min(10, dimension)]) / total),
        "logdet_covariance": float(np.sum(np.log(safe))),
        "trace_covariance": float(np.sum(eigenvalues)),
        "n_over_d": float(len(x) / dimension),
        "n_over_d_plus_1": float(len(x) / (dimension + 1)),
        "n_over_d_squared": float(len(x) / (dimension * dimension)),
    }


def radius_statistics(train: np.ndarray, validation: np.ndarray, centroid: np.ndarray) -> dict[str, float]:
    train_radius = np.linalg.norm(train - centroid, axis=1)
    val_radius = np.linalg.norm(validation - centroid, axis=1)
    train_mean = float(np.mean(train_radius))
    train_std = float(np.std(train_radius))
    train_median = float(np.median(train_radius))
    train_mad = float(np.median(np.abs(train_radius - train_median)))
    return {
        "train_radius_mean": train_mean,
        "train_radius_median": train_median,
        "train_radius_p90": float(np.quantile(train_radius, 0.90)),
        "train_radius_p95": float(np.quantile(train_radius, 0.95)),
        "train_radius_mad": train_mad,
        "train_radius_cv": float(train_std / (abs(train_mean) + EPS)),
        "validation_radius_mean": float(np.mean(val_radius)),
        "validation_radius_median": float(np.median(val_radius)),
        "validation_radius_p95": float(np.quantile(val_radius, 0.95)),
        "radius_ks_statistic": float(ks_2samp(train_radius, val_radius).statistic),
        "radius_wasserstein": float(wasserstein_distance(train_radius, val_radius)),
        "radius_median_shift": float(np.median(val_radius) - train_median),
        "radius_p95_shift": float(np.quantile(val_radius, 0.95) - np.quantile(train_radius, 0.95)),
    }


def centroid_scores(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.sum((x - centroid) ** 2, axis=1)


def score_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
    }


def spherical_nll(x: np.ndarray, mean: np.ndarray, variance: float) -> np.ndarray:
    variance = max(float(variance), EPS)
    dimension = x.shape[1]
    squared = np.sum((x - mean) ** 2, axis=1)
    return 0.5 * (dimension * math.log(2.0 * math.pi * variance) + squared / variance)


def margin_statistics(validation: np.ndarray, labels: np.ndarray, centroids: np.ndarray, class_index: int) -> dict[str, float]:
    mask = labels == class_index
    samples = validation[mask]
    distances = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
    own = distances[:, class_index]
    other = distances.copy()
    other[:, class_index] = np.inf
    nearest_other = np.min(other, axis=1)
    margin = nearest_other - own
    ratio = own / (nearest_other + EPS)
    return {
        "validation_margin_mean": float(np.mean(margin)),
        "validation_margin_median": float(np.median(margin)),
        "validation_margin_p10": float(np.quantile(margin, 0.10)),
        "validation_negative_margin_fraction": float(np.mean(margin < 0)),
        "validation_distance_ratio_mean": float(np.mean(ratio)),
        "validation_distance_ratio_median": float(np.median(ratio)),
    }


def bootstrap_covariance_stability(x: np.ndarray, iterations: int = 100, seed: int = 0) -> dict[str, float | int | str]:
    """Fixed PCA64 train-only nonparametric covariance bootstrap."""

    _, reference = mle_covariance(x)
    ref_values, ref_vectors = np.linalg.eigh(reference)
    order = np.argsort(ref_values)[::-1]
    ref_values = np.maximum(ref_values[order], EPS)
    ref_vectors = ref_vectors[:, order]
    ref_norm = float(np.linalg.norm(reference, ord="fro")) + EPS
    ref_condition = float(ref_values[0] / ref_values[-1])
    ref_logdet = float(np.sum(np.log(ref_values)))
    rng = np.random.default_rng(seed)
    metrics = {"frobenius_relative": [], "eigenvalue_relative": [], "condition_log10_abs": [], "logdet_abs": [], "subspace_top5": [], "subspace_top10": []}
    n = len(x)
    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        _, covariance = mle_covariance(x[indices])
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], EPS)
        vectors = vectors[:, order]
        metrics["frobenius_relative"].append(float(np.linalg.norm(covariance - reference, ord="fro") / ref_norm))
        metrics["eigenvalue_relative"].append(float(np.linalg.norm(values - ref_values) / (np.linalg.norm(ref_values) + EPS)))
        condition = float(values[0] / values[-1])
        metrics["condition_log10_abs"].append(abs(math.log10(condition) - math.log10(ref_condition)))
        metrics["logdet_abs"].append(abs(float(np.sum(np.log(values))) - ref_logdet))
        for k in (5, 10):
            angles = subspace_angles(ref_vectors[:, :k], vectors[:, :k])
            metrics[f"subspace_top{k}"].append(float(np.sqrt(np.mean(np.sin(angles) ** 2))))
    result: dict[str, float | int | str] = {"space": "pca64", "bootstrap_iterations": iterations, "bootstrap_seed": seed}
    for name, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        result[f"{name}_mean"] = float(np.mean(array))
        result[f"{name}_std"] = float(np.std(array))
        result[f"{name}_p95"] = float(np.quantile(array, 0.95))
    return result


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_table(rows: list[dict], prefix: str, excluded: set[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in rows[0]:
        if key in excluded:
            continue
        try:
            values = [float(row[key]) for row in rows]
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(values)):
            continue
        for statistic, value in numeric_summary(values).items():
            output[f"{prefix}_{key}_{statistic}"] = value
    return output
