#!/usr/bin/env python3
"""Shared, frozen-representation utilities for Stage 13A-1."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.distance import cdist


STAGE_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_ROOT = STAGE_ROOT.parent
STAGE11B_ROOT = UNKNOWN_ROOT / "stage11b_decoupled_support_readout"
STAGE11B_SCRIPTS = STAGE11B_ROOT / "scripts"
if str(STAGE11B_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE11B_SCRIPTS))

from stage11b_common import (  # noqa: E402
    centroid_scores,
    classification_metrics,
    detection_metrics,
    sha256_file,
)


CONFIG_PATH = STAGE_ROOT / "configs" / "stage13a1_config.json"
SUMMARY = STAGE_ROOT / "outputs" / "summary"
METHODS = ("CENTROID", "KNN_5", "KNN_10")
METRICS = ("auroc", "auprc", "ufar", "known_frr", "binary_f1", "known_macro_f1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV schema for empty rows: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def empirical_centroids(train_mu: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(train_labels, dtype=np.int64)
    n_classes = int(labels.max()) + 1
    if sorted(np.unique(labels).tolist()) != list(range(n_classes)):
        raise AssertionError("Known Train labels are not a complete reindexed class range")
    return np.vstack([np.asarray(train_mu)[labels == index].mean(axis=0) for index in range(n_classes)])


def local_knn_scores(
    query_mu: np.ndarray,
    predicted_classes: np.ndarray,
    train_mu: np.ndarray,
    train_labels: np.ndarray,
    ks: Iterable[int] = (5, 10),
    *,
    query_train_indices: np.ndarray | None = None,
    chunk_size: int = 512,
) -> dict[int, np.ndarray]:
    """Mean class-conditional Euclidean kNN distance with exact self masking."""

    query = np.asarray(query_mu, dtype=np.float64)
    train = np.asarray(train_mu, dtype=np.float64)
    predictions = np.asarray(predicted_classes, dtype=np.int64)
    labels = np.asarray(train_labels, dtype=np.int64)
    k_values = tuple(sorted({int(k) for k in ks}))
    if not k_values or k_values[0] <= 0:
        raise ValueError("k values must be positive")
    if set(k_values) != {5, 10}:
        raise ValueError("Stage13A-1 permits only k=5 and k=10")
    if len(query) != len(predictions):
        raise ValueError("query/prediction length mismatch")
    if query_train_indices is not None and len(query_train_indices) != len(query):
        raise ValueError("query_train_indices length mismatch")

    maximum_k = max(k_values)
    scores = {k: np.full(len(query), np.nan, dtype=np.float64) for k in k_values}
    for class_index in sorted(np.unique(predictions).tolist()):
        reference_indices = np.flatnonzero(labels == class_index)
        query_indices = np.flatnonzero(predictions == class_index)
        if len(reference_indices) < maximum_k:
            raise ValueError(f"Class {class_index} has only {len(reference_indices)} support samples")
        reference = train[reference_indices]
        for start in range(0, len(query_indices), chunk_size):
            selected = query_indices[start : start + chunk_size]
            distances = cdist(query[selected], reference, metric="euclidean")
            if query_train_indices is not None:
                global_indices = np.asarray(query_train_indices, dtype=np.int64)[selected]
                positions = np.searchsorted(reference_indices, global_indices)
                valid = positions < len(reference_indices)
                matched = np.zeros(len(selected), dtype=bool)
                matched[valid] = reference_indices[positions[valid]] == global_indices[valid]
                rows = np.flatnonzero(matched)
                distances[rows, positions[matched]] = np.inf
            finite_counts = np.sum(np.isfinite(distances), axis=1)
            if np.any(finite_counts < maximum_k):
                raise ValueError(f"Insufficient non-self neighbours in class {class_index}")
            nearest = np.partition(distances, maximum_k - 1, axis=1)[:, :maximum_k]
            nearest.sort(axis=1)
            for k in k_values:
                scores[k][selected] = nearest[:, :k].mean(axis=1)
    if any(np.any(~np.isfinite(values)) for values in scores.values()):
        raise AssertionError("Non-finite kNN score generated")
    return scores


def method_metrics(
    validation_scores: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_labels: np.ndarray,
    known_predictions: np.ndarray,
    eval_seed: int,
) -> dict[str, object]:
    detection = detection_metrics(
        np.asarray(validation_scores),
        np.asarray(known_scores),
        np.asarray(unknown_scores),
        eval_seed,
    )
    detection["threshold_source"] = "Known Validation score P95 only; numpy method=higher"
    classification = classification_metrics(
        np.asarray(known_labels, dtype=np.int64),
        np.asarray(known_predictions, dtype=np.int64),
    )
    return {
        "threshold": detection["threshold"],
        "threshold_source": detection["threshold_source"],
        "auroc": detection["auroc"],
        "auprc": detection["auprc"],
        "ufar": detection["ufar"],
        "known_frr": detection["known_frr"],
        "binary_f1": detection["binary_f1"],
        "known_accuracy": classification["accuracy"],
        "known_macro_f1": classification["macro_f1"],
        "validation_known_acceptance": detection["validation_known_acceptance"],
        "all_known_test_frr": detection["all_known_test_frr"],
        "all_unknown_test_ufar": detection["all_unknown_test_ufar"],
        "sample_count_known": detection["sample_count_known"],
        "sample_count_unknown": detection["sample_count_unknown"],
    }


def delta_row(result: dict[str, object], lhs: str, rhs: str) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": result["scenario"],
        "fold": result["fold"],
        "seed": result["seed"],
        "comparison": f"{lhs}-{rhs}",
        "lhs": lhs,
        "rhs": rhs,
        "checkpoint_sha256": result["frozen_inputs"]["checkpoint_sha256"],
    }
    for metric in METRICS:
        row[f"delta_{metric}"] = float(result["methods"][lhs][metric]) - float(
            result["methods"][rhs][metric]
        )
    return row


def summary_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def apply_gate(
    knn10_rows: list[dict[str, object]],
    knn5_rows: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    scenarios = ("a1", "a2", "a3")
    gate = config["gate"]

    def characterize(rows: list[dict[str, object]]) -> dict[str, object]:
        scenario_values: dict[str, dict[str, object]] = {}
        for scenario in scenarios:
            source = [row for row in rows if row["scenario"] == scenario]
            if len(source) != 5:
                raise AssertionError(f"Expected five paired seeds for {scenario}")
            scenario_values[scenario] = {
                "mean_delta_auroc": float(np.mean([row["delta_auroc"] for row in source])),
                "mean_delta_ufar": float(np.mean([row["delta_ufar"] for row in source])),
                "mean_delta_known_frr": float(
                    np.mean([row["delta_known_frr"] for row in source])
                ),
                "positive_seed_count": int(sum(float(row["delta_auroc"]) > 0 for row in source)),
            }
        pooled = np.asarray([row["delta_auroc"] for row in rows], dtype=np.float64)
        return {
            "scenarios": scenario_values,
            "material_scenario_count": int(
                sum(
                    value["mean_delta_auroc"] >= float(gate["material_auroc_delta"])
                    for value in scenario_values.values()
                )
            ),
            "positive_scenario_count": int(
                sum(value["mean_delta_auroc"] > 0 for value in scenario_values.values())
            ),
            "clear_harm_scenario_count": int(
                sum(
                    value["mean_delta_auroc"] <= float(gate["clear_harm_auroc_delta"])
                    for value in scenario_values.values()
                )
            ),
            "positive_seed_count": int(np.sum(pooled > 0)),
            "pooled_delta_auroc_std": float(pooled.std(ddof=1)),
        }

    primary = characterize(knn10_rows)
    sensitivity = characterize(knn5_rows)
    material = [
        scenario
        for scenario, value in primary["scenarios"].items()
        if value["mean_delta_auroc"] >= float(gate["material_auroc_delta"])
    ]
    no_clear_harm = primary["clear_harm_scenario_count"] == 0
    majority_seeds = primary["positive_seed_count"] >= int(
        gate["majority_positive_seed_count_overall"]
    )
    frr_safe = all(
        value["mean_delta_known_frr"] <= float(gate["max_mean_delta_known_frr"])
        for value in primary["scenarios"].values()
    )
    ufar_improves_in_material = all(
        primary["scenarios"][scenario]["mean_delta_ufar"] < 0 for scenario in material
    )
    go = (
        len(material) >= int(gate["material_scenarios_required_for_go"])
        and no_clear_harm
        and majority_seeds
        and frr_safe
        and ufar_improves_in_material
    )
    conditional = (
        len(material) >= int(gate["material_scenarios_required_for_conditional_go"])
        and no_clear_harm
        and frr_safe
    )
    final_gate = "GO" if go else "CONDITIONAL_GO" if conditional else "NO_GO"

    def stability_key(value: dict[str, object]) -> tuple[float, ...]:
        return (
            -float(value["clear_harm_scenario_count"]),
            float(value["positive_scenario_count"]),
            float(value["positive_seed_count"]),
            -float(value["pooled_delta_auroc_std"]),
        )

    key5, key10 = stability_key(sensitivity), stability_key(primary)
    stability = "KNN_10" if key10 > key5 else "KNN_5" if key5 > key10 else "TIE"
    return {
        "final_gate": final_gate,
        "primary": "KNN_10-CENTROID",
        "primary_characterization": primary,
        "knn5_sensitivity_characterization": sensitivity,
        "conditions": {
            "at_least_two_material_scenarios": len(material)
            >= int(gate["material_scenarios_required_for_go"]),
            "no_clear_harm_scenario": no_clear_harm,
            "overall_majority_positive_seeds": majority_seeds,
            "known_frr_safe": frr_safe,
            "ufar_lower_in_each_material_scenario": ufar_improves_in_material,
        },
        "material_scenarios": material,
        "more_stable_k": stability,
        "stability_definition": config["stability_definition"],
        "fusion_study_worthwhile": final_gate in {"GO", "CONDITIONAL_GO"},
        "new_encoder_training": False,
        "external_test_datasets_read": [],
        "frozen_experiment_modified": False,
        "next_stage_started": False,
    }

