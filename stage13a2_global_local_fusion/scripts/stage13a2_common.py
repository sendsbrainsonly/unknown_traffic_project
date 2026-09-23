#!/usr/bin/env python3
"""Shared utilities for Stage 13A-2 frozen-score fusion."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_ROOT = STAGE_ROOT.parent
STAGE13A1_ROOT = UNKNOWN_ROOT / "stage13a1_knn_local_support"
STAGE13A1_SCRIPTS = STAGE13A1_ROOT / "scripts"
if str(STAGE13A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE13A1_SCRIPTS))

from stage13_common import method_metrics, sha256_file  # noqa: E402


CONFIG_PATH = STAGE_ROOT / "configs" / "stage13a2_config.json"
SUMMARY = STAGE_ROOT / "outputs" / "summary"
METHODS = ("CENTROID", "KNN_10", "GL_FUSION")
METRICS = ("auroc", "auprc", "ufar", "known_frr", "binary_f1", "known_macro_f1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
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


def robust_parameters(validation_scores: np.ndarray, eps: float) -> dict[str, float]:
    values = np.asarray(validation_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("Known Validation score must be a finite non-empty vector")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0:
        raise ValueError("Known Validation MAD must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return {"median": median, "mad": mad, "eps": float(eps), "denominator": mad + float(eps)}


def robust_normalize(scores: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    normalized = (values - parameters["median"]) / parameters["denominator"]
    if np.any(~np.isfinite(normalized)):
        raise AssertionError("Non-finite robust-normalized score")
    return normalized


def fixed_gl_fusion(
    centroid_scores: np.ndarray,
    knn10_scores: np.ndarray,
    centroid_parameters: dict[str, float],
    knn10_parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zc = robust_normalize(centroid_scores, centroid_parameters)
    zk = robust_normalize(knn10_scores, knn10_parameters)
    if zc.shape != zk.shape:
        raise ValueError("Centroid and kNN-10 score shapes differ")
    fused = 0.5 * zc + 0.5 * zk
    return zc, zk, fused


def delta_row(result: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": result["scenario"],
        "fold": result["fold"],
        "seed": result["seed"],
        "comparison": "GL_FUSION-CENTROID",
        "lhs": "GL_FUSION",
        "rhs": "CENTROID",
        "checkpoint_sha256": result["frozen_inputs"]["checkpoint_sha256"],
    }
    for metric in METRICS:
        row[f"delta_{metric}"] = float(result["methods"]["GL_FUSION"][metric]) - float(
            result["methods"]["CENTROID"][metric]
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


def apply_gate(rows: list[dict[str, object]], config: dict[str, object]) -> dict[str, object]:
    scenario_values: dict[str, dict[str, object]] = {}
    for scenario in ("a1", "a2", "a3"):
        source = [row for row in rows if row["scenario"] == scenario]
        if len(source) != 5:
            raise AssertionError(f"Expected five paired seeds for {scenario}")
        scenario_values[scenario] = {
            "mean_delta_auroc": float(np.mean([float(row["delta_auroc"]) for row in source])),
            "mean_delta_auprc": float(np.mean([float(row["delta_auprc"]) for row in source])),
            "mean_delta_ufar": float(np.mean([float(row["delta_ufar"]) for row in source])),
            "mean_delta_known_frr": float(
                np.mean([float(row["delta_known_frr"]) for row in source])
            ),
            "positive_seed_count": int(sum(float(row["delta_auroc"]) > 0 for row in source)),
        }
    positive_count = int(sum(float(row["delta_auroc"]) > 0 for row in rows))
    gate = config["gate"]
    conditions = {
        "a1_preserved": scenario_values["a1"]["mean_delta_auroc"]
        > float(gate["a1_min_delta_auroc_exclusive"]),
        "a2_material_improvement": scenario_values["a2"]["mean_delta_auroc"]
        >= float(gate["a2_min_delta_auroc_inclusive"]),
        "a3_preserved": scenario_values["a3"]["mean_delta_auroc"]
        > float(gate["a3_min_delta_auroc_exclusive"]),
        "at_least_10_of_15_positive_seeds": positive_count
        >= int(gate["min_positive_seed_count"]),
        "known_frr_safe": all(
            value["mean_delta_known_frr"] <= float(gate["max_mean_delta_known_frr"])
            for value in scenario_values.values()
        ),
        "a2_ufar_decreased": scenario_values["a2"]["mean_delta_ufar"] < 0,
    }
    conditional = config["conditional_gate"]
    conditional_conditions = {
        "a1_not_materially_harmed": scenario_values["a1"]["mean_delta_auroc"]
        > float(conditional["a1_min_delta_auroc_exclusive"]),
        "a2_positive": scenario_values["a2"]["mean_delta_auroc"]
        > float(conditional["a2_min_delta_auroc_exclusive"]),
        "a3_not_materially_harmed": scenario_values["a3"]["mean_delta_auroc"]
        > float(conditional["a3_min_delta_auroc_exclusive"]),
        "at_least_8_of_15_positive_seeds": positive_count
        >= int(conditional["min_positive_seed_count"]),
        "known_frr_safe": all(
            value["mean_delta_known_frr"]
            <= float(conditional["max_mean_delta_known_frr"])
            for value in scenario_values.values()
        ),
        "a2_ufar_decreased": scenario_values["a2"]["mean_delta_ufar"] < 0,
    }
    if all(conditions.values()):
        decision = "GO"
    elif all(conditional_conditions.values()):
        decision = "CONDITIONAL_GO"
    else:
        decision = "NO_GO"
    return {
        "final_gate": decision,
        "primary": "GL_FUSION-CENTROID",
        "scenario_characterization": scenario_values,
        "positive_seed_count": positive_count,
        "conditions": conditions,
        "conditional_conditions": conditional_conditions,
        "new_encoder_training": False,
        "external_test_datasets_read": [],
        "cipherspectrum_test_read": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
    }

