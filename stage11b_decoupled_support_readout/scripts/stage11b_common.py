#!/usr/bin/env python3
"""Shared Stage 11B frozen-inference and support-readout implementation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score


STAGE_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_ROOT = STAGE_ROOT.parent
STAGE11A_ROOT = UNKNOWN_ROOT / "stage11a_data_anchored_prototype"
STAGE11A_SCRIPTS = STAGE11A_ROOT / "scripts"
if str(STAGE11A_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE11A_SCRIPTS))

from stage11a_common import (  # noqa: E402
    LatentOutputs,
    array_hashes,
    build_loaders,
    classification_metrics,
    detection_metrics,
    extract_outputs,
    load_checkpoint,
    model_for_method,
    prototype_alignment,
    set_reproducible,
    sha256_file,
)


CONFIG_PATH = STAGE_ROOT / "configs" / "stage11b_config.json"
METHODS = ("R0", "R1", "R2", "R3")


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


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass
class ReadoutArrays:
    scores: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]


def empirical_centroids(train: LatentOutputs) -> np.ndarray:
    n_classes = int(train.labels.max()) + 1
    if sorted(np.unique(train.labels).tolist()) != list(range(n_classes)):
        raise AssertionError("Known Train labels do not form a complete reindexed class range")
    return np.vstack([train.mu[train.labels == index].mean(axis=0) for index in range(n_classes)])


def centroid_scores(values: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sum((values[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return distances.min(axis=1), distances.argmin(axis=1).astype(np.int64, copy=False)


def density_scores(
    values: np.ndarray,
    scaler: object,
    pca: object,
    models: list[object],
) -> tuple[np.ndarray, np.ndarray]:
    transformed = pca.transform(scaler.transform(values))
    nll = np.column_stack([-model.score_samples(transformed) for model in models])
    return nll.min(axis=1), nll.argmin(axis=1).astype(np.int64, copy=False)


def _method_arrays(
    role: str,
    outputs: LatentOutputs,
    centroids: np.ndarray,
    frozen_density: dict[str, object],
) -> dict[str, ReadoutArrays]:
    r1_scores, r1_predictions = centroid_scores(outputs.mu, centroids)
    r2_scores, r2_predictions = density_scores(
        outputs.mu,
        frozen_density["scaler"],
        frozen_density["pca"],
        frozen_density["models"]["K1"],
    )
    r3_scores, r3_predictions = density_scores(
        outputs.mu,
        frozen_density["scaler"],
        frozen_density["pca"],
        frozen_density["models"]["K2"],
    )
    del role
    return {
        "R0": ReadoutArrays(outputs.native_scores, outputs.predictions),
        "R1": ReadoutArrays(r1_scores, r1_predictions),
        "R2": ReadoutArrays(r2_scores, r2_predictions),
        "R3": ReadoutArrays(r3_scores, r3_predictions),
    }


def _detector_metadata(method: str) -> dict[str, object]:
    if method == "R0":
        return {
            "support_source": "frozen learned B0 prototypes",
            "score": "forward Gaussian KL with query logvar and unit prototype covariance",
        }
    if method == "R1":
        return {
            "support_source": "Known Train deterministic mu empirical centroids",
            "score": "minimum squared Euclidean distance",
        }
    components = 1 if method == "R2" else 2
    return {
        "support_source": "read-only Stage11A B0 Known-Train density artifact",
        "score": "negative maximum per-class log likelihood",
        "representation": "Known-Train StandardScaler -> Known-Train randomized PCA64",
        "components": components,
        "covariance_type": "full",
        "reg_covar": 1e-3,
        "n_init": 3,
        "max_iter": 300,
        "random_state": 0,
    }


def balanced_indices(n_known: int, n_unknown: int, eval_seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(n_known, n_unknown)
    rng = np.random.default_rng(eval_seed)
    known_idx = rng.choice(n_known, size=count, replace=False)
    unknown_idx = rng.choice(n_unknown, size=count, replace=False)
    return known_idx, unknown_idx


def per_class_f1(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    return f1_score(
        labels,
        predictions,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0,
    )


def evaluate_readouts(
    model: torch.nn.Module,
    loaders: object,
    device: torch.device,
    frozen_density: dict[str, object],
    eval_seed: int,
    class_names: dict[int, str],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, np.ndarray],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    outputs = {
        "train": extract_outputs(model, loaders.anchor_train, device),
        "validation": extract_outputs(model, loaders.validation, device),
        "known_test": extract_outputs(model, loaders.known_test, device),
        "unknown_test": extract_outputs(model, loaders.unknown_test, device),
    }
    centroids = empirical_centroids(outputs["train"])
    arrays_by_role = {
        role: _method_arrays(role, latent, centroids, frozen_density)
        for role, latent in outputs.items()
    }
    detector_results: dict[str, object] = {}
    thresholds: dict[str, object] = {}
    for method in METHODS:
        detection = detection_metrics(
            arrays_by_role["validation"][method].scores,
            arrays_by_role["known_test"][method].scores,
            arrays_by_role["unknown_test"][method].scores,
            eval_seed,
        )
        validation_classification = classification_metrics(
            outputs["validation"].labels,
            arrays_by_role["validation"][method].predictions,
        )
        known_classification = classification_metrics(
            outputs["known_test"].labels,
            arrays_by_role["known_test"][method].predictions,
        )
        detector_results[method] = {
            **_detector_metadata(method),
            "validation_known": {
                **validation_classification,
                "acceptance": detection["validation_known_acceptance"],
                "frr": 1.0 - detection["validation_known_acceptance"],
            },
            "known_test": {
                **known_classification,
                "acceptance": 1.0 - detection["all_known_test_frr"],
                "frr": detection["all_known_test_frr"],
            },
            "unknown_test": {
                "ufar": detection["all_unknown_test_ufar"],
                "rejection": 1.0 - detection["all_unknown_test_ufar"],
            },
            "combined_balanced_1to1": detection,
        }
        thresholds[method] = {
            "threshold": detection["threshold"],
            "source": "Known Validation anomaly score P95 only",
            "numpy_method": "higher",
            "validation_known_acceptance": detection["validation_known_acceptance"],
        }

    alignment_rows, alignment_summary = prototype_alignment(
        model, outputs["train"], loaders.known_classes
    )
    alignment_by_index = {int(row["class_index"]): row for row in alignment_rows}
    n_classes = len(loaders.known_classes)
    known_f1 = {
        method: per_class_f1(
            outputs["known_test"].labels,
            arrays_by_role["known_test"][method].predictions,
            n_classes,
        )
        for method in METHODS
    }
    unknown_scores = {
        method: arrays_by_role["unknown_test"][method].scores for method in METHODS
    }
    unknown_predictions = {
        method: arrays_by_role["unknown_test"][method].predictions for method in METHODS
    }
    full_models = frozen_density["models"]["K1"]
    prototypes = model.prototypes.detach().cpu().numpy().astype(np.float64)
    per_class_rows: list[dict[str, object]] = []
    for class_index, original_class_id in enumerate(loaders.known_classes):
        model_k1 = full_models[class_index]
        covariance = np.asarray(model_k1.covariances_[0], dtype=np.float64)
        sign, logdet = np.linalg.slogdet(covariance)
        row: dict[str, object] = {
            "class_index": class_index,
            "original_class_id": original_class_id,
            "class_name": class_names[original_class_id],
            "train_samples": int(np.sum(outputs["train"].labels == class_index)),
            "learned_prototype_vector_json": json.dumps(prototypes[class_index].tolist()),
            "empirical_centroid_vector_json": json.dumps(centroids[class_index].astype(float).tolist()),
            "r2_mean_pca64_vector_json": json.dumps(np.asarray(model_k1.means_[0], dtype=float).tolist()),
            "r2_covariance_shape": "64x64",
            "r2_covariance_trace": float(np.trace(covariance)),
            "r2_covariance_logdet": float(logdet) if sign > 0 else float("nan"),
            "r2_covariance_condition_number": float(np.linalg.cond(covariance)),
            "full_covariance_storage": "support_models.joblib:R2",
            "prototype_gap": alignment_by_index[class_index]["gap"],
            "median_within_class_radius": alignment_by_index[class_index]["median_within_class_radius"],
            "normalized_gap": alignment_by_index[class_index]["normalized_gap"],
        }
        for method in METHODS:
            threshold = float(thresholds[method]["threshold"])
            accepted = unknown_scores[method] < threshold
            absorbed = int(np.sum(accepted & (unknown_predictions[method] == class_index)))
            accepted_total = int(np.sum(accepted))
            row[f"{method.lower()}_known_test_f1"] = float(known_f1[method][class_index])
            row[f"{method.lower()}_unknown_absorbed_count"] = absorbed
            row[f"{method.lower()}_unknown_absorption_rate_all_unknown"] = (
                absorbed / len(outputs["unknown_test"].labels)
            )
            row[f"{method.lower()}_unknown_absorption_share_of_accepted"] = (
                absorbed / accepted_total if accepted_total else 0.0
            )
        per_class_rows.append(row)

    absorption_rows: list[dict[str, object]] = []
    unknown_labels = outputs["unknown_test"].labels
    for method in METHODS:
        threshold = float(thresholds[method]["threshold"])
        accepted = unknown_scores[method] < threshold
        for true_unknown_class in loaders.unknown_classes:
            true_mask = unknown_labels == true_unknown_class
            total = int(np.sum(true_mask))
            accepted_total = int(np.sum(true_mask & accepted))
            for class_index, predicted_original_id in enumerate(loaders.known_classes):
                count = int(
                    np.sum(true_mask & accepted & (unknown_predictions[method] == class_index))
                )
                absorption_rows.append(
                    {
                        "method": method,
                        "true_unknown_class": true_unknown_class,
                        "true_unknown_name": class_names[true_unknown_class],
                        "predicted_known_class_index": class_index,
                        "predicted_known_class": predicted_original_id,
                        "predicted_known_name": class_names[predicted_original_id],
                        "true_unknown_samples": total,
                        "accepted_total_for_true_unknown": accepted_total,
                        "accepted_count": count,
                        "absorption_rate": count / total if total else 0.0,
                        "share_of_accepted_true_unknown": count / accepted_total if accepted_total else 0.0,
                    }
                )

    support_bundle = {
        "centroids": centroids,
        "known_class_ids": np.asarray(loaders.known_classes, dtype=np.int64),
        "class_names": class_names,
        "R2_R3_frozen_density": frozen_density,
        "fit_policy": {
            "R1": "computed from current frozen checkpoint Known Train deterministic mu",
            "R2_R3": "reused read-only Stage11A B0 models fit from the same Known Train mu",
            "checkpoint_write_back": False,
        },
    }
    sample_arrays: dict[str, np.ndarray] = {
        "train_mu": outputs["train"].mu,
        "train_logvar": outputs["train"].logvar,
        "train_labels_reindexed": outputs["train"].labels,
        "validation_mu": outputs["validation"].mu,
        "validation_logvar": outputs["validation"].logvar,
        "validation_labels_reindexed": outputs["validation"].labels,
        "known_test_mu": outputs["known_test"].mu,
        "known_test_logvar": outputs["known_test"].logvar,
        "known_test_labels_reindexed": outputs["known_test"].labels,
        "known_test_labels_original": np.asarray(loaders.known_classes, dtype=np.int64)[
            outputs["known_test"].labels
        ],
        "unknown_test_mu": outputs["unknown_test"].mu,
        "unknown_test_logvar": outputs["unknown_test"].logvar,
        "unknown_test_labels_original": outputs["unknown_test"].labels,
    }
    for role in ("validation", "known_test", "unknown_test"):
        for method in METHODS:
            sample_arrays[f"{role}_{method.lower()}_scores"] = arrays_by_role[role][method].scores
            sample_arrays[f"{role}_{method.lower()}_predictions_reindexed"] = arrays_by_role[role][
                method
            ].predictions
    known_idx, unknown_idx = balanced_indices(
        len(outputs["known_test"].labels), len(outputs["unknown_test"].labels), eval_seed
    )
    sample_arrays["balanced_known_indices"] = known_idx
    sample_arrays["balanced_unknown_indices"] = unknown_idx
    evaluation = {
        "sample_counts": {
            "train_known": len(outputs["train"].labels),
            "validation_known": len(outputs["validation"].labels),
            "test_known": len(outputs["known_test"].labels),
            "test_unknown": len(outputs["unknown_test"].labels),
            "balanced_each_side": len(known_idx),
        },
        "prototype_alignment": alignment_summary,
        "prototype_alignment_by_class": alignment_rows,
        "detectors": detector_results,
    }
    return evaluation, support_bundle, sample_arrays, per_class_rows, absorption_rows


def metric_abs_errors(observed: dict[str, object], expected: dict[str, object]) -> dict[str, float]:
    observed_detection = observed["combined_balanced_1to1"]
    expected_detection = expected["detection"]
    return {
        "auroc_abs_error": abs(float(observed_detection["auroc"]) - float(expected_detection["auroc"])),
        "threshold_abs_error": abs(
            float(observed_detection["threshold"]) - float(expected_detection["threshold"])
        ),
        "binary_f1_abs_error": abs(
            float(observed_detection["binary_f1"]) - float(expected_detection["binary_f1"])
        ),
        "known_accuracy_abs_error": abs(
            float(observed["known_test"]["accuracy"]) - float(expected["known_test"]["accuracy"])
        ),
        "known_macro_f1_abs_error": abs(
            float(observed["known_test"]["macro_f1"]) - float(expected["known_test"]["macro_f1"])
        ),
    }


def stage11a_expected(stage11a_results: dict[str, object], method: str) -> dict[str, object]:
    if method == "R0":
        return stage11a_results["native"]
    key = "K1" if method == "R2" else "K2"
    return stage11a_results["density_diagnostics"]["methods"][key]

