#!/usr/bin/env python3
"""Fit train-only PCA64 K1/K2 models and Known-Validation-only boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import (
    AccessLedger,
    CONFIG_PATH,
    STAGE8_ROOT,
    VALID_SETTINGS,
    conservative_empirical_threshold,
    linear_p05,
    load_fold,
    require,
    score_density_matrix,
    score_k2_matrices,
    sha256_file,
    true_class_component_assignments,
    verify_all_provenance,
    write_csv,
    write_json,
)


def fit_one(values: np.ndarray, n_components: int) -> tuple[GaussianMixture, list[dict[str, str]]]:
    caught_rows: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-3,
            n_init=3,
            max_iter=300,
            random_state=0,
        ).fit(values)
    for warning in caught:
        caught_rows.append(
            {
                "warning_category": warning.category.__name__,
                "warning_message": str(warning.message),
            }
        )
    return model, caught_rows


def fit_class(class_index: int, class_name: str, train_values: np.ndarray) -> dict[str, object]:
    k1, k1_warnings = fit_one(train_values, 1)
    k2, k2_warnings = fit_one(train_values, 2)
    return {
        "class_index": class_index,
        "class_name": class_name,
        "k1": k1,
        "k2": k2,
        "k1_warnings": k1_warnings,
        "k2_warnings": k2_warnings,
    }


def read_manifest(path: Path, ledger: AccessLedger, role: str) -> list[dict[str, str]]:
    ledger.record(path, "csv", role)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(all(row["source_role"] == role for row in rows), f"{role}: manifest role mismatch")
    require([int(row["row_index"]) for row in rows] == list(range(len(rows))), f"{role}: manifest row order mismatch")
    return rows


def assignment_arrays(
    models: dict[str, GaussianMixture],
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    components = np.empty(len(values), dtype=np.int64)
    local_scores = np.empty(len(values), dtype=np.float64)
    class_scores = np.empty(len(values), dtype=np.float64)
    posteriors = np.empty(len(values), dtype=np.float64)
    for class_name, model in models.items():
        positions = np.flatnonzero(labels == class_name)
        require(len(positions) > 0, f"missing samples for class {class_name}")
        block = np.asarray(model._estimate_weighted_log_prob(values[positions]), dtype=np.float64)
        selected = np.argmax(block, axis=1)
        selected_scores = block[np.arange(len(positions)), selected]
        normalization = np.logaddexp(block[:, 0], block[:, 1])
        components[positions] = selected
        local_scores[positions] = selected_scores
        class_scores[positions] = normalization
        posteriors[positions] = np.exp(selected_scores - normalization)
    return components, local_scores, class_scores, posteriors


def write_assignments(
    path: Path,
    setting: str,
    manifest: list[dict[str, str]],
    components: np.ndarray,
    local_scores: np.ndarray,
    class_scores: np.ndarray,
    posteriors: np.ndarray,
) -> None:
    fields = [
        "setting", "sample_id", "row_index", "class_name", "component_id",
        "posterior_probability", "weighted_local_log_score", "true_class_k2_log_score",
        "assignment_scope", "fit_split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(manifest):
            writer.writerow(
                {
                    "setting": setting,
                    "sample_id": row["sample_id"],
                    "row_index": index,
                    "class_name": row["class_name"],
                    "component_id": int(components[index]),
                    "posterior_probability": float(posteriors[index]),
                    "weighted_local_log_score": float(local_scores[index]),
                    "true_class_k2_log_score": float(class_scores[index]),
                    "assignment_scope": "corresponding true-class train-fitted K2 only",
                    "fit_split": "KNOWN_TRAIN_ONLY",
                }
            )


def covariance_diagnostics(model: GaussianMixture) -> tuple[list[float], list[float]]:
    logdets: list[float] = []
    conditions: list[float] = []
    for covariance in model.covariances_:
        sign, logdet = np.linalg.slogdet(covariance)
        require(sign > 0 and np.isfinite(logdet), "non-positive-definite covariance")
        logdets.append(float(logdet))
        conditions.append(float(np.linalg.cond(covariance)))
    return logdets, conditions


def class_and_component_coverage(
    setting: str,
    class_names: list[str],
    true_labels: np.ndarray,
    true_components: np.ndarray,
    decisions: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for method, accepted in decisions.items():
        class_acceptance: list[float] = []
        class_frr: list[float] = []
        component_gaps: list[float] = []
        for class_name in class_names:
            class_mask = true_labels == class_name
            acceptance = float(np.mean(accepted[class_mask]))
            class_acceptance.append(acceptance)
            class_frr.append(1.0 - acceptance)
            component_acceptance: list[float] = []
            for component in range(2):
                mask = class_mask & (true_components == component)
                value = float(np.mean(accepted[mask])) if np.any(mask) else math.nan
                component_acceptance.append(value)
                component_rows.append(
                    {
                        "setting": setting,
                        "method": method,
                        "class_name": class_name,
                        "component_id": component,
                        "support": int(np.sum(mask)),
                        "acceptance": value,
                        "FRR": 1.0 - value if np.isfinite(value) else math.nan,
                    }
                )
            finite_values = [value for value in component_acceptance if np.isfinite(value)]
            gap = abs(finite_values[0] - finite_values[1]) if len(finite_values) == 2 else math.nan
            if np.isfinite(gap):
                component_gaps.append(gap)
            class_rows.append(
                {
                    "setting": setting,
                    "method": method,
                    "class_name": class_name,
                    "support": int(np.sum(class_mask)),
                    "acceptance": acceptance,
                    "FRR": 1.0 - acceptance,
                    "component_0_acceptance": component_acceptance[0],
                    "component_1_acceptance": component_acceptance[1],
                    "component_acceptance_gap": gap,
                }
            )
        overall = float(np.mean(accepted))
        summary.append(
            {
                "setting": setting,
                "method": method,
                "overall_acceptance": overall,
                "overall_FRR": 1.0 - overall,
                "per_class_acceptance_min": float(np.min(class_acceptance)),
                "per_class_acceptance_max": float(np.max(class_acceptance)),
                "per_class_FRR_std": float(np.std(class_frr)),
                "mean_per_class_component_acceptance_gap": float(np.mean(component_gaps)),
                "max_per_class_component_acceptance_gap": float(np.max(component_gaps)),
                "calibration_split": "KNOWN_VALIDATION_ONLY",
                "test_samples_used": 0,
                "unknown_samples_used": 0,
            }
        )
    return summary, class_rows, component_rows


def threshold_payload(setting: str, detector: str, score: str, method: str, threshold: float, actual: float) -> dict[str, object]:
    return {
        "setting": setting,
        "detector": detector,
        "score_definition": score,
        "score_semantics": "higher_is_more_known",
        "threshold": float(threshold),
        "target_known_validation_acceptance": 0.95,
        "actual_known_validation_acceptance": float(actual),
        "known_validation_FRR": float(1.0 - actual),
        "selection_method": method,
        "fit_split": "KNOWN_VALIDATION_ONLY",
        "known_test_samples_used": 0,
        "unknown_samples_used": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    require(args.jobs >= 1, "--jobs must be positive")
    verify_all_provenance()
    setting = args.setting
    fold = load_fold(setting)
    class_names = [str(value) for value in fold["known_classes"]]
    artifact_dir = STAGE8_ROOT / "artifacts" / setting
    output_dir = STAGE8_ROOT / "outputs" / setting
    require((artifact_dir / "mu_train.npy").is_file() and (artifact_dir / "mu_val.npy").is_file(), "mu_x extraction must complete first")
    require(not (artifact_dir / "scaler.joblib").exists(), f"{setting}: refusing to overwrite fitted artifacts")
    ledger = AccessLedger(setting, output_dir / "file_access_audit.json")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    train_manifest = read_manifest(artifact_dir / "train_manifest.csv", ledger, "KNOWN_TRAIN")
    val_manifest = read_manifest(artifact_dir / "val_manifest.csv", ledger, "KNOWN_VALIDATION")
    train_mu_path = artifact_dir / "mu_train.npy"
    val_mu_path = artifact_dir / "mu_val.npy"
    native_path = artifact_dir / "native_val_known_scores.npy"
    for path, role in ((train_mu_path, "KNOWN_TRAIN"), (val_mu_path, "KNOWN_VALIDATION"), (native_path, "KNOWN_VALIDATION")):
        ledger.record(path, "npy", role)
    train_mu = np.load(train_mu_path, mmap_mode="r", allow_pickle=False)
    val_mu = np.load(val_mu_path, mmap_mode="r", allow_pickle=False)
    native_val = np.load(native_path, mmap_mode="r", allow_pickle=False)
    require(train_mu.shape == (int(fold["known_train_count"]), 128), "train mu shape mismatch")
    require(val_mu.shape == (int(fold["known_validation_count"]), 128), "validation mu shape mismatch")
    require(np.isfinite(train_mu).all() and np.isfinite(val_mu).all(), "non-finite mu_x")
    train_labels = np.asarray([row["class_name"] for row in train_manifest], dtype=object)
    val_labels = np.asarray([row["class_name"] for row in val_manifest], dtype=object)
    require(set(train_labels) == set(class_names) and set(val_labels) == set(class_names), "Known class set mismatch")

    scaler = StandardScaler(copy=True).fit(np.asarray(train_mu, dtype=np.float64))
    train_scaled = scaler.transform(train_mu)
    val_scaled = scaler.transform(val_mu)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(train_scaled)
    train_z = pca.transform(train_scaled)
    val_z = pca.transform(val_scaled)
    require(train_z.shape == (len(train_mu), 64) and val_z.shape == (len(val_mu), 64), "PCA64 shape mismatch")
    require(np.isfinite(train_z).all() and np.isfinite(val_z).all(), "non-finite PCA64 values")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(pca, artifact_dir / "pca64.joblib")
    np.save(artifact_dir / "z_train_pca64.npy", train_z, allow_pickle=False)
    np.save(artifact_dir / "z_val_pca64.npy", val_z, allow_pickle=False)
    write_json(
        output_dir / "scaler_pca_audit.json",
        {
            "setting": setting,
            "scaler_fit_split": "KNOWN_TRAIN_ONLY",
            "pca_fit_split": "KNOWN_TRAIN_ONLY",
            "pca_dimension": 64,
            "pca_svd_solver": "randomized",
            "pca_random_state": 0,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "known_test_samples_used": 0,
            "unknown_samples_used": 0,
        },
    )

    fitted = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(fit_class)(index, class_name, train_z[train_labels == class_name])
        for index, class_name in enumerate(class_names)
    )
    fitted.sort(key=lambda row: int(row["class_index"]))
    k1_models = {str(row["class_name"]): row["k1"] for row in fitted}
    k2_models = {str(row["class_name"]): row["k2"] for row in fitted}
    require(list(k1_models) == class_names and list(k2_models) == class_names, "model class ordering changed")
    single_dir = artifact_dir / "single_k1"
    multi_dir = artifact_dir / "multi_k2"
    (single_dir / "classes").mkdir(parents=True)
    (multi_dir / "classes").mkdir(parents=True)
    joblib.dump(k1_models, single_dir / "models.joblib")
    joblib.dump(k2_models, multi_dir / "models.joblib")

    train_components, train_local_true, train_class_true, train_posteriors = assignment_arrays(k2_models, train_z, train_labels)
    val_components, val_local_true, val_class_true, val_posteriors = assignment_arrays(k2_models, val_z, val_labels)
    write_assignments(output_dir / "component_assignments_train.csv", setting, train_manifest, train_components, train_local_true, train_class_true, train_posteriors)
    write_assignments(output_dir / "component_assignments_val.csv", setting, val_manifest, val_components, val_local_true, val_class_true, val_posteriors)

    diagnostic_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    for row in fitted:
        class_index = int(row["class_index"])
        class_name = str(row["class_name"])
        k1: GaussianMixture = row["k1"]
        k2: GaussianMixture = row["k2"]
        np.savez(
            single_dir / f"classes/class_{class_index:02d}.npz",
            class_name=np.asarray(class_name), means=k1.means_, covariances=k1.covariances_, weights=k1.weights_,
        )
        np.savez(
            multi_dir / f"classes/class_{class_index:02d}.npz",
            class_name=np.asarray(class_name), means=k2.means_, covariances=k2.covariances_, weights=k2.weights_,
        )
        train_mask = train_labels == class_name
        val_mask = val_labels == class_name
        train_counts = np.bincount(train_components[train_mask], minlength=2)
        val_counts = np.bincount(val_components[val_mask], minlength=2)
        train_ratio = train_counts / train_counts.sum()
        val_ratio = val_counts / val_counts.sum()
        logdets, conditions = covariance_diagnostics(k2)
        delta = k2.means_[0] - k2.means_[1]
        pooled = 0.5 * (k2.covariances_[0] + k2.covariances_[1])
        euclidean = float(np.linalg.norm(delta))
        mahalanobis = float(np.sqrt(max(0.0, delta @ np.linalg.solve(pooled, delta))))
        k1_nll = float(-np.mean(k1.score_samples(val_z[val_mask])))
        k2_nll = float(-np.mean(k2.score_samples(val_z[val_mask])))
        density_rows.append(
            {
                "setting": setting,
                "class_name": class_name,
                "validation_samples": int(np.sum(val_mask)),
                "NLL_K1": k1_nll,
                "NLL_K2": k2_nll,
                "DeltaNLL2": k1_nll - k2_nll,
                "K2_validation_NLL_better": k2_nll < k1_nll,
                "interpretation_limit": "density fit only; not Unknown utility",
            }
        )
        diagnostic_rows.append(
            {
                "setting": setting,
                "class_name": class_name,
                "converged": bool(k2.converged_),
                "n_iter": int(k2.n_iter_),
                "weights": json.dumps(k2.weights_.tolist()),
                "component_train_counts": json.dumps(train_counts.tolist()),
                "component_validation_counts": json.dumps(val_counts.tolist()),
                "min_component_weight": float(np.min(k2.weights_)),
                "train_val_component_TV": float(0.5 * np.abs(train_ratio - val_ratio).sum()),
                "covariance_logdets": json.dumps(logdets),
                "covariance_condition_numbers": json.dumps(conditions),
                "mean_euclidean_separation": euclidean,
                "pooled_mahalanobis_separation": mahalanobis,
                "warning_count": len(row["k2_warnings"]),
                "warning_categories": json.dumps([item["warning_category"] for item in row["k2_warnings"]]),
                "warning_messages": json.dumps([item["warning_message"] for item in row["k2_warnings"]]),
            }
        )
        for detector, model, caught in (("Single-Full-K1", k1, row["k1_warnings"]), ("Multi-Full-K2", k2, row["k2_warnings"])):
            if caught:
                for item in caught:
                    warning_rows.append({"setting": setting, "class_name": class_name, "detector": detector, **item, "converged": bool(model.converged_), "n_iter": int(model.n_iter_)})
            else:
                warning_rows.append({"setting": setting, "class_name": class_name, "detector": detector, "warning_category": "", "warning_message": "", "converged": bool(model.converged_), "n_iter": int(model.n_iter_)})
        for component in range(2):
            val_n = int(val_counts[component])
            capacity_rows.append(
                {
                    "setting": setting,
                    "class_name": class_name,
                    "component_id": component,
                    "train_n": int(train_counts[component]),
                    "val_n": val_n,
                    "weight": float(k2.weights_[component]),
                    "val_n_lt_10": val_n < 10,
                    "val_n_lt_20": val_n < 20,
                    "val_n_lt_30": val_n < 30,
                    "val_n_lt_50": val_n < 50,
                    "val_n_lt_100": val_n < 100,
                }
            )
    write_csv(output_dir / "k2_component_diagnostics.csv", diagnostic_rows)
    write_csv(output_dir / "component_calibration_capacity.csv", capacity_rows)
    write_csv(output_dir / "gmm_warnings.csv", warning_rows)
    write_csv(output_dir / "k1_k2_density_comparison.csv", density_rows)

    _, k1_val_matrix = score_density_matrix(k1_models, val_z)
    scored_names, k2_val_matrix, k2_local_matrix = score_k2_matrices(k2_models, val_z)
    require(scored_names == class_names, "K2 score ordering changed")
    replay_components, replay_local, _ = true_class_component_assignments(val_labels, class_names, k2_local_matrix)
    require(np.array_equal(replay_components, val_components), "true-class component assignment replay failed")
    require(np.allclose(replay_local, val_local_true, rtol=0, atol=0), "true-class local score replay failed")
    single_global_score = k1_val_matrix.max(axis=1)
    multi_global_score = k2_val_matrix.max(axis=1)
    native_threshold = linear_p05(np.asarray(native_val, dtype=np.float64))
    native_accept = np.asarray(native_val) >= native_threshold
    single_threshold, single_actual = conservative_empirical_threshold(single_global_score, 0.95)
    multi_threshold, multi_actual = conservative_empirical_threshold(multi_global_score, 0.95)
    single_accept = single_global_score >= single_threshold
    multi_accept = multi_global_score >= multi_threshold

    class_thresholds = np.empty(len(class_names), dtype=np.float64)
    class_threshold_rows: list[dict[str, object]] = []
    component_thresholds = np.empty(2 * len(class_names), dtype=np.float64)
    component_threshold_rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        positions = np.flatnonzero(val_labels == class_name)
        own_class = k2_val_matrix[positions, class_index]
        class_tau = linear_p05(own_class)
        class_thresholds[class_index] = class_tau
        class_threshold_rows.append(
            {
                "setting": setting,
                "class_name": class_name,
                "class_index": class_index,
                "validation_n": len(positions),
                "threshold": class_tau,
                "quantile": 0.05,
                "true_class_validation_acceptance": float(np.mean(own_class >= class_tau)),
                "fit_split": "KNOWN_VALIDATION_ONLY",
            }
        )
        for component in range(2):
            selected = positions[val_components[positions] == component]
            scores = val_local_true[selected]
            raw_tau = linear_p05(scores) if len(scores) else math.nan
            fallback = len(scores) < 30
            effective = class_tau if fallback else raw_tau
            component_thresholds[2 * class_index + component] = effective
            component_threshold_rows.append(
                {
                    "setting": setting,
                    "class_name": class_name,
                    "class_index": class_index,
                    "component_id": component,
                    "validation_n": len(scores),
                    "raw_component_p05": raw_tau,
                    "class_p05_threshold": class_tau,
                    "effective_threshold": effective,
                    "quantile": 0.05,
                    "fallback": fallback,
                    "fallback_reason": "component validation n < 30" if fallback else "",
                    "fallback_min_validation_n": 30,
                    "fit_split": "KNOWN_VALIDATION_ONLY",
                }
            )
    write_json(
        artifact_dir / "native_threshold.json",
        threshold_payload(setting, "Native", "negative minimum KL divergence to frozen prototypes", "Stage3 linear P05 quantile", native_threshold, float(np.mean(native_accept))),
    )
    write_json(
        artifact_dir / "single_global_threshold.json",
        threshold_payload(setting, "Single-Full-K1", "maximum class-conditional K1 log density", "closest empirical acceptance; ties choose higher threshold", single_threshold, single_actual),
    )
    write_json(
        artifact_dir / "multi_global_threshold.json",
        threshold_payload(setting, "Multi-Global-K2", "maximum class-conditional K2 log density", "closest empirical acceptance; ties choose higher threshold", multi_threshold, multi_actual),
    )
    write_csv(artifact_dir / "class_p05_thresholds.csv", class_threshold_rows)
    write_csv(artifact_dir / "component_p05_thresholds.csv", component_threshold_rows)

    class_margins = k2_val_matrix - class_thresholds[None, :]
    component_margins = k2_local_matrix - component_thresholds[None, :]
    class_accept = class_margins.max(axis=1) >= 0
    component_accept = component_margins.max(axis=1) >= 0
    decisions = {
        "Native": native_accept,
        "Single-Full-K1": single_accept,
        "Multi-Global-K2": multi_accept,
        "Class-P05-K2": class_accept,
        "Component-P05-K2": component_accept,
    }
    boundary_summary, boundary_class, boundary_component = class_and_component_coverage(
        setting, class_names, val_labels, val_components, decisions
    )
    write_csv(output_dir / "known_val_boundary_summary.csv", boundary_summary)
    write_csv(output_dir / "known_val_boundary_per_class.csv", boundary_class)
    write_csv(output_dir / "known_val_boundary_per_component.csv", boundary_component)
    fallback_count = sum(bool(row["fallback"]) for row in component_threshold_rows)
    write_csv(
        output_dir / "fallback_summary.csv",
        [
            {
                "setting": setting,
                "components_total": len(component_threshold_rows),
                "components_val_n_lt_30": fallback_count,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / len(component_threshold_rows),
                "fallback_rule": "component validation n < 30 -> class P05 threshold",
                "rule_frozen_before_test": True,
            }
        ],
    )
    write_json(
        output_dir / "density_boundary_metadata.json",
        {
            "setting": setting,
            "status": "PASS",
            "scaler_fit_split": "KNOWN_TRAIN_ONLY",
            "pca_fit_split": "KNOWN_TRAIN_ONLY",
            "k1_fit_split": "KNOWN_TRAIN_ONLY",
            "k2_fit_split": "KNOWN_TRAIN_ONLY",
            "threshold_fit_split": "KNOWN_VALIDATION_ONLY",
            "pca_dimension": 64,
            "pca_cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "k1_all_converged": all(bool(model.converged_) for model in k1_models.values()),
            "k2_all_converged": all(bool(model.converged_) for model in k2_models.values()),
            "gmm_warning_count": sum(bool(row["warning_category"]) for row in warning_rows),
            "minimum_k2_component_weight": min(float(np.min(model.weights_)) for model in k2_models.values()),
            "component_val_n_lt_30": fallback_count,
            "component_fallback_rate": fallback_count / len(component_threshold_rows),
            "mean_DeltaNLL2": float(np.mean([float(row["DeltaNLL2"]) for row in density_rows])),
            "median_DeltaNLL2": float(np.median([float(row["DeltaNLL2"]) for row in density_rows])),
            "classes_K2_validation_NLL_better": sum(bool(row["K2_validation_NLL_better"]) for row in density_rows),
            "known_test_opened": 0,
            "unknown_test_opened": 0,
            "unknown_inference_executed": False,
            "interpretation_limit": "K2 validation NLL improvement is density-fit evidence, not Unknown utility",
        },
    )
    ledger.record(train_mu_path, "npy", "KNOWN_TRAIN", len(train_mu))
    ledger.record(val_mu_path, "npy", "KNOWN_VALIDATION", len(val_mu))
    ledger.record(native_path, "npy", "KNOWN_VALIDATION", len(native_val))
    ledger.save()
    print(
        json.dumps(
            {
                "setting": setting,
                "status": "PASS",
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "k1_all_converged": all(bool(model.converged_) for model in k1_models.values()),
                "k2_all_converged": all(bool(model.converged_) for model in k2_models.values()),
                "fallback_count": fallback_count,
                "known_test_opened": 0,
                "unknown_test_opened": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
