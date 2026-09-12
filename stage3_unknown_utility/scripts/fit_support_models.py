#!/usr/bin/env python3
"""Fit train-only PCA64 and frozen K1/K2 full-covariance support models."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import (
    EXPECTED_EXECUTION_PLAN_RAW_SHA256,
    EXPECTED_PROTOCOL_CANONICAL_SHA256,
    STAGE3_ROOT,
    known_acceptance_threshold,
    latent_columns,
    load_fold,
    score_density_models,
    sha256_file,
    verify_frozen_inputs,
)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_one(
    values: np.ndarray,
    components: int,
    setting: str,
    class_name: str,
    detector: str,
    warning_rows: list[dict[str, object]],
) -> GaussianMixture:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-3,
            n_init=3,
            max_iter=300,
            random_state=0,
        ).fit(values)
    if not caught:
        warning_rows.append(
            {
                "setting": setting,
                "class_name": class_name,
                "detector": detector,
                "warning_category": "",
                "warning_message": "",
                "converged": model.converged_,
                "n_iter": model.n_iter_,
            }
        )
    else:
        for warning in caught:
            warning_rows.append(
                {
                    "setting": setting,
                    "class_name": class_name,
                    "detector": detector,
                    "warning_category": warning.category.__name__,
                    "warning_message": str(warning.message),
                    "converged": model.converged_,
                    "n_iter": model.n_iter_,
                }
            )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    verify_frozen_inputs()
    fold = load_fold(args.setting)
    output_dir = STAGE3_ROOT / "outputs" / args.setting
    artifact_dir = STAGE3_ROOT / "artifacts" / args.setting
    frozen_path = output_dir / "frozen_model_hashes.json"
    if frozen_path.exists():
        raise RuntimeError("refusing to overwrite already frozen support models")
    train_path = artifact_dir / "train_known_mu.parquet"
    val_path = artifact_dir / "val_known_mu.parquet"
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    columns = latent_columns(train.columns)
    if columns != latent_columns(val.columns):
        raise RuntimeError("train/validation latent columns differ")
    if set(train["class_name"]) != set(fold["known_classes"]):
        raise RuntimeError("train latent class set differs from frozen Known classes")
    if set(val["class_name"]) != set(fold["known_classes"]):
        raise RuntimeError("validation latent class set differs from frozen Known classes")
    if len(train) != fold["formal_counts"]["known_train"] or len(val) != fold["formal_counts"]["known_validation"]:
        raise RuntimeError("train/validation latent count differs from frozen fold")
    train_mu = train[columns].to_numpy(dtype=np.float64, copy=True)
    val_mu = val[columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(train_mu).all() or not np.isfinite(val_mu).all():
        raise RuntimeError("non-finite train/validation mu")

    scaler = StandardScaler(copy=True).fit(train_mu)
    train_scaled = scaler.transform(train_mu)
    val_scaled = scaler.transform(val_mu)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(train_scaled)
    train_pca = pca.transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    scaler_path = artifact_dir / "standard_scaler.joblib"
    pca_path = artifact_dir / "pca64.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    k1_models: dict[str, GaussianMixture] = {}
    k2_models: dict[str, GaussianMixture] = {}
    warning_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    validation_loglik_rows: list[dict[str, object]] = []
    train_names = train["class_name"].to_numpy()
    val_names = val["class_name"].to_numpy()
    for class_name in fold["known_classes"]:
        train_values = train_pca[train_names == class_name]
        val_values = val_pca[val_names == class_name]
        k1 = fit_one(train_values, 1, args.setting, class_name, "Single-Full", warning_rows)
        k2 = fit_one(train_values, 2, args.setting, class_name, "Multi-Full-K2", warning_rows)
        k1_models[class_name] = k1
        k2_models[class_name] = k2
        k1_val_loglik = k1.score_samples(val_values)
        k2_val_loglik = k2.score_samples(val_values)
        validation_loglik_rows.append(
            {
                "setting": args.setting,
                "class_name": class_name,
                "validation_samples": len(val_values),
                "single_avg_loglik": float(k1_val_loglik.mean()),
                "multi_avg_loglik": float(k2_val_loglik.mean()),
                "delta_avg_loglik_multi_minus_single": float(
                    k2_val_loglik.mean() - k1_val_loglik.mean()
                ),
            }
        )
        assignments = k2.predict(val_values)
        val_counts = np.bincount(assignments, minlength=2)
        empirical = val_counts / len(val_values)
        train_weights = np.asarray(k2.weights_, dtype=np.float64)
        minimum = float(train_weights.min())
        stability_rows.append(
            {
                "setting": args.setting,
                "class_name": class_name,
                "train_samples": len(train_values),
                "validation_samples": len(val_values),
                "train_component_weights": json.dumps(train_weights.tolist()),
                "validation_empirical_weights": json.dumps(empirical.tolist()),
                "validation_component_counts": json.dumps(val_counts.tolist()),
                "total_variation": float(0.5 * np.abs(train_weights - empirical).sum()),
                "min_component_weight": minimum,
                "tiny_component_lt_1pct": minimum < 0.01,
                "very_tiny_component_lt_0_5pct": minimum < 0.005,
                "validation_empty_component": bool(np.any(val_counts == 0)),
                "converged": bool(k2.converged_),
                "n_iter": int(k2.n_iter_),
            }
        )
        print(
            json.dumps(
                {
                    "setting": args.setting,
                    "class": class_name,
                    "k1_converged": bool(k1.converged_),
                    "k2_converged": bool(k2.converged_),
                    "k2_weights": train_weights.tolist(),
                }
            ),
            flush=True,
        )
    k1_path = artifact_dir / "single_full_k1_models.joblib"
    k2_path = artifact_dir / "multi_full_k2_models.joblib"
    joblib.dump(k1_models, k1_path)
    joblib.dump(k2_models, k2_path)
    write_csv(output_dir / "density_fit_warnings.csv", warning_rows)
    write_csv(output_dir / "k2_stability.csv", stability_rows)
    weighted_single = float(
        np.average(
            [row["single_avg_loglik"] for row in validation_loglik_rows],
            weights=[row["validation_samples"] for row in validation_loglik_rows],
        )
    )
    weighted_multi = float(
        np.average(
            [row["multi_avg_loglik"] for row in validation_loglik_rows],
            weights=[row["validation_samples"] for row in validation_loglik_rows],
        )
    )
    validation_loglik_rows.append(
        {
            "setting": args.setting,
            "class_name": "__ALL_WEIGHTED__",
            "validation_samples": len(val),
            "single_avg_loglik": weighted_single,
            "multi_avg_loglik": weighted_multi,
            "delta_avg_loglik_multi_minus_single": weighted_multi - weighted_single,
        }
    )
    write_csv(output_dir / "density_validation_loglik.csv", validation_loglik_rows)

    single_val, _ = score_density_models(k1_models, val_pca)
    multi_val, _ = score_density_models(k2_models, val_pca)
    native_val = -val["native_unknown_score"].to_numpy(dtype=np.float64)
    score_map = {
        "Native": native_val,
        "Single-Full": single_val,
        "Multi-Full-K2": multi_val,
    }
    threshold_rows: list[dict[str, object]] = []
    val_scores = pd.DataFrame(
        {
            "flow_id": val["flow_id"],
            "class_name": val["class_name"],
            "native_known_score": native_val,
            "single_known_score": single_val,
            "multi_known_score": multi_val,
        }
    )
    val_scores_path = artifact_dir / "val_known_scores.parquet"
    val_scores.to_parquet(val_scores_path, index=False, engine="pyarrow", compression="zstd")
    for detector, scores in score_map.items():
        threshold = known_acceptance_threshold(scores, 0.95)
        actual = float(np.mean(scores >= threshold))
        threshold_rows.append(
            {
                "setting": args.setting,
                "detector": detector,
                "score_semantics": "higher_is_more_known",
                "threshold": threshold,
                "target_known_acceptance": 0.95,
                "actual_known_acceptance": actual,
                "known_val_FRR": 1.0 - actual,
                "calibration_split": "Known Validation only",
                "unknown_samples_used": 0,
            }
        )
    threshold_path = output_dir / "threshold_calibration.csv"
    write_csv(threshold_path, threshold_rows)

    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    training_config_path = output_dir / "training_config.json"
    frozen_files = [
        checkpoint_path,
        scaler_path,
        pca_path,
        k1_path,
        k2_path,
        threshold_path,
        training_config_path,
        train_path,
        val_path,
        val_scores_path,
    ]
    frozen = {
        "setting": args.setting,
        "created_before_final_test": True,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_canonical_sha256": EXPECTED_PROTOCOL_CANONICAL_SHA256,
        "execution_plan_sha256": EXPECTED_EXECUTION_PLAN_RAW_SHA256,
        "fit_data": "Known Train only",
        "threshold_data": "Known Validation only",
        "test_data_read": False,
        "unknown_samples_used_for_fit_or_calibration": 0,
        "pca_dim": 64,
        "reg_covar": 0.001,
        "k1": 1,
        "k2": 2,
        "gmm_n_init": 3,
        "gmm_max_iter": 300,
        "gmm_seed": 0,
        "files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in frozen_files],
    }
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"setting": args.setting, "frozen_model_hashes": str(frozen_path), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
