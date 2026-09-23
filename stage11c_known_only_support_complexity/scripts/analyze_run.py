#!/usr/bin/env python3
"""Compute one Stage 11C run using Known Train/Validation only."""

from __future__ import annotations

import argparse
import json
import traceback

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from stage11c_common import (
    CONFIG_PATH,
    bootstrap_covariance_stability,
    centroid_scores,
    load_known_inputs,
    margin_statistics,
    mle_covariance,
    output_run_dir,
    radius_statistics,
    read_json,
    score_summary,
    sha256_file,
    spectrum_statistics,
    spherical_nll,
    summarize_table,
    write_csv,
    write_json,
)


def analyze(scenario: str, fold: int, seed: int) -> dict:
    config = read_json(CONFIG_PATH)
    loaded = load_known_inputs(scenario, fold, seed)
    source = loaded["source"]
    arrays = loaded["arrays"]
    support = loaded["support"]
    thresholds = loaded["thresholds"]
    source_result = loaded["result"]

    train_source = arrays["train_mu"]
    validation_source = arrays["validation_mu"]
    train_raw = train_source.astype(np.float64, copy=False)
    validation_raw = validation_source.astype(np.float64, copy=False)
    train_labels = arrays["train_labels_reindexed"].astype(np.int64, copy=False)
    validation_labels = arrays["validation_labels_reindexed"].astype(np.int64, copy=False)
    known_ids = np.asarray(support["known_class_ids"], dtype=np.int64)
    density = support["R2_R3_frozen_density"]
    scaler = density["scaler"]
    pca = density["pca"]
    k1_models = density["models"]["K1"]
    train_pca = pca.transform(scaler.transform(train_source)).astype(np.float64, copy=False)
    validation_pca = pca.transform(scaler.transform(validation_source)).astype(np.float64, copy=False)
    spaces = {"raw128": (train_raw, validation_raw), "pca64": (train_pca, validation_pca)}

    source_centroids = np.vstack([np.mean(train_source[train_labels == index], axis=0) for index in range(len(known_ids))])
    if not np.array_equal(source_centroids, np.asarray(support["centroids"])):
        raise RuntimeError("Stage11B empirical centroid parity failed")
    centroids_by_space = {
        "raw128": np.asarray(support["centroids"], dtype=np.float64),
        "pca64": np.vstack([np.mean(train_pca[train_labels == index], axis=0) for index in range(len(known_ids))]),
    }

    geometry_rows: list[dict] = []
    stability_rows: list[dict] = []
    fit_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for class_index, original_class_id in enumerate(known_ids):
        train_mask = train_labels == class_index
        validation_mask = validation_labels == class_index
        for space, (train, validation) in spaces.items():
            class_train = train[train_mask]
            class_validation = validation[validation_mask]
            centroid = centroids_by_space[space][class_index]
            row = {
                "scenario": scenario,
                "fold": fold,
                "seed": seed,
                "class_index": class_index,
                "original_class_id": int(original_class_id),
                "train_samples": len(class_train),
                "validation_samples": len(class_validation),
                **spectrum_statistics(class_train, space),
                **radius_statistics(class_train, class_validation, centroid),
                **margin_statistics(validation, validation_labels, centroids_by_space[space], class_index),
            }
            geometry_rows.append(row)

        class_train_pca = train_pca[train_mask]
        class_validation_pca = validation_pca[validation_mask]
        stability_rows.append(
            {
                "scenario": scenario,
                "fold": fold,
                "seed": seed,
                "class_index": class_index,
                "original_class_id": int(original_class_id),
                "train_samples": len(class_train_pca),
                **bootstrap_covariance_stability(
                    class_train_pca,
                    iterations=int(config["bootstrap"]["iterations"]),
                    seed=int(config["bootstrap"]["seed"]),
                ),
            }
        )

        mean_pca, covariance_pca = mle_covariance(class_train_pca)
        scalar_variance = float(np.trace(covariance_pca) / covariance_pca.shape[0])
        spherical_train = spherical_nll(class_train_pca, mean_pca, scalar_variance)
        spherical_validation = spherical_nll(class_validation_pca, mean_pca, scalar_variance)
        full_train = -k1_models[class_index].score_samples(class_train_pca)
        full_validation = -k1_models[class_index].score_samples(class_validation_pca)
        raw_centroid = centroids_by_space["raw128"][class_index]
        centroid_train = centroid_scores(train_raw[train_mask], raw_centroid)
        centroid_validation = centroid_scores(validation_raw[validation_mask], raw_centroid)
        fit_rows.append(
            {
                "scenario": scenario,
                "fold": fold,
                "seed": seed,
                "class_index": class_index,
                "original_class_id": int(original_class_id),
                "train_samples": int(np.sum(train_mask)),
                "validation_samples": int(np.sum(validation_mask)),
                "space": "pca64",
                "spherical_variance_train_mle": scalar_variance,
                **score_summary("centroid_train_score", centroid_train),
                **score_summary("centroid_validation_score", centroid_validation),
                "spherical_train_nll": float(np.mean(spherical_train)),
                "spherical_validation_nll": float(np.mean(spherical_validation)),
                "full_k1_train_nll": float(np.mean(full_train)),
                "full_k1_validation_nll": float(np.mean(full_validation)),
                "spherical_train_to_validation_gap": float(np.mean(spherical_validation) - np.mean(spherical_train)),
                "full_k1_train_to_validation_gap": float(np.mean(full_validation) - np.mean(full_train)),
                "delta_nll_full_validation": float(np.mean(spherical_validation) - np.mean(full_validation)),
                "relative_delta_nll_full_validation": float((np.mean(spherical_validation) - np.mean(full_validation)) / (abs(np.mean(spherical_validation)) + 1e-12)),
                "full_k1_source": "FROZEN_STAGE11B_K1_NO_REFIT",
            }
        )

        for method in ("R1", "R2"):
            scores = arrays[f"validation_{method.lower()}_scores"][validation_mask]
            threshold = float(thresholds[method]["threshold"])
            coverage_rows.append(
                {
                    "scenario": scenario,
                    "fold": fold,
                    "seed": seed,
                    "class_index": class_index,
                    "original_class_id": int(original_class_id),
                    "validation_samples": int(np.sum(validation_mask)),
                    "method": method,
                    "threshold": threshold,
                    "threshold_source": "FROZEN_STAGE11B_KNOWN_VALIDATION_P95",
                    "validation_coverage": float(np.mean(scores <= threshold)),
                }
            )

    centroid_distance = np.sum((validation_raw[:, None, :] - centroids_by_space["raw128"][None, :, :]) ** 2, axis=2)
    centroid_prediction = np.argmin(centroid_distance, axis=1)
    full_loglik = np.column_stack([model.score_samples(validation_pca) for model in k1_models])
    full_prediction = np.argmax(full_loglik, axis=1)
    centroid_accuracy = float(accuracy_score(validation_labels, centroid_prediction))
    centroid_macro_f1 = float(f1_score(validation_labels, centroid_prediction, labels=np.arange(len(known_ids)), average="macro", zero_division=0))
    full_accuracy = float(accuracy_score(validation_labels, full_prediction))
    full_macro_f1 = float(f1_score(validation_labels, full_prediction, labels=np.arange(len(known_ids)), average="macro", zero_division=0))
    classification = {
        "scenario": scenario,
        "fold": fold,
        "seed": seed,
        "validation_samples": len(validation_labels),
        "centroid_accuracy": centroid_accuracy,
        "centroid_macro_f1": centroid_macro_f1,
        "full_k1_accuracy": full_accuracy,
        "full_k1_macro_f1": full_macro_f1,
        "delta_accuracy_full_minus_centroid": full_accuracy - centroid_accuracy,
        "delta_macro_f1_full_minus_centroid": full_macro_f1 - centroid_macro_f1,
        "data_role": "KNOWN_VALIDATION_ONLY",
    }

    fit_weights = np.asarray([row["validation_samples"] for row in fit_rows], dtype=np.float64)
    spherical_val = np.average([row["spherical_validation_nll"] for row in fit_rows], weights=fit_weights)
    full_val = np.average([row["full_k1_validation_nll"] for row in fit_rows], weights=fit_weights)
    run_features: dict[str, float | str | int | bool] = {
        "scenario": scenario,
        "fold": fold,
        "seed": seed,
        "known_class_count": len(known_ids),
        "known_train_samples": len(train_labels),
        "known_validation_samples": len(validation_labels),
        "validation_nll_spherical": float(spherical_val),
        "validation_nll_full_k1": float(full_val),
        "delta_nll_full_validation": float(spherical_val - full_val),
        "relative_delta_nll_full_validation": float((spherical_val - full_val) / (abs(spherical_val) + 1e-12)),
        "validation_centroid_accuracy": centroid_accuracy,
        "validation_centroid_macro_f1": centroid_macro_f1,
        "validation_full_k1_accuracy": full_accuracy,
        "validation_full_k1_macro_f1": full_macro_f1,
        "delta_validation_macro_f1_full_minus_centroid": full_macro_f1 - centroid_macro_f1,
        "covariance_bootstrap_frobenius_relative_mean": float(np.mean([row["frobenius_relative_mean"] for row in stability_rows])),
        "features_use_unknown": False,
        "features_use_known_test": False,
        "features_use_unknown_test": False,
    }
    identifiers = {"scenario", "fold", "seed", "class_index", "original_class_id", "space", "method", "threshold_source", "full_k1_source"}
    for space in ("raw128", "pca64"):
        rows = [row for row in geometry_rows if row["space"] == space]
        run_features.update(summarize_table(rows, f"geometry_{space}", identifiers))
    run_features.update(summarize_table(stability_rows, "stability_pca64", identifiers))
    run_features.update(summarize_table(fit_rows, "fit_pca64", identifiers))
    for method in ("R1", "R2"):
        rows = [row for row in coverage_rows if row["method"] == method]
        coverage = np.asarray([row["validation_coverage"] for row in rows], dtype=np.float64)
        run_features.update(
            {
                f"coverage_{method.lower()}_mean": float(np.mean(coverage)),
                f"coverage_{method.lower()}_variance": float(np.var(coverage)),
                f"coverage_{method.lower()}_std": float(np.std(coverage)),
                f"coverage_{method.lower()}_min": float(np.min(coverage)),
                f"coverage_{method.lower()}_max": float(np.max(coverage)),
                f"coverage_{method.lower()}_range": float(np.max(coverage) - np.min(coverage)),
                f"coverage_{method.lower()}_cv": float(np.std(coverage) / (abs(np.mean(coverage)) + 1e-12)),
            }
        )
    run_features["delta_coverage_variance_r1_minus_r2"] = float(run_features["coverage_r1_variance"] - run_features["coverage_r2_variance"])

    r1 = source_result["evaluation"]["detectors"]["R1"]["combined_balanced_1to1"]
    r2 = source_result["evaluation"]["detectors"]["R2"]["combined_balanced_1to1"]
    outcome = {
        "scenario": scenario,
        "fold": fold,
        "seed": seed,
        "role": "OUTCOME_ONLY_NOT_A_CRITERION_FEATURE",
        "delta_cov_auroc": float(r2["auroc"] - r1["auroc"]),
        "delta_cov_ufar": float(r2["ufar"] - r1["ufar"]),
        "delta_cov_binary_f1": float(r2["binary_f1"] - r1["binary_f1"]),
        "r0_auroc": float(source_result["evaluation"]["detectors"]["R0"]["combined_balanced_1to1"]["auroc"]),
        "r1_auroc": float(r1["auroc"]),
        "r2_auroc": float(r2["auroc"]),
    }

    output = output_run_dir(scenario, fold, seed)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "class_level_geometry.csv", geometry_rows)
    write_csv(output / "covariance_stability.csv", stability_rows)
    write_csv(output / "validation_fit_comparison.csv", fit_rows)
    write_csv(output / "coverage_stability.csv", coverage_rows)
    write_csv(output / "validation_classification_comparison.csv", [classification])
    write_json(output / "known_only_features.json", run_features)
    write_json(output / "outcome_only.json", outcome)
    source_hashes = {
        "stage11b_results_sha256": sha256_file(source / "results.json"),
        "stage11b_sample_outputs_sha256": sha256_file(source / "sample_outputs.npz"),
        "stage11b_support_models_sha256": sha256_file(source / "support_models.joblib"),
        "stage11b_thresholds_sha256": sha256_file(source / "thresholds.json"),
    }
    write_json(output / "source_hashes.json", source_hashes)
    result = {
        "status": "SUCCESS",
        "scenario": scenario,
        "fold": fold,
        "seed": seed,
        "known_only_feature_construction": True,
        "unknown_feature_used": False,
        "known_test_feature_used": False,
        "unknown_test_feature_used": False,
        "new_encoder_training": False,
        "new_detector_fitting": False,
        "new_threshold_fitting": False,
        "frozen_k1_reused_without_refit": True,
        "bootstrap_iterations": int(config["bootstrap"]["iterations"]),
        "classification": classification,
        "priority_features": {
            "delta_nll_full_validation": run_features["delta_nll_full_validation"],
            "delta_validation_macro_f1_full_minus_centroid": run_features["delta_validation_macro_f1_full_minus_centroid"],
            "covariance_bootstrap_frobenius_relative_mean": run_features["covariance_bootstrap_frobenius_relative_mean"],
        },
        "outcome_role": outcome["role"],
        "source_hashes": source_hashes,
        "next_stage_started": False,
    }
    write_json(output / "results.json", result)
    (output / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a1", "a2", "a3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = output_run_dir(args.scenario, args.fold, args.seed)
    if (output / "SUCCESS").exists() or (output / "FAILURE.json").exists():
        raise RuntimeError(f"Run directory already terminal; preserve it and use a new attempt: {output}")
    try:
        result = analyze(args.scenario, args.fold, args.seed)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILURE.json", {"status": "FAILURE", "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
