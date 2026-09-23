#!/usr/bin/env python3
"""Run one fixed Stage 10B score-chain diagnosis setting.

This script fits nothing. Empirical centroids are deterministic summaries of
Known Train only; all Test rows are evaluation-only. S5/S6 scores and
predictions are read directly from the frozen Stage 9 per-sample evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

from diagnosis_common import (
    CONFIG_PATH,
    PROJECT_ROOT,
    ROOT,
    SCORE_NAMES,
    SCORE_ORDER,
    SETTINGS,
    SCOPE,
    anomaly_metrics,
    load_json,
    require,
    safe_spearman,
    score_stats,
    squared_distance_matrix,
    write_json,
)


STAGE9_SCRIPTS = PROJECT_ROOT / "stage9_cipherspectrum_final_test/scripts"
sys.path.insert(0, str(STAGE9_SCRIPTS))
from stage9_common import (  # noqa: E402
    STAGE8A_ROOT,
    STAGE9_ROOT,
    assert_test_open_record,
    load_fold,
    stage7_run,
    verify_all_provenance,
)


STAGE9_METHOD = {"S0": "Native", "S5": "Single-Full-K1", "S6": "Multi-Global-K2"}


def write_frame(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows), f"refusing to write empty table: {path}")
    pd.DataFrame(rows).to_csv(path, index=False)


def manifest(path: Path, role_field: str, expected_role: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("row_index").reset_index(drop=True)
    require(frame["row_index"].tolist() == list(range(len(frame))), f"row order mismatch: {path}")
    if expected_role is not None:
        require((frame[role_field].astype(str) == expected_role).all(), f"role mismatch: {path}")
    return frame


def empirical_centers(values: np.ndarray, labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    centers = []
    for name in class_names:
        mask = labels == name
        require(bool(np.any(mask)), f"Known Train class absent: {name}")
        centers.append(np.mean(np.asarray(values[mask], dtype=np.float64), axis=0))
    return np.asarray(centers, dtype=np.float64)


def nearest(values: np.ndarray, centers: np.ndarray, half: bool = False) -> tuple[np.ndarray, np.ndarray]:
    matrix = squared_distance_matrix(values, centers)
    predictions = np.argmin(matrix, axis=1)
    scores = matrix[np.arange(len(matrix)), predictions]
    return (0.5 * scores if half else scores), predictions


def native_parts(mu: np.ndarray, logvar: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the released float32 algebra and accumulation order in blocks."""
    proto = np.asarray(prototypes, dtype=np.float32)
    proto_square = np.sum(proto * proto, axis=1)
    distances: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for start in range(0, len(mu), 2048):
        x = np.asarray(mu[start:start + 2048], dtype=np.float32)
        lv = np.asarray(logvar[start:start + 2048], dtype=np.float32)
        matrix = np.sum(x * x, axis=1, keepdims=True) + proto_square[None, :] - np.float32(2.0) * (x @ proto.T)
        pred = np.argmin(matrix, axis=1)
        d = np.float32(0.5) * matrix[np.arange(len(x)), pred]
        v = np.float32(0.5) * np.sum(np.exp(lv) - lv - np.float32(1.0), axis=1)
        distances.append(d.astype(np.float64))
        variances.append(v.astype(np.float64))
        predictions.append(pred.astype(np.int64))
    d_all = np.concatenate(distances)
    v_all = np.concatenate(variances)
    return d_all, v_all, np.concatenate(predictions)


def load_stage9_role(setting: str, role: str, expected_ids: np.ndarray) -> dict[str, np.ndarray]:
    detail = pd.read_parquet(STAGE9_ROOT / "outputs" / setting / f"{role}_sample_details.parquet")
    result: dict[str, np.ndarray] = {}
    for key, method in (("S5", "Single-Full-K1"), ("S6", "Multi-Global-K2")):
        block = detail[detail["method"] == method].reset_index(drop=True)
        require(len(block) == len(expected_ids), f"{setting}/{role}/{method}: row count mismatch")
        require(np.array_equal(block["sample_id"].astype(str).to_numpy(), expected_ids), f"{setting}/{role}/{method}: sample order mismatch")
        result[f"{key}_score"] = -block["detector_score"].to_numpy(dtype=np.float64)
        result[f"{key}_prediction"] = block["predicted_known_class"].astype(str).to_numpy()
    return result


def classification_row(setting: str, method: str, truth: np.ndarray, prediction: np.ndarray,
                       class_names: list[str]) -> dict[str, object]:
    return {
        "analysis_scope": SCOPE,
        "setting": setting,
        "method": method,
        "known_test_n": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=class_names, average="macro", zero_division=0)),
        "training_performed": False,
        "threshold_used": False,
    }


def stratified_indices(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == name) for name in sorted(set(labels))]


def paired_bootstrap(setting: str, scores: dict[str, tuple[np.ndarray, np.ndarray]],
                     known_labels: np.ndarray, unknown_labels: np.ndarray,
                     iterations: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    known_strata = stratified_indices(known_labels)
    unknown_strata = stratified_indices(unknown_labels)
    exact = {key: anomaly_metrics(*values)["AUROC"] for key, values in scores.items()}
    replicates: list[dict[str, object]] = []
    comparisons = list(zip(SCORE_ORDER[:-1], SCORE_ORDER[1:]))
    for replicate in range(iterations):
        ki = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in known_strata])
        ui = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in unknown_strata])
        aucs = {}
        labels = np.concatenate([np.zeros(len(ki), dtype=np.int8), np.ones(len(ui), dtype=np.int8)])
        for key in SCORE_ORDER:
            known, unknown = scores[key]
            aucs[key] = float(roc_auc_score(labels, np.concatenate([known[ki], unknown[ui]])))
        for left, right in comparisons:
            replicates.append({
                "analysis_scope": SCOPE,
                "setting": setting,
                "replicate": replicate,
                "comparison": f"{right}-{left}",
                "delta_auroc": aucs[right] - aucs[left],
                "seed": seed,
                "resampling": "paired_class_stratified",
            })
        if (replicate + 1) % 100 == 0:
            print(json.dumps({"event": "bootstrap_progress", "setting": setting, "replicates": replicate + 1}), flush=True)
    frame = pd.DataFrame(replicates)
    summary: list[dict[str, object]] = []
    for left, right in comparisons:
        values = frame.loc[frame["comparison"] == f"{right}-{left}", "delta_auroc"].to_numpy(dtype=np.float64)
        lo, hi = np.quantile(values, [0.025, 0.975])
        summary.append({
            "analysis_scope": SCOPE,
            "setting": setting,
            "comparison": f"{right}-{left}",
            "delta_auroc": exact[right] - exact[left],
            "bootstrap_mean_delta": float(np.mean(values)),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "bootstrap_iterations": iterations,
            "seed": seed,
            "resampling": "paired class-stratified within Known and Unknown; shared indices for all S0-S6",
        })
    return replicates, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    args = parser.parse_args()
    setting = args.setting
    config = load_json(CONFIG_PATH)
    verify_all_provenance()
    assert_test_open_record()
    provenance = load_json(ROOT / "outputs/summary/provenance_verification.json")
    require(provenance["status"] == "PASS", "Stage 10B provenance gate is not PASS")
    inference = load_json(ROOT / "outputs" / setting / "frozen_inference_audit.json")
    require(inference["status"] == "PASS", f"{setting}: frozen-inference gate is not PASS")

    output = ROOT / "outputs" / setting
    artifact_out = ROOT / "artifacts" / setting
    require(not (output / "score_chain_metrics.csv").exists(), f"{setting}: refusing to overwrite diagnosis")
    source = STAGE9_ROOT / "artifacts" / setting
    stage8 = STAGE8A_ROOT / "artifacts" / setting
    fold = load_fold(setting)
    class_names = [str(value) for value in fold["known_classes"]]

    train_manifest = manifest(stage8 / "train_manifest.csv", "source_role", "KNOWN_TRAIN")
    known_manifest = manifest(source / "known_test_manifest.csv", "role")
    unknown_manifest = manifest(source / "unknown_test_manifest.csv", "role")
    train_labels = train_manifest["class_name"].astype(str).to_numpy()
    known_labels = known_manifest["true_class"].astype(str).to_numpy()
    unknown_labels = unknown_manifest["true_class"].astype(str).to_numpy()
    known_ids = known_manifest["sample_id"].astype(str).to_numpy()
    unknown_ids = unknown_manifest["sample_id"].astype(str).to_numpy()
    require(np.array_equal(known_ids, np.load(source / "known_test_sample_ids.npy", allow_pickle=False).astype(str)), "Known sample ID mismatch")
    require(np.array_equal(unknown_ids, np.load(source / "unknown_test_sample_ids.npy", allow_pickle=False).astype(str)), "Unknown sample ID mismatch")
    require(set(train_labels) == set(class_names) and set(known_labels) == set(class_names), "Known class mapping mismatch")
    require(not set(unknown_labels) & set(class_names), "Known/Unknown Test label overlap")

    train_mu = np.load(stage8 / "mu_train.npy", mmap_mode="r", allow_pickle=False)
    known_mu = np.load(source / "mu_known_test.npy", mmap_mode="r", allow_pickle=False)
    unknown_mu = np.load(source / "mu_unknown_test.npy", mmap_mode="r", allow_pickle=False)
    known_logvar = np.load(artifact_out / "logvar_known_test.npy", mmap_mode="r", allow_pickle=False)
    unknown_logvar = np.load(artifact_out / "logvar_unknown_test.npy", mmap_mode="r", allow_pickle=False)
    require(train_mu.shape == (len(train_labels), 128), "Known Train mu shape mismatch")
    require(known_mu.shape == known_logvar.shape == (len(known_labels), 128), "Known Test latent shape mismatch")
    require(unknown_mu.shape == unknown_logvar.shape == (len(unknown_labels), 128), "Unknown Test latent shape mismatch")

    checkpoint = torch.load(stage7_run(setting) / "artifacts/training_best_checkpoint.pt", map_location="cpu", weights_only=False)
    prototypes = checkpoint["model_state_dict"]["prototypes"].detach().cpu().numpy().astype(np.float32)
    require(prototypes.shape == (len(class_names), 128), "learned prototype shape mismatch")
    d_known, v_known, learned_idx_known = native_parts(known_mu, known_logvar, prototypes)
    d_unknown, v_unknown, learned_idx_unknown = native_parts(unknown_mu, unknown_logvar, prototypes)
    native_saved_known = -np.asarray(np.load(source / "native_known_test_scores.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
    native_saved_unknown = -np.asarray(np.load(source / "native_unknown_test_scores.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
    native_pred_known = np.load(source / "native_known_test_predictions.npy", mmap_mode="r", allow_pickle=False)
    native_pred_unknown = np.load(source / "native_unknown_test_predictions.npy", mmap_mode="r", allow_pickle=False)
    reconstructed_known = d_known + v_known
    reconstructed_unknown = d_unknown + v_unknown
    tolerance = config["numerical_gates"]
    identity_roles: dict[str, object] = {}
    all_errors = []
    for role, reconstructed, saved, prediction, frozen_prediction in (
        ("known_test", reconstructed_known, native_saved_known, learned_idx_known, native_pred_known),
        ("unknown_test", reconstructed_unknown, native_saved_unknown, learned_idx_unknown, native_pred_unknown),
    ):
        errors = np.abs(reconstructed - saved)
        all_errors.append(errors)
        close = np.allclose(reconstructed, saved, atol=float(tolerance["native_identity_atol"]), rtol=float(tolerance["native_identity_rtol"]))
        identity_roles[role] = {
            "samples": len(saved),
            "max_abs_error": float(np.max(errors)),
            "mean_abs_error": float(np.mean(errors)),
            "p99_abs_error": float(np.quantile(errors, 0.99)),
            "prediction_mismatch_count": int(np.sum(prediction != frozen_prediction)),
            "status": "PASS" if close and np.array_equal(prediction, frozen_prediction) else "FAIL",
        }
    combined_error = np.concatenate(all_errors)
    identity_status = "PASS" if all(row["status"] == "PASS" for row in identity_roles.values()) else "NATIVE_SCORE_RECONSTRUCTION_FAILED"
    identity = {
        "analysis_scope": SCOPE,
        "setting": setting,
        "status": identity_status,
        "formula": "A_native = 0.5 min_y ||mu-p_y||^2 + 0.5 sum(exp(logvar)-logvar-1)",
        "stage9_relation": "A_native approximately equals -score_native_stage9",
        "atol": tolerance["native_identity_atol"],
        "rtol": tolerance["native_identity_rtol"],
        "max_abs_error": float(np.max(combined_error)),
        "mean_abs_error": float(np.mean(combined_error)),
        "p99_abs_error": float(np.quantile(combined_error, 0.99)),
        "roles": identity_roles,
    }
    write_json(output / "native_identity_check.json", identity)
    require(identity_status == "PASS", f"{setting}: NATIVE_SCORE_RECONSTRUCTION_FAILED")

    raw_centers = empirical_centers(train_mu, train_labels, class_names)
    s2_known, raw_idx_known = nearest(known_mu, raw_centers)
    s2_unknown, raw_idx_unknown = nearest(unknown_mu, raw_centers)
    scaler = joblib.load(stage8 / "scaler.joblib")
    pca = joblib.load(stage8 / "pca64.joblib")
    train_scaled = scaler.transform(np.asarray(train_mu, dtype=np.float64))
    known_scaled = scaler.transform(np.asarray(known_mu, dtype=np.float64))
    unknown_scaled = scaler.transform(np.asarray(unknown_mu, dtype=np.float64))
    scaled_centers = empirical_centers(train_scaled, train_labels, class_names)
    s3_known, scaled_idx_known = nearest(known_scaled, scaled_centers)
    s3_unknown, scaled_idx_unknown = nearest(unknown_scaled, scaled_centers)
    train_z = np.load(stage8 / "z_train_pca64.npy", mmap_mode="r", allow_pickle=False)
    known_z = np.load(source / "z_known_test_pca64.npy", mmap_mode="r", allow_pickle=False)
    unknown_z = np.load(source / "z_unknown_test_pca64.npy", mmap_mode="r", allow_pickle=False)
    pca_known_check = pca.transform(known_scaled).astype(np.float32)
    pca_unknown_check = pca.transform(unknown_scaled).astype(np.float32)
    pca_known_error = np.abs(pca_known_check.astype(np.float64) - np.asarray(known_z, dtype=np.float64))
    pca_unknown_error = np.abs(pca_unknown_check.astype(np.float64) - np.asarray(unknown_z, dtype=np.float64))
    require(np.array_equal(pca_known_check, np.asarray(known_z)) and np.array_equal(pca_unknown_check, np.asarray(unknown_z)), "frozen scaler/PCA Test transform parity failed")
    pca_centers = empirical_centers(train_z, train_labels, class_names)
    s4_known, pca_idx_known = nearest(known_z, pca_centers)
    s4_unknown, pca_idx_unknown = nearest(unknown_z, pca_centers)

    frozen_known = load_stage9_role(setting, "known", known_ids)
    frozen_unknown = load_stage9_role(setting, "unknown", unknown_ids)
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "S0": (native_saved_known, native_saved_unknown),
        "S1": (d_known, d_unknown),
        "S2": (s2_known, s2_unknown),
        "S3": (s3_known, s3_unknown),
        "S4": (s4_known, s4_unknown),
        "S5": (frozen_known["S5_score"], frozen_unknown["S5_score"]),
        "S6": (frozen_known["S6_score"], frozen_unknown["S6_score"]),
    }

    stage9_auc = pd.read_csv(STAGE9_ROOT / "outputs" / setting / "detection_auc_metrics.csv").set_index("method")
    metric_rows: list[dict[str, object]] = []
    auc_lookup: dict[str, float] = {}
    for key in SCORE_ORDER:
        known_score, unknown_score = scores[key]
        metrics = anomaly_metrics(known_score, unknown_score)
        ks = ks_2samp(known_score, unknown_score, alternative="two-sided", method="auto")
        auc_lookup[key] = metrics["AUROC"]
        method = STAGE9_METHOD.get(key)
        reference_auc = float(stage9_auc.loc[method, "AUROC"]) if method else np.nan
        reference_auprc = float(stage9_auc.loc[method, "AUPRC"]) if method else np.nan
        row = {
            "analysis_scope": SCOPE,
            "setting": setting,
            "score_id": key,
            "score_name": SCORE_NAMES[key],
            "positive_class": "Unknown",
            "score_orientation": "higher_is_more_unknown",
            **metrics,
            "KS_statistic": float(ks.statistic),
            "KS_pvalue": float(ks.pvalue),
            **score_stats(known_score, "known"),
            **score_stats(unknown_score, "unknown"),
            "stage9_method_reference": method or "",
            "stage9_AUROC_reference": reference_auc,
            "AUROC_reproduction_abs_error": abs(metrics["AUROC"] - reference_auc) if method else np.nan,
            "stage9_AUPRC_known_positive_reference": reference_auprc,
            "note": "AUPRC here is Unknown-positive; Stage9 reference AUPRC is Known-positive",
        }
        metric_rows.append(row)
        if method:
            require(abs(metrics["AUROC"] - reference_auc) <= float(tolerance["stage9_auroc_atol"]), f"{setting}/{key}: Stage9 AUROC reproduction failed")
            expected = float(config["expected_stage9_auroc"][setting][key])
            require(abs(metrics["AUROC"] - expected) <= 1e-6, f"{setting}/{key}: AUROC differs from task-stated reference")
    write_frame(output / "score_chain_metrics.csv", metric_rows)

    decomposition_rows = []
    for left, right in zip(SCORE_ORDER[:-1], SCORE_ORDER[1:]):
        decomposition_rows.append({
            "analysis_scope": SCOPE,
            "setting": setting,
            "contrast": f"{right}-{left}",
            "from_score": left,
            "to_score": right,
            "from_AUROC": auc_lookup[left],
            "to_AUROC": auc_lookup[right],
            "delta_AUROC": auc_lookup[right] - auc_lookup[left],
        })
    write_frame(output / "auroc_decomposition.csv", decomposition_rows)

    posterior_rows: list[dict[str, object]] = []
    for quantity, known_score, unknown_score in (
        ("D_proto", d_known, d_unknown),
        ("V_post", v_known, v_unknown),
        ("A_native", native_saved_known, native_saved_unknown),
    ):
        metric = anomaly_metrics(known_score, unknown_score)
        posterior_rows.append({"analysis_scope": SCOPE, "setting": setting, "row_type": "score_metric", "quantity": quantity, "group": "Known_vs_Unknown", **metric})
    for group, d_values, v_values in (
        ("Known", d_known, v_known),
        ("Unknown", d_unknown, v_unknown),
        ("Combined", np.concatenate([d_known, d_unknown]), np.concatenate([v_known, v_unknown])),
    ):
        rho, pvalue, status = safe_spearman(d_values, v_values)
        posterior_rows.append({
            "analysis_scope": SCOPE, "setting": setting, "row_type": "D_V_spearman", "quantity": "D_proto_vs_V_post", "group": group,
            "spearman_rho": rho, "spearman_pvalue": pvalue, "correlation_status": status,
            "median_D_proto": float(np.median(d_values)), "median_V_post": float(np.median(v_values)),
        })
    for group, d_values, v_values in (("Known", d_known, v_known), ("Unknown", d_unknown, v_unknown)):
        fraction = v_values / (d_values + v_values + np.finfo(np.float64).eps)
        posterior_rows.append({
            "analysis_scope": SCOPE, "setting": setting, "row_type": "variance_fraction", "quantity": "V_post/(D_proto+V_post+eps)", "group": group,
            "variance_fraction_mean": float(np.mean(fraction)), "variance_fraction_median": float(np.median(fraction)),
            "variance_fraction_p25": float(np.quantile(fraction, 0.25)), "variance_fraction_p75": float(np.quantile(fraction, 0.75)),
            "median_D_proto": float(np.median(d_values)), "median_V_post": float(np.median(v_values)),
        })
    variance_delta = auc_lookup["S1"] - auc_lookup["S0"]
    posterior_rows.append({
        "analysis_scope": SCOPE, "setting": setting, "row_type": "ranking_verdict", "quantity": "AUROC(D_proto)-AUROC(A_native)", "group": "Known_vs_Unknown",
        "delta_AUROC": variance_delta,
        "ranking_verdict": "VARIANCE_TERM_HURTS_RANKING" if variance_delta > 0 else ("VARIANCE_TERM_HELPS_RANKING" if variance_delta < 0 else "VARIANCE_TERM_NEUTRAL"),
    })
    write_frame(output / "posterior_variance_diagnosis.csv", posterior_rows)

    gap_rows: list[dict[str, object]] = []
    for index, name in enumerate(class_names):
        class_values = np.asarray(train_mu[train_labels == name], dtype=np.float64)
        center = raw_centers[index]
        gap = float(np.linalg.norm(np.asarray(prototypes[index], dtype=np.float64) - center))
        radius = float(np.median(np.linalg.norm(class_values - center, axis=1)))
        gap_rows.append({
            "analysis_scope": SCOPE, "setting": setting, "class": name, "class_index": index,
            "train_n": len(class_values), "gap": gap, "within_radius": radius,
            "normalized_gap": gap / (radius + np.finfo(np.float64).eps),
            "prototype_source": "frozen Stage7 best checkpoint", "centroid_source": "Known Train mu only",
        })
    write_frame(output / "prototype_centroid_gap.csv", gap_rows)

    names = np.asarray(class_names, dtype=object)
    predictions = {
        "LearnedPrototype": names[learned_idx_known],
        "EmpiricalCentroidRaw": names[raw_idx_known],
        "EmpiricalCentroidScaled": names[scaled_idx_known],
        "EmpiricalCentroidPCA64": names[pca_idx_known],
        "FullK1": frozen_known["S5_prediction"],
        "FullK2": frozen_known["S6_prediction"],
    }
    classification_rows = [classification_row(setting, method, known_labels, prediction, class_names) for method, prediction in predictions.items()]
    write_frame(output / "known_geometry_classification.csv", classification_rows)

    per_unknown_stage9 = pd.read_csv(STAGE9_ROOT / "outputs" / setting / "per_unknown_class_ufar.csv")
    ufar_lookup = {(str(row.method), str(row.unknown_class)): (int(row.accepted_as_known), float(row.UFAR)) for row in per_unknown_stage9.itertuples(index=False)}
    per_unknown_rows = []
    for unknown_class in sorted(set(unknown_labels)):
        mask = unknown_labels == unknown_class
        for key in SCORE_ORDER:
            known_score, unknown_score = scores[key]
            subset = unknown_score[mask]
            metric = anomaly_metrics(known_score, subset)
            stage9_method = STAGE9_METHOD.get(key)
            accepted, ufar = ufar_lookup[(stage9_method, unknown_class)] if stage9_method else (-1, np.nan)
            per_unknown_rows.append({
                "analysis_scope": SCOPE, "setting": setting, "unknown_class": unknown_class,
                "unknown_n": int(np.sum(mask)), "score_id": key, "score_name": SCORE_NAMES[key],
                "AUROC": metric["AUROC"], "AUPRC": metric["AUPRC"],
                "median_anomaly_score": float(np.median(subset)),
                "known_reference_percentile_of_unknown_median": float(100.0 * np.mean(known_score <= np.median(subset))),
                "delta_AUROC_vs_S0": metric["AUROC"] - anomaly_metrics(scores["S0"][0], scores["S0"][1][mask])["AUROC"],
                "stage9_method_reference": stage9_method or "NONE_NO_FROZEN_THRESHOLD",
                "stage9_accepted_as_known": accepted,
                "stage9_UFAR": ufar,
                "threshold_note": "reused Stage9 decision" if stage9_method else "not applicable; no threshold defined or tuned",
            })
    write_frame(output / "per_unknown_score_chain.csv", per_unknown_rows)

    gap_frame = pd.DataFrame(gap_rows).set_index("class")
    stage9_known_class = pd.read_csv(STAGE9_ROOT / "outputs" / setting / "known_test_per_class_metrics.csv")
    native_known_class = stage9_known_class[stage9_known_class["method"] == "Native"].set_index("class")
    absorption = pd.read_csv(STAGE9_ROOT / "outputs" / setting / "unknown_to_known_absorption_matrix.csv")
    native_absorption = absorption[absorption["method"] == "Native"].groupby("predicted_known_class")["count"].sum()
    aligned = pd.DataFrame(index=class_names)
    aligned["normalized_gap"] = gap_frame.loc[class_names, "normalized_gap"]
    aligned["known_test_f1"] = native_known_class.loc[class_names, "f1"]
    aligned["unknown_absorbed_count"] = native_absorption.reindex(class_names, fill_value=0).astype(int)
    aligned["unknown_absorbed_rate"] = aligned["unknown_absorbed_count"] / len(unknown_labels)
    rho_f1, p_f1, status_f1 = safe_spearman(aligned["normalized_gap"].to_numpy(), aligned["known_test_f1"].to_numpy())
    rho_abs, p_abs, status_abs = safe_spearman(aligned["normalized_gap"].to_numpy(), aligned["unknown_absorbed_count"].to_numpy())
    absorption_rows = []
    for name, row in aligned.iterrows():
        absorption_rows.append({
            "analysis_scope": SCOPE, "setting": setting, "known_class": name,
            "normalized_gap": row["normalized_gap"], "known_test_f1": row["known_test_f1"],
            "known_test_macro_f1": float(native_known_class["f1"].mean()),
            "unknown_absorbed_count": int(row["unknown_absorbed_count"]), "unknown_absorbed_rate": row["unknown_absorbed_rate"],
            "spearman_gap_vs_known_f1": rho_f1, "p_gap_vs_known_f1": p_f1, "gap_vs_known_f1_status": status_f1,
            "spearman_gap_vs_unknown_absorption": rho_abs, "p_gap_vs_unknown_absorption": p_abs, "gap_vs_absorption_status": status_abs,
            "absorption_source": "frozen Stage9 Native Unknown-to-Known absorption matrix",
        })
    write_frame(output / "prototype_gap_vs_absorption.csv", absorption_rows)

    known_lv = np.asarray(known_logvar, dtype=np.float64)
    unknown_lv = np.asarray(unknown_logvar, dtype=np.float64)
    known_exp = np.exp(known_lv)
    unknown_exp = np.exp(unknown_lv)
    known_vdim = 0.5 * (known_exp - known_lv - 1.0)
    unknown_vdim = 0.5 * (unknown_exp - unknown_lv - 1.0)
    logvar_rows = []
    closer = np.mean(unknown_vdim, axis=0) <= np.mean(known_vdim, axis=0)
    for dimension in range(128):
        logvar_rows.append({
            "analysis_scope": SCOPE, "setting": setting, "row_type": "dimension", "dimension": dimension,
            "known_mean_logvar": float(np.mean(known_lv[:, dimension])), "known_median_logvar": float(np.median(known_lv[:, dimension])),
            "unknown_mean_logvar": float(np.mean(unknown_lv[:, dimension])), "unknown_median_logvar": float(np.median(unknown_lv[:, dimension])),
            "known_mean_exp_logvar": float(np.mean(known_exp[:, dimension])), "unknown_mean_exp_logvar": float(np.mean(unknown_exp[:, dimension])),
            "known_mean_V_contribution": float(np.mean(known_vdim[:, dimension])), "unknown_mean_V_contribution": float(np.mean(unknown_vdim[:, dimension])),
            "unknown_closer_to_unit_variance": bool(closer[dimension]),
        })
    logvar_rows.append({
        "analysis_scope": SCOPE, "setting": setting, "row_type": "overall", "dimension": -1,
        "known_mean_logvar": float(np.mean(known_lv)), "known_median_logvar": float(np.median(known_lv)),
        "unknown_mean_logvar": float(np.mean(unknown_lv)), "unknown_median_logvar": float(np.median(unknown_lv)),
        "known_mean_exp_logvar": float(np.mean(known_exp)), "unknown_mean_exp_logvar": float(np.mean(unknown_exp)),
        "known_mean_V_contribution": float(np.mean(v_known)), "unknown_mean_V_contribution": float(np.mean(v_unknown)),
        "unknown_closer_to_unit_variance": bool(np.median(v_unknown) <= np.median(v_known)),
        "dimensions_unknown_closer_count": int(np.sum(closer)), "dimensions_unknown_closer_rate": float(np.mean(closer)),
    })
    write_frame(output / "logvar_distribution_audit.csv", logvar_rows)

    bootstrap_cfg = config["bootstrap"]
    replicate_rows, bootstrap_rows = paired_bootstrap(
        setting, scores, known_labels, unknown_labels,
        int(bootstrap_cfg["iterations"]), int(bootstrap_cfg["seed"]),
    )
    pd.DataFrame(replicate_rows).to_parquet(output / "paired_auroc_bootstrap_replicates.parquet", index=False, engine="pyarrow", compression="zstd")
    write_frame(output / "paired_auroc_bootstrap.csv", bootstrap_rows)

    np.savez_compressed(artifact_out / "diagnostic_scores.npz", **{f"{key}_{role}": values[index] for key, values in scores.items() for index, role in enumerate(("known", "unknown"))})
    np.savez_compressed(
        artifact_out / "diagnostic_predictions.npz",
        learned_known=names[learned_idx_known].astype(str), learned_unknown=names[learned_idx_unknown].astype(str),
        raw_centroid_known=names[raw_idx_known].astype(str), raw_centroid_unknown=names[raw_idx_unknown].astype(str),
        scaled_centroid_known=names[scaled_idx_known].astype(str), scaled_centroid_unknown=names[scaled_idx_unknown].astype(str),
        pca64_centroid_known=names[pca_idx_known].astype(str), pca64_centroid_unknown=names[pca_idx_unknown].astype(str),
        k1_known=frozen_known["S5_prediction"].astype(str), k1_unknown=frozen_unknown["S5_prediction"].astype(str),
        k2_known=frozen_known["S6_prediction"].astype(str), k2_unknown=frozen_unknown["S6_prediction"].astype(str),
    )
    write_json(output / "setting_completion.json", {
        "analysis_scope": SCOPE,
        "setting": setting,
        "status": "PASS",
        "sample_order": "PASS",
        "native_identity": "PASS",
        "stage9_AUROC_reproduction": {key: auc_lookup[key] for key in ("S0", "S5", "S6")},
        "score_chain_complete": list(scores) == list(SCORE_ORDER),
        "pca_transform_parity": {
            "status": "PASS",
            "known_max_abs_error": float(np.max(pca_known_error)), "unknown_max_abs_error": float(np.max(pca_unknown_error)),
        },
        "centroid_fit_data": "Known Train only",
        "test_usage": "evaluation only",
        "model_training_performed": False,
        "threshold_tuning_performed": False,
        "frozen_model_refit_performed": False,
        "bootstrap_iterations": int(bootstrap_cfg["iterations"]),
        "bootstrap_seed": int(bootstrap_cfg["seed"]),
    })
    print(json.dumps({"setting": setting, "status": "PASS", "AUROC": auc_lookup}, indent=2), flush=True)


if __name__ == "__main__":
    main()
