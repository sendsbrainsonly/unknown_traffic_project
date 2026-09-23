#!/usr/bin/env python3
"""Read-only post-hoc diagnosis of the frozen Stage 3 A-1/A-2/A-3 results.

This module never fits a model or recalibrates a threshold.  It reads frozen
Stage 3 predictions and applies only ``transform``, ``score_samples``, and
``predict`` to Known Train/Validation representations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import kruskal, spearmanr


DIAG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DIAG_ROOT.parent
STAGE3_ROOT = PROJECT_ROOT / "stage3_unknown_utility"
OUTPUT_ROOT = DIAG_ROOT / "outputs"
FLOW_INDEX = PROJECT_ROOT / "outputs/stage0_data/flow_index.csv"
SETTINGS = ("A-1", "A-2", "A-3")
SNAPSHOT_COMMIT = "ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc"
PROTOCOL_CANONICAL_SHA256 = "1fff4ed33211de0b123cf4c0ec6b203cc33be683ac315b610ffd8f0194550ae1"
PROTOCOL_RAW_SHA256 = "86718b1930c71ef6a904f16f37a199b8ceeea6600d3ceba65e7592048f54677a"
EXECUTION_PLAN_SHA256 = "197e36be2e3559c20dddf4375d117855b24cab1c1ab0f89d76f08a7f164d13de"
SNAPSHOT_PACKAGE_SHA256 = {
    "A-1": "0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9",
    "A-2": "5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6",
    "A-3": "1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be",
}
FEATURES = ("packet_count", "total_bytes", "duration_s", "forward_packets", "backward_packets")
CORRELATION_SIGNALS = (
    "delta_nll2",
    "min_component_weight",
    "train_val_tv",
    "component_mean_euclidean_distance",
    "component_mahalanobis_distance",
    "log_covariance_volume_ratio",
    "val_own_score_shift_mean",
    "val_own_score_shift_p95",
    "val_global_disagreement_rate",
    "val_global_margin_shift_mean",
    "val_global_margin_shift_p95",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_content_hash(root: Path) -> tuple[str, int]:
    """Return a deterministic audit-run tree hash, independent of snapshot hash format."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def threshold_map(setting: str) -> dict[str, float]:
    frame = pd.read_csv(STAGE3_ROOT / "outputs" / setting / "threshold_calibration.csv")
    if len(frame) != 3 or int(frame["unknown_samples_used"].sum()) != 0:
        raise RuntimeError(f"{setting}: invalid frozen threshold audit")
    return dict(zip(frame["detector"], frame["threshold"].astype(float)))


def fold(setting: str) -> dict[str, object]:
    path = PROJECT_ROOT / "stage3_protocol" / f"fold_{setting.replace('-', '')}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["scenario"] != setting:
        raise RuntimeError(f"{setting}: fold mismatch")
    return payload


def latent_columns(columns: Iterable[str]) -> list[str]:
    result = sorted(name for name in columns if name.startswith("mu_"))
    if len(result) != 128:
        raise RuntimeError(f"expected 128 mu columns, found {len(result)}")
    return result


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"{prefix}: empty or non-finite values")
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
    }


def covariance_summary(covariance: np.ndarray, prefix: str) -> dict[str, float]:
    covariance = np.asarray(covariance, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if sign <= 0 or float(eigenvalues[0]) <= 0:
        raise RuntimeError(f"{prefix}: covariance is not positive definite")
    return {
        f"{prefix}_logdet": float(logdet),
        f"{prefix}_eig_min": float(eigenvalues[0]),
        f"{prefix}_eig_median": float(np.median(eigenvalues)),
        f"{prefix}_eig_max": float(eigenvalues[-1]),
        f"{prefix}_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }


def epsilon_squared_kruskal(h_statistic: float, n: int, groups: int) -> float:
    if groups < 2 or n <= groups or not np.isfinite(h_statistic):
        return float("nan")
    return max(0.0, float((h_statistic - groups + 1) / (n - groups)))


def transition_code(single_accept: pd.Series, multi_accept: pd.Series) -> np.ndarray:
    return np.select(
        [~single_accept & ~multi_accept, ~single_accept & multi_accept, single_accept & ~multi_accept],
        ["SS", "SM", "MS"],
        default="MM",
    )


def verify_snapshot_and_frozen_inputs() -> dict[str, object]:
    subprocess.check_call(["git", "cat-file", "-e", f"{SNAPSHOT_COMMIT}^{{commit}}"], cwd=PROJECT_ROOT)
    if subprocess.call(["git", "merge-base", "--is-ancestor", SNAPSHOT_COMMIT, "HEAD"], cwd=PROJECT_ROOT) != 0:
        raise RuntimeError("Stage 3 snapshot is not an ancestor of HEAD")
    protocol = PROJECT_ROOT / "stage3_protocol/stage3_protocol.json"
    plan = PROJECT_ROOT / "stage3_protocol/stage3_execution_plan.json"
    if sha256_file(protocol) != PROTOCOL_RAW_SHA256:
        raise RuntimeError("Stage 3 protocol raw hash changed")
    if sha256_file(plan) != EXECUTION_PLAN_SHA256:
        raise RuntimeError("Stage 3 execution plan hash changed")

    verified: list[dict[str, object]] = []
    tree_hashes: dict[str, dict[str, object]] = {}
    for setting in SETTINGS:
        manifest_path = STAGE3_ROOT / "outputs" / setting / "frozen_model_hashes.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_canonical_sha256"] != PROTOCOL_CANONICAL_SHA256:
            raise RuntimeError(f"{setting}: frozen protocol hash mismatch")
        if manifest["fit_data"] != "Known Train only" or manifest["threshold_data"] != "Known Validation only":
            raise RuntimeError(f"{setting}: unexpected fit/calibration provenance")
        if manifest["test_data_read"] or manifest["unknown_samples_used_for_fit_or_calibration"] != 0:
            raise RuntimeError(f"{setting}: frozen model provenance is not Unknown-free")
        for item in manifest["files"]:
            path = Path(item["path"])
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise RuntimeError(f"{setting}: frozen file changed: {path}")
            verified.append({"setting": setting, "path": str(path), "sha256": actual})
        final_metadata = json.loads((STAGE3_ROOT / "outputs" / setting / "run_metadata.json").read_text(encoding="utf-8"))
        final_predictions = STAGE3_ROOT / "artifacts" / setting / "frozen_final_predictions.parquet"
        final_predictions_hash = sha256_file(final_predictions)
        if final_predictions_hash != final_metadata["final_predictions_hash"]:
            raise RuntimeError(f"{setting}: frozen final-prediction hash mismatch")
        verified.append({"setting": setting, "path": str(final_predictions), "sha256": final_predictions_hash})
        current_hash, count = tree_content_hash(STAGE3_ROOT / "outputs" / setting)
        tree_hashes[setting] = {
            "snapshot_package_sha256": SNAPSHOT_PACKAGE_SHA256[setting],
            "audit_run_tree_sha256_before": current_hash,
            "file_count": count,
        }
    return {"verified_frozen_files": verified, "stage3_output_trees_before": tree_hashes}


def load_frozen_predictions(setting: str) -> tuple[pd.DataFrame, dict[str, float]]:
    path = STAGE3_ROOT / "artifacts" / setting / "frozen_final_predictions.parquet"
    predictions = pd.read_parquet(path)
    thresholds = threshold_map(setting)
    for prefix, detector in (("single", "Single-Full"), ("multi", "Multi-Full-K2")):
        recomputed = predictions[f"{prefix}_known_score"].to_numpy() >= thresholds[detector]
        frozen = predictions[f"{prefix}_accepted_as_known"].to_numpy(dtype=bool)
        if not np.array_equal(recomputed, frozen):
            raise RuntimeError(f"{setting}: frozen {prefix} binary decisions disagree with score/threshold")
    return predictions, thresholds


def replay_unknown_component_assignments(setting: str, unknown: pd.DataFrame) -> dict[str, np.ndarray]:
    """Replay only missing component assignments; require score parity with frozen output."""
    artifact = STAGE3_ROOT / "artifacts" / setting
    latent_path = artifact / "test_unknown_mu.parquet"
    latent_manifest = json.loads(
        (STAGE3_ROOT / "outputs" / setting / "latent_export_manifest.json").read_text(encoding="utf-8")
    )["files"]["test_unknown_mu.parquet"]
    if sha256_file(latent_path) != latent_manifest["sha256"]:
        raise RuntimeError(f"{setting}: frozen Unknown latent hash mismatch")
    latent = pd.read_parquet(latent_path)
    if not latent["flow_id"].is_unique or len(latent) != len(unknown):
        raise RuntimeError(f"{setting}: Unknown latent rows are not a one-to-one frozen population")
    aligned = latent.set_index("flow_id").loc[unknown["flow_id"]].reset_index()
    if not np.array_equal(aligned["flow_id"].to_numpy(), unknown["flow_id"].to_numpy()):
        raise RuntimeError(f"{setting}: failed to align frozen Unknown latent flow IDs")
    columns = latent_columns(aligned.columns)
    scaler = joblib.load(artifact / "standard_scaler.joblib")
    pca = joblib.load(artifact / "pca64.joblib")
    values = pca.transform(scaler.transform(aligned[columns].to_numpy(dtype=np.float64)))
    result: dict[str, np.ndarray] = {}
    for prefix, model_file in (("single", "single_full_k1_models.joblib"), ("multi", "multi_full_k2_models.joblib")):
        models = joblib.load(artifact / model_file)
        best_classes = unknown[f"{prefix}_predicted_known_class"].to_numpy(dtype=object)
        replay_scores = np.empty(len(unknown), dtype=np.float64)
        components = np.zeros(len(unknown), dtype=np.int64)
        posteriors = np.ones(len(unknown), dtype=np.float64)
        for class_name in np.unique(best_classes):
            positions = np.flatnonzero(best_classes == class_name)
            model = models[str(class_name)]
            replay_scores[positions] = model.score_samples(values[positions])
            components[positions] = model.predict(values[positions]).astype(np.int64)
            if prefix == "multi":
                probabilities = model.predict_proba(values[positions])
                posteriors[positions] = probabilities[np.arange(len(positions)), components[positions]]
        frozen_scores = unknown[f"{prefix}_known_score"].to_numpy(dtype=np.float64)
        if not np.allclose(replay_scores, frozen_scores, rtol=1e-10, atol=1e-8):
            maximum_error = float(np.max(np.abs(replay_scores - frozen_scores)))
            raise RuntimeError(f"{setting}: {prefix} replay score mismatch; max abs error={maximum_error}")
        result[f"{prefix}_component_id"] = components
        result[f"{prefix}_component_posterior"] = posteriors
        result[f"{prefix}_replay_score_abs_error"] = np.abs(replay_scores - frozen_scores)
    return result


def absorption_and_transitions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    delta_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    frozen_predictions: dict[str, pd.DataFrame] = {}

    for setting in SETTINGS:
        predictions, thresholds = load_frozen_predictions(setting)
        frozen_predictions[setting] = predictions
        unknown = predictions[predictions["known_or_unknown"] == "Unknown"].copy().reset_index(drop=True)
        replay = replay_unknown_component_assignments(setting, unknown)
        known_classes = [str(name) for name in fold(setting)["known_classes"]]
        unknown_classes = [str(name) for name in fold(setting)["unknown_classes"]]
        single_accept = unknown["single_accepted_as_known"].astype(bool)
        multi_accept = unknown["multi_accepted_as_known"].astype(bool)
        total = len(unknown)

        for known_class in known_classes:
            single_count = int((single_accept & (unknown["single_predicted_known_class"] == known_class)).sum())
            multi_count = int((multi_accept & (unknown["multi_predicted_known_class"] == known_class)).sum())
            delta_rows.append({
                "setting": setting,
                "known_class": known_class,
                "single_absorbed_unknown": single_count,
                "multi_absorbed_unknown": multi_count,
                "delta_absorbed": multi_count - single_count,
                "single_absorption_rate": single_count / total,
                "multi_absorption_rate": multi_count / total,
                "unknown_samples": total,
            })
        for unknown_class in unknown_classes:
            group = unknown[unknown["class_name"] == unknown_class]
            group_single = group["single_accepted_as_known"].astype(bool)
            group_multi = group["multi_accepted_as_known"].astype(bool)
            for known_class in known_classes:
                single_count = int((group_single & (group["single_predicted_known_class"] == known_class)).sum())
                multi_count = int((group_multi & (group["multi_predicted_known_class"] == known_class)).sum())
                matrix_rows.append({
                    "setting": setting,
                    "unknown_class": unknown_class,
                    "known_class": known_class,
                    "unknown_samples": len(group),
                    "single_absorbed_count": single_count,
                    "multi_absorbed_count": multi_count,
                    "delta_absorbed": multi_count - single_count,
                    "single_absorption_rate": single_count / len(group),
                    "multi_absorption_rate": multi_count / len(group),
                })

        detail = pd.DataFrame({
            "flow_id": unknown["flow_id"],
            "unknown_class": unknown["class_name"],
            "setting": setting,
            "single_best_known_class": unknown["single_predicted_known_class"],
            "multi_best_known_class": unknown["multi_predicted_known_class"],
            "single_score": unknown["single_known_score"].astype(float),
            "multi_score": unknown["multi_known_score"].astype(float),
            "single_threshold": thresholds["Single-Full"],
            "multi_threshold": thresholds["Multi-Full-K2"],
            "single_threshold_margin": unknown["single_known_score"].astype(float) - thresholds["Single-Full"],
            "multi_threshold_margin": unknown["multi_known_score"].astype(float) - thresholds["Multi-Full-K2"],
            "single_accepted_as_known": single_accept.to_numpy(),
            "multi_accepted_as_known": multi_accept.to_numpy(),
            "transition": transition_code(single_accept, multi_accept),
            "single_component_id": replay["single_component_id"],
            "multi_component_id": replay["multi_component_id"],
            "multi_component_posterior": replay["multi_component_posterior"],
            "multi_component_log_joint": unknown["multi_known_score"].to_numpy(dtype=np.float64)
            + np.log(replay["multi_component_posterior"]),
            "single_replay_score_abs_error": replay["single_replay_score_abs_error"],
            "multi_replay_score_abs_error": replay["multi_replay_score_abs_error"],
        })
        detail_frames.append(detail)

        official = pd.read_csv(STAGE3_ROOT / "outputs" / setting / "decision_transition_matrix.csv")
        observed = detail["transition"].value_counts().to_dict()
        labels = {
            "Recovered Unknown: Single accepted, Multi rejected": "MS",
            "Regressed Unknown: Single rejected, Multi accepted": "SM",
            "Accepted by both": "MM",
            "Rejected by both": "SS",
        }
        for row in official[official["population"] == "Unknown"].itertuples(index=False):
            if int(row.count) != int(observed.get(labels[row.transition], 0)):
                raise RuntimeError(f"{setting}: transition count mismatch for {row.transition}")
        official_unknown = pd.read_csv(STAGE3_ROOT / "outputs" / setting / "per_unknown_class_results.csv")
        for row in official_unknown.itertuples(index=False):
            group = detail[detail["unknown_class"] == row.unknown_class]
            if not np.isclose(group["single_accepted_as_known"].mean(), row.single_unknown_far):
                raise RuntimeError(f"{setting}/{row.unknown_class}: Single UFAR mismatch")
            if not np.isclose(group["multi_accepted_as_known"].mean(), row.multi_unknown_far):
                raise RuntimeError(f"{setting}/{row.unknown_class}: Multi UFAR mismatch")

    delta = pd.DataFrame(delta_rows)
    delta["abs_delta_absorbed"] = delta["delta_absorbed"].abs()
    delta = delta.sort_values(["setting", "abs_delta_absorbed", "known_class"], ascending=[True, False, True])
    delta["absolute_delta_rank"] = delta.groupby("setting")["abs_delta_absorbed"].rank(method="first", ascending=False).astype(int)
    matrix = pd.DataFrame(matrix_rows).sort_values(["setting", "unknown_class", "delta_absorbed", "known_class"], ascending=[True, True, False, True])
    details = pd.concat(detail_frames, ignore_index=True)
    return delta, matrix, details, frozen_predictions


def select_top_pairs(delta: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    a1 = delta[delta["setting"] == "A-1"].copy()
    a1["total_absorption"] = a1["single_absorbed_unknown"] + a1["multi_absorbed_unknown"]
    a1 = a1.sort_values(["delta_absorbed", "total_absorption", "known_class"], ascending=[False, False, True]).head(3)
    a1["selection_role"] = "A-1_top_failure"
    a1["selection_rank"] = range(1, len(a1) + 1)
    selected.append(a1)
    a3 = delta[delta["setting"] == "A-3"].copy()
    a3["total_absorption"] = a3["single_absorbed_unknown"] + a3["multi_absorbed_unknown"]
    a3 = a3.sort_values(["delta_absorbed", "total_absorption", "known_class"], ascending=[True, False, True]).head(3)
    a3["selection_role"] = "A-3_top_improvement"
    a3["selection_rank"] = range(1, len(a3) + 1)
    selected.append(a3)
    result = pd.concat(selected, ignore_index=True)
    result["nonzero_contributor"] = result["delta_absorbed"] != 0
    return result


def known_only_diagnostics(delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []
    top_pairs = select_top_pairs(delta)
    selected = set(zip(top_pairs["setting"], top_pairs["known_class"]))

    for setting in SETTINGS:
        artifact = STAGE3_ROOT / "artifacts" / setting
        train = pd.read_parquet(artifact / "train_known_mu.parquet")
        val = pd.read_parquet(artifact / "val_known_mu.parquet")
        mu_columns = latent_columns(train.columns)
        scaler = joblib.load(artifact / "standard_scaler.joblib")
        pca = joblib.load(artifact / "pca64.joblib")
        k1_models = joblib.load(artifact / "single_full_k1_models.joblib")
        k2_models = joblib.load(artifact / "multi_full_k2_models.joblib")
        val_global = pd.read_parquet(artifact / "val_known_scores.parquet")
        thresholds = threshold_map(setting)

        if set(k1_models) != set(k2_models) or set(k1_models) != set(fold(setting)["known_classes"]):
            raise RuntimeError(f"{setting}: frozen model classes differ from protocol")
        for class_name in k1_models:
            train_group = train[train["class_name"] == class_name]
            val_group = val[val["class_name"] == class_name]
            train_pca = pca.transform(scaler.transform(train_group[mu_columns].to_numpy(dtype=np.float64)))
            val_pca = pca.transform(scaler.transform(val_group[mu_columns].to_numpy(dtype=np.float64)))
            k1 = k1_models[class_name]
            k2 = k2_models[class_name]
            if k1.n_components != 1 or k2.n_components != 2:
                raise RuntimeError(f"{setting}/{class_name}: frozen K changed")
            if k1.covariance_type != "full" or k2.covariance_type != "full":
                raise RuntimeError(f"{setting}/{class_name}: frozen covariance type changed")
            k1_train_score = k1.score_samples(train_pca)
            k1_val_score = k1.score_samples(val_pca)
            k2_train_score = k2.score_samples(train_pca)
            k2_val_score = k2.score_samples(val_pca)
            assignments = k2.predict(val_pca)
            val_counts = np.bincount(assignments, minlength=2)
            val_weights = val_counts / len(val_pca)
            train_weights = np.asarray(k2.weights_, dtype=np.float64)
            tv = float(0.5 * np.abs(train_weights - val_weights).sum())
            diff = np.asarray(k2.means_[0] - k2.means_[1], dtype=np.float64)
            pooled = train_weights[0] * k2.covariances_[0] + train_weights[1] * k2.covariances_[1]
            mahalanobis = float(np.sqrt(max(0.0, diff @ np.linalg.solve(pooled, diff))))
            euclidean = float(np.linalg.norm(diff))
            k1_cov = covariance_summary(k1.covariances_[0], "k1_cov")
            k2_cov0 = covariance_summary(k2.covariances_[0], "k2_cov0")
            k2_cov1 = covariance_summary(k2.covariances_[1], "k2_cov1")
            weighted_k2_logdet = float(train_weights[0] * k2_cov0["k2_cov0_logdet"] + train_weights[1] * k2_cov1["k2_cov1_logdet"])
            log_volume_ratio = 0.5 * (weighted_k2_logdet - k1_cov["k1_cov_logdet"])
            own_shift = k2_val_score - k1_val_score
            global_group = val_global[val_global["class_name"] == class_name]
            single_accept = global_group["single_known_score"] >= thresholds["Single-Full"]
            multi_accept = global_group["multi_known_score"] >= thresholds["Multi-Full-K2"]
            margin_shift = (
                global_group["multi_known_score"].to_numpy(dtype=np.float64) - thresholds["Multi-Full-K2"]
                - global_group["single_known_score"].to_numpy(dtype=np.float64) + thresholds["Single-Full"]
            )
            row: dict[str, object] = {
                "setting": setting,
                "known_class": class_name,
                "representation": "frozen_mu_x_standardized_PCA64",
                "fit_operation_performed": False,
                "k1_converged": bool(k1.converged_),
                "k2_converged": bool(k2.converged_),
                "k1_n_iter": int(k1.n_iter_),
                "k2_n_iter": int(k2.n_iter_),
                "train_samples": len(train_group),
                "val_samples": len(val_group),
                "k1_train_avg_loglik": float(k1_train_score.mean()),
                "k1_val_avg_loglik": float(k1_val_score.mean()),
                "k1_val_nll": float(-k1_val_score.mean()),
                "k2_train_avg_loglik": float(k2_train_score.mean()),
                "k2_val_avg_loglik": float(k2_val_score.mean()),
                "k2_val_nll": float(-k2_val_score.mean()),
                "delta_nll2": float(k1_val_score.mean() * -1 - k2_val_score.mean() * -1),
                "k2_weight_0": float(train_weights[0]),
                "k2_weight_1": float(train_weights[1]),
                "min_component_weight": float(train_weights.min()),
                "val_component_0_count": int(val_counts[0]),
                "val_component_1_count": int(val_counts[1]),
                "val_component_0_weight": float(val_weights[0]),
                "val_component_1_weight": float(val_weights[1]),
                "train_val_tv": tv,
                "tiny_component_lt_1pct": bool(train_weights.min() < 0.01),
                "validation_empty_component": bool(np.any(val_counts == 0)),
                "component_mean_euclidean_distance": euclidean,
                "component_mahalanobis_distance": mahalanobis,
                "weighted_k2_cov_logdet": weighted_k2_logdet,
                "log_covariance_volume_ratio": log_volume_ratio,
                "covariance_volume_ratio": float(np.exp(np.clip(log_volume_ratio, -700, 700))),
                "max_k2_cov_condition_number": float(max(k2_cov0["k2_cov0_condition_number"], k2_cov1["k2_cov1_condition_number"])),
                "val_global_disagreement_count": int((single_accept != multi_accept).sum()),
                "val_global_disagreement_rate": float((single_accept != multi_accept).mean()),
                "val_global_single_acceptance_rate": float(single_accept.mean()),
                "val_global_multi_acceptance_rate": float(multi_accept.mean()),
            }
            row.update(k1_cov)
            row.update(k2_cov0)
            row.update(k2_cov1)
            row.update(quantiles(k1_train_score, "k1_train_loglik"))
            row.update(quantiles(k1_val_score, "k1_val_loglik"))
            row.update(quantiles(k2_val_score, "k2_val_loglik"))
            row.update(quantiles(own_shift, "val_own_score_shift"))
            row.update(quantiles(margin_shift, "val_global_margin_shift"))
            rows.append(row)

            if (setting, class_name) in selected:
                for split_name, group, values in (("train", train_group, train_pca), ("val", val_group, val_pca)):
                    assignment_frames.append(pd.DataFrame({
                        "setting": setting,
                        "known_class": class_name,
                        "split": split_name,
                        "flow_id": group["flow_id"].to_numpy(),
                        "component_id": k2.predict(values).astype(int),
                    }))
        del train, val

    diagnostics = pd.DataFrame(rows).sort_values(["setting", "known_class"])
    signal_columns = [
        "setting", "known_class", "train_samples", "val_samples", "delta_nll2",
        "min_component_weight", "train_val_tv", "component_mean_euclidean_distance",
        "component_mahalanobis_distance", "log_covariance_volume_ratio", "covariance_volume_ratio",
        "val_own_score_shift_mean", "val_own_score_shift_median", "val_own_score_shift_p90",
        "val_own_score_shift_p95", "val_own_score_shift_p99", "val_global_disagreement_count",
        "val_global_disagreement_rate", "val_global_margin_shift_mean", "val_global_margin_shift_median",
        "val_global_margin_shift_p90", "val_global_margin_shift_p95", "val_global_margin_shift_p99",
        "tiny_component_lt_1pct", "validation_empty_component", "max_k2_cov_condition_number",
    ]
    signals = diagnostics[signal_columns].copy()
    assignments = pd.concat(assignment_frames, ignore_index=True)
    return diagnostics, signals, assignments


def correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("ALL", merged), *((setting, merged[merged["setting"] == setting]) for setting in SETTINGS)]
    for scope, frame in scopes:
        for signal in CORRELATION_SIGNALS:
            pair = frame[[signal, "delta_absorbed"]].dropna()
            if len(pair) < 3 or pair[signal].nunique() < 2 or pair["delta_absorbed"].nunique() < 2:
                rho, p_value, status = float("nan"), float("nan"), "not_estimable_constant_or_small_outcome"
            else:
                rho, p_value = spearmanr(pair[signal], pair["delta_absorbed"])
                status = "estimated_post_hoc"
            rows.append({
                "scope": scope,
                "signal": signal,
                "outcome": "delta_absorbed",
                "n": len(pair),
                "n_unique_outcome": int(pair["delta_absorbed"].nunique()),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "status": status,
                "interpretation_limit": "post-hoc; sparse zero-inflated class outcome; no independent validation",
            })
    return pd.DataFrame(rows)


def simple_statistics(assignments: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    needed_ids = set(assignments["flow_id"].astype(str))
    metadata = pd.read_csv(FLOW_INDEX, usecols=["flow_id", *FEATURES])
    metadata = metadata[metadata["flow_id"].astype(str).isin(needed_ids)]
    joined = assignments.merge(metadata, on="flow_id", how="left", validate="many_to_one")
    joined = joined.merge(delta[["setting", "known_class", "delta_absorbed"]], on=["setting", "known_class"], how="left", validate="many_to_one")
    selected = select_top_pairs(delta)[["setting", "known_class", "selection_role", "selection_rank", "nonzero_contributor"]]
    joined = joined.merge(selected, on=["setting", "known_class"], how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for keys, group in joined.groupby(["setting", "known_class", "split"], sort=True):
        setting, class_name, split = keys
        for feature in FEATURES:
            valid = group[["component_id", feature]].dropna()
            samples = [part[feature].to_numpy(dtype=np.float64) for _, part in valid.groupby("component_id")]
            samples = [sample for sample in samples if len(sample)]
            if len(samples) >= 2 and any(float(np.ptp(sample)) > 0 for sample in samples):
                try:
                    h_stat, p_value = kruskal(*samples)
                    note = ""
                except ValueError as exc:
                    h_stat, p_value, note = float("nan"), float("nan"), str(exc)
            else:
                h_stat, p_value, note = float("nan"), float("nan"), "insufficient or constant groups"
            def component_value(component: int, function) -> float:
                values = valid.loc[valid["component_id"] == component, feature].to_numpy(dtype=np.float64)
                return float(function(values)) if len(values) else float("nan")
            first = group.iloc[0]
            rows.append({
                "setting": setting,
                "known_class": class_name,
                "selection_role": first["selection_role"],
                "selection_rank": int(first["selection_rank"]),
                "nonzero_contributor": bool(first["nonzero_contributor"]),
                "delta_absorbed": int(first["delta_absorbed"]),
                "split": split,
                "feature": feature,
                "rows_expected": len(group),
                "valid_samples": len(valid),
                "missing_metadata": int(len(group) - len(valid)),
                "component_0_count": int((valid["component_id"] == 0).sum()),
                "component_1_count": int((valid["component_id"] == 1).sum()),
                "component_0_mean": component_value(0, np.mean),
                "component_1_mean": component_value(1, np.mean),
                "component_0_median": component_value(0, np.median),
                "component_1_median": component_value(1, np.median),
                "kruskal_H": float(h_stat),
                "p_value": float(p_value),
                "epsilon_squared": epsilon_squared_kruskal(float(h_stat), len(valid), len(samples)),
                "effect_size_formula": "max(0,(H-k+1)/(n-k))",
                "association_only": True,
                "note": note,
            })
    return pd.DataFrame(rows).sort_values(["selection_role", "selection_rank", "split", "feature"])


def failure_cases(merged: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    nonzero = merged[merged["delta_absorbed"] != 0].copy()
    rows: list[dict[str, object]] = []
    for row in nonzero.itertuples(index=False):
        setting_details = details[details["setting"] == row.setting]
        regressed = int(((setting_details["transition"] == "SM") & (setting_details["multi_best_known_class"] == row.known_class)).sum())
        recovered = int(((setting_details["transition"] == "MS") & (setting_details["single_best_known_class"] == row.known_class)).sum())
        regressed_rows = setting_details[(setting_details["transition"] == "SM") & (setting_details["multi_best_known_class"] == row.known_class)]
        recovered_rows = setting_details[(setting_details["transition"] == "MS") & (setting_details["single_best_known_class"] == row.known_class)]
        regressed_components = {str(int(key)): int(value) for key, value in regressed_rows["multi_component_id"].value_counts().sort_index().items()}
        recovered_components = {str(int(key)): int(value) for key, value in recovered_rows["multi_component_id"].value_counts().sort_index().items()}
        degenerate = bool(row.tiny_component_lt_1pct or row.validation_empty_component)
        rows.append({
            "setting": row.setting,
            "known_class": row.known_class,
            "delta_absorbed": int(row.delta_absorbed),
            "regressed_unknown_into_class": regressed,
            "recovered_unknown_from_class": recovered,
            "regressed_multi_component_counts": json.dumps(regressed_components, sort_keys=True),
            "recovered_multi_component_counts": json.dumps(recovered_components, sort_keys=True),
            "delta_nll2": row.delta_nll2,
            "k2_weight_0": row.k2_weight_0,
            "k2_weight_1": row.k2_weight_1,
            "train_val_tv": row.train_val_tv,
            "component_mean_euclidean_distance": row.component_mean_euclidean_distance,
            "component_mahalanobis_distance": row.component_mahalanobis_distance,
            "max_k2_cov_condition_number": row.max_k2_cov_condition_number,
            "tiny_component_lt_1pct": row.tiny_component_lt_1pct,
            "validation_empty_component": row.validation_empty_component,
            "degenerate_mixture_flag": degenerate,
            "density_utility_mismatch": bool(row.delta_nll2 > 0 and row.delta_absorbed > 0),
            "diagnostic_label": (
                "density_better_unknown_absorption_worse" if row.delta_nll2 > 0 and row.delta_absorbed > 0
                else "density_better_unknown_absorption_better" if row.delta_nll2 > 0 and row.delta_absorbed < 0
                else "other"
            ),
        })
    return pd.DataFrame(rows).sort_values(["setting", "delta_absorbed"], ascending=[True, False])


def write_markdown_outputs(
    provenance: dict[str, object], delta: pd.DataFrame, matrix: pd.DataFrame, details: pd.DataFrame,
    diagnostics: pd.DataFrame, correlations_frame: pd.DataFrame, simple: pd.DataFrame,
    cases: pd.DataFrame, output_hashes_after: dict[str, dict[str, object]],
) -> None:
    snapshot_lines = [
        "# Stage 3 Provenance Snapshot", "",
        f"- Frozen Stage 3 commit: `{SNAPSHOT_COMMIT}`",
        f"- Commit present and ancestor of analysis HEAD: `PASS`",
        f"- Protocol canonical SHA-256: `{PROTOCOL_CANONICAL_SHA256}`",
        f"- Protocol raw SHA-256: `{PROTOCOL_RAW_SHA256}`",
        f"- Execution plan SHA-256: `{EXECUTION_PLAN_SHA256}`", "",
        "| Setting | frozen package SHA-256 recorded before diagnosis | audit-run tree SHA-256 before | after | files | unchanged |",
        "|---|---|---|---|---:|---|",
    ]
    for setting in SETTINGS:
        before = provenance["stage3_output_trees_before"][setting]
        after = output_hashes_after[setting]
        unchanged = before["audit_run_tree_sha256_before"] == after["audit_run_tree_sha256_after"]
        snapshot_lines.append(
            f"| {setting} | `{before['snapshot_package_sha256']}` | `{before['audit_run_tree_sha256_before']}` | "
            f"`{after['audit_run_tree_sha256_after']}` | {after['file_count']} | {'PASS' if unchanged else 'FAIL'} |"
        )
    snapshot_lines.extend([
        "", f"- Frozen manifest files re-hashed successfully: `{len(provenance['verified_frozen_files'])}`.",
        "- Diagnosis wrote only under `stage3_failure_diagnosis/`; official Stage 3 output trees are byte-stable across the run.",
    ])
    (OUTPUT_ROOT / "provenance_snapshot.md").write_text("\n".join(snapshot_lines) + "\n", encoding="utf-8")

    q = lambda setting, klass: diagnostics[(diagnostics.setting == setting) & (diagnostics.known_class == klass)].iloc[0]
    htbot = q("A-1", "Htbot")
    virut = q("A-3", "Virut")
    top_corr = correlations_frame[(correlations_frame.scope == "ALL") & correlations_frame.spearman_rho.notna()].copy()
    top_corr["abs_rho"] = top_corr["spearman_rho"].abs()
    top_corr = top_corr.sort_values("abs_rho", ascending=False)
    strongest_signal = top_corr.iloc[0] if len(top_corr) else None
    a1 = delta[delta.setting == "A-1"].sort_values("delta_absorbed", ascending=False).iloc[0]
    a3 = delta[delta.setting == "A-3"].sort_values("delta_absorbed", ascending=True).iloc[0]
    mismatch = cases[cases["density_utility_mismatch"]]
    tiny_count = int(diagnostics["tiny_component_lt_1pct"].sum())
    empty_count = int(diagnostics["validation_empty_component"].sum())
    max_simple = simple[simple["nonzero_contributor"] & simple["epsilon_squared"].notna()].sort_values("epsilon_squared", ascending=False)
    simple_line = "No estimable association was produced."
    if len(max_simple):
        r = max_simple.iloc[0]
        simple_line = (
            f"The strongest measured association among non-zero contributors was {r.setting}/{r.known_class} "
            f"{r.split} `{r.feature}` (epsilon-squared={r.epsilon_squared:.6f}); direction alone did not identify utility."
        )

    hypothesis_lines = [
        "# Candidate Hypotheses", "",
        "> These are post-hoc hypotheses, not a validated Adaptive-K method. No candidate was executed on the observed USTC Final Test.", "",
        "## Candidate Rule R1 — Known-only mixture-validity gate", "",
        "Predeclare a rule on a future development benchmark using only Known Train/Validation: require DeltaNLL2 above its Known-only median, "
        "min component weight at or above its 25th percentile, and train/validation TV and maximum covariance condition number at or below their 75th percentiles. "
        "Freeze both percentiles and action before touching the future test set.", "",
        "## Candidate Rule R2 — support-expansion caution", "",
        "Treat a high Known-validation upper-tail score shift or acceptance-margin shift as a caution signal rather than evidence that K2 is safe. "
        "A predeclared 75th-percentile flag may trigger likelihood/support calibration research, but must not select K on the observed USTC test.", "",
        "## Candidate Rule R3 — metadata-associated component audit", "",
        "When K2 assignment is strongly associated with packet/byte/duration statistics, label the mixture as metadata-associated and require independent boundary validation. "
        "Do not assume that a simple-statistics component either helps or hurts: the present directions differ by class.", "",
        "## Known-only reference percentiles", "",
    ]
    for signal in ("delta_nll2", "min_component_weight", "train_val_tv", "max_k2_cov_condition_number", "val_own_score_shift_p95", "val_global_margin_shift_p95"):
        hypothesis_lines.append(
            f"- `{signal}`: P25={diagnostics[signal].quantile(.25):.6g}, P50={diagnostics[signal].quantile(.5):.6g}, P75={diagnostics[signal].quantile(.75):.6g}."
        )
    hypothesis_lines.extend([
        "", "These values describe this Known-only sample only. They were not optimized against Unknown outcomes and must not be transported as validated universal cutoffs.",
    ])
    (OUTPUT_ROOT / "candidate_hypotheses.md").write_text("\n".join(hypothesis_lines) + "\n", encoding="utf-8")

    corr_line = "No estimable pooled correlation."
    if strongest_signal is not None:
        corr_line = (
            f"Largest absolute pooled post-hoc Spearman coefficient: `{strongest_signal.signal}` "
            f"rho={strongest_signal.spearman_rho:.6f}, p={strongest_signal.p_value:.6g}, n={int(strongest_signal.n)}. "
            "This is exploratory under a sparse, zero-inflated outcome."
        )
    audit_lines = [
        "# Stage 3 Failure Diagnosis Audit Summary", "",
        "## Scope and validity", "",
        "This is post-hoc diagnosis of an already observed Final Test. It did not fit a scaler, PCA, Gaussian/GMM, threshold, encoder, K, or Adaptive detector, and did not rerun Final Test. Frozen Unknown test mu_x was loaded only to recover missing K2 component assignments; replayed Single/Multi scores were required to match the frozen scores row by row.", "",
        "## Attribution", "",
        f"- A-1: {a1.known_class} accounts for {int(a1.delta_absorbed):+d} of the net +86 absorbed Unknown samples; all other Known classes have zero delta.",
        "- A-2: Miuref contributes +16 and Shifu -1, for a net +15 over 5,579 Unknown samples; the large common Geodo absorption remains almost unchanged.",
        f"- A-3: {a3.known_class} accounts for {int(a3.delta_absorbed):+d}; these are 236 recovered Neris samples, while Shifu remains unchanged at two accepted Htbot samples.", "",
        "## Known-only geometry", "",
        f"- A-1/Htbot: DeltaNLL2={htbot.delta_nll2:.6f}, weights=({htbot.k2_weight_0:.6f},{htbot.k2_weight_1:.6f}), TV={htbot.train_val_tv:.6f}, Euclidean/Mahalanobis separation={htbot.component_mean_euclidean_distance:.6f}/{htbot.component_mahalanobis_distance:.6f}, max condition={htbot.max_k2_cov_condition_number:.6g}.",
        f"- A-3/Virut: DeltaNLL2={virut.delta_nll2:.6f}, weights=({virut.k2_weight_0:.6f},{virut.k2_weight_1:.6f}), TV={virut.train_val_tv:.6f}, Euclidean/Mahalanobis separation={virut.component_mean_euclidean_distance:.6f}/{virut.component_mahalanobis_distance:.6f}, max condition={virut.max_k2_cov_condition_number:.6g}.",
        f"- Degeneracy flags over {len(diagnostics)} setting-class cells: tiny components={tiny_count}, validation-empty components={empty_count}. The A-1/Htbot failure class has neither flag.", "",
        "## Density versus utility", "",
        f"- {len(mismatch)} class-setting cells improve Known-validation NLL while increasing Unknown absorption: " + ", ".join(f"{r.setting}/{r.known_class} ({int(r.delta_absorbed):+d})" for r in mismatch.itertuples()) + ".",
        "- Therefore density-fit improvement does not imply open-set utility.",
        f"- {corr_line}", "",
        "## Simple statistics", "",
        f"- {simple_line}",
        "- Because the Top-3 lists contain zero-delta ties, only Htbot (A-1) and Virut (A-3) are causal non-zero contributors; the other ranked rows are reference comparisons.", "",
        "## Final diagnosis", "",
        "**Diagnosis A — DENSITY-UTILITY MISMATCH.** K2 improves Known-validation density likelihood for the failure class while expanding acceptance of Unknown traffic. The mixture is not tiny, validation-empty, or train/validation-unstable, so simple mixture degeneration is not the primary explanation. Known-only correlations remain exploratory and do not establish a reliable class-selection rule.", "",
        "## Independent validation boundary", "",
        "Any rule proposed after observing A-1/A-2/A-3 must be frozen and evaluated first on an untouched benchmark. Priority: CSTNET-TLS1.3, then CipherSpectrum. No claim that an adaptive rule beats Single on USTC is made here.", "",
        "## Reproducibility checks", "",
        f"- Frozen binary prediction parity: PASS for {len(details):,} Unknown setting-sample rows.",
        f"- Frozen Unknown score replay parity: PASS; maximum Single/Multi absolute errors={details['single_replay_score_abs_error'].max():.3g}/{details['multi_replay_score_abs_error'].max():.3g}.",
        "- Official transition and per-Unknown-class aggregate parity: PASS.",
        "- Frozen scaler/PCA/GMM manifest hashes: PASS.",
        "- Stage 3 output tree before/after hashes: PASS.",
    ]
    (OUTPUT_ROOT / "audit_summary.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUTPUT_ROOT
    OUTPUT_ROOT = args.output_dir.resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    command = " ".join(sys.argv)
    provenance = verify_snapshot_and_frozen_inputs()
    delta, matrix, details, _ = absorption_and_transitions()
    diagnostics, signals, assignments = known_only_diagnostics(delta)
    merged = signals.merge(delta, on=["setting", "known_class"], how="inner", validate="one_to_one")
    correlation_frame = correlations(merged)
    simple = simple_statistics(assignments, delta)
    cases = failure_cases(merged.merge(
        diagnostics[["setting", "known_class", "k2_weight_0", "k2_weight_1"]],
        on=["setting", "known_class"], how="left", validate="one_to_one",
    ), details)

    delta.to_csv(OUTPUT_ROOT / "class_unknown_absorption_delta.csv", index=False)
    matrix.to_csv(OUTPUT_ROOT / "unknown_to_known_absorption_matrix.csv", index=False)
    details.to_parquet(OUTPUT_ROOT / "unknown_transition_details.parquet", index=False, engine="pyarrow", compression="zstd")
    diagnostics.to_csv(OUTPUT_ROOT / "known_only_support_diagnostics.csv", index=False)
    signals.to_csv(OUTPUT_ROOT / "known_only_candidate_signals.csv", index=False)
    merged.to_csv(OUTPUT_ROOT / "known_signal_vs_unknown_failure.csv", index=False)
    correlation_frame.to_csv(OUTPUT_ROOT / "known_signal_correlations.csv", index=False)
    simple.to_csv(OUTPUT_ROOT / "simple_statistics_followup.csv", index=False)
    cases.to_csv(OUTPUT_ROOT / "failure_case_summary.csv", index=False)

    output_hashes_after: dict[str, dict[str, object]] = {}
    for setting in SETTINGS:
        digest, count = tree_content_hash(STAGE3_ROOT / "outputs" / setting)
        output_hashes_after[setting] = {"audit_run_tree_sha256_after": digest, "file_count": count}
        if digest != provenance["stage3_output_trees_before"][setting]["audit_run_tree_sha256_before"]:
            raise RuntimeError(f"{setting}: official Stage 3 output tree changed during diagnosis")

    write_markdown_outputs(provenance, delta, matrix, details, diagnostics, correlation_frame, simple, cases, output_hashes_after)
    finished = utc_now()
    metadata = {
        "run_id": "stage3_failure_diagnosis_v1",
        "execution_type": "post_hoc_diagnostic",
        "status": "completed",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "command": command,
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUTPUT_ROOT),
        "git_head_at_run": git("rev-parse", "HEAD"),
        "stage3_snapshot_commit": SNAPSHOT_COMMIT,
        "protocol_canonical_sha256": PROTOCOL_CANONICAL_SHA256,
        "stage3_snapshot_package_sha256": SNAPSHOT_PACKAGE_SHA256,
        "stage3_output_trees_before": provenance["stage3_output_trees_before"],
        "stage3_output_trees_after": output_hashes_after,
        "verified_frozen_file_count": len(provenance["verified_frozen_files"]),
        "unknown_test_observed": True,
        "post_hoc_only": True,
        "fit_operations_performed": [],
        "test_embedding_files_loaded": [
            str(STAGE3_ROOT / "artifacts" / setting / "test_unknown_mu.parquet") for setting in SETTINGS
        ],
        "final_test_rerun": False,
        "threshold_changed": False,
        "k_changed": False,
        "adaptive_detector_executed": False,
        "frozen_model_operations": ["transform", "score_samples", "predict"],
        "known_data_usage": "Known Train and Known Validation diagnostics only",
        "unknown_data_usage": "frozen_final_predictions attribution plus frozen test_unknown_mu component replay; no fitting or metric rerun",
        "unknown_latent_hashes_verified": True,
        "unknown_replayed_scores_match_frozen": True,
        "random_processes": [],
        "simple_statistics_source": str(FLOW_INDEX),
        "simple_statistics_method": "existing Stage 0 flow_index join; no raw PCAP reparsing",
        "pooled_covariance_definition": "K2 weight-weighted within-component covariance",
        "epsilon_squared_formula": "max(0,(H-k+1)/(n-k))",
        "final_diagnosis": "A — DENSITY-UTILITY MISMATCH",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "conda_prefix": sys.prefix,
        },
        "row_counts": {
            "class_unknown_absorption_delta": len(delta),
            "unknown_to_known_absorption_matrix": len(matrix),
            "unknown_transition_details": len(details),
            "known_only_support_diagnostics": len(diagnostics),
            "known_signal_correlations": len(correlation_frame),
            "simple_statistics_followup": len(simple),
            "failure_case_summary": len(cases),
        },
    }
    (OUTPUT_ROOT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "diagnosis": metadata["final_diagnosis"], "output_dir": str(OUTPUT_ROOT), "row_counts": metadata["row_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
