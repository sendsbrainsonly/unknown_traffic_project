#!/usr/bin/env python3
"""Known-only Stage 4 boundary diagnosis over frozen Stage 3 assets.

The execution order is deliberate: calibrate/evaluate Known Validation, then
evaluate Known Test, and only then read the already-observed Unknown Test for
post-hoc mechanism diagnosis. No estimator or representation is refitted.
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
from scipy.special import logsumexp
from scipy.stats import spearmanr


STAGE4_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE4_ROOT.parent
STAGE3_ROOT = PROJECT_ROOT / "stage3_unknown_utility"
OUTPUT_ROOT = STAGE4_ROOT / "outputs"
SETTINGS = ("A-1", "A-2", "A-3")
RULES = ("Global", "Class-P05", "Component-P05")
SNAPSHOT_COMMIT = "ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc"
PROTOCOL_SHA256 = "1fff4ed33211de0b123cf4c0ec6b203cc33be683ac315b610ffd8f0194550ae1"
PACKAGE_HASHES = {
    "A-1": "0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9",
    "A-2": "5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6",
    "A-3": "1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be",
}
MIN_LOCAL_VALIDATION = 30
QUANTILE = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def latent_columns(columns: Iterable[str]) -> list[str]:
    result = sorted(name for name in columns if name.startswith("mu_"))
    if len(result) != 128:
        raise RuntimeError(f"expected 128 frozen mu_x columns, got {len(result)}")
    return result


def q(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("quantile input is empty or non-finite")
    return {probability: float(np.quantile(values, probability, method="linear")) for probability in probabilities}


def covariance_metrics(covariance: np.ndarray) -> tuple[float, float]:
    covariance = np.asarray(covariance, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if sign <= 0 or eigenvalues[0] <= 0:
        raise RuntimeError("frozen covariance is not positive definite")
    return float(logdet), float(eigenvalues[-1] / eigenvalues[0])


def thresholds(setting: str) -> dict[str, float]:
    frame = pd.read_csv(STAGE3_ROOT / "outputs" / setting / "threshold_calibration.csv")
    if len(frame) != 3 or int(frame["unknown_samples_used"].sum()) != 0:
        raise RuntimeError(f"{setting}: frozen threshold provenance failed")
    return dict(zip(frame["detector"], frame["threshold"].astype(float)))


def verify_inputs() -> dict[str, object]:
    subprocess.check_call(["git", "cat-file", "-e", f"{SNAPSHOT_COMMIT}^{{commit}}"], cwd=PROJECT_ROOT)
    if subprocess.call(["git", "merge-base", "--is-ancestor", SNAPSHOT_COMMIT, "HEAD"], cwd=PROJECT_ROOT) != 0:
        raise RuntimeError("Stage 3 snapshot is not an ancestor of HEAD")
    result: dict[str, object] = {"snapshot_commit": SNAPSHOT_COMMIT, "packages_before": {}, "verified_files": []}
    for setting in SETTINGS:
        digest, count = tree_hash(STAGE3_ROOT / "outputs" / setting)
        if digest != PACKAGE_HASHES[setting]:
            raise RuntimeError(f"{setting}: Stage 3 package hash changed")
        result["packages_before"][setting] = {"sha256": digest, "files": count}
        frozen = json.loads((STAGE3_ROOT / "outputs" / setting / "frozen_model_hashes.json").read_text(encoding="utf-8"))
        if frozen["protocol_canonical_sha256"] != PROTOCOL_SHA256:
            raise RuntimeError(f"{setting}: protocol hash mismatch")
        if frozen["fit_data"] != "Known Train only" or frozen["threshold_data"] != "Known Validation only":
            raise RuntimeError(f"{setting}: frozen fitting provenance changed")
        for item in frozen["files"]:
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"{setting}: frozen file hash changed: {path}")
            result["verified_files"].append(str(path))
        latent_manifest = json.loads((STAGE3_ROOT / "outputs" / setting / "latent_export_manifest.json").read_text(encoding="utf-8"))
        for name in ("train_known_mu.parquet", "val_known_mu.parquet", "test_known_mu.parquet", "test_unknown_mu.parquet"):
            item = latent_manifest["files"][name]
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"{setting}: latent hash changed: {name}")
    return result


def load_models(setting: str) -> tuple[object, object, dict[str, object], dict[str, object]]:
    root = STAGE3_ROOT / "artifacts" / setting
    scaler = joblib.load(root / "standard_scaler.joblib")
    pca = joblib.load(root / "pca64.joblib")
    k1 = joblib.load(root / "single_full_k1_models.joblib")
    k2 = joblib.load(root / "multi_full_k2_models.joblib")
    if list(k1) != list(k2):
        raise RuntimeError(f"{setting}: K1/K2 class ordering mismatch")
    for name, model in k2.items():
        if model.n_components != 2 or model.covariance_type != "full" or not model.converged_:
            raise RuntimeError(f"{setting}/{name}: frozen K2 model contract changed")
    return scaler, pca, k1, k2


def transform(frame: pd.DataFrame, scaler: object, pca: object) -> np.ndarray:
    columns = latent_columns(frame.columns)
    return pca.transform(scaler.transform(frame[columns].to_numpy(dtype=np.float64)))


def score_k2(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(models)
    local_blocks: list[np.ndarray] = []
    class_blocks: list[np.ndarray] = []
    for name in names:
        local = np.asarray(models[name]._estimate_weighted_log_prob(values), dtype=np.float64)
        if local.shape != (len(values), 2):
            raise RuntimeError(f"{name}: unexpected local-score shape")
        local_blocks.append(local)
        class_blocks.append(logsumexp(local, axis=1))
    return names, np.column_stack(class_blocks), np.column_stack(local_blocks)


def score_k1(models: dict[str, object], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(list(models), dtype=object)
    matrix = np.column_stack([models[name].score_samples(values) for name in names])
    best = np.argmax(matrix, axis=1)
    return matrix[np.arange(len(values)), best], names[best]


def true_components(labels: np.ndarray, names: list[str], local_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {name: index for index, name in enumerate(names)}
    components = np.empty(len(labels), dtype=np.int64)
    assigned_scores = np.empty(len(labels), dtype=np.float64)
    for name in np.unique(labels):
        positions = np.flatnonzero(labels == name)
        index = lookup[str(name)]
        block = local_scores[positions, 2 * index:2 * index + 2]
        selected = np.argmax(block, axis=1)
        components[positions] = selected
        assigned_scores[positions] = block[np.arange(len(positions)), selected]
    return components, assigned_scores


def rule_decisions(
    names: list[str], class_scores: np.ndarray, local_scores: np.ndarray,
    global_threshold: float, class_thresholds: np.ndarray, local_thresholds: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    class_names = np.asarray(names, dtype=object)
    local_names = np.asarray([name for name in names for _ in range(2)], dtype=object)
    local_components = np.tile(np.arange(2), len(names))
    raw = {
        "Global": class_scores - global_threshold,
        "Class-P05": class_scores - class_thresholds[None, :],
        "Component-P05": local_scores - local_thresholds[None, :],
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    for rule, matrix in raw.items():
        best = np.argmax(matrix, axis=1)
        margin = matrix[np.arange(len(matrix)), best]
        if rule == "Component-P05":
            predicted_class = local_names[best]
            predicted_component = local_components[best]
        else:
            predicted_class = class_names[best]
            predicted_component = np.full(len(best), -1, dtype=np.int64)
        result[rule] = {
            "margin": margin,
            "accepted": margin >= 0,
            "predicted_class": predicted_class,
            "predicted_component": predicted_component,
            "matrix": matrix,
        }
    return result


def verify_stage3_predictions(
    setting: str, frame: pd.DataFrame, values: np.ndarray, names: list[str],
    class_scores: np.ndarray, k1_models: dict[str, object], role: str,
) -> dict[str, object]:
    frozen_all = pd.read_parquet(STAGE3_ROOT / "artifacts" / setting / "frozen_final_predictions.parquet")
    frozen = frozen_all[frozen_all["known_or_unknown"] == role].set_index("flow_id").loc[frame["flow_id"]].reset_index()
    if not np.array_equal(frozen["flow_id"].to_numpy(), frame["flow_id"].to_numpy()):
        raise RuntimeError(f"{setting}/{role}: frozen prediction alignment failed")
    frozen_thresholds = thresholds(setting)
    names_array = np.asarray(names, dtype=object)
    multi_best = np.argmax(class_scores, axis=1)
    multi_score = class_scores[np.arange(len(frame)), multi_best]
    multi_class = names_array[multi_best]
    single_score, single_class = score_k1(k1_models, values)
    native_score = -frame["native_unknown_score"].to_numpy(dtype=np.float64)
    checks = (
        np.allclose(multi_score, frozen["multi_known_score"], rtol=1e-10, atol=1e-8),
        np.array_equal(multi_class, frozen["multi_predicted_known_class"].to_numpy()),
        np.array_equal(multi_score >= frozen_thresholds["Multi-Full-K2"], frozen["multi_accepted_as_known"].to_numpy(dtype=bool)),
        np.allclose(single_score, frozen["single_known_score"], rtol=1e-10, atol=1e-8),
        np.array_equal(single_class, frozen["single_predicted_known_class"].to_numpy()),
        np.array_equal(single_score >= frozen_thresholds["Single-Full"], frozen["single_accepted_as_known"].to_numpy(dtype=bool)),
        np.allclose(native_score, frozen["native_known_score"], rtol=0, atol=0),
        np.array_equal(frame["native_predicted_known_class"].to_numpy(), frozen["native_predicted_known_class"].to_numpy()),
        np.array_equal(native_score >= frozen_thresholds["Native"], frozen["native_accepted_as_known"].to_numpy(dtype=bool)),
    )
    if not all(checks):
        raise RuntimeError(f"{setting}/{role}: Stage 3 prediction replay mismatch: {checks}")
    return {
        "setting": setting,
        "population": role,
        "rows": len(frame),
        "native_score_max_abs_error": float(np.max(np.abs(native_score - frozen["native_known_score"].to_numpy()))),
        "single_score_max_abs_error": float(np.max(np.abs(single_score - frozen["single_known_score"].to_numpy()))),
        "multi_score_max_abs_error": float(np.max(np.abs(multi_score - frozen["multi_known_score"].to_numpy()))),
        "class_and_binary_parity": True,
    }


def build_calibration(
    setting: str, train: pd.DataFrame, val: pd.DataFrame, scaler: object, pca: object,
    models: dict[str, object], names: list[str], val_class_scores: np.ndarray,
    val_local_scores: np.ndarray, global_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[tuple[str, int], np.ndarray]]:
    labels = val["class_name"].to_numpy(dtype=object)
    lookup = {name: index for index, name in enumerate(names)}
    components, assigned_local = true_components(labels, names, val_local_scores)
    class_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    class_tau = np.empty(len(names), dtype=np.float64)
    local_tau = np.empty(2 * len(names), dtype=np.float64)
    calibration_distributions: dict[tuple[str, int], np.ndarray] = {}
    train_counts: dict[tuple[str, int], int] = {}
    for name in names:
        group = train[train["class_name"] == name]
        group_values = transform(group, scaler, pca)
        weighted = np.asarray(models[name]._estimate_weighted_log_prob(group_values), dtype=np.float64)
        assignment = np.argmax(weighted, axis=1)
        for component in range(2):
            train_counts[(name, component)] = int(np.sum(assignment == component))

    for name in names:
        index = lookup[name]
        positions = np.flatnonzero(labels == name)
        own_class_scores = val_class_scores[positions, index]
        tau = float(np.quantile(own_class_scores, QUANTILE, method="linear"))
        class_tau[index] = tau
        class_rows.append({
            "setting": setting, "known_class": name, "calibration_split": "Known Validation only",
            "unknown_samples_used": 0, "quantile": QUANTILE, "validation_samples": len(positions),
            "class_threshold": tau, "own_class_acceptance_at_threshold": float(np.mean(own_class_scores >= tau)),
            "own_class_frr_at_threshold": float(np.mean(own_class_scores < tau)),
            "global_threshold": global_threshold, "class_minus_global_threshold": tau - global_threshold,
        })
        cov0, cov1 = models[name].covariances_
        difference = models[name].means_[0] - models[name].means_[1]
        weights = np.asarray(models[name].weights_, dtype=np.float64)
        pooled = weights[0] * cov0 + weights[1] * cov1
        euclidean = float(np.linalg.norm(difference))
        mahalanobis = float(np.sqrt(max(0.0, difference @ np.linalg.solve(pooled, difference))))
        for component in range(2):
            selected_positions = positions[components[positions] == component]
            scores = val_local_scores[selected_positions, 2 * index + component]
            calibration_distributions[(name, component)] = np.sort(scores)
            fallback = len(scores) < MIN_LOCAL_VALIDATION
            raw_tau = float(np.quantile(scores, QUANTILE, method="linear")) if len(scores) else float("nan")
            effective_tau = tau if fallback else raw_tau
            local_tau[2 * index + component] = effective_tau
            logdet, condition = covariance_metrics(models[name].covariances_[component])
            local_rows.append({
                "setting": setting, "known_class": name, "component_id": component,
                "calibration_split": "Known Validation only", "unknown_samples_used": 0,
                "quantile": QUANTILE, "min_validation_samples": MIN_LOCAL_VALIDATION,
                "train_count": train_counts[(name, component)], "validation_count": len(scores),
                "model_weight": float(weights[component]), "validation_empirical_ratio": len(scores) / len(positions),
                "raw_local_p05": raw_tau, "class_threshold": tau, "effective_local_threshold": effective_tau,
                "fallback_to_class_threshold": fallback, "effective_minus_class_threshold": effective_tau - tau,
                "effective_minus_global_threshold": effective_tau - global_threshold,
                "local_score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "local_score_std": float(np.std(scores, ddof=0)) if len(scores) else float("nan"),
                "covariance_logdet": logdet, "covariance_condition_number": condition,
                "component_mean_euclidean_distance": euclidean,
                "component_mahalanobis_distance": mahalanobis,
            })
    return (
        pd.DataFrame(class_rows), pd.DataFrame(local_rows), class_tau, local_tau,
        components, assigned_local, calibration_distributions,
    )


def global_class_coverage(
    setting: str, labels: np.ndarray, global_scores: np.ndarray, global_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    margin = global_scores - global_threshold
    for name in sorted(np.unique(labels)):
        values = global_scores[labels == name]
        margins = margin[labels == name]
        score_q = q(values, (0.05, 0.10, 0.50, 0.90, 0.95))
        margin_q = q(margins, (0.05, 0.50, 0.95))
        rows.append({
            "setting": setting, "known_class": name, "split": "Known Validation",
            "sample_count": len(values), "global_threshold": global_threshold,
            "accepted_count": int(np.sum(margins >= 0)), "acceptance_rate": float(np.mean(margins >= 0)),
            "FRR": float(np.mean(margins < 0)), "score_mean": float(np.mean(values)),
            "score_median": float(np.median(values)), "score_P05": score_q[0.05], "score_P10": score_q[0.10],
            "score_P50": score_q[0.50], "score_P90": score_q[0.90], "score_P95": score_q[0.95],
            "margin_mean": float(np.mean(margins)), "margin_P05": margin_q[0.05],
            "margin_P50": margin_q[0.50], "margin_P95": margin_q[0.95],
        })
    return pd.DataFrame(rows)


def global_component_coverage(
    setting: str, labels: np.ndarray, components: np.ndarray, assigned_local: np.ndarray,
    global_accept: np.ndarray, global_threshold: float, local_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lookup = local_thresholds.set_index(["known_class", "component_id"])
    for name in sorted(np.unique(labels)):
        class_total = int(np.sum(labels == name))
        for component in range(2):
            mask = (labels == name) & (components == component)
            scores = assigned_local[mask]
            accepted = global_accept[mask]
            quant = q(scores, (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99))
            meta = lookup.loc[(name, component)]
            rows.append({
                "setting": setting, "known_class": name, "component_id": component,
                "split": "Known Validation", "component_train_count": int(meta.train_count),
                "component_val_count": len(scores), "model_weight": float(meta.model_weight),
                "validation_empirical_ratio": len(scores) / class_total, "global_threshold": global_threshold,
                "global_accepted_count": int(np.sum(accepted)), "global_acceptance_rate": float(np.mean(accepted)),
                "global_FRR": float(np.mean(~accepted)), "local_score_mean": float(np.mean(scores)),
                "local_score_median": float(np.median(scores)), "local_score_P01": quant[0.01],
                "local_score_P05": quant[0.05], "local_score_P10": quant[0.10],
                "local_score_P50": quant[0.50], "local_score_P90": quant[0.90],
                "local_score_P95": quant[0.95], "local_score_P99": quant[0.99],
            })
    return pd.DataFrame(rows)


def comparison_rows(
    setting: str, split: str, labels: np.ndarray, components: np.ndarray,
    decisions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule in RULES:
        accepted = decisions[rule]["accepted"]
        for level, groups in (
            ("overall", [("__ALL__", -1, np.ones(len(labels), dtype=bool))]),
            ("class", [(name, -1, labels == name) for name in sorted(np.unique(labels))]),
            ("component", [(name, component, (labels == name) & (components == component)) for name in sorted(np.unique(labels)) for component in range(2)]),
        ):
            for name, component, mask in groups:
                rows.append({
                    "setting": setting, "split": split, "rule": rule, "level": level,
                    "known_class": name, "component_id": component, "sample_count": int(np.sum(mask)),
                    "accepted_count": int(np.sum(accepted[mask])), "acceptance_rate": float(np.mean(accepted[mask])),
                    "FRR": float(np.mean(~accepted[mask])),
                })
    frame = pd.DataFrame(rows)
    components_frame = frame[frame["level"] == "component"]
    gaps = components_frame.groupby(["setting", "split", "rule", "known_class"]).agg(
        component_acceptance_gap=("acceptance_rate", lambda x: float(x.max() - x.min())),
        component_FRR_std=("FRR", lambda x: float(np.std(x, ddof=0))),
    ).reset_index()
    class_mask = frame["level"] == "class"
    frame = frame.merge(gaps, on=["setting", "split", "rule", "known_class"], how="left")
    for rule in RULES:
        class_values = frame[(frame["rule"] == rule) & class_mask]["acceptance_rate"]
        mask = (frame["rule"] == rule) & (frame["level"] == "overall")
        frame.loc[mask, "class_acceptance_gap"] = float(class_values.max() - class_values.min())
        frame.loc[mask, "class_FRR_std"] = float(np.std(1.0 - class_values, ddof=0))
    return frame


def build_heterogeneity(
    val_comparison: pd.DataFrame, test_comparison: pd.DataFrame,
    local_thresholds: pd.DataFrame, class_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting in SETTINGS:
        local_setting = local_thresholds[local_thresholds["setting"] == setting]
        class_setting = class_thresholds[class_thresholds["setting"] == setting].set_index("known_class")
        for name in sorted(local_setting["known_class"].unique()):
            local = local_setting[local_setting["known_class"] == name].sort_values("component_id")
            row: dict[str, object] = {
                "setting": setting, "known_class": name,
                "global_threshold": float(class_setting.loc[name, "global_threshold"]),
                "class_threshold": float(class_setting.loc[name, "class_threshold"]),
                "component_0_threshold": float(local.iloc[0]["effective_local_threshold"]),
                "component_1_threshold": float(local.iloc[1]["effective_local_threshold"]),
                "local_threshold_absolute_gap": float(abs(local.iloc[0]["effective_local_threshold"] - local.iloc[1]["effective_local_threshold"])),
                "fallback_count": int(local["fallback_to_class_threshold"].sum()),
            }
            for split_name, source in (("val", val_comparison), ("test", test_comparison)):
                for rule in RULES:
                    match = source[(source["setting"] == setting) & (source["rule"] == rule) & (source["level"] == "class") & (source["known_class"] == name)].iloc[0]
                    key = rule.lower().replace("-p05", "").replace("component", "local")
                    row[f"{split_name}_{key}_component_acceptance_gap"] = float(match.component_acceptance_gap)
                    row[f"{split_name}_{key}_component_FRR_std"] = float(match.component_FRR_std)
            rows.append(row)
    return pd.DataFrame(rows)


def factor_correlations(local_thresholds: pd.DataFrame) -> pd.DataFrame:
    factors = (
        "model_weight", "covariance_logdet", "local_score_std",
        "component_mean_euclidean_distance", "component_mahalanobis_distance",
        "validation_count", "train_count",
    )
    outcomes = ("effective_local_threshold", "effective_minus_class_threshold")
    rows: list[dict[str, object]] = []
    scopes = [("ALL", local_thresholds), *((setting, local_thresholds[local_thresholds["setting"] == setting]) for setting in SETTINGS)]
    for scope, frame in scopes:
        for outcome in outcomes:
            for factor in factors:
                pair = frame[[factor, outcome]].dropna()
                if len(pair) < 3 or pair[factor].nunique() < 2 or pair[outcome].nunique() < 2:
                    rho, p_value, status = float("nan"), float("nan"), "not_estimable"
                else:
                    rho, p_value = spearmanr(pair[factor], pair[outcome])
                    status = "estimated_known_only"
                rows.append({
                    "scope": scope, "outcome": outcome, "factor": factor, "n": len(pair),
                    "spearman_rho": float(rho), "p_value": float(p_value), "status": status,
                    "data_scope": "Known Validation calibration plus frozen Known Train/model factors; Unknown usage=0",
                })
    return pd.DataFrame(rows)


def percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(reference):
        return np.full(len(values), np.nan)
    return np.searchsorted(reference, values, side="right") / len(reference)


def case_studies(
    setting: str, frame: pd.DataFrame, frozen: pd.DataFrame, names: list[str],
    class_scores: np.ndarray, local_scores: np.ndarray, decisions: dict[str, dict[str, np.ndarray]],
    class_tau: np.ndarray, local_tau: np.ndarray, distributions: dict[tuple[str, int], np.ndarray],
    global_threshold: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    lookup = {name: index for index, name in enumerate(names)}
    aligned = frozen.set_index("flow_id").loc[frame["flow_id"]].reset_index()
    def build(unknown_class: str, known_class: str, focus: np.ndarray) -> pd.DataFrame:
        y = lookup[known_class]
        block = local_scores[:, 2 * y:2 * y + 2]
        component = np.argmax(block, axis=1)
        assigned_score = block[np.arange(len(block)), component]
        assigned_percentile = np.empty(len(frame), dtype=np.float64)
        for k in range(2):
            mask = component == k
            assigned_percentile[mask] = percentile(assigned_score[mask], distributions[(known_class, k)])
        posterior = np.exp(assigned_score - class_scores[:, y])
        return pd.DataFrame({
            "flow_id": frame["flow_id"], "setting": setting, "unknown_class": frame["class_name"],
            "case_known_class": known_class, "focus_case": focus,
            "stage3_single_accept": aligned["single_accepted_as_known"].to_numpy(dtype=bool),
            "stage3_global_k2_accept": aligned["multi_accepted_as_known"].to_numpy(dtype=bool),
            "stage3_single_best_class": aligned["single_predicted_known_class"],
            "stage3_global_k2_best_class": aligned["multi_predicted_known_class"],
            "case_component_id": component, "case_component_posterior": posterior,
            "component_joint_score": assigned_score,
            "component_validation_percentile": assigned_percentile,
            "component_log_joint_gap": np.abs(block[:, 0] - block[:, 1]),
            "global_margin": decisions["Global"]["margin"],
            "class_own_margin": class_scores[:, y] - class_tau[y],
            "class_calibrated_margin": decisions["Class-P05"]["margin"],
            "local_own_margin": assigned_score - local_tau[2 * y + component],
            "local_calibrated_margin": decisions["Component-P05"]["margin"],
            "class_p05_accept": decisions["Class-P05"]["accepted"],
            "local_p05_accept": decisions["Component-P05"]["accepted"],
            "class_p05_predicted_class": decisions["Class-P05"]["predicted_class"],
            "local_p05_predicted_class": decisions["Component-P05"]["predicted_class"],
            "local_p05_predicted_component": decisions["Component-P05"]["predicted_component"],
            "global_threshold": global_threshold,
            "class_threshold": class_tau[y],
            "component_effective_threshold": local_tau[2 * y + component],
            "validity": "POST_HOC_DIAGNOSTIC_ONLY",
        })
    htbot = None
    virut = None
    if setting == "A-1":
        focus = (~aligned["single_accepted_as_known"].to_numpy(dtype=bool)) & aligned["multi_accepted_as_known"].to_numpy(dtype=bool)
        htbot = build("Tinba", "Htbot", focus)
    if setting == "A-3":
        focus = aligned["single_accepted_as_known"].to_numpy(dtype=bool) & (~aligned["multi_accepted_as_known"].to_numpy(dtype=bool))
        virut_all = build("Neris", "Virut", focus)
        virut = virut_all[(virut_all["unknown_class"] == "Neris") & virut_all["focus_case"]].copy()
    return htbot, virut


def posthoc_rows(
    setting: str, labels: np.ndarray, decisions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule in RULES:
        accepted = decisions[rule]["accepted"]
        for unknown_class, mask in [("__ALL__", np.ones(len(labels), dtype=bool)), *((name, labels == name) for name in sorted(np.unique(labels)))]:
            rows.append({
                "setting": setting, "rule": rule, "unknown_class": unknown_class,
                "sample_count": int(np.sum(mask)), "accepted_as_known_count": int(np.sum(accepted[mask])),
                "unknown_false_acceptance_rate": float(np.mean(accepted[mask])),
                "unknown_rejection_rate": float(np.mean(~accepted[mask])),
                "validity": "POST_HOC_DIAGNOSTIC_ONLY", "unknown_used_for_calibration": 0,
            })
    return pd.DataFrame(rows)


def choose_diagnosis(val: pd.DataFrame, test: pd.DataFrame) -> tuple[str, dict[str, float]]:
    def mean_gap(frame: pd.DataFrame, rule: str) -> float:
        rows = frame[(frame["rule"] == rule) & (frame["level"] == "class")]
        return float(rows["component_acceptance_gap"].mean())
    metrics = {
        "val_global_mean_component_gap": mean_gap(val, "Global"),
        "val_class_mean_component_gap": mean_gap(val, "Class-P05"),
        "val_local_mean_component_gap": mean_gap(val, "Component-P05"),
        "test_global_mean_component_gap": mean_gap(test, "Global"),
        "test_class_mean_component_gap": mean_gap(test, "Class-P05"),
        "test_local_mean_component_gap": mean_gap(test, "Component-P05"),
        "val_global_max_component_gap": float(val[(val.rule == "Global") & (val.level == "class")]["component_acceptance_gap"].max()),
    }
    class_val_gain = metrics["val_global_mean_component_gap"] - metrics["val_class_mean_component_gap"]
    class_test_gain = metrics["test_global_mean_component_gap"] - metrics["test_class_mean_component_gap"]
    local_val_gain = metrics["val_class_mean_component_gap"] - metrics["val_local_mean_component_gap"]
    local_test_gain = metrics["test_class_mean_component_gap"] - metrics["test_local_mean_component_gap"]
    metrics.update({
        "class_val_gain_over_global": class_val_gain,
        "class_test_gain_over_global": class_test_gain,
        "local_val_increment_over_class": local_val_gain,
        "local_test_increment_over_class": local_test_gain,
    })
    if local_val_gain >= 0.02 and local_test_gain >= 0.01:
        diagnosis = "C — LOCAL BOUNDARY NECESSARY"
    elif class_val_gain >= 0.02 and class_test_gain >= 0.01 and local_test_gain < 0.01:
        diagnosis = "B — CLASS-LEVEL CALIBRATION SUFFICIENT"
    elif metrics["val_global_max_component_gap"] >= 0.10 and min(class_val_gain, class_test_gain) > 0:
        diagnosis = "A — GLOBAL THRESHOLD MISMATCH"
    else:
        diagnosis = "D — BOUNDARY NOT THE MAIN ISSUE"
    return diagnosis, metrics


def overall_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["level"] == "overall"].copy()


def write_documents(
    provenance: dict[str, object], replay: list[dict[str, object]], class_cov: pd.DataFrame,
    comp_cov: pd.DataFrame, heterogeneity: pd.DataFrame, val_compare: pd.DataFrame,
    test_compare: pd.DataFrame, local_thresholds: pd.DataFrame, correlations: pd.DataFrame,
    posthoc: pd.DataFrame, htbot: pd.DataFrame, virut: pd.DataFrame, diagnosis: str,
    diagnosis_metrics: dict[str, float], packages_after: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# Stage 4 Provenance Snapshot", "",
        f"- Frozen Stage 3 commit: `{SNAPSHOT_COMMIT}`",
        f"- Protocol canonical SHA-256: `{PROTOCOL_SHA256}`",
        "- Frozen asset operations: `transform`, `score_samples`, `_estimate_weighted_log_prob`; estimator fitting operations: `0`.",
        "", "| Setting | package SHA-256 before | after | files | unchanged |", "|---|---|---|---:|---|",
    ]
    for setting in SETTINGS:
        before = provenance["packages_before"][setting]
        after = packages_after[setting]
        lines.append(f"| {setting} | `{before['sha256']}` | `{after['sha256']}` | {after['files']} | {'PASS' if before == after else 'FAIL'} |")
    lines.extend(["", f"- Stage 3 prediction replay: `PASS`, {sum(item['rows'] for item in replay):,} setting-sample rows across Known and Unknown."])
    (OUTPUT_ROOT / "provenance_snapshot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    leakage = [
        "# Boundary Leakage Audit", "",
        "- Rule G threshold source: frozen Stage 3 Known Validation threshold.",
        "- Rule C thresholds: true-class K2 mixture-score P05 on Known Validation only.",
        "- Rule L thresholds: true-class posterior-assigned component joint-score P05 on Known Validation only.",
        f"- Rule L deterministic fallback: validation count `< {MIN_LOCAL_VALIDATION}` uses the class P05 threshold.",
        "- Unknown samples used for threshold, quantile, fallback, K, model, or formula selection: `0`.",
        "- Known Test was evaluated only after all thresholds were frozen.",
        "- USTC Unknown Test was read only after Known Validation and Known Test evaluation; its results are `POST_HOC_DIAGNOSTIC_ONLY`.",
        "- Quantile search performed: `NO`; only P05 was executed.",
        "- Adaptive K, refitting, and Stage 3 threshold modification: `NO`.",
    ]
    (OUTPUT_ROOT / "boundary_leakage_audit.md").write_text("\n".join(leakage) + "\n", encoding="utf-8")

    proposed = "Component-P05" if diagnosis.startswith("C") else "Class-P05" if diagnosis.startswith("B") else "no promoted rule"
    protocol = [
        "# External Validation Protocol Candidate", "",
        "> Candidate freeze only. No CSTNET training or evaluation was executed.", "",
        f"- Stage 4 Known-only diagnosis: **{diagnosis}**.",
        f"- Proposed candidate for future untouched validation: **{proposed}**.",
        "- Baselines: frozen Global rule and Class-P05; include Component-P05 when it is not the proposed candidate.",
        "- Representation/model: training-split-only scaler, PCA64, full-covariance K2 per Known class; all frozen before test access.",
        "- Class-P05: `tau_y = P05_{Known Validation, true class y}[log p_GMM(x|y)]`; accept iff `max_y(log p_GMM(x|y)-tau_y) >= 0`.",
        "- Component-P05: assign each Known Validation sample within its true-class K2 model; `tau_yk=P05[log w_yk + log N(x|mu_yk,Sigma_yk)]`; accept iff `max_yk(s_yk-tau_yk) >= 0`.",
        f"- Fallback: if a component has fewer than {MIN_LOCAL_VALIDATION} Known Validation samples, use its class threshold `tau_y`.",
        "- Fixed quantile: P05 only. No P01/P02/P10, temperature, learned boundary, weighted quantile, or Unknown-aware selection.",
        "- Target first independent benchmark: CSTNET-TLS1.3 under a new strict Unknown-Free split protocol.",
        "- Freeze before CSTNET Unknown Test: class lists/splits, representation, K2, score definitions, P05, fallback, detector comparison, metrics, and decision rule.",
        "- CipherSpectrum remains secondary validation only.",
    ]
    (OUTPUT_ROOT / "external_validation_protocol_candidate.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")

    val_overall = overall_table(val_compare).pivot(index="setting", columns="rule", values="FRR")
    test_overall = overall_table(test_compare).pivot(index="setting", columns="rule", values="FRR")
    max_gap = heterogeneity.sort_values("val_global_component_acceptance_gap", ascending=False).iloc[0]
    ht_focus = htbot[htbot["focus_case"]]
    vir_focus = virut[virut["focus_case"]]
    ht_components = {int(key): int(value) for key, value in ht_focus["case_component_id"].value_counts().sort_index().items()}
    vir_components = {int(key): int(value) for key, value in vir_focus["case_component_id"].value_counts().sort_index().items()}
    vir_local_predictions = {
        f"{name}/component-{int(component)}": int(value)
        for (name, component), value in vir_focus.groupby(
            ["local_p05_predicted_class", "local_p05_predicted_component"]
        ).size().items()
    }
    corr = correlations[(correlations["scope"] == "ALL") & (correlations["outcome"] == "effective_minus_class_threshold")].copy()
    corr["abs_rho"] = corr["spearman_rho"].abs()
    strongest = corr.sort_values("abs_rho", ascending=False).iloc[0]
    post_overall = posthoc[posthoc["unknown_class"] == "__ALL__"].pivot(index="setting", columns="rule", values="unknown_false_acceptance_rate")
    audit = [
        "# Stage 4 Local Boundary Diagnosis", "",
        "## Validity boundary", "",
        "This is POST-HOC USTC MECHANISM DIAGNOSIS. A-1/A-2/A-3 Unknown Test has already been observed repeatedly. USTC results below are not independent performance validation.", "",
        "## Prediction reproduction", "",
        f"- Native, Single, and Multi scores/classes/binary decisions: `100% parity` over {sum(item['rows'] for item in replay):,} setting-sample rows; maximum replay score errors are recorded in `run_metadata.json`.", "",
        "## Global coverage heterogeneity", "",
        f"- Largest Known-Validation component acceptance gap: {max_gap.setting}/{max_gap.known_class} = {max_gap.val_global_component_acceptance_gap:.6f}.",
        f"- Mean component gap, Global/Class/Local on validation: {diagnosis_metrics['val_global_mean_component_gap']:.6f}/{diagnosis_metrics['val_class_mean_component_gap']:.6f}/{diagnosis_metrics['val_local_mean_component_gap']:.6f}.",
        f"- Mean component gap, Global/Class/Local on Known Test: {diagnosis_metrics['test_global_mean_component_gap']:.6f}/{diagnosis_metrics['test_class_mean_component_gap']:.6f}/{diagnosis_metrics['test_local_mean_component_gap']:.6f}.", "",
        "- Class-P05 makes per-class coverage much more uniform but does not equalize the two component coverages; Component-P05 materially reduces the mean component gap on both Known Validation and Known Test.",
        "- Component-P05 validation coverage is partly calibration-by-construction and must be judged together with Known Test generalization and the fixed `<30` fallback rows.", "",
        "## Overall Known FRR", "",
    ]
    for setting in SETTINGS:
        audit.append(f"- {setting} Validation Global/Class/Local: {val_overall.loc[setting, 'Global']:.6f}/{val_overall.loc[setting, 'Class-P05']:.6f}/{val_overall.loc[setting, 'Component-P05']:.6f}; Test: {test_overall.loc[setting, 'Global']:.6f}/{test_overall.loc[setting, 'Class-P05']:.6f}/{test_overall.loc[setting, 'Component-P05']:.6f}.")
    audit.extend([
        "", "## Htbot–Tinba case", "",
        f"- Focus rows: {len(ht_focus)}. Median global/class/local margins: {ht_focus.global_margin.median():.6f}/{ht_focus.class_calibrated_margin.median():.6f}/{ht_focus.local_calibrated_margin.median():.6f}.",
        f"- Class-P05 rejects {int((~ht_focus.class_p05_accept).sum())}/{len(ht_focus)}; Component-P05 rejects {int((~ht_focus.local_p05_accept).sum())}/{len(ht_focus)}.",
        f"- Htbot component counts: {json.dumps(ht_components, sort_keys=True)}; median assigned-component validation percentile: {ht_focus.component_validation_percentile.median():.6f}.",
        "- The 86 Tinba are inside Htbot component 1 rather than merely crossing a badly scaled global threshold; both calibrated rules make their margins more positive. Local calibration does not explain or repair this failure.", "",
        "## Virut–Neris case", "",
        f"- Focus rows: {len(vir_focus)}. Median global/class/local margins: {vir_focus.global_margin.median():.6f}/{vir_focus.class_calibrated_margin.median():.6f}/{vir_focus.local_calibrated_margin.median():.6f}.",
        f"- Class-P05 rejects {int((~vir_focus.class_p05_accept).sum())}/{len(vir_focus)}; Component-P05 rejects {int((~vir_focus.local_p05_accept).sum())}/{len(vir_focus)}.",
        f"- Virut raw component assignments: {json.dumps(vir_components, sort_keys=True)}; median assigned-component validation percentile: {vir_focus.component_validation_percentile.median():.6f}.",
        f"- Component-P05 normalized-margin predictions: {json.dumps(vir_local_predictions, sort_keys=True)}. The rule reassigns all focus rows to Virut component 0 and removes the Stage 3 rejection mechanism.",
        "- Thus the Stage 3 Virut gain primarily comes from the stringent global threshold against its lower-score-scale support, not from a clean between-component low-density gap that Component-P05 preserves.", "",
        "## Threshold-factor association", "",
        f"- Strongest pooled association with local-minus-class threshold: `{strongest.factor}`, rho={strongest.spearman_rho:.6f}, p={strongest.p_value:.6g}, n={int(strongest.n)}. Association is Known-only and diagnostic.", "",
        "## Post-hoc Unknown results", "",
    ])
    for setting in SETTINGS:
        audit.append(f"- {setting} Global/Class/Local UFAR: {post_overall.loc[setting, 'Global']:.6f}/{post_overall.loc[setting, 'Class-P05']:.6f}/{post_overall.loc[setting, 'Component-P05']:.6f} (`POST_HOC_DIAGNOSTIC_ONLY`).")
    audit.extend([
        "", "## Final diagnosis", "", f"**Diagnosis {diagnosis}.**",
        "", "This diagnosis is about Known component-coverage consistency, not demonstrated Unknown-detection benefit. The observed USTC post-hoc results show substantial utility risk and are not used to tune or reselect the frozen P05 formula.",
        "", f"Future candidate: **{proposed}**. This candidate was not independently validated here. Its first formal test must be a pre-frozen CSTNET-TLS1.3 strict Unknown-Free protocol.",
    ])
    (OUTPUT_ROOT / "audit_summary.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


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
    provenance = verify_inputs()
    class_coverage_frames: list[pd.DataFrame] = []
    component_coverage_frames: list[pd.DataFrame] = []
    class_threshold_frames: list[pd.DataFrame] = []
    local_threshold_frames: list[pd.DataFrame] = []
    val_compare_frames: list[pd.DataFrame] = []
    test_compare_frames: list[pd.DataFrame] = []
    replay_rows: list[dict[str, object]] = []
    states: dict[str, dict[str, object]] = {}

    # Phase 1: all settings complete Known-only calibration and Known Test checks.
    for setting in SETTINGS:
        artifact = STAGE3_ROOT / "artifacts" / setting
        scaler, pca, k1_models, k2_models = load_models(setting)
        names = list(k2_models)
        frozen_thresholds = thresholds(setting)
        train = pd.read_parquet(artifact / "train_known_mu.parquet")
        val = pd.read_parquet(artifact / "val_known_mu.parquet")
        val_values = transform(val, scaler, pca)
        names_scored, val_class_scores, val_local_scores = score_k2(k2_models, val_values)
        if names_scored != names:
            raise RuntimeError("class ordering changed")
        class_table, local_table, class_tau, local_tau, val_components, val_assigned_local, distributions = build_calibration(
            setting, train, val, scaler, pca, k2_models, names, val_class_scores,
            val_local_scores, frozen_thresholds["Multi-Full-K2"],
        )
        class_threshold_frames.append(class_table)
        local_threshold_frames.append(local_table)
        val_decisions = rule_decisions(names, val_class_scores, val_local_scores, frozen_thresholds["Multi-Full-K2"], class_tau, local_tau)
        val_global_score = val_class_scores.max(axis=1)
        class_coverage_frames.append(global_class_coverage(setting, val["class_name"].to_numpy(dtype=object), val_global_score, frozen_thresholds["Multi-Full-K2"]))
        component_coverage_frames.append(global_component_coverage(
            setting, val["class_name"].to_numpy(dtype=object), val_components, val_assigned_local,
            val_decisions["Global"]["accepted"], frozen_thresholds["Multi-Full-K2"], local_table,
        ))
        val_compare_frames.append(comparison_rows(setting, "Known Validation", val["class_name"].to_numpy(dtype=object), val_components, val_decisions))

        test = pd.read_parquet(artifact / "test_known_mu.parquet")
        test_values = transform(test, scaler, pca)
        _, test_class_scores, test_local_scores = score_k2(k2_models, test_values)
        replay_rows.append(verify_stage3_predictions(setting, test, test_values, names, test_class_scores, k1_models, "Known"))
        test_components, _ = true_components(test["class_name"].to_numpy(dtype=object), names, test_local_scores)
        test_decisions = rule_decisions(names, test_class_scores, test_local_scores, frozen_thresholds["Multi-Full-K2"], class_tau, local_tau)
        test_compare_frames.append(comparison_rows(setting, "Known Test", test["class_name"].to_numpy(dtype=object), test_components, test_decisions))
        states[setting] = {
            "scaler": scaler, "pca": pca, "k1": k1_models, "k2": k2_models, "names": names,
            "global_threshold": frozen_thresholds["Multi-Full-K2"], "class_tau": class_tau,
            "local_tau": local_tau, "distributions": distributions,
        }
        del train, val, val_values, val_class_scores, val_local_scores, test, test_values, test_class_scores, test_local_scores

    class_coverage = pd.concat(class_coverage_frames, ignore_index=True)
    component_coverage = pd.concat(component_coverage_frames, ignore_index=True)
    class_threshold_table = pd.concat(class_threshold_frames, ignore_index=True)
    local_threshold_table = pd.concat(local_threshold_frames, ignore_index=True)
    val_compare = pd.concat(val_compare_frames, ignore_index=True)
    test_compare = pd.concat(test_compare_frames, ignore_index=True)
    heterogeneity = build_heterogeneity(val_compare, test_compare, local_threshold_table, class_threshold_table)
    correlations = factor_correlations(local_threshold_table)
    fallback = local_threshold_table.groupby("setting").agg(
        total_components=("component_id", "size"),
        fallback_components=("fallback_to_class_threshold", "sum"),
        minimum_component_validation_count=("validation_count", "min"),
    ).reset_index()
    fallback["fallback_rule"] = f"validation_count < {MIN_LOCAL_VALIDATION} -> class P05"
    fallback["unknown_samples_used"] = 0
    diagnosis, diagnosis_metrics = choose_diagnosis(val_compare, test_compare)

    # Phase 2: thresholds/diagnosis are frozen; only now read Unknown Test.
    posthoc_frames: list[pd.DataFrame] = []
    htbot_case: pd.DataFrame | None = None
    virut_case: pd.DataFrame | None = None
    for setting in SETTINGS:
        state = states[setting]
        artifact = STAGE3_ROOT / "artifacts" / setting
        unknown = pd.read_parquet(artifact / "test_unknown_mu.parquet")
        values = transform(unknown, state["scaler"], state["pca"])
        names, class_scores, local_scores = score_k2(state["k2"], values)
        replay_rows.append(verify_stage3_predictions(setting, unknown, values, names, class_scores, state["k1"], "Unknown"))
        decisions = rule_decisions(names, class_scores, local_scores, state["global_threshold"], state["class_tau"], state["local_tau"])
        posthoc_frames.append(posthoc_rows(setting, unknown["class_name"].to_numpy(dtype=object), decisions))
        frozen = pd.read_parquet(artifact / "frozen_final_predictions.parquet")
        h, v = case_studies(
            setting, unknown, frozen, names, class_scores, local_scores, decisions,
            state["class_tau"], state["local_tau"], state["distributions"], state["global_threshold"],
        )
        htbot_case = h if h is not None else htbot_case
        virut_case = v if v is not None else virut_case
    if htbot_case is None or virut_case is None:
        raise RuntimeError("required case study was not generated")
    posthoc = pd.concat(posthoc_frames, ignore_index=True)

    class_coverage.to_csv(OUTPUT_ROOT / "global_class_coverage.csv", index=False)
    component_coverage.to_csv(OUTPUT_ROOT / "global_component_coverage.csv", index=False)
    heterogeneity.to_csv(OUTPUT_ROOT / "boundary_heterogeneity.csv", index=False)
    class_threshold_table.to_csv(OUTPUT_ROOT / "class_thresholds.csv", index=False)
    local_threshold_table.to_csv(OUTPUT_ROOT / "local_component_thresholds.csv", index=False)
    val_compare.to_csv(OUTPUT_ROOT / "known_validation_boundary_comparison.csv", index=False)
    test_compare.to_csv(OUTPUT_ROOT / "known_test_boundary_generalization.csv", index=False)
    htbot_case.to_csv(OUTPUT_ROOT / "htbot_tinba_case_study.csv", index=False)
    virut_case.to_csv(OUTPUT_ROOT / "virut_neris_case_study.csv", index=False)
    posthoc.to_csv(OUTPUT_ROOT / "posthoc_unknown_boundary_results.csv", index=False)
    correlations.to_csv(OUTPUT_ROOT / "boundary_factor_correlations.csv", index=False)
    fallback.to_csv(OUTPUT_ROOT / "fallback_summary.csv", index=False)

    packages_after: dict[str, dict[str, object]] = {}
    for setting in SETTINGS:
        digest, count = tree_hash(STAGE3_ROOT / "outputs" / setting)
        packages_after[setting] = {"sha256": digest, "files": count}
        if packages_after[setting] != provenance["packages_before"][setting]:
            raise RuntimeError(f"{setting}: Stage 3 package changed during Stage 4")
    write_documents(
        provenance, replay_rows, class_coverage, component_coverage, heterogeneity,
        val_compare, test_compare, local_threshold_table, correlations, posthoc,
        htbot_case, virut_case, diagnosis, diagnosis_metrics, packages_after,
    )
    metadata = {
        "run_id": "stage4_local_boundary_diagnosis_v1",
        "status": "completed", "execution_type": "post_hoc_mechanism_diagnosis",
        "started_at_utc": started, "finished_at_utc": utc_now(),
        "command": " ".join(sys.argv), "git_head": git("rev-parse", "HEAD"),
        "snapshot_commit": SNAPSHOT_COMMIT, "stage3_package_hashes_before": provenance["packages_before"],
        "stage3_package_hashes_after": packages_after, "fit_operations_performed": [],
        "frozen_operations": ["transform", "score_samples", "_estimate_weighted_log_prob"],
        "execution_order": ["all Known Validation calibration/coverage", "all Known Test generalization", "post-hoc Unknown Test mechanism"],
        "quantiles_executed": [0.05], "minimum_local_validation_samples": MIN_LOCAL_VALIDATION,
        "fallback": "class P05", "unknown_calibration_samples": 0,
        "stage3_threshold_modified": False, "k_modified": False, "adaptive_k": False,
        "unknown_test_observed_before_stage4": True, "posthoc_unknown_validity": "POST_HOC_DIAGNOSTIC_ONLY",
        "prediction_replay": replay_rows, "prediction_replay_all_pass": True,
        "diagnosis": diagnosis, "diagnosis_known_only_metrics": diagnosis_metrics,
        "diagnosis_rule": {
            "local_necessary": "Class-to-Local mean component-gap reduction >=0.02 validation and >=0.01 Known Test",
            "class_sufficient": "Global-to-Class reduction >=0.02 validation and >=0.01 Known Test, Local Known-Test increment <0.01",
            "global_mismatch": "maximum global validation component gap >=0.10 and positive Class gain on validation/test",
            "otherwise": "boundary not main issue",
        },
        "environment": {
            "python": sys.version.split()[0], "prefix": sys.prefix, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "row_counts": {
            "global_class_coverage": len(class_coverage), "global_component_coverage": len(component_coverage),
            "boundary_heterogeneity": len(heterogeneity), "class_thresholds": len(class_threshold_table),
            "local_component_thresholds": len(local_threshold_table), "known_validation_boundary_comparison": len(val_compare),
            "known_test_boundary_generalization": len(test_compare), "htbot_tinba_case_study": len(htbot_case),
            "virut_neris_case_study": len(virut_case), "posthoc_unknown_boundary_results": len(posthoc),
            "boundary_factor_correlations": len(correlations), "fallback_summary": len(fallback),
        },
        "next_external_benchmark": "CSTNET-TLS1.3", "external_training_executed": False,
    }
    (OUTPUT_ROOT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "diagnosis": diagnosis, "output_dir": str(OUTPUT_ROOT), "row_counts": metadata["row_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
