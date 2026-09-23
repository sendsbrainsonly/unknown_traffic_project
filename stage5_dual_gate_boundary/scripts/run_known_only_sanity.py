#!/usr/bin/env python3
"""Freeze and sanity-check DGSB using frozen Stage 3 Known assets only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


STAGE5_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE5_ROOT.parent
STAGE3_ROOT = PROJECT_ROOT / "stage3_unknown_utility"
STAGE4_ROOT = PROJECT_ROOT / "stage4_local_boundary_diagnosis"
CONFIG_PATH = STAGE5_ROOT / "configs" / "stage5_freeze_config.json"
SETTINGS = ("A-1", "A-2", "A-3")
BASELINE_RULES = ("Global", "Class-P05", "Component-P05")
ALL_RULES = (*BASELINE_RULES, "DGSB")
BLOCKED_INPUT_TOKENS = (
    "test_unknown",
    "frozen_final_predictions",
    "posthoc_unknown",
    "unknown_boundary_results",
    "transition",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous != content:
            raise RuntimeError(f"refusing to modify frozen file: {path}")
        return
    path.write_text(content, encoding="utf-8")


class InputLedger:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def add(self, path: Path) -> Path:
        resolved = path.resolve()
        lowered = resolved.as_posix().lower()
        if any(token in lowered for token in BLOCKED_INPUT_TOKENS):
            raise RuntimeError(f"blocked Stage 5 input path: {resolved}")
        self.paths.append(resolved)
        return resolved

    def unique(self) -> list[Path]:
        return sorted(set(self.paths))


def read_json(path: Path, ledger: InputLedger) -> object:
    return json.loads(ledger.add(path).read_text(encoding="utf-8"))


def read_csv(path: Path, ledger: InputLedger) -> pd.DataFrame:
    return pd.read_csv(ledger.add(path))


def read_parquet(path: Path, ledger: InputLedger) -> pd.DataFrame:
    return pd.read_parquet(ledger.add(path))


def load_joblib(path: Path, ledger: InputLedger) -> object:
    return joblib.load(ledger.add(path))


def latent_columns(columns: Iterable[str]) -> list[str]:
    result = sorted(name for name in columns if name.startswith("mu_"))
    if len(result) != 128:
        raise RuntimeError(f"expected 128 frozen mu_x columns, got {len(result)}")
    return result


def transform(frame: pd.DataFrame, scaler: object, pca: object) -> np.ndarray:
    values = frame[latent_columns(frame.columns)].to_numpy(dtype=np.float64)
    return pca.transform(scaler.transform(values))


def score_k2(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(models)
    local_blocks: list[np.ndarray] = []
    class_blocks: list[np.ndarray] = []
    for name in names:
        local = np.asarray(models[name]._estimate_weighted_log_prob(values), dtype=np.float64)
        if local.shape != (len(values), 2):
            raise RuntimeError(f"{name}: unexpected local score shape {local.shape}")
        local_blocks.append(local)
        class_blocks.append(logsumexp(local, axis=1))
    return names, np.column_stack(class_blocks), np.column_stack(local_blocks)


def true_component_assignments(
    labels: np.ndarray, names: list[str], local_scores: np.ndarray
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    components = np.empty(len(labels), dtype=np.int64)
    for name in np.unique(labels):
        positions = np.flatnonzero(labels == name)
        index = lookup[str(name)]
        block = local_scores[positions, 2 * index : 2 * index + 2]
        components[positions] = np.argmax(block, axis=1)
    return components


def build_known_calibration(
    setting: str,
    labels: np.ndarray,
    names: list[str],
    class_scores: np.ndarray,
    local_scores: np.ndarray,
    quantile: float,
    minimum_local_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    lookup = {name: index for index, name in enumerate(names)}
    components = true_component_assignments(labels, names, local_scores)
    class_thresholds = np.empty(len(names), dtype=np.float64)
    local_thresholds = np.empty(2 * len(names), dtype=np.float64)
    class_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    for name in names:
        index = lookup[name]
        positions = np.flatnonzero(labels == name)
        own_class = class_scores[positions, index]
        class_tau = float(np.quantile(own_class, quantile, method="linear"))
        class_thresholds[index] = class_tau
        class_rows.append(
            {
                "setting": setting,
                "known_class": name,
                "validation_samples": len(positions),
                "quantile": quantile,
                "class_threshold": class_tau,
                "unknown_samples_used": 0,
            }
        )
        for component in range(2):
            selected = positions[components[positions] == component]
            scores = local_scores[selected, 2 * index + component]
            if not len(scores):
                raise RuntimeError(f"{setting}/{name}/component-{component}: no validation samples")
            raw_tau = float(np.quantile(scores, quantile, method="linear"))
            fallback = len(scores) < minimum_local_samples
            effective_tau = class_tau if fallback else raw_tau
            local_thresholds[2 * index + component] = effective_tau
            local_rows.append(
                {
                    "setting": setting,
                    "known_class": name,
                    "component_id": component,
                    "validation_samples": len(scores),
                    "quantile": quantile,
                    "raw_local_threshold": raw_tau,
                    "effective_local_threshold": effective_tau,
                    "fallback_to_class_p05": fallback,
                    "unknown_samples_used": 0,
                }
            )
    return class_thresholds, local_thresholds, components, class_rows, local_rows


def calibrate_dual_global_threshold(
    global_scores: np.ndarray,
    local_gate_pass: np.ndarray,
    target_acceptance: float,
) -> tuple[float, float, int]:
    """Choose an empirical finite threshold closest to the requested final rate.

    Candidate thresholds are the distinct G scores among Local-Gate passing
    Known Validation samples. Ties in absolute target error use the larger,
    more conservative threshold.
    """
    scores = np.asarray(global_scores, dtype=np.float64)
    local_pass = np.asarray(local_gate_pass, dtype=bool)
    eligible = np.sort(scores[local_pass])
    if not len(eligible) or not np.isfinite(eligible).all():
        raise RuntimeError("DGSB global calibration has no finite Local-Gate passing scores")
    thresholds, first_positions = np.unique(eligible, return_index=True)
    accepted_counts = len(eligible) - first_positions
    rates = accepted_counts / len(scores)
    errors = np.abs(rates - target_acceptance)
    best_error = float(errors.min())
    candidates = np.flatnonzero(np.isclose(errors, best_error, rtol=0, atol=1e-15))
    selected = int(candidates[-1])
    return float(thresholds[selected]), float(rates[selected]), int(accepted_counts[selected])


def decisions(
    names: list[str],
    class_scores: np.ndarray,
    local_scores: np.ndarray,
    old_global_threshold: float,
    class_thresholds: np.ndarray,
    local_thresholds: np.ndarray,
    dual_global_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    global_score = class_scores.max(axis=1)
    local_winner = np.argmax(local_scores, axis=1)
    winning_local_score = local_scores[np.arange(len(local_scores)), local_winner]
    winning_local_pass = winning_local_score >= local_thresholds[local_winner]
    result = {
        "Global": global_score >= old_global_threshold,
        "Class-P05": np.max(class_scores - class_thresholds[None, :], axis=1) >= 0,
        "Component-P05": np.max(local_scores - local_thresholds[None, :], axis=1) >= 0,
        "DGSB": (global_score >= dual_global_threshold) & winning_local_pass,
    }
    return result, local_winner, winning_local_pass


def coverage_rows(
    setting: str,
    split: str,
    labels: np.ndarray,
    components: np.ndarray,
    rule_acceptance: dict[str, np.ndarray],
    dual_global_threshold: float,
    local_gate_pass: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    classes = sorted(np.unique(labels))
    for rule in ALL_RULES:
        accepted = np.asarray(rule_acceptance[rule], dtype=bool)
        class_rates: dict[str, float] = {}
        component_rates: dict[tuple[str, int], float] = {}
        component_gaps: dict[str, float] = {}
        for name in classes:
            class_mask = labels == name
            class_rates[name] = float(np.mean(accepted[class_mask]))
            rates: list[float] = []
            for component in range(2):
                mask = class_mask & (components == component)
                if not np.any(mask):
                    raise RuntimeError(f"{setting}/{split}/{name}/{component}: empty component")
                rate = float(np.mean(accepted[mask]))
                component_rates[(name, component)] = rate
                rates.append(rate)
            component_gaps[name] = max(rates) - min(rates)
        rows.append(
            {
                "setting": setting,
                "split": split,
                "rule": rule,
                "level": "overall",
                "known_class": "__ALL__",
                "component_id": -1,
                "sample_count": len(labels),
                "accepted_count": int(accepted.sum()),
                "acceptance_rate": float(accepted.mean()),
                "FRR": float((~accepted).mean()),
                "per_class_acceptance_min": min(class_rates.values()),
                "per_class_acceptance_max": max(class_rates.values()),
                "per_class_acceptance_range": max(class_rates.values()) - min(class_rates.values()),
                "mean_component_acceptance_gap": float(np.mean(list(component_gaps.values()))),
                "max_component_acceptance_gap": max(component_gaps.values()),
                "dual_global_threshold": dual_global_threshold if rule == "DGSB" else np.nan,
                "local_gate_acceptance_rate": float(local_gate_pass.mean()) if rule == "DGSB" else np.nan,
            }
        )
        for name in classes:
            mask = labels == name
            rows.append(
                {
                    "setting": setting,
                    "split": split,
                    "rule": rule,
                    "level": "class",
                    "known_class": name,
                    "component_id": -1,
                    "sample_count": int(mask.sum()),
                    "accepted_count": int(accepted[mask].sum()),
                    "acceptance_rate": class_rates[name],
                    "FRR": 1.0 - class_rates[name],
                    "per_class_acceptance_min": np.nan,
                    "per_class_acceptance_max": np.nan,
                    "per_class_acceptance_range": np.nan,
                    "mean_component_acceptance_gap": component_gaps[name],
                    "max_component_acceptance_gap": component_gaps[name],
                    "dual_global_threshold": np.nan,
                    "local_gate_acceptance_rate": np.nan,
                }
            )
            for component in range(2):
                component_mask = mask & (components == component)
                rate = component_rates[(name, component)]
                rows.append(
                    {
                        "setting": setting,
                        "split": split,
                        "rule": rule,
                        "level": "component",
                        "known_class": name,
                        "component_id": component,
                        "sample_count": int(component_mask.sum()),
                        "accepted_count": int(accepted[component_mask].sum()),
                        "acceptance_rate": rate,
                        "FRR": 1.0 - rate,
                        "per_class_acceptance_min": np.nan,
                        "per_class_acceptance_max": np.nan,
                        "per_class_acceptance_range": np.nan,
                        "mean_component_acceptance_gap": np.nan,
                        "max_component_acceptance_gap": np.nan,
                        "dual_global_threshold": np.nan,
                        "local_gate_acceptance_rate": np.nan,
                    }
                )
    return rows


def validate_stage4_baselines(current: pd.DataFrame, split: str, ledger: InputLedger) -> None:
    filename = (
        "known_validation_boundary_comparison.csv"
        if split == "Known Validation"
        else "known_test_boundary_generalization.csv"
    )
    previous = read_csv(STAGE4_ROOT / "outputs" / filename, ledger)
    current = current[current["rule"].isin(BASELINE_RULES)].copy()
    keys = ["setting", "split", "rule", "level", "known_class", "component_id"]
    previous["component_id"] = previous["component_id"].fillna(-1).astype(int)
    current["component_id"] = current["component_id"].astype(int)
    merged = current.merge(
        previous[keys + ["sample_count", "acceptance_rate", "FRR"]],
        on=keys,
        how="outer",
        suffixes=("_stage5", "_stage4"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"Stage 4 baseline row parity failed for {split}")
    for column in ("sample_count", "acceptance_rate", "FRR"):
        left = merged[f"{column}_stage5"].to_numpy()
        right = merged[f"{column}_stage4"].to_numpy()
        if not np.allclose(left, right, rtol=0, atol=1e-12):
            raise RuntimeError(f"Stage 4 baseline metric parity failed: {split}/{column}")


def expected_hashes(setting: str, ledger: InputLedger) -> dict[str, str]:
    output_root = STAGE3_ROOT / "outputs" / setting
    frozen = read_json(output_root / "frozen_model_hashes.json", ledger)
    result = {Path(item["path"]).name: item["sha256"] for item in frozen["files"]}
    latent = read_json(output_root / "latent_export_manifest.json", ledger)
    result["test_known_mu.parquet"] = latent["files"]["test_known_mu.parquet"]["sha256"]
    return result


def load_old_global_threshold(setting: str, ledger: InputLedger) -> float:
    frame = read_csv(STAGE3_ROOT / "outputs" / setting / "threshold_calibration.csv", ledger)
    if int(frame["unknown_samples_used"].sum()) != 0:
        raise RuntimeError(f"{setting}: Stage 3 threshold used Unknown samples")
    row = frame[frame["detector"] == "Multi-Full-K2"]
    if len(row) != 1 or row.iloc[0]["calibration_split"] != "Known Validation only":
        raise RuntimeError(f"{setting}: invalid frozen Multi-Full-K2 threshold provenance")
    return float(row.iloc[0]["threshold"])


def assert_stage4_threshold_parity(
    setting: str,
    class_rows: list[dict[str, object]],
    local_rows: list[dict[str, object]],
    ledger: InputLedger,
) -> None:
    expected_class = read_csv(STAGE4_ROOT / "outputs" / "class_thresholds.csv", ledger)
    expected_local = read_csv(STAGE4_ROOT / "outputs" / "local_component_thresholds.csv", ledger)
    expected_class = expected_class[expected_class["setting"] == setting].sort_values("known_class")
    expected_local = expected_local[expected_local["setting"] == setting].sort_values(["known_class", "component_id"])
    actual_class = pd.DataFrame(class_rows).sort_values("known_class")
    actual_local = pd.DataFrame(local_rows).sort_values(["known_class", "component_id"])
    if not np.array_equal(actual_class["known_class"].to_numpy(), expected_class["known_class"].to_numpy()):
        raise RuntimeError(f"{setting}: class threshold label parity failed")
    if not np.allclose(actual_class["class_threshold"], expected_class["class_threshold"], rtol=0, atol=1e-12):
        raise RuntimeError(f"{setting}: class threshold value parity failed")
    if not np.array_equal(
        actual_local[["known_class", "component_id"]].to_numpy(),
        expected_local[["known_class", "component_id"]].to_numpy(),
    ):
        raise RuntimeError(f"{setting}: local threshold label parity failed")
    if not np.allclose(
        actual_local["effective_local_threshold"],
        expected_local["effective_local_threshold"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{setting}: local threshold value parity failed")


def run(output_root: Path) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    method = config["method"]
    rule_root = output_root / "rule_freeze"
    rule_root.mkdir(parents=True, exist_ok=True)
    ledger = InputLedger()
    started = utc_now()
    all_rows: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    all_local_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    setting_freezes: dict[str, object] = {}
    verified_hashes: list[dict[str, object]] = []

    for setting in SETTINGS:
        artifact_root = STAGE3_ROOT / "artifacts" / setting
        expected = expected_hashes(setting, ledger)
        input_names = (
            "standard_scaler.joblib",
            "pca64.joblib",
            "multi_full_k2_models.joblib",
            "val_known_mu.parquet",
            "test_known_mu.parquet",
        )
        for filename in input_names:
            path = artifact_root / filename
            actual = sha256_file(ledger.add(path))
            if filename not in expected or actual != expected[filename]:
                raise RuntimeError(f"{setting}: frozen Known asset hash mismatch: {filename}")
            verified_hashes.append({"setting": setting, "file": str(path.resolve()), "sha256": actual})

        scaler = load_joblib(artifact_root / "standard_scaler.joblib", ledger)
        pca = load_joblib(artifact_root / "pca64.joblib", ledger)
        models = load_joblib(artifact_root / "multi_full_k2_models.joblib", ledger)
        if int(pca.n_components_) != method["pca_dimension"]:
            raise RuntimeError(f"{setting}: PCA dimension changed")
        for name, model in models.items():
            if (
                model.n_components != method["k"]
                or model.covariance_type != method["covariance_type"]
                or not np.isclose(model.reg_covar, method["reg_covar"])
                or not model.converged_
            ):
                raise RuntimeError(f"{setting}/{name}: frozen K2 model contract changed")

        val = read_parquet(artifact_root / "val_known_mu.parquet", ledger)
        test = read_parquet(artifact_root / "test_known_mu.parquet", ledger)
        for split, frame in (("Known Validation", val), ("Known Test", test)):
            if set(frame["known_or_unknown"]) != {"Known"}:
                raise RuntimeError(f"{setting}/{split}: non-Known row encountered")

        val_values = transform(val, scaler, pca)
        names, val_class_scores, val_local_scores = score_k2(models, val_values)
        val_labels = val["class_name"].to_numpy(dtype=object)
        if set(val_labels) != set(names):
            raise RuntimeError(f"{setting}: validation/model class mismatch")
        class_tau, local_tau, val_components, class_rows, local_rows = build_known_calibration(
            setting,
            val_labels,
            names,
            val_class_scores,
            val_local_scores,
            float(method["local_quantile"]),
            int(method["minimum_local_validation_samples"]),
        )
        assert_stage4_threshold_parity(setting, class_rows, local_rows, ledger)
        all_class_rows.extend(class_rows)
        all_local_rows.extend(local_rows)
        old_global = load_old_global_threshold(setting, ledger)

        raw_winner = np.argmax(val_local_scores, axis=1)
        val_local_gate = (
            val_local_scores[np.arange(len(val)), raw_winner] >= local_tau[raw_winner]
        )
        val_global_scores = val_class_scores.max(axis=1)
        dual_tau, achieved, accepted_count = calibrate_dual_global_threshold(
            val_global_scores,
            val_local_gate,
            float(method["global_target_known_validation_acceptance"]),
        )
        val_decisions, _, val_local_gate_check = decisions(
            names,
            val_class_scores,
            val_local_scores,
            old_global,
            class_tau,
            local_tau,
            dual_tau,
        )
        if not np.array_equal(val_local_gate, val_local_gate_check):
            raise RuntimeError(f"{setting}: Local-Gate replay mismatch")
        all_rows.extend(
            coverage_rows(
                setting,
                "Known Validation",
                val_labels,
                val_components,
                val_decisions,
                dual_tau,
                val_local_gate,
            )
        )

        test_values = transform(test, scaler, pca)
        test_names, test_class_scores, test_local_scores = score_k2(models, test_values)
        if test_names != names:
            raise RuntimeError(f"{setting}: model ordering changed during Known Test")
        test_labels = test["class_name"].to_numpy(dtype=object)
        test_components = true_component_assignments(test_labels, names, test_local_scores)
        test_decisions, _, test_local_gate = decisions(
            names,
            test_class_scores,
            test_local_scores,
            old_global,
            class_tau,
            local_tau,
            dual_tau,
        )
        all_rows.extend(
            coverage_rows(
                setting,
                "Known Test",
                test_labels,
                test_components,
                test_decisions,
                dual_tau,
                test_local_gate,
            )
        )
        calibration_rows.append(
            {
                "setting": setting,
                "calibration_split": "Known Validation only",
                "validation_samples": len(val),
                "target_acceptance": method["global_target_known_validation_acceptance"],
                "local_gate_accepted_count": int(val_local_gate.sum()),
                "local_gate_acceptance_rate": float(val_local_gate.mean()),
                "tau_global_dual": dual_tau,
                "dual_accepted_count": accepted_count,
                "dual_acceptance_rate": achieved,
                "absolute_target_error": abs(
                    achieved - float(method["global_target_known_validation_acceptance"])
                ),
                "unknown_samples_used": 0,
            }
        )
        setting_freezes[setting] = {
            "tau_global_dual": dual_tau,
            "target_known_validation_acceptance": method[
                "global_target_known_validation_acceptance"
            ],
            "actual_known_validation_acceptance": achieved,
            "class_thresholds": {
                row["known_class"]: row["class_threshold"] for row in class_rows
            },
            "component_thresholds": [
                {
                    "class": row["known_class"],
                    "component": row["component_id"],
                    "threshold": row["effective_local_threshold"],
                    "validation_samples": row["validation_samples"],
                    "fallback_to_class_p05": row["fallback_to_class_p05"],
                }
                for row in local_rows
            ],
        }

    sanity = pd.DataFrame(all_rows)
    validate_stage4_baselines(sanity[sanity["split"] == "Known Validation"], "Known Validation", ledger)
    validate_stage4_baselines(sanity[sanity["split"] == "Known Test"], "Known Test", ledger)
    calibration = pd.DataFrame(calibration_rows)
    tolerance = float(method["known_validation_target_tolerance"])
    if (calibration["absolute_target_error"] > tolerance).any():
        raise RuntimeError("DGSB Known Validation acceptance is outside the frozen tolerance")
    overall = sanity[sanity["level"] == "overall"]
    if (overall["acceptance_rate"] < float(method["minimum_noncollapse_acceptance"])).any():
        raise RuntimeError("Known-only sanity detected a frozen non-collapse violation")

    write_frozen(rule_root / "known_only_sanity.csv", sanity.to_csv(index=False, lineterminator="\n"))
    write_frozen(rule_root / "dgsb_calibration.csv", calibration.to_csv(index=False, lineterminator="\n"))
    write_frozen(
        rule_root / "class_thresholds.csv",
        pd.DataFrame(all_class_rows).to_csv(index=False, lineterminator="\n"),
    )
    write_frozen(
        rule_root / "local_component_thresholds.csv",
        pd.DataFrame(all_local_rows).to_csv(index=False, lineterminator="\n"),
    )

    rule = {
        "method_name": method["name"],
        "short_name": method["short_name"],
        "created_before_cstnet_unknown_evaluation": True,
        "claim_scope": "METHOD_FREEZE_AND_USTC_KNOWN_ONLY_SANITY",
        "model": {
            "K": method["k"],
            "covariance_type": method["covariance_type"],
            "reg_covar": method["reg_covar"],
            "pca_dimension": method["pca_dimension"],
            "refit_performed": False,
        },
        "scores": {
            "component_joint": "s_yk(z) = log(w_yk) + log N(z | mu_yk, Sigma_yk)",
            "class_density": "log p(z|y) = logsumexp_k s_yk(z)",
            "global_absolute": "G(z) = max_y log p(z|y)",
            "winning_local_component": "(y*,k*) = argmax_y,k s_yk(z)",
            "winning_local": "L(z) = s_y*k*(z)",
        },
        "local_gate": {
            "assignment": "true-class Known Validation samples assigned by posterior under frozen train-fitted K2 GMM",
            "quantile": method["local_quantile"],
            "threshold": "tau_local_yk = empirical P05 of assigned Known Validation component joint scores",
            "minimum_validation_samples": method["minimum_local_validation_samples"],
            "fallback": "if n_yk < 30, tau_local_yk = class-level Known Validation P05 tau_y",
            "decision": "L(z) >= tau_local_y*k* for the raw joint-score winning component",
        },
        "global_gate": {
            "calibration_data": "Known Validation only after Local Gate is frozen",
            "target_final_acceptance": method["global_target_known_validation_acceptance"],
            "candidate_thresholds": "distinct finite G values among Local-Gate passing Known Validation samples",
            "selection": "minimize absolute final acceptance error; ties choose the highest threshold",
            "decision": "G(z) >= tau_global_dual",
        },
        "final_decision": "accept Known iff Global Gate AND winning Local Component Gate",
        "class_prediction": "predict y* from argmax_y,k s_yk(z)",
        "forbidden_fusion": ["lambda", "alpha", "beta", "temperature", "learned fusion", "MLP", "SVM"],
        "unknown_samples_used_for_calibration": 0,
        "ustc_unknown_inputs_accessed": 0,
        "ustc_known_only_setting_thresholds": setting_freezes,
    }
    rule_text = stable_json(rule)
    write_frozen(rule_root / "dgsb_rule.json", rule_text)
    rule_sha = sha256_file(rule_root / "dgsb_rule.json")
    write_frozen(rule_root / "dgsb_rule.sha256", f"{rule_sha}  dgsb_rule.json\n")

    markdown = f"""# Frozen Dual-Gate Support Boundary (DGSB)

## Scope

This is a method freeze plus USTC Known-only sanity check. It contains no USTC
Unknown calibration or evaluation and no CSTNET training or inference.

## Frozen Representation and Density Model

- PCA dimension: `{method['pca_dimension']}`
- Per-class GMM: `K={method['k']}`, covariance=`{method['covariance_type']}`,
  `reg_covar={method['reg_covar']}`
- Estimators are frozen Stage 3 assets; no fit operation is performed.

## Scores and Gates

`s_yk(z) = log(w_yk) + log N(z | mu_yk, Sigma_yk)`

`G(z) = max_y logsumexp_k s_yk(z)`

`(y*, k*) = argmax_y,k s_yk(z)` and `L(z) = s_y*k*(z)`.

The Local Gate is `L(z) >= tau_local_y*k*`. Each local threshold is the fixed
P05 of Known Validation samples assigned to that component under the true-class
frozen GMM. If the assigned validation count is below 30, the class P05 is used.

After the Local Gate is frozen, `tau_global_dual` is selected only on Known
Validation from the distinct finite `G` values of Local-Gate passing samples to
make final AND acceptance closest to 95%; ties choose the higher threshold.

Final decision:

`Known iff G(z) >= tau_global_dual AND L(z) >= tau_local_y*k*`.

The predicted class is `y*`, the class of the raw maximum component joint score.

## Freeze Boundary

- `created_before_cstnet_unknown_evaluation = true`
- Unknown samples used or accessed: `0`
- Rule JSON SHA-256: `{rule_sha}`
"""
    write_frozen(rule_root / "dgsb_rule.md", markdown)

    input_paths = ledger.unique()
    audit_lines = [
        "# Unknown Access Audit",
        "",
        "- USTC Unknown data access during Stage 5 rule calibration: `0`.",
        "- USTC Unknown flow, label, score, UFAR, transition, and prediction files read: `0`.",
        "- Calibration source: Known Validation only.",
        "- Evaluation source: Known Validation and Known Test only.",
        "- Fit operations: `0`.",
        "",
        "## Explicit Input Ledger",
        "",
    ]
    audit_lines.extend(f"- `{path}`" for path in input_paths)
    audit_lines.extend(
        [
            "",
            "Every executable input path is checked against blocked Unknown-result path tokens before reading.",
        ]
    )
    write_frozen(rule_root / "unknown_access_audit.md", "\n".join(audit_lines) + "\n")

    metadata = {
        "run_id": "stage5_dgsb_known_only_sanity_v1",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "status": "completed",
        "claim_scope": "METHOD_FREEZE_AND_USTC_KNOWN_ONLY_SANITY",
        "environment": {
            "python": platform.python_version(),
            "prefix": sys.prefix,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "fit_operations_performed": [],
        "accessed_input_files": [str(path) for path in input_paths],
        "verified_known_asset_hashes": verified_hashes,
        "ustc_unknown_inputs_accessed": 0,
        "cstnet_training_executed": False,
        "cstnet_unknown_scores_accessed": False,
        "rule_sha256": rule_sha,
        "stage4_known_baseline_replay": "PASS",
        "calibration": calibration_rows,
    }
    write_frozen(rule_root / "known_only_run_metadata.json", stable_json(metadata))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=STAGE5_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run(args.output_root.resolve())
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "rule_sha256": metadata["rule_sha256"],
                "ustc_unknown_inputs_accessed": metadata["ustc_unknown_inputs_accessed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
