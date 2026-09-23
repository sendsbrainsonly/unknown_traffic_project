#!/usr/bin/env python3
"""Evaluate all six frozen methods for one one-shot CipherSpectrum setting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

from stage9_common import (
    EVALUATION_CONFIG_PATH,
    METHOD_ORDER,
    STAGE8A_ROOT,
    STAGE8B_ROOT,
    STAGE9_ROOT,
    VALID_SETTINGS,
    assert_test_open_record,
    load_fold,
    load_frozen_models,
    load_json,
    read_csv,
    require,
    score_density_matrix,
    score_k2_matrices,
    verify_all_provenance,
    write_csv,
    write_json,
)


def json_threshold(path: Path) -> float:
    return float(load_json(path)["threshold"])


def threshold_arrays(setting: str, class_names: list[str]) -> dict[str, np.ndarray | float]:
    stage8a = STAGE8A_ROOT / "artifacts" / setting
    stage8b = STAGE8B_ROOT / "artifacts" / setting
    class_rows = sorted(read_csv(stage8a / "class_p05_thresholds.csv"), key=lambda row: int(row["class_index"]))
    component_rows = sorted(read_csv(stage8a / "component_p05_thresholds.csv"), key=lambda row: (int(row["class_index"]), int(row["component_id"])))
    dgsb_rows = sorted(read_csv(stage8b / "dgsbv2_local_thresholds.csv"), key=lambda row: (int(row["class_index"]), int(row["component_id"])))
    require([row["class_name"] for row in class_rows] == class_names, f"{setting}: class threshold order mismatch")
    require([row["class_name"] for row in component_rows[::2]] == class_names, f"{setting}: component threshold order mismatch")
    require([row["class_name"] for row in dgsb_rows[::2]] == class_names, f"{setting}: DGSB threshold order mismatch")
    return {
        "native": json_threshold(stage8a / "native_threshold.json"),
        "single": json_threshold(stage8a / "single_global_threshold.json"),
        "multi": json_threshold(stage8a / "multi_global_threshold.json"),
        "class": np.asarray([float(row["threshold"]) for row in class_rows]),
        "component": np.asarray([float(row["effective_threshold"]) for row in component_rows]),
        "dgsb_global": json_threshold(stage8b / "dgsbv2_global_threshold.json"),
        "dgsb_local": np.asarray([float(row["effective_threshold"]) for row in dgsb_rows]),
    }


def evaluate_role(
    z: np.ndarray,
    native_scores: np.ndarray,
    native_predictions: np.ndarray,
    class_names: list[str],
    models: dict[str, object],
    thresholds: dict[str, np.ndarray | float],
) -> dict[str, dict[str, np.ndarray]]:
    names1, k1_scores = score_density_matrix(models["k1"], z)
    names2, k2_scores, local_scores = score_k2_matrices(models["k2"], z)
    require(names1 == class_names and names2 == class_names, "density model class order mismatch")
    names = np.asarray(class_names, dtype=object)
    n = len(z)

    native_threshold = float(thresholds["native"])
    native_margin = np.asarray(native_scores, dtype=np.float64) - native_threshold
    single_idx = np.argmax(k1_scores, axis=1)
    single_raw = k1_scores[np.arange(n), single_idx]
    single_threshold = float(thresholds["single"])
    single_margin = single_raw - single_threshold
    multi_idx = np.argmax(k2_scores, axis=1)
    multi_raw = k2_scores[np.arange(n), multi_idx]
    multi_threshold = float(thresholds["multi"])
    multi_margin = multi_raw - multi_threshold

    class_tau = np.asarray(thresholds["class"], dtype=np.float64)
    class_margin_matrix = k2_scores - class_tau[None, :]
    class_idx = np.argmax(class_margin_matrix, axis=1)
    class_margin = class_margin_matrix[np.arange(n), class_idx]
    class_raw = k2_scores[np.arange(n), class_idx]
    class_selected_tau = class_tau[class_idx]

    component_tau = np.asarray(thresholds["component"], dtype=np.float64)
    component_margin_matrix = local_scores - component_tau[None, :]
    component_flat_idx = np.argmax(component_margin_matrix, axis=1)
    component_class_idx = component_flat_idx // 2
    component_ids = component_flat_idx % 2
    component_margin = component_margin_matrix[np.arange(n), component_flat_idx]
    component_raw = local_scores[np.arange(n), component_flat_idx]
    component_selected_tau = component_tau[component_flat_idx]

    dgsb_global_tau = float(thresholds["dgsb_global"])
    dgsb_global_score = multi_raw
    dgsb_global_margin = dgsb_global_score - dgsb_global_tau
    same_class_blocks = local_scores.reshape(n, len(class_names), 2)[np.arange(n), multi_idx]
    dgsb_components = np.argmax(same_class_blocks, axis=1)
    dgsb_local_score = same_class_blocks[np.arange(n), dgsb_components]
    dgsb_flat_idx = 2 * multi_idx + dgsb_components
    dgsb_local_tau = np.asarray(thresholds["dgsb_local"], dtype=np.float64)[dgsb_flat_idx]
    dgsb_local_margin = dgsb_local_score - dgsb_local_tau
    dgsb_score = np.minimum(dgsb_global_margin, dgsb_local_margin)

    def result(
        prediction: np.ndarray,
        detector_score: np.ndarray,
        decision_margin: np.ndarray,
        raw_score: np.ndarray,
        selected_threshold: np.ndarray,
        component: np.ndarray | None = None,
        **extra: np.ndarray,
    ) -> dict[str, np.ndarray]:
        payload = {
            "prediction": np.asarray(prediction, dtype=object),
            "detector_score": np.asarray(detector_score, dtype=np.float64),
            "margin": np.asarray(decision_margin, dtype=np.float64),
            "raw_score": np.asarray(raw_score, dtype=np.float64),
            "selected_threshold": np.asarray(selected_threshold, dtype=np.float64),
            "accept": np.asarray(decision_margin >= 0, dtype=bool),
            "predicted_component": np.full(n, -1, dtype=np.int64) if component is None else np.asarray(component, dtype=np.int64),
        }
        payload.update(extra)
        return payload

    return {
        "Native": result(names[np.asarray(native_predictions, dtype=np.int64)], native_scores, native_margin, native_scores, np.full(n, native_threshold)),
        "Single-Full-K1": result(names[single_idx], single_raw, single_margin, single_raw, np.full(n, single_threshold)),
        "Multi-Global-K2": result(names[multi_idx], multi_raw, multi_margin, multi_raw, np.full(n, multi_threshold)),
        "Class-P05-K2": result(names[class_idx], class_margin, class_margin, class_raw, class_selected_tau),
        "Component-P05-K2": result(names[component_class_idx], component_margin, component_margin, component_raw, component_selected_tau, component_ids),
        "DGSB-v2": result(
            names[multi_idx],
            dgsb_score,
            dgsb_score,
            dgsb_score,
            np.zeros(n),
            dgsb_components,
            global_score=dgsb_global_score,
            global_threshold=np.full(n, dgsb_global_tau),
            global_margin=dgsb_global_margin,
            local_score=dgsb_local_score,
            local_threshold=dgsb_local_tau,
            local_margin=dgsb_local_margin,
            global_pass=dgsb_global_margin >= 0,
            local_pass=dgsb_local_margin >= 0,
        ),
    }


def auc_metrics(known_score: np.ndarray, unknown_score: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.ones(len(known_score), dtype=np.int8), np.zeros(len(unknown_score), dtype=np.int8)])
    scores = np.concatenate([known_score, unknown_score])
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def classification_rows(setting: str, method: str, true_names: np.ndarray, result: dict[str, np.ndarray], class_names: list[str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    predicted = result["prediction"]
    precision, recall, f1, support = precision_recall_fscore_support(true_names, predicted, labels=class_names, zero_division=0)
    rows = []
    for index, class_name in enumerate(class_names):
        mask = true_names == class_name
        acceptance = float(np.mean(result["accept"][mask]))
        rows.append({
            "setting": setting,
            "method": method,
            "class": class_name,
            "support": int(support[index]),
            "accuracy": float(recall[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "acceptance": acceptance,
            "FRR": 1.0 - acceptance,
        })
    acceptance = float(np.mean(result["accept"]))
    summary = {
        "setting": setting,
        "method": method,
        "known_test_n": len(true_names),
        "known_accuracy": float(accuracy_score(true_names, predicted)),
        "known_macro_f1": float(f1_score(true_names, predicted, labels=class_names, average="macro", zero_division=0)),
        "known_acceptance": acceptance,
        "known_FRR": 1.0 - acceptance,
    }
    return summary, rows


def sample_details(
    manifests: pd.DataFrame,
    results: dict[str, dict[str, np.ndarray]],
    unknown: bool,
) -> pd.DataFrame:
    frames = []
    for method in METHOD_ORDER:
        result = results[method]
        n = len(manifests)
        frame = pd.DataFrame({
            "sample_id": manifests["sample_id"].astype(str).to_numpy(),
            "true_unknown_class" if unknown else "true_class": manifests["true_class"].astype(str).to_numpy(),
            "method": method,
            "accepted_as_known": result["accept"],
            "predicted_known_class": result["prediction"],
            "detector_score": result["detector_score"],
            "threshold": np.zeros(n) if method in {"Class-P05-K2", "Component-P05-K2", "DGSB-v2"} else result["selected_threshold"],
            "margin": result["margin"],
            "raw_score": result["raw_score"],
            "selected_threshold": result["selected_threshold"],
            "global_score": result.get("global_score", np.full(n, np.nan)),
            "global_threshold": result.get("global_threshold", np.full(n, np.nan)),
            "global_margin": result.get("global_margin", np.full(n, np.nan)),
            "predicted_component": result["predicted_component"],
            "local_score": result.get("local_score", np.full(n, np.nan)),
            "local_threshold": result.get("local_threshold", np.full(n, np.nan)),
            "local_margin": result.get("local_margin", np.full(n, np.nan)),
            "global_pass": result.get("global_pass", np.full(n, False)),
            "local_pass": result.get("local_pass", np.full(n, False)),
            "final_accept": result["accept"],
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def gate_attribution(setting: str, values: dict[str, np.ndarray], role: str) -> list[dict[str, object]]:
    global_pass = values["global_pass"]
    local_pass = values["local_pass"]
    specs = (
        ("Global FAIL / Local PASS", ~global_pass & local_pass, "Global-exclusive reject"),
        ("Global PASS / Local FAIL", global_pass & ~local_pass, "Local-exclusive reject"),
        ("Both FAIL", ~global_pass & ~local_pass, "Both gates reject"),
        ("Both PASS", global_pass & local_pass, "Both gates accept"),
    )
    return [{"setting": setting, "role": role, "category": name, "interpretation": interpretation, "count": int(mask.sum()), "rate": float(mask.mean())} for name, mask, interpretation in specs]


def decision_transitions(setting: str, multi: np.ndarray, dgsb: np.ndarray, role: str) -> list[dict[str, object]]:
    specs = (
        ("RR", ~multi & ~dgsb, "both reject"),
        ("RA", ~multi & dgsb, "Multi rejects; DGSB accepts"),
        ("AR", multi & ~dgsb, "Multi accepts; DGSB rejects"),
        ("AA", multi & dgsb, "both accept"),
    )
    return [{"setting": setting, "role": role, "transition": name, "definition": definition, "count": int(mask.sum()), "rate": float(mask.mean())} for name, mask, definition in specs]


def bootstrap_primary(
    known_true: np.ndarray,
    unknown_true: np.ndarray,
    known: dict[str, dict[str, np.ndarray]],
    unknown: dict[str, dict[str, np.ndarray]],
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    require(iterations == 1000 and seed == 0, "formal bootstrap is frozen at 1000 iterations and seed 0")
    rng = np.random.default_rng(seed)
    known_strata = [np.flatnonzero(known_true == name) for name in sorted(set(known_true))]
    unknown_strata = [np.flatnonzero(unknown_true == name) for name in sorted(set(unknown_true))]
    replicates = []
    for replicate in range(iterations):
        ki = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in known_strata])
        ui = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in unknown_strata])
        multi_k = known["Multi-Global-K2"]
        dgsb_k = known["DGSB-v2"]
        multi_u = unknown["Multi-Global-K2"]
        dgsb_u = unknown["DGSB-v2"]
        multi_auc, multi_ap = auc_metrics(multi_k["detector_score"][ki], multi_u["detector_score"][ui])
        dgsb_auc, dgsb_ap = auc_metrics(dgsb_k["detector_score"][ki], dgsb_u["detector_score"][ui])
        replicates.append({
            "replicate": replicate,
            "DeltaUFAR": float(np.mean(dgsb_u["accept"][ui]) - np.mean(multi_u["accept"][ui])),
            "Delta Known FRR": float(np.mean(~dgsb_k["accept"][ki]) - np.mean(~multi_k["accept"][ki])),
            "DeltaAUROC": dgsb_auc - multi_auc,
            "DeltaAUPRC": dgsb_ap - multi_ap,
        })
    arrays = {metric: np.asarray([row[metric] for row in replicates], dtype=np.float64) for metric in ("DeltaUFAR", "Delta Known FRR", "DeltaAUROC", "DeltaAUPRC")}
    exact_multi_auc, exact_multi_ap = auc_metrics(known["Multi-Global-K2"]["detector_score"], unknown["Multi-Global-K2"]["detector_score"])
    exact_dgsb_auc, exact_dgsb_ap = auc_metrics(known["DGSB-v2"]["detector_score"], unknown["DGSB-v2"]["detector_score"])
    exact = {
        "DeltaUFAR": float(np.mean(unknown["DGSB-v2"]["accept"]) - np.mean(unknown["Multi-Global-K2"]["accept"])),
        "Delta Known FRR": float(np.mean(~known["DGSB-v2"]["accept"]) - np.mean(~known["Multi-Global-K2"]["accept"])),
        "DeltaAUROC": exact_dgsb_auc - exact_multi_auc,
        "DeltaAUPRC": exact_dgsb_ap - exact_multi_ap,
    }
    summary = []
    for metric, values in arrays.items():
        low, high = np.quantile(values, [0.025, 0.975])
        summary.append({
            "metric": metric,
            "delta": exact[metric],
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_iterations": iterations,
            "seed": seed,
        })
    return replicates, summary


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan, math.nan, "UNDEFINED_CONSTANT_OR_INSUFFICIENT"
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), "OK"


def difficulty_vs_absorption(setting: str, unknown_n: int, details: pd.DataFrame) -> list[dict[str, object]]:
    validation = pd.read_csv(STAGE8A_ROOT / f"outputs/{setting}/known_val_per_class_metrics.csv")
    rows = []
    for method in METHOD_ORDER:
        accepted = details[(details["method"] == method) & details["accepted_as_known"]]
        counts = accepted.groupby("predicted_known_class").size().to_dict()
        values = validation.copy()
        values["absorbed_count"] = values["class_name"].map(counts).fillna(0).astype(int)
        values["absorbed_rate_all_unknown"] = values["absorbed_count"] / unknown_n
        rho_f1_count, p_f1_count, status1 = safe_spearman(values["f1"].to_numpy(), values["absorbed_count"].to_numpy())
        rho_f1_rate, p_f1_rate, status2 = safe_spearman(values["f1"].to_numpy(), values["absorbed_rate_all_unknown"].to_numpy())
        rho_acc_count, p_acc_count, status3 = safe_spearman(values["accuracy"].to_numpy(), values["absorbed_count"].to_numpy())
        rho_acc_rate, p_acc_rate, status4 = safe_spearman(values["accuracy"].to_numpy(), values["absorbed_rate_all_unknown"].to_numpy())
        for _, row in values.iterrows():
            rows.append({
                "setting": setting,
                "method": method,
                "known_class": row["class_name"],
                "known_validation_f1": row["f1"],
                "known_validation_accuracy": row["accuracy"],
                "unknown_absorbed_count": int(row["absorbed_count"]),
                "unknown_absorbed_rate": row["absorbed_rate_all_unknown"],
                "rho_f1_vs_count": rho_f1_count,
                "p_f1_vs_count": p_f1_count,
                "rho_f1_vs_rate": rho_f1_rate,
                "p_f1_vs_rate": p_f1_rate,
                "rho_accuracy_vs_count": rho_acc_count,
                "p_accuracy_vs_count": p_acc_count,
                "rho_accuracy_vs_rate": rho_acc_rate,
                "p_accuracy_vs_rate": p_acc_rate,
                "correlation_status": ";".join((status1, status2, status3, status4)),
                "interpretation_limit": "post-test explanatory correlation only",
            })
    return rows


def density_vs_utility(setting: str, unknown_n: int, details: pd.DataFrame) -> list[dict[str, object]]:
    density = pd.read_csv(STAGE8A_ROOT / f"outputs/{setting}/k1_k2_density_comparison.csv")
    counts_by_method = {
        method: details[(details["method"] == method) & details["accepted_as_known"]].groupby("predicted_known_class").size().to_dict()
        for method in METHOD_ORDER
    }
    rows = []
    for comparison, reference, candidate in (
        ("Multi-Global-K2 vs Single-Full-K1", "Single-Full-K1", "Multi-Global-K2"),
        ("DGSB-v2 vs Multi-Global-K2", "Multi-Global-K2", "DGSB-v2"),
    ):
        reductions = []
        for _, row in density.iterrows():
            name = row["class_name"]
            reductions.append(int(counts_by_method[reference].get(name, 0)) - int(counts_by_method[candidate].get(name, 0)))
        density_values = density["DeltaNLL2"].to_numpy(dtype=np.float64)
        reduction_values = np.asarray(reductions, dtype=np.float64)
        rho_count, p_count, status1 = safe_spearman(density_values, reduction_values)
        rho_rate, p_rate, status2 = safe_spearman(density_values, reduction_values / unknown_n)
        for index, (_, row) in enumerate(density.iterrows()):
            name = row["class_name"]
            rows.append({
                "setting": setting,
                "comparison": comparison,
                "known_class": name,
                "DeltaNLL2": float(row["DeltaNLL2"]),
                "reference_absorbed_count": int(counts_by_method[reference].get(name, 0)),
                "candidate_absorbed_count": int(counts_by_method[candidate].get(name, 0)),
                "absorption_reduction_count": int(reductions[index]),
                "absorption_reduction_rate": float(reductions[index] / unknown_n),
                "spearman_rho_count": rho_count,
                "spearman_p_count": p_count,
                "spearman_rho_rate": rho_rate,
                "spearman_p_rate": p_rate,
                "correlation_status": f"{status1};{status2}",
                "interpretation_limit": "post-test diagnosis; cannot modify frozen method",
            })
    return rows


def primary_conclusion(rows: list[dict[str, object]]) -> str:
    lookup = {row["metric"]: row for row in rows}
    du = lookup["DeltaUFAR"]
    df = lookup["Delta Known FRR"]
    ufar_improves = float(du["delta"]) < 0
    ufar_harms = float(du["delta"]) > 0
    ufar_supported_improvement = float(du["ci_high"]) < 0
    ufar_supported_harm = float(du["ci_low"]) > 0
    frr_supported_harm = float(df["ci_low"]) > 0
    frr_supported_improvement = float(df["ci_high"]) < 0
    if ufar_supported_improvement and not frr_supported_harm:
        return "WIN"
    if (ufar_improves and frr_supported_harm) or (ufar_harms and frr_supported_improvement):
        return "TRADEOFF"
    if ufar_supported_harm and not frr_supported_improvement:
        return "HARM"
    return "NO GAIN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    args = parser.parse_args()
    verify_all_provenance()
    assert_test_open_record()
    setting = args.setting
    output = STAGE9_ROOT / "outputs" / setting
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "known_test_metrics.csv").exists(), f"{setting}: refusing to overwrite final Test metrics")
    artifact = STAGE9_ROOT / "artifacts" / setting
    integrity = load_json(artifact / "test_representation_integrity.json")
    require(integrity["status"] == "PASS", f"{setting}: representation integrity gate failed")
    fold = load_fold(setting)
    class_names = [str(value) for value in fold["known_classes"]]
    known_manifest = pd.read_csv(artifact / "known_test_manifest.csv").sort_values("row_index")
    unknown_manifest = pd.read_csv(artifact / "unknown_test_manifest.csv").sort_values("row_index")
    require(known_manifest["row_index"].tolist() == list(range(len(known_manifest))), "Known Test row indices changed")
    require(unknown_manifest["row_index"].tolist() == list(range(len(unknown_manifest))), "Unknown Test row indices changed")
    true_known = known_manifest["true_class"].astype(str).to_numpy()
    true_unknown = unknown_manifest["true_class"].astype(str).to_numpy()
    models = load_frozen_models(setting)
    thresholds = threshold_arrays(setting, class_names)
    known = evaluate_role(
        np.load(artifact / "z_known_test_pca64.npy", mmap_mode="r", allow_pickle=False),
        np.load(artifact / "native_known_test_scores.npy", mmap_mode="r", allow_pickle=False),
        np.load(artifact / "native_known_test_predictions.npy", mmap_mode="r", allow_pickle=False),
        class_names, models, thresholds,
    )
    unknown = evaluate_role(
        np.load(artifact / "z_unknown_test_pca64.npy", mmap_mode="r", allow_pickle=False),
        np.load(artifact / "native_unknown_test_scores.npy", mmap_mode="r", allow_pickle=False),
        np.load(artifact / "native_unknown_test_predictions.npy", mmap_mode="r", allow_pickle=False),
        class_names, models, thresholds,
    )
    known_rows = []
    per_class_rows = []
    unknown_rows = []
    auc_rows = []
    per_unknown_rows = []
    auc_lookup = {}
    for method in METHOD_ORDER:
        known_summary, class_rows = classification_rows(setting, method, true_known, known[method], class_names)
        known_rows.append(known_summary)
        per_class_rows.extend(class_rows)
        unknown_acceptance = float(np.mean(unknown[method]["accept"]))
        unknown_rows.append({
            "setting": setting,
            "method": method,
            "unknown_sample_count": len(true_unknown),
            "unknown_accepted_as_known_count": int(np.sum(unknown[method]["accept"])),
            "UFAR": unknown_acceptance,
            "unknown_rejection_rate": 1.0 - unknown_acceptance,
        })
        auroc, auprc = auc_metrics(known[method]["detector_score"], unknown[method]["detector_score"])
        auc_lookup[method] = (auroc, auprc)
        auc_rows.append({
            "setting": setting,
            "method": method,
            "positive_class": "Known",
            "score_orientation": "higher_is_more_known",
            "AUROC": auroc,
            "AUPRC": auprc,
        })
        for unknown_class in sorted(set(true_unknown)):
            mask = true_unknown == unknown_class
            accepted = int(np.sum(unknown[method]["accept"][mask]))
            per_unknown_rows.append({
                "setting": setting,
                "method": method,
                "unknown_class": unknown_class,
                "n": int(mask.sum()),
                "accepted_as_known": accepted,
                "rejected": int(mask.sum()) - accepted,
                "UFAR": accepted / int(mask.sum()),
            })
    write_csv(output / "known_test_metrics.csv", known_rows)
    write_csv(output / "known_test_per_class_metrics.csv", per_class_rows)
    write_csv(output / "unknown_test_metrics.csv", unknown_rows)
    write_csv(output / "per_unknown_class_ufar.csv", per_unknown_rows)
    write_csv(output / "detection_auc_metrics.csv", auc_rows)

    unknown_details = sample_details(unknown_manifest, unknown, unknown=True)
    known_details = sample_details(known_manifest, known, unknown=False)
    unknown_details.to_parquet(output / "unknown_sample_details.parquet", index=False, engine="pyarrow", compression="zstd")
    known_details.to_parquet(output / "known_sample_details.parquet", index=False, engine="pyarrow", compression="zstd")

    absorption_rows = []
    for method in METHOD_ORDER:
        accepted = unknown_details[(unknown_details["method"] == method) & unknown_details["accepted_as_known"]]
        counts = accepted.groupby(["true_unknown_class", "predicted_known_class"]).size()
        for (unknown_class, known_class), count in counts.items():
            class_n = int(np.sum(true_unknown == unknown_class))
            absorption_rows.append({
                "setting": setting,
                "true_unknown_class": unknown_class,
                "predicted_known_class": known_class,
                "method": method,
                "count": int(count),
                "rate": int(count) / class_n,
                "rate_within_true_unknown_class": int(count) / class_n,
                "rate_of_all_unknown": int(count) / len(true_unknown),
            })
    write_csv(output / "unknown_to_known_absorption_matrix.csv", absorption_rows, ["setting", "true_unknown_class", "predicted_known_class", "method", "count", "rate", "rate_within_true_unknown_class", "rate_of_all_unknown"])
    write_csv(output / "unknown_decision_transitions.csv", decision_transitions(setting, unknown["Multi-Global-K2"]["accept"], unknown["DGSB-v2"]["accept"], "UNKNOWN_TEST"))
    write_csv(output / "known_decision_transitions.csv", decision_transitions(setting, known["Multi-Global-K2"]["accept"], known["DGSB-v2"]["accept"], "KNOWN_TEST"))
    write_csv(output / "dgsbv2_unknown_gate_attribution.csv", gate_attribution(setting, unknown["DGSB-v2"], "UNKNOWN_TEST"))
    write_csv(output / "dgsbv2_known_gate_attribution.csv", gate_attribution(setting, known["DGSB-v2"], "KNOWN_TEST"))

    stage9_config = load_json(STAGE9_ROOT / "configs/stage9_config.json")
    bootstrap_config = stage9_config["bootstrap"]
    replicate_rows, bootstrap_rows = bootstrap_primary(true_known, true_unknown, known, unknown, int(bootstrap_config["iterations"]), int(bootstrap_config["seed"]))
    pd.DataFrame(replicate_rows).to_parquet(output / "paired_bootstrap_primary_replicates.parquet", index=False, engine="pyarrow", compression="zstd")
    write_csv(output / "paired_bootstrap_primary.csv", bootstrap_rows)
    conclusion = primary_conclusion(bootstrap_rows)
    known_lookup = {row["method"]: row for row in known_rows}
    unknown_lookup = {row["method"]: row for row in unknown_rows}
    secondary = []
    for reference in ("Native", "Single-Full-K1", "Class-P05-K2", "Component-P05-K2"):
        secondary.append({
            "setting": setting,
            "candidate": "DGSB-v2",
            "reference": reference,
            "Delta Known FRR": known_lookup["DGSB-v2"]["known_FRR"] - known_lookup[reference]["known_FRR"],
            "DeltaUFAR": unknown_lookup["DGSB-v2"]["UFAR"] - unknown_lookup[reference]["UFAR"],
            "DeltaAUROC": auc_lookup["DGSB-v2"][0] - auc_lookup[reference][0],
            "DeltaAUPRC": auc_lookup["DGSB-v2"][1] - auc_lookup[reference][1],
            "comparison_scope": "secondary",
        })
    write_csv(output / "secondary_comparisons.csv", secondary)
    write_csv(output / "known_difficulty_vs_absorption.csv", difficulty_vs_absorption(setting, len(true_unknown), unknown_details))
    write_csv(output / "density_vs_unknown_utility.csv", density_vs_utility(setting, len(true_unknown), unknown_details))

    primary = {
        "setting": setting,
        "reference": "Multi-Global-K2",
        "candidate": "DGSB-v2",
        "Multi UFAR": unknown_lookup["Multi-Global-K2"]["UFAR"],
        "DGSB UFAR": unknown_lookup["DGSB-v2"]["UFAR"],
        "DeltaUFAR": unknown_lookup["DGSB-v2"]["UFAR"] - unknown_lookup["Multi-Global-K2"]["UFAR"],
        "Multi Known FRR": known_lookup["Multi-Global-K2"]["known_FRR"],
        "DGSB Known FRR": known_lookup["DGSB-v2"]["known_FRR"],
        "Delta Known FRR": known_lookup["DGSB-v2"]["known_FRR"] - known_lookup["Multi-Global-K2"]["known_FRR"],
        "Multi AUROC": auc_lookup["Multi-Global-K2"][0],
        "DGSB AUROC": auc_lookup["DGSB-v2"][0],
        "DeltaAUROC": auc_lookup["DGSB-v2"][0] - auc_lookup["Multi-Global-K2"][0],
        "Multi AUPRC": auc_lookup["Multi-Global-K2"][1],
        "DGSB AUPRC": auc_lookup["DGSB-v2"][1],
        "DeltaAUPRC": auc_lookup["DGSB-v2"][1] - auc_lookup["Multi-Global-K2"][1],
        "primary_conclusion": conclusion,
    }
    write_csv(output / "primary_comparison.csv", [primary])
    write_json(output / "setting_summary.json", {
        "setting": setting,
        "status": "COMPLETE",
        "known_test_n": len(true_known),
        "unknown_test_n": len(true_unknown),
        "known_positive_for_auc": True,
        "methods": list(METHOD_ORDER),
        "primary": primary,
        "bootstrap": bootstrap_rows,
        "post_test_tuning_performed": False,
        "methods_modified_after_test_open": False,
    })
    print(json.dumps({"setting": setting, "status": "COMPLETE", "primary": primary}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
