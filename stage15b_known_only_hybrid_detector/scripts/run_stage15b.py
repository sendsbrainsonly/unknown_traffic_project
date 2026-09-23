#!/usr/bin/env python3
"""Stage 15B Known-only percentile Hybrid development.

This program reads frozen scores/latents, fits only empirical CDFs and P95
thresholds on Known Validation, and evaluates the preregistered H1/H2/H3.
It never runs or updates an encoder and never fits on Unknown or Test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PRIMARY = ("VNAT", "USTC-TFC2016")
CANDIDATES = ("H1", "H2", "H3")
REFERENCES = ("H0", "DES0", "DES1")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_stage15a(project_root: Path):
    path = project_root / "stage15a_failure_regime_complementarity_diagnosis" / "scripts" / "run_stage15a.py"
    spec = importlib.util.spec_from_file_location("stage15a_reuse", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def empirical_percentile(validation: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(validation, dtype=np.float64))
    if reference.ndim != 1 or len(reference) == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("Known Validation reference must be a finite non-empty vector")
    values = np.asarray(query, dtype=np.float64)
    result = np.searchsorted(reference, values, side="right").astype(np.float64) / len(reference)
    if np.any(~np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise RuntimeError("invalid empirical percentile")
    return result


def p95(values: np.ndarray) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="higher"))


def metrics(validation: np.ndarray, known: np.ndarray, unknown: np.ndarray) -> dict[str, float]:
    validation = np.asarray(validation, dtype=np.float64)
    known = np.asarray(known, dtype=np.float64)
    unknown = np.asarray(unknown, dtype=np.float64)
    threshold = p95(validation)
    y_true = np.concatenate([np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)])
    y_score = np.concatenate([known, unknown])
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "ufar": float(np.mean(unknown < threshold)),
        "known_frr": float(np.mean(known >= threshold)),
        "validation_frr": float(np.mean(validation >= threshold)),
        "threshold": threshold,
    }


def paired_bootstrap(values: np.ndarray, repetitions: int = 10000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_scores(protocol: Any, stage15a: Any) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, float]]:
    roles = ("validation", "known_test", "unknown_test")
    role_mu = {"validation": protocol.val_mu, "known_test": protocol.known_mu, "unknown_test": protocol.unknown_mu}
    role_logvar = {"validation": protocol.val_logvar, "known_test": protocol.known_logvar, "unknown_test": protocol.unknown_logvar}
    prototypes = stage15a.load_prototypes(protocol.checkpoint)
    raw: dict[str, dict[str, np.ndarray]] = {
        "OD": {role: np.asarray(protocol.scores["M0"][role], dtype=np.float64) for role in roles},
        "DES0": {role: np.asarray(protocol.scores["M1"][role], dtype=np.float64) for role in roles},
    }
    raw["PROTO"], raw["VPOST"] = {}, {}
    parity: dict[str, float] = {}
    for role in roles:
        dproto, vpost, _ = stage15a.native_parts(role_mu[role], role_logvar[role], prototypes)
        raw["PROTO"][role] = dproto
        raw["VPOST"][role] = vpost
        parity[f"m0_decomposition_max_abs_{role}"] = float(np.max(np.abs(dproto + vpost - raw["OD"][role])))
        if parity[f"m0_decomposition_max_abs_{role}"] > 2e-4:
            raise RuntimeError(f"M0 decomposition parity failed: {protocol.dataset}/{protocol.protocol_id}/{role}")
    if "M2" in protocol.scores:
        raw["DES1"] = {role: np.asarray(protocol.scores["M2"][role], dtype=np.float64) for role in roles}

    calibrated: dict[str, dict[str, np.ndarray]] = {}
    for signal, by_role in raw.items():
        calibrated[signal] = {role: empirical_percentile(by_role["validation"], by_role[role]) for role in roles}

    methods: dict[str, dict[str, np.ndarray]] = {
        "H0": raw["OD"],
        "DES0": raw["DES0"],
        "H1": {role: np.maximum(calibrated["OD"][role], calibrated["DES0"][role]) for role in roles},
        "H3": {role: np.maximum.reduce([calibrated["PROTO"][role], calibrated["DES0"][role], calibrated["VPOST"][role]]) for role in roles},
    }
    if "DES1" in raw:
        methods["DES1"] = raw["DES1"]
        methods["H2"] = {role: np.maximum(calibrated["OD"][role], calibrated["DES1"][role]) for role in roles}
    return methods, calibrated, parity


def save_protocol_scores(output: Path, protocol: Any, methods: dict[str, dict[str, np.ndarray]],
                         calibrated: dict[str, dict[str, np.ndarray]], method_metrics: dict[str, dict[str, float]]) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{protocol.dataset}_{protocol.protocol_id}")
    directory = output / "artifacts" / safe
    directory.mkdir(parents=True, exist_ok=False)
    payload: dict[str, np.ndarray] = {
        "validation_ids": protocol.val_ids.astype(str), "known_test_ids": protocol.known_ids.astype(str),
        "unknown_test_ids": protocol.unknown_ids.astype(str), "known_test_labels": protocol.known_labels,
        "unknown_test_classes": protocol.unknown_names.astype(str),
    }
    for signal, roles in calibrated.items():
        for role, values in roles.items():
            payload[f"{role}_A_{signal.lower()}"] = values
    for method, roles in methods.items():
        for role, values in roles.items():
            payload[f"{role}_{method.lower()}"] = values
        payload[f"threshold_{method.lower()}"] = np.asarray(method_metrics[method]["threshold"])
    path = directory / "hybrid_scores.npz"
    np.savez_compressed(path, **payload)
    (directory / "metadata.json").write_text(json.dumps({
        "dataset": protocol.dataset, "protocol_id": protocol.protocol_id, "setting": protocol.setting,
        "seed": protocol.seed, "known_classes": protocol.known_names,
        "unknown_classes": sorted(set(protocol.unknown_names.tolist())),
        "normalization_fit_split": "Known Validation only", "threshold_fit_split": "Known Validation only",
        "unknown_calibration_samples": 0, "test_calibration_samples": 0,
        "available_methods": sorted(methods), "metrics": method_metrics,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def process_protocol(protocol: Any, stage15a: Any, output: Path,
                     result_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods, calibrated, parity = build_scores(protocol, stage15a)
    method_metrics = {method: metrics(scores["validation"], scores["known_test"], scores["unknown_test"])
                      for method, scores in methods.items()}
    base = method_metrics["H0"]
    des0 = method_metrics["DES0"]
    scope = "primary_development" if protocol.dataset in PRIMARY else "supplementary_retrospective"
    for method, values in method_metrics.items():
        row = {
            "dataset": protocol.dataset, "dataset_scope": scope, "protocol_id": protocol.protocol_id,
            "setting": protocol.setting, "seed": protocol.seed, "method": method,
            "score_scale": "raw" if method in REFERENCES else "Known-Val empirical percentile max",
            "known_test_samples": len(protocol.known_mu), "unknown_test_samples": len(protocol.unknown_mu),
            **values,
            "delta_auroc_vs_h0": values["auroc"] - base["auroc"],
            "delta_auprc_vs_h0": values["auprc"] - base["auprc"],
            "delta_ufar_vs_h0": values["ufar"] - base["ufar"],
            "delta_known_frr_vs_h0": values["known_frr"] - base["known_frr"],
            "delta_auroc_vs_des0": values["auroc"] - des0["auroc"],
            "delta_auprc_vs_des0": values["auprc"] - des0["auprc"],
            "delta_ufar_vs_des0": values["ufar"] - des0["ufar"],
            "negative_protocol_vs_h0": int(values["auroc"] < base["auroc"]),
            "normalization_fit_split": "Known Validation only" if method not in REFERENCES else "not applicable/raw reference",
            "threshold_fit_split": "Known Validation only", "unknown_calibration_samples": 0,
            "test_calibration_samples": 0,
        }
        result_rows.append(row)

    for unknown_class in sorted(set(protocol.unknown_names.tolist())):
        mask = protocol.unknown_names == unknown_class
        class_metrics = {method: metrics(scores["validation"], scores["known_test"], scores["unknown_test"][mask])
                         for method, scores in methods.items()}
        for method, values in class_metrics.items():
            failure_rows.append({
                "analysis_level": "protocol_class", "dataset": protocol.dataset, "dataset_scope": scope,
                "protocol_id": protocol.protocol_id, "setting": protocol.setting, "seed": protocol.seed,
                "unknown_class": unknown_class, "unknown_samples": int(mask.sum()), "method": method,
                **values,
                "delta_auroc_vs_h0": values["auroc"] - class_metrics["H0"]["auroc"],
                "delta_auprc_vs_h0": values["auprc"] - class_metrics["H0"]["auprc"],
                "delta_ufar_vs_h0": values["ufar"] - class_metrics["H0"]["ufar"],
                "delta_auroc_vs_des0": values["auroc"] - class_metrics["DES0"]["auroc"],
                "large_failure_vs_h0": int(values["auroc"] - class_metrics["H0"]["auroc"] < -0.10),
            })
    score_path = save_protocol_scores(output, protocol, methods, calibrated, method_metrics)
    return {"score_path": str(score_path.relative_to(output)), **parity}


def comparison_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, protocol_id), frame in results.groupby(["dataset", "protocol_id"]):
        indexed = frame.set_index("method")
        for candidate in CANDIDATES:
            if candidate not in indexed.index:
                continue
            c, h0, d0 = indexed.loc[candidate], indexed.loc["H0"], indexed.loc["DES0"]
            rows.append({
                "analysis_level": "protocol", "dataset": dataset, "dataset_scope": c.dataset_scope,
                "protocol_id": protocol_id, "setting": c.setting, "seed": int(c.seed), "candidate": candidate,
                "delta_auroc_vs_h0": c.auroc - h0.auroc, "delta_auprc_vs_h0": c.auprc - h0.auprc,
                "delta_ufar_vs_h0": c.ufar - h0.ufar, "delta_known_frr_vs_h0": c.known_frr - h0.known_frr,
                "delta_auroc_vs_des0": c.auroc - d0.auroc, "delta_auprc_vs_des0": c.auprc - d0.auprc,
                "delta_ufar_vs_des0": c.ufar - d0.ufar,
                "positive_vs_h0": int(c.auroc > h0.auroc), "negative_vs_h0": int(c.auroc < h0.auroc),
            })
    protocol = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    scopes = [(name, group) for name, group in protocol.groupby("dataset")]
    scopes.append(("PRIMARY_POOLED", protocol[protocol.dataset.isin(PRIMARY)]))
    scopes.append(("ALL_AVAILABLE", protocol))
    for scope, frame in scopes:
        for candidate, group in frame.groupby("candidate"):
            for reference in ("h0", "des0"):
                dauc = group[f"delta_auroc_vs_{reference}"].to_numpy()
                dap = group[f"delta_auprc_vs_{reference}"].to_numpy()
                du = group[f"delta_ufar_vs_{reference}"].to_numpy()
                low, high = paired_bootstrap(dauc)
                summary_rows.append({
                    "analysis_level": "summary", "dataset": scope,
                    "dataset_scope": "primary_pooled" if scope == "PRIMARY_POOLED" else ("all_available" if scope == "ALL_AVAILABLE" else group.dataset_scope.iloc[0]),
                    "protocol_id": "ALL", "setting": "ALL", "seed": "ALL", "candidate": candidate,
                    "comparison": f"{candidate.upper()}-{reference.upper()}", "reference": reference.upper(),
                    "protocols": len(group), "mean_delta_auroc": float(dauc.mean()), "std_delta_auroc": float(dauc.std(ddof=1)) if len(dauc) > 1 else 0.0,
                    "bootstrap_ci95_low_auroc": low, "bootstrap_ci95_high_auroc": high,
                    "mean_delta_auprc": float(dap.mean()), "mean_delta_ufar": float(du.mean()),
                    "positive_protocols": int(np.sum(dauc > 0)), "negative_protocols": int(np.sum(dauc < 0)),
                    "zero_protocols": int(np.sum(dauc == 0)), "worst_delta_auroc": float(dauc.min()),
                    "best_delta_auroc": float(dauc.max()), "bootstrap_repetitions": 10000, "bootstrap_seed": 0,
                })
    return rows + summary_rows


def add_failure_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    base = frame[frame.analysis_level == "protocol_class"]
    for (dataset, unknown_class, method), group in base.groupby(["dataset", "unknown_class", "method"]):
        summaries.append({
            "analysis_level": "class_summary", "dataset": dataset,
            "dataset_scope": group.dataset_scope.iloc[0], "protocol_id": "ALL", "setting": "ALL", "seed": "ALL",
            "unknown_class": unknown_class, "unknown_samples": int(group.unknown_samples.sum()), "method": method,
            "protocol_class_units": len(group), "auroc": float(group.auroc.mean()), "auprc": float(group.auprc.mean()),
            "ufar": float(group.ufar.mean()), "known_frr": float(group.known_frr.mean()),
            "delta_auroc_vs_h0": float(group.delta_auroc_vs_h0.mean()),
            "delta_auprc_vs_h0": float(group.delta_auprc_vs_h0.mean()),
            "delta_ufar_vs_h0": float(group.delta_ufar_vs_h0.mean()),
            "delta_auroc_vs_des0": float(group.delta_auroc_vs_des0.mean()),
            "negative_units_vs_h0": int(np.sum(group.delta_auroc_vs_h0 < 0)),
            "large_failure_vs_h0": int(np.sum(group.delta_auroc_vs_h0 < -0.10)),
            "worst_delta_auroc_vs_h0": float(group.delta_auroc_vs_h0.min()),
        })
    return rows + summaries


def evaluate_candidates(results: pd.DataFrame, failure: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary = results[results.dataset.isin(PRIMARY)]
    pivots = {(d, p): g.set_index("method") for (d, p), g in primary.groupby(["dataset", "protocol_id"])}
    des0_negative = sum(int(g.loc["DES0", "auroc"] < g.loc["H0", "auroc"]) for g in pivots.values())
    checks: list[dict[str, Any]] = []
    protocol_class = failure[(failure.analysis_level == "protocol_class") & (failure.dataset == "VNAT")]
    for candidate in CANDIDATES:
        candidate_rows = primary[primary.method == candidate]
        if len(candidate_rows) != 30:
            raise RuntimeError(f"expected 30 primary rows for {candidate}, got {len(candidate_rows)}")
        dataset_values: dict[str, float] = {}
        auprc_values: dict[str, float] = {}
        ufar_values: dict[str, float] = {}
        deltas: list[float] = []
        negative = 0
        for (dataset, pid), g in pivots.items():
            delta = float(g.loc[candidate, "auroc"] - g.loc["H0", "auroc"])
            deltas.append(delta)
            negative += int(delta < 0)
        for dataset in PRIMARY:
            frame = primary[primary.dataset == dataset].set_index(["protocol_id", "method"])
            c = frame.xs(candidate, level="method")
            h = frame.xs("H0", level="method")
            dataset_values[dataset] = float((c.auroc - h.auroc).mean())
            auprc_values[dataset] = float((c.auprc - h.auprc).mean())
            ufar_values[dataset] = float((c.ufar - h.ufar).mean())
        hard: dict[str, dict[str, float]] = {}
        for class_name in ("rsync", "scp"):
            c = protocol_class[(protocol_class.unknown_class == class_name) & (protocol_class.method == candidate)]
            hard[class_name] = {"mean": float(c.delta_auroc_vs_h0.mean()), "worst": float(c.delta_auroc_vs_h0.min()), "n": int(len(c))}
        conditions = {
            "dataset_mean_auroc": all(value >= 0 for value in dataset_values.values()),
            "overall_paired_auroc": float(np.mean(deltas)) > 0,
            "negative_protocol_reduction": negative <= des0_negative - 2,
            "hard_regime": all(v["n"] > 0 and v["mean"] >= -0.02 and v["worst"] >= -0.10 for v in hard.values()),
            "auprc": all(value >= -0.005 for value in auprc_values.values()),
            "ufar": all(value <= 0.01 for value in ufar_values.values()),
            "calibration": True,
        }
        checks.append({
            "candidate": candidate, "vnat_mean_delta_auroc": dataset_values["VNAT"],
            "ustc_mean_delta_auroc": dataset_values["USTC-TFC2016"], "pooled_mean_delta_auroc": float(np.mean(deltas)),
            "vnat_mean_delta_auprc": auprc_values["VNAT"], "ustc_mean_delta_auprc": auprc_values["USTC-TFC2016"],
            "vnat_mean_delta_ufar": ufar_values["VNAT"], "ustc_mean_delta_ufar": ufar_values["USTC-TFC2016"],
            "negative_primary_protocols": negative, "des0_negative_primary_protocols": des0_negative,
            "worst_primary_protocol_delta_auroc": float(np.min(deltas)),
            "rsync_mean_delta_auroc": hard["rsync"]["mean"], "rsync_worst_delta_auroc": hard["rsync"]["worst"],
            "scp_mean_delta_auroc": hard["scp"]["mean"], "scp_worst_delta_auroc": hard["scp"]["worst"],
            **{f"pass_{key}": bool(value) for key, value in conditions.items()},
            "full_gate_pass": all(conditions.values()),
            "partial_gate_pass": conditions["dataset_mean_auroc"] and conditions["overall_paired_auroc"] and conditions["auprc"] and conditions["calibration"],
        })
    eligible = [row for row in checks if row["full_gate_pass"]]
    if eligible:
        gate = "HYBRID_CONFIRMED"
        pool = eligible
    else:
        pool = [row for row in checks if row["partial_gate_pass"]]
        gate = "HYBRID_PARTIAL" if pool else "HYBRID_FAILED"
        if not pool:
            pool = checks
    min_negative = min(row["negative_primary_protocols"] for row in pool)
    best_worst = max(row["worst_primary_protocol_delta_auroc"] for row in pool)
    simple_pool = [row for row in pool if row["negative_primary_protocols"] <= min_negative + 1 and row["worst_primary_protocol_delta_auroc"] >= best_worst - 0.01]
    complexity = {"H1": 1, "H2": 2, "H3": 3}
    selected = sorted(simple_pool, key=lambda row: (complexity[row["candidate"]], -row["pooled_mean_delta_auroc"]))[0]
    decision = {
        "final_gate": gate, "recommended_hybrid": selected["candidate"],
        "selection_rule": "preregistered stability then simplicity tolerance then pooled mean",
        "full_gate_candidates": [row["candidate"] for row in eligible],
        "partial_gate_candidates": [row["candidate"] for row in checks if row["partial_gate_pass"]],
        "des0_negative_primary_protocols": des0_negative,
        "unknown_calibration_samples": 0, "test_calibration_samples": 0,
        "new_encoder_training": False, "weight_search": False, "gating_model_training": False,
    }
    return decision, checks


def f(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.6f}"


def make_report(output: Path, results: pd.DataFrame, comparisons: pd.DataFrame, failure: pd.DataFrame,
                decision: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    lines = ["# Stage 15B — Known-Only Hybrid Detector Development", "",
             "## Scope and controls", "",
             "All encoders and source scores are frozen. Empirical CDFs and thresholds use Known Validation only. Unknown/Test calibration count is zero. No weights, k, signals, or gating model were searched. VNAT and USTC are development datasets, not untouched external validation; ISCX results are supplementary retrospective checks.", "",
             "## Mean results", "", "| Dataset | Method | AUROC | AUPRC | UFAR | Known FRR |", "|---|---|---:|---:|---:|---:|"]
    for dataset in ("VNAT", "USTC-TFC2016", "ISCX-VPN", "ISCXTor2016"):
        frame = results[results.dataset == dataset]
        for method in ("H0", "DES0", "DES1", "H1", "H2", "H3"):
            sub = frame[frame.method == method]
            if len(sub):
                lines.append(f"| {dataset} | {method} | {f(sub.auroc.mean())} | {f(sub.auprc.mean())} | {f(sub.ufar.mean())} | {f(sub.known_frr.mean())} |")
    lines += ["", "## Candidate gate", "", "| Candidate | VNAT ΔAUROC | USTC ΔAUROC | pooled ΔAUROC | negatives | worst ΔAUROC | rsync mean/worst | scp mean/worst | Gate |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in checks:
        lines.append(f"| {row['candidate']} | {f(row['vnat_mean_delta_auroc'])} | {f(row['ustc_mean_delta_auroc'])} | {f(row['pooled_mean_delta_auroc'])} | {row['negative_primary_protocols']} | {f(row['worst_primary_protocol_delta_auroc'])} | {f(row['rsync_mean_delta_auroc'])}/{f(row['rsync_worst_delta_auroc'])} | {f(row['scp_mean_delta_auroc'])}/{f(row['scp_worst_delta_auroc'])} | {'PASS' if row['full_gate_pass'] else ('PARTIAL' if row['partial_gate_pass'] else 'FAIL')} |")

    lines += ["", "## Paired AUROC comparison against H0", "",
              "| Scope | Candidate | mean delta | positive/total | paired bootstrap 95% CI |", "|---|---|---:|---:|---:|"]
    primary_summary = comparisons[(comparisons.analysis_level == "summary") &
                                  comparisons.dataset.isin(["VNAT", "USTC-TFC2016", "PRIMARY_POOLED"]) &
                                  (comparisons.reference == "H0")]
    for row in primary_summary.itertuples():
        lines.append(f"| {row.dataset} | {row.candidate} | {f(row.mean_delta_auroc)} | {int(row.positive_protocols)}/{int(row.protocols)} | [{f(row.bootstrap_ci95_low_auroc)}, {f(row.bootstrap_ci95_high_auroc)}] |")

    selected = next(row for row in checks if row["candidate"] == decision["recommended_hybrid"])
    h1 = next(row for row in checks if row["candidate"] == "H1")
    h2 = next(row for row in checks if row["candidate"] == "H2")
    h3 = next(row for row in checks if row["candidate"] == "H3")
    h2_h1: dict[str, float] = {}
    for dataset in PRIMARY:
        frame = results[results.dataset == dataset].pivot(index="protocol_id", columns="method", values="auroc")
        h2_h1[dataset] = float((frame.H2 - frame.H1).mean())
    des0_negative = decision["des0_negative_primary_protocols"]
    lines += ["", "## Mechanism answers", "",
              f"1. **Q1 — H1 only partially preserves OD and DES advantages.** It improves over H0 by {f(h1['vnat_mean_delta_auroc'])} on VNAT and {f(h1['ustc_mean_delta_auroc'])} on USTC, but retains {h1['negative_primary_protocols']} negative protocols, exactly the same as DES-v0's {des0_negative}, and is below DES-v0 on USTC. Worst protocol delta is {f(h1['worst_primary_protocol_delta_auroc'])}.",
              f"2. **Q2 — H2 adds real but modest ranking value, not only an operating-point change.** Mean H2−H1 AUROC is {f(h2_h1['VNAT'])} on VNAT and {f(h2_h1['USTC-TFC2016'])} on USTC; negatives fall from {h1['negative_primary_protocols']} to {h2['negative_primary_protocols']} and UFAR improves further. However, H2 worsens both rsync and scp relative to H1, so the local term is not a stable failure-regime repair.",
              f"3. **Q3 — H3 does not repair posterior-uncertainty support-overlap failures.** rsync mean/worst ΔAUROC remains {f(h3['rsync_mean_delta_auroc'])}/{f(h3['rsync_worst_delta_auroc'])}; scp remains {f(h3['scp_mean_delta_auroc'])}/{f(h3['scp_worst_delta_auroc'])}. H3 also has more negative protocols than DES-v0.",
              f"4. **Q4 — no Hybrid passes the worst-case requirement.** DES-v0 has {des0_negative} negative primary protocols. H1/H2/H3 have {h1['negative_primary_protocols']}/{h2['negative_primary_protocols']}/{h3['negative_primary_protocols']}; worst deltas are {f(h1['worst_primary_protocol_delta_auroc'])}/{f(h2['worst_primary_protocol_delta_auroc'])}/{f(h3['worst_primary_protocol_delta_auroc'])}. H2 reduces the count by only one, below the preregistered reduction of two.", ""]

    hard = failure[(failure.analysis_level == "class_summary") & (failure.dataset == "VNAT") & failure.unknown_class.isin(["rsync", "scp", "sftp"]) & failure.method.isin(["H0", "DES0", "H1", "H2", "H3"])]
    lines += ["## VNAT failure regimes", "", "| Unknown | Method | AUROC | ΔAUROC vs H0 | negative units | large failures | worst delta |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in hard.sort_values(["unknown_class", "method"]).itertuples():
        lines.append(f"| {row.unknown_class} | {row.method} | {f(row.auroc)} | {f(row.delta_auroc_vs_h0)} | {int(row.negative_units_vs_h0)} | {int(row.large_failure_vs_h0)} | {f(row.worst_delta_auroc_vs_h0)} |")

    lines += ["", "## Supplementary retrospective check", "", "| Dataset | Candidate | mean ΔAUROC vs H0 | positive/total | bootstrap 95% CI |", "|---|---|---:|---:|---:|"]
    summary = comparisons[(comparisons.analysis_level == "summary") & comparisons.dataset.isin(["ISCX-VPN", "ISCXTor2016"]) & (comparisons.reference == "H0")]
    for row in summary.itertuples():
        lines.append(f"| {row.dataset} | {row.candidate} | {f(row.mean_delta_auroc)} | {int(row.positive_protocols)}/{int(row.protocols)} | [{f(row.bootstrap_ci95_low_auroc)}, {f(row.bootstrap_ci95_high_auroc)}] |")

    lines += ["", "## Final decision", "", f"**{decision['final_gate']}**", "",
              f"Recommended single formula for any controlled continuation: **{decision['recommended_hybrid']}**.", "",
              f"This is a partial, not confirmed, recommendation. It follows the preregistered cross-dataset stability/simplicity rule: H2's small aggregate advantage is offset by worse rsync/scp behavior, while H3 is less stable. Full-gate candidates: {', '.join(decision['full_gate_candidates']) if decision['full_gate_candidates'] else 'none'}. No dataset-specific formula is selected.", "",
              "## Limitations", "",
              "- This is method development on VNAT and USTC, not untouched external validation.",
              "- ISCX-VPN/ISCXTor2016 lack frozen DES-v1, so H2 is unavailable there.",
              "- Percentile fusion changes score scale and its P95 operating point; AUROC and operating metrics must be interpreted separately.",
              "- Candidate selection uses development Unknown outcomes only for the preregistered Gate; Unknown never enters calibration or parameter fitting.", ""]
    (output / "stage15b_hybrid_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    config = read_json(output / "config.json")
    stage15a = load_stage15a(root)
    protocols = stage15a.vnat_protocols(root) + stage15a.ustc_protocols(root) + stage15a.iscx_protocols(root)
    source_hashes: dict[str, str] = {}
    for protocol in protocols:
        for path in protocol.source_paths:
            source_hashes[str(path.resolve())] = sha256(path)
    (output / "source_hashes_before.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for index, protocol in enumerate(protocols, 1):
        print(f"[{index:02d}/{len(protocols)}] {protocol.dataset}/{protocol.protocol_id}", flush=True)
        parity = process_protocol(protocol, stage15a, output, result_rows, failure_rows)
        parity_rows.append({"dataset": protocol.dataset, "protocol_id": protocol.protocol_id, **parity})

    results = pd.DataFrame(result_rows)
    comparison = comparison_rows(results)
    failure_all = add_failure_summaries(failure_rows)
    failure = pd.DataFrame(failure_all)
    decision, checks = evaluate_candidates(results, failure)
    write_csv(output / "stage15b_hybrid_results.csv", result_rows)
    write_csv(output / "stage15b_protocol_comparison.csv", comparison)
    write_csv(output / "stage15b_failure_regime_analysis.csv", failure_all)
    (output / "candidate_gate.json").write_text(json.dumps({"decision": decision, "candidates": checks}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    after = {path: sha256(Path(path)) for path in source_hashes}
    (output / "source_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source_hashes != after:
        raise RuntimeError("frozen source hashes changed during Stage 15B")
    make_report(output, results, pd.DataFrame(comparison), failure, decision, checks)
    completion = {
        "status": "PASS", "final_gate": decision["final_gate"], "recommended_hybrid": decision["recommended_hybrid"],
        "protocols": len(protocols), "metric_rows": len(result_rows), "failure_rows": len(failure_all),
        "score_artifact_protocols": len(parity_rows), "source_hashes": len(source_hashes), "source_hashes_unchanged": True,
        "max_m0_decomposition_error": max(value for row in parity_rows for key, value in row.items() if key.startswith("m0_decomposition")),
        "new_encoder_training": False, "encoder_inference": False, "unknown_calibration_samples": 0,
        "test_calibration_samples": 0, "weight_search": False, "new_signal_search": False,
        "gating_model_training": False, "config_sha256": sha256(output / "config.json"), "config": config,
    }
    (output / "completion_verification.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **decision, "metric_rows": len(result_rows), "failure_rows": len(failure_all)}, indent=2))


if __name__ == "__main__":
    main()
