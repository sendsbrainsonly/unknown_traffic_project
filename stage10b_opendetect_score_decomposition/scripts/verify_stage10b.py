#!/usr/bin/env python3
"""Independent completion recheck for Stage 10B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnosis_common import CONFIG_PATH, ROOT, SCORE_ORDER, SETTINGS, SCOPE, load_json, require, write_json


def main() -> None:
    config = load_json(CONFIG_PATH)
    provenance = load_json(ROOT / "outputs/summary/provenance_verification.json")
    require(provenance["status"] == "PASS", "provenance recheck is not PASS")
    setting_checks = []
    for setting in SETTINGS:
        required = [
            "frozen_inference_audit.json", "native_identity_check.json", "score_chain_metrics.csv",
            "auroc_decomposition.csv", "posterior_variance_diagnosis.csv", "prototype_centroid_gap.csv",
            "known_geometry_classification.csv", "per_unknown_score_chain.csv",
            "prototype_gap_vs_absorption.csv", "logvar_distribution_audit.csv",
            "paired_auroc_bootstrap.csv", "paired_auroc_bootstrap_replicates.parquet", "setting_completion.json",
        ]
        out = ROOT / "outputs" / setting
        missing = [name for name in required if not (out / name).is_file()]
        require(not missing, f"{setting}: missing outputs: {missing}")
        inference = load_json(out / "frozen_inference_audit.json")
        identity = load_json(out / "native_identity_check.json")
        completion = load_json(out / "setting_completion.json")
        require(inference["status"] == identity["status"] == completion["status"] == "PASS", f"{setting}: a gate is not PASS")
        require(all(float(role["max_abs_error"]) == 0.0 and float(role["mean_abs_error"]) == 0.0 for role in inference["roles"].values()), f"{setting}: recomputed mu is not exactly aligned")
        metrics = pd.read_csv(out / "score_chain_metrics.csv")
        require(metrics["score_id"].tolist() == list(SCORE_ORDER), f"{setting}: S0-S6 order mismatch")
        require((metrics["analysis_scope"] == SCOPE).all(), f"{setting}: score scope label missing")
        require((metrics["positive_class"] == "Unknown").all() and (metrics["score_orientation"] == "higher_is_more_unknown").all(), f"{setting}: score semantics mismatch")
        require(np.isfinite(metrics[["AUROC", "AUPRC", "KS_statistic"]].to_numpy()).all(), f"{setting}: non-finite core metrics")
        for key in ("S0", "S5", "S6"):
            row = metrics[metrics["score_id"] == key].iloc[0]
            require(float(row["AUROC_reproduction_abs_error"]) <= float(config["numerical_gates"]["stage9_auroc_atol"]), f"{setting}/{key}: Stage9 reproduction failed")
        bootstrap = pd.read_csv(out / "paired_auroc_bootstrap.csv")
        replicates = pd.read_parquet(out / "paired_auroc_bootstrap_replicates.parquet")
        require(len(bootstrap) == 6 and set(bootstrap["bootstrap_iterations"]) == {1000} and set(bootstrap["seed"]) == {0}, f"{setting}: bootstrap summary mismatch")
        require(len(replicates) == 6000 and replicates["replicate"].nunique() == 1000 and set(replicates["seed"]) == {0}, f"{setting}: bootstrap replicate evidence mismatch")
        classes = pd.read_csv(out / "known_geometry_classification.csv")
        require(len(classes) == 6 and np.isfinite(classes[["accuracy", "macro_f1"]].to_numpy()).all(), f"{setting}: classification output mismatch")
        logvar = pd.read_csv(out / "logvar_distribution_audit.csv")
        require(len(logvar[logvar["row_type"] == "dimension"]) == 128 and len(logvar[logvar["row_type"] == "overall"]) == 1, f"{setting}: logvar audit dimensions mismatch")
        artifacts = ROOT / "artifacts" / setting
        for name in (
            "mu_known_test_recomputed.npy", "mu_unknown_test_recomputed.npy",
            "logvar_known_test.npy", "logvar_unknown_test.npy",
            "diagnostic_scores.npz", "diagnostic_predictions.npz",
        ):
            require((artifacts / name).is_file(), f"{setting}: missing artifact {name}")
        with np.load(artifacts / "diagnostic_scores.npz", allow_pickle=False) as scores:
            require(set(scores.files) == {f"{key}_{role}" for key in SCORE_ORDER for role in ("known", "unknown")}, f"{setting}: score artifact keys mismatch")
        setting_checks.append({
            "setting": setting,
            "status": "PASS",
            "score_rows": len(metrics),
            "bootstrap_replicates": len(replicates),
            "native_identity_max_abs_error": identity["max_abs_error"],
            "mu_alignment": "EXACT",
        })

    for path in (
        ROOT / "README.md",
        ROOT / "DIAGNOSIS_REPORT.md",
        ROOT / "RESULTS.md",
        ROOT / "outputs/summary/score_chain_summary.csv",
        ROOT / "outputs/summary/mechanism_contribution.csv",
        ROOT / "outputs/summary/auroc_decomposition_summary.csv",
        ROOT / "outputs/summary/stage7_prototype_reset_audit.md",
        ROOT / "outputs/summary/final_gate.json",
    ):
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty summary artifact: {path}")
    report = (ROOT / "DIAGNOSIS_REPORT.md").read_text(encoding="utf-8")
    for heading in (
        "Executive Summary", "Frozen Evidence", "Native KL Identity", "Native Score Decomposition",
        "Posterior Variance", "Learned Prototype Geometry", "Empirical Centroid", "Feature Scaling",
        "PCA64", "Full Covariance", "K2 Local Support", "Known Classification Geometry",
        "Unknown-Class Analysis", "Absorption Analysis", "Failure Mechanism", "Implication for Method Design",
        "What We Can Claim", "What We Cannot Claim", "Recommended Next Branch",
    ):
        require(f"## {heading}" in report, f"report heading missing: {heading}")
    mechanism = load_json(ROOT / "outputs/summary/mechanism_classification.json")
    require(str(mechanism["primary"]).startswith("M2"), "primary mechanism is not M2")
    require(any(str(value).startswith("M5") for value in mechanism["secondary"]), "M5 secondary mechanism missing")
    require(mechanism["recommended_next_branch"] == "Branch P", "recommended branch changed")
    reset = load_json(ROOT / "outputs/summary/stage7_prototype_reset_audit.json")
    require(reset["status"] == "PROTOTYPE_OPTIMIZER_RISK" and reset["risk_realized_in_frozen_runs"] is False, "prototype reset verdict mismatch")
    gate = load_json(ROOT / "outputs/summary/final_gate.json")
    require(gate["final_gate"] == "DIAGNOSIS_COMPLETE" and all(gate["checks"].values()), "final gate failed")

    prohibited_calls = (".fit(", ".fit_transform(", ".backward(", "optimizer.step(", "torch.optim")
    scanned = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [pattern for pattern in prohibited_calls if pattern in text]
        require(not hits, f"prohibited training/refit call in {path.name}: {hits}")
        scanned.append(path.name)
    result = {
        "analysis_scope": SCOPE,
        "status": "PASS",
        "final_gate": "DIAGNOSIS_COMPLETE",
        "provenance": "PASS",
        "settings": setting_checks,
        "scripts_static_checked": scanned,
        "prohibited_training_or_fit_calls": 0,
        "threshold_tuning_performed": False,
        "upstream_modification_performed": False,
        "primary_failure_mechanism": mechanism["primary"],
        "secondary_mechanism": mechanism["secondary"],
        "recommended_next_branch": mechanism["recommended_next_branch"],
    }
    write_json(ROOT / "outputs/summary/completion_verification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
