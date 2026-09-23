#!/usr/bin/env python3
"""Fresh completion checks for the materialized Stage 3 diagnosis bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DIAG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DIAG_ROOT.parent
OUTPUT_ROOT = DIAG_ROOT / "outputs"
sys.path.insert(0, str(DIAG_ROOT / "scripts"))
import analyze_failure as af  # noqa: E402


REQUIRED = {
    "provenance_snapshot.md",
    "class_unknown_absorption_delta.csv",
    "unknown_to_known_absorption_matrix.csv",
    "unknown_transition_details.parquet",
    "known_only_support_diagnostics.csv",
    "known_only_candidate_signals.csv",
    "known_signal_vs_unknown_failure.csv",
    "known_signal_correlations.csv",
    "simple_statistics_followup.csv",
    "failure_case_summary.csv",
    "candidate_hypotheses.md",
    "audit_summary.md",
    "run_metadata.json",
    "RESULTS.md",
    "manifest.json",
}
README_HEADINGS = {
    "## Background",
    "## Stage3 Gate D",
    "## Why This Is Post-Hoc Diagnosis",
    "## A-1 Failure Attribution",
    "## A-2 Neutral Result",
    "## A-3 Improvement Attribution",
    "## Known-Only Signals",
    "## Density vs Utility",
    "## Candidate Hypotheses",
    "## Final Diagnosis",
    "## Validity Boundary",
    "## Next Experiment",
}


def require_columns(frame: pd.DataFrame, names: set[str], label: str) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise AssertionError(f"{label}: missing columns {sorted(missing)}")


def main() -> None:
    missing_files = sorted(name for name in REQUIRED if not (OUTPUT_ROOT / name).is_file())
    assert not missing_files, f"missing output files: {missing_files}"
    readme = (DIAG_ROOT / "README.md").read_text(encoding="utf-8")
    assert README_HEADINGS <= set(readme.splitlines())

    delta = pd.read_csv(OUTPUT_ROOT / "class_unknown_absorption_delta.csv")
    require_columns(delta, {
        "setting", "known_class", "single_absorbed_unknown", "multi_absorbed_unknown",
        "delta_absorbed", "single_absorption_rate", "multi_absorption_rate",
    }, "class absorption")
    assert len(delta) == 51
    assert delta.groupby("setting")["delta_absorbed"].sum().to_dict() == {"A-1": 86, "A-2": 15, "A-3": -236}
    nonzero = delta[delta["delta_absorbed"] != 0].set_index(["setting", "known_class"])["delta_absorbed"].to_dict()
    assert nonzero == {("A-1", "Htbot"): 86, ("A-2", "Miuref"): 16, ("A-2", "Shifu"): -1, ("A-3", "Virut"): -236}

    matrix = pd.read_csv(OUTPUT_ROOT / "unknown_to_known_absorption_matrix.csv")
    require_columns(matrix, {"setting", "unknown_class", "known_class", "single_absorbed_count", "multi_absorbed_count", "delta_absorbed"}, "absorption matrix")
    assert len(matrix) == 145

    details = pd.read_parquet(OUTPUT_ROOT / "unknown_transition_details.parquet")
    require_columns(details, {
        "flow_id", "unknown_class", "setting", "single_best_known_class", "multi_best_known_class",
        "single_score", "multi_score", "single_threshold_margin", "multi_threshold_margin",
        "transition", "multi_component_id", "multi_component_posterior",
    }, "transition details")
    assert len(details) == 16735
    assert set(details["transition"]) == {"SS", "SM", "MS", "MM"}
    assert float(details["single_replay_score_abs_error"].max()) == 0.0
    assert float(details["multi_replay_score_abs_error"].max()) == 0.0
    assert details.groupby(["setting", "transition"]).size().to_dict() == {
        ("A-1", "MM"): 444, ("A-1", "SM"): 86, ("A-1", "SS"): 320,
        ("A-2", "MM"): 3401, ("A-2", "MS"): 48, ("A-2", "SM"): 63, ("A-2", "SS"): 2067,
        ("A-3", "MM"): 964, ("A-3", "MS"): 236, ("A-3", "SS"): 9106,
    }

    diagnostics = pd.read_csv(OUTPUT_ROOT / "known_only_support_diagnostics.csv")
    require_columns(diagnostics, {
        "delta_nll2", "min_component_weight", "train_val_tv",
        "component_mean_euclidean_distance", "component_mahalanobis_distance",
        "log_covariance_volume_ratio", "val_own_score_shift_p95",
        "val_global_disagreement_rate", "val_global_margin_shift_p95",
        "k1_converged", "k2_converged", "tiny_component_lt_1pct", "validation_empty_component",
    }, "Known-only diagnostics")
    assert len(diagnostics) == 51
    assert not diagnostics["fit_operation_performed"].astype(bool).any()
    assert diagnostics["k1_converged"].astype(bool).all() and diagnostics["k2_converged"].astype(bool).all()

    signals = pd.read_csv(OUTPUT_ROOT / "known_only_candidate_signals.csv")
    assert "delta_absorbed" not in signals.columns
    merged = pd.read_csv(OUTPUT_ROOT / "known_signal_vs_unknown_failure.csv")
    assert len(merged) == 51 and "delta_absorbed" in merged.columns
    correlations = pd.read_csv(OUTPUT_ROOT / "known_signal_correlations.csv")
    delta_nll = correlations[(correlations["scope"] == "ALL") & (correlations["signal"] == "delta_nll2")].iloc[0]
    assert np.isclose(delta_nll["spearman_rho"], -0.0215453649784472)

    simple = pd.read_csv(OUTPUT_ROOT / "simple_statistics_followup.csv")
    assert len(simple) == 60 and int(simple["missing_metadata"].sum()) == 0
    assert set(simple["effect_size_formula"]) == {"max(0,(H-k+1)/(n-k))"}

    metadata = json.loads((OUTPUT_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["fit_operations_performed"] == []
    assert metadata["final_test_rerun"] is False
    assert metadata["unknown_replayed_scores_match_frozen"] is True
    assert metadata["final_diagnosis"] == "A — DENSITY-UTILITY MISMATCH"
    for setting in af.SETTINGS:
        digest, count = af.tree_content_hash(af.STAGE3_ROOT / "outputs" / setting)
        assert digest == af.SNAPSHOT_PACKAGE_SHA256[setting]
        assert count == metadata["stage3_output_trees_after"][setting]["file_count"]

    root_readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Stage 3 Failure Diagnosis | ✅ 完成" in root_readme
    index = (PROJECT_ROOT / "EXPERIMENT_RESULTS.md").read_text(encoding="utf-8")
    assert "stage3-failure-diagnosis-20260912" in index
    print(json.dumps({"status": "PASS", "required_outputs": len(REQUIRED), "diagnostic_cells": len(diagnostics), "unknown_rows": len(details)}, sort_keys=True))


if __name__ == "__main__":
    main()
