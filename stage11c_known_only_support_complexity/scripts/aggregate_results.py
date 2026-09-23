#!/usr/bin/env python3
"""Aggregate Known-only features, fixed rules, LOSO checks and Stage 11C gate."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from stage11c_common import (
    SCENARIOS,
    SEEDS,
    STAGE_ROOT,
    SUMMARY_ROOT,
    output_run_dir,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


def as_number(value: object) -> float:
    return float(value)


def decision_label(delta: float) -> str:
    if delta >= 0.02:
        return "FULL_K1_MATERIAL_BENEFIT"
    if delta <= -0.02:
        return "FULL_K1_HARM"
    return "NEUTRAL"


def choice_correct(choice: str, label: str) -> bool:
    return label == "NEUTRAL" or (label == "FULL_K1_MATERIAL_BENEFIT" and choice == "R2") or (label == "FULL_K1_HARM" and choice == "R1")


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([as_number(row[key]) for row in rows]))


def scenario_known_summary(features: list[dict], outcomes: list[dict]) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        subset = [row for row in features if row["scenario"] == scenario]
        outcome_subset = [row for row in outcomes if row["scenario"] == scenario]
        rows.append(
            {
                "scenario": scenario,
                "runs": len(subset),
                "pca64_margin_median_mean": mean(subset, "geometry_pca64_validation_margin_median_mean"),
                "pca64_margin_p10_mean": mean(subset, "geometry_pca64_validation_margin_p10_mean"),
                "pca64_negative_margin_fraction_mean": mean(subset, "geometry_pca64_validation_negative_margin_fraction_mean"),
                "raw128_condition_number_median": mean(subset, "geometry_raw128_condition_number_median"),
                "pca64_condition_number_median": mean(subset, "geometry_pca64_condition_number_median"),
                "raw128_effective_rank_median": mean(subset, "geometry_raw128_effective_rank_median"),
                "pca64_effective_rank_median": mean(subset, "geometry_pca64_effective_rank_median"),
                "raw128_n_over_d_median": mean(subset, "geometry_raw128_n_over_d_median"),
                "pca64_n_over_d_median": mean(subset, "geometry_pca64_n_over_d_median"),
                "covariance_bootstrap_frobenius_relative_mean": mean(subset, "covariance_bootstrap_frobenius_relative_mean"),
                "validation_delta_nll_full_mean": mean(subset, "delta_nll_full_validation"),
                "validation_relative_delta_nll_full_mean": mean(subset, "relative_delta_nll_full_validation"),
                "validation_delta_macro_f1_full_mean": mean(subset, "delta_validation_macro_f1_full_minus_centroid"),
                "coverage_r1_variance_mean": mean(subset, "coverage_r1_variance"),
                "coverage_r2_variance_mean": mean(subset, "coverage_r2_variance"),
                "delta_coverage_variance_r1_minus_r2_mean": mean(subset, "delta_coverage_variance_r1_minus_r2"),
                "outcome_r2_minus_r1_auroc_mean": mean(outcome_subset, "delta_cov_auroc"),
            }
        )
    return rows


def build_correlations(features: list[dict], outcomes: list[dict]) -> list[dict]:
    outcome_by_key = {(row["scenario"], int(row["fold"]), int(row["seed"])): float(row["delta_cov_auroc"]) for row in outcomes}
    y = np.asarray([outcome_by_key[(row["scenario"], int(row["fold"]), int(row["seed"]))] for row in features], dtype=np.float64)
    excluded = {"scenario", "fold", "seed", "features_use_unknown", "features_use_known_test", "features_use_unknown_test"}
    rows = []
    for feature in features[0]:
        if feature in excluded:
            continue
        try:
            x = np.asarray([float(row[feature]) for row in features], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
            continue
        sr = spearmanr(x, y)
        pr = pearsonr(x, y)
        rows.append(
            {
                "feature": feature,
                "n": len(x),
                "spearman_rho": float(sr.statistic),
                "spearman_p_value": float(sr.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p_value": float(pr.pvalue),
                "outcome": "OUTCOME_ONLY_delta_cov_auroc",
                "analysis_scope": "EXPLORATORY_ONLY",
                "used_for_threshold_mining": False,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["spearman_rho"])), reverse=True)


def build_rules(features: list[dict], outcomes: list[dict]) -> tuple[list[dict], list[dict], dict]:
    outcome_by_key = {(row["scenario"], int(row["fold"]), int(row["seed"])): row for row in outcomes}
    instability_by_scenario = defaultdict(list)
    for row in features:
        instability_by_scenario[row["scenario"]].append(float(row["covariance_bootstrap_frobenius_relative_mean"]))
    loso_threshold = {
        held: float(np.median([value for scenario, values in instability_by_scenario.items() if scenario != held for value in values]))
        for held in SCENARIOS
    }
    rows: list[dict] = []
    for feature in features:
        key = (feature["scenario"], int(feature["fold"]), int(feature["seed"]))
        outcome = outcome_by_key[key]
        label = decision_label(float(outcome["delta_cov_auroc"]))
        choices = {
            "Rule-NLL": "R2" if float(feature["delta_nll_full_validation"]) > 0 else "R1",
            "Rule-NLL-1pct": "R2" if float(feature["relative_delta_nll_full_validation"]) > 0.01 else "R1",
            "Rule-F1": "R2" if float(feature["delta_validation_macro_f1_full_minus_centroid"]) > 0.005 else "R1",
            "Rule-Stability": "R2" if float(feature["delta_nll_full_validation"]) > 0 and float(feature["covariance_bootstrap_frobenius_relative_mean"]) < loso_threshold[feature["scenario"]] else "R1",
        }
        for rule, choice in choices.items():
            chosen_auroc = float(outcome["r2_auroc"] if choice == "R2" else outcome["r1_auroc"])
            rows.append(
                {
                    "row_type": "run",
                    "rule": rule,
                    "scenario": feature["scenario"],
                    "fold": feature["fold"],
                    "seed": feature["seed"],
                    "choice": choice,
                    "decision_label": label,
                    "choice_correct_or_neutral": choice_correct(choice, label),
                    "adaptive_auroc": chosen_auroc,
                    "always_r0_auroc": outcome["r0_auroc"],
                    "always_r1_auroc": outcome["r1_auroc"],
                    "always_r2_auroc": outcome["r2_auroc"],
                    "delta_vs_r1": chosen_auroc - float(outcome["r1_auroc"]),
                    "delta_vs_r2": chosen_auroc - float(outcome["r2_auroc"]),
                    "outcome_delta_cov_auroc": outcome["delta_cov_auroc"],
                    "stability_threshold_source": "OTHER_TWO_SCENARIOS_ONLY" if rule == "Rule-Stability" else "NOT_APPLICABLE",
                    "stability_threshold": loso_threshold[feature["scenario"]] if rule == "Rule-Stability" else "",
                    "post_hoc_label": "POST_HOC_DEVELOPMENT_DIAGNOSIS",
                }
            )
    summary_rows: list[dict] = []
    for rule in ("Rule-NLL", "Rule-NLL-1pct", "Rule-F1", "Rule-Stability"):
        for scenario in SCENARIOS:
            subset = [row for row in rows if row["rule"] == rule and row["scenario"] == scenario]
            summary_rows.append(
                {
                    "row_type": "summary",
                    "rule": rule,
                    "scenario": scenario,
                    "fold": "",
                    "seed": "",
                    "choice": "",
                    "decision_label": "",
                    "choice_correct_or_neutral": float(np.mean([str(row["choice_correct_or_neutral"]).lower() == "true" if isinstance(row["choice_correct_or_neutral"], str) else bool(row["choice_correct_or_neutral"]) for row in subset])),
                    "adaptive_auroc": mean(subset, "adaptive_auroc"),
                    "always_r0_auroc": mean(subset, "always_r0_auroc"),
                    "always_r1_auroc": mean(subset, "always_r1_auroc"),
                    "always_r2_auroc": mean(subset, "always_r2_auroc"),
                    "delta_vs_r1": mean(subset, "delta_vs_r1"),
                    "delta_vs_r2": mean(subset, "delta_vs_r2"),
                    "r2_choice_count": sum(row["choice"] == "R2" for row in subset),
                    "r1_choice_count": sum(row["choice"] == "R1" for row in subset),
                    "post_hoc_label": "POST_HOC_DEVELOPMENT_DIAGNOSIS",
                }
            )
    all_rows = rows + summary_rows
    loso_rows = []
    for held in SCENARIOS:
        for rule in ("Rule-NLL", "Rule-NLL-1pct", "Rule-F1", "Rule-Stability"):
            subset = [row for row in rows if row["scenario"] == held and row["rule"] == rule]
            for row in subset:
                loso_rows.append({"row_type": "run", "train_scenarios": "+".join(s for s in SCENARIOS if s != held), "test_scenario": held, **{k: row[k] for k in ("rule", "fold", "seed", "choice", "decision_label", "choice_correct_or_neutral", "adaptive_auroc", "always_r1_auroc", "always_r2_auroc", "delta_vs_r1")}})
            loso_rows.append(
                {
                    "row_type": "summary",
                    "train_scenarios": "+".join(s for s in SCENARIOS if s != held),
                    "test_scenario": held,
                    "rule": rule,
                    "fold": "",
                    "seed": "",
                    "choice": "",
                    "decision_label": "",
                    "choice_correct_or_neutral": float(np.mean([bool(row["choice_correct_or_neutral"]) for row in subset])),
                    "adaptive_auroc": mean(subset, "adaptive_auroc"),
                    "always_r1_auroc": mean(subset, "always_r1_auroc"),
                    "always_r2_auroc": mean(subset, "always_r2_auroc"),
                    "delta_vs_r1": mean(subset, "delta_vs_r1"),
                    "r2_choice_count": sum(row["choice"] == "R2" for row in subset),
                    "r1_choice_count": sum(row["choice"] == "R1" for row in subset),
                }
            )
    return all_rows, loso_rows, loso_threshold


def gate_rules(rule_rows: list[dict]) -> tuple[dict, str, str]:
    run_rows = [row for row in rule_rows if row["row_type"] == "run"]
    evaluations = {}
    for rule in ("Rule-NLL", "Rule-F1", "Rule-Stability"):
        by_scenario = {scenario: [row for row in run_rows if row["rule"] == rule and row["scenario"] == scenario] for scenario in SCENARIOS}
        a1_full = sum(row["choice"] == "R2" for row in by_scenario["a1"])
        a2_full = sum(row["choice"] == "R2" for row in by_scenario["a2"])
        a3_severe_errors = sum(not bool(row["choice_correct_or_neutral"]) for row in by_scenario["a3"])
        adaptive_mean = float(np.mean([float(row["adaptive_auroc"]) for row in run_rows if row["rule"] == rule]))
        r1_mean = float(np.mean([float(row["always_r1_auroc"]) for row in run_rows if row["rule"] == rule]))
        checks = {
            "avoids_a1_harm": a1_full <= 1,
            "retains_a2_gain": a2_full >= 3,
            "a3_no_severe_misselection": a3_severe_errors <= 1,
            "loso_direction_reasonable": a1_full <= 1 and a2_full >= 3,
            "adaptive_not_below_always_r1": adaptive_mean >= r1_mean - 1e-12,
            "known_only": True,
        }
        evaluations[rule] = {"checks": checks, "passed_checks": sum(checks.values()), "all_checks_pass": all(checks.values()), "a1_r2_count": a1_full, "a2_r2_count": a2_full, "a3_severe_errors": a3_severe_errors, "adaptive_mean_auroc": adaptive_mean, "always_r1_mean_auroc": r1_mean}
    passing = [rule for rule, value in evaluations.items() if value["all_checks_pass"]]
    if passing:
        gate = "SIGNAL_FOUND"
        mechanism = "C1" if any(rule in passing for rule in ("Rule-NLL", "Rule-F1")) else "C2"
    elif max(value["passed_checks"] for value in evaluations.values()) >= 4:
        gate = "WEAK_SIGNAL"
        mechanism = "C4"
    else:
        gate = "NO_SIGNAL"
        mechanism = "C5"
    return evaluations, gate, mechanism


def diagnosis_markdown(scenario: str, row: dict, gate: str) -> str:
    title = {"a1": "A1 Full-K1 failure diagnosis", "a2": "A2 covariance benefit diagnosis", "a3": "A3 covariance diagnosis"}[scenario]
    if scenario == "a3":
        delta = float(row["validation_delta_macro_f1_full_mean"])
        nll = float(row["validation_delta_nll_full_mean"])
        interpretation = "COVARIANCE_GAIN_CEILING_LIMITED" if delta > 0 and nll > 0 else ("CENTROID_ALREADY_SUFFICIENT" if float(row["outcome_r2_minus_r1_auroc_mean"]) >= 0 else "COVARIANCE_NOT_USEFUL")
    elif scenario == "a1":
        interpretation = "Known-only covariance fit does not transfer to open-set ranking; centroid geometry is already sufficient" if float(row["outcome_r2_minus_r1_auroc_mean"]) < 0 else "No observed Full-K1 harm"
    else:
        interpretation = "Full covariance captures useful Known-side structure" if float(row["outcome_r2_minus_r1_auroc_mean"]) > 0 else "No observed Full-K1 benefit"
    return f"""# {title}

## Verified Known-only evidence

- PCA64 centroid median margin: `{float(row['pca64_margin_median_mean']):.6f}`; negative-margin fraction: `{float(row['pca64_negative_margin_fraction_mean']):.6f}`.
- PCA64 covariance condition-number median: `{float(row['pca64_condition_number_median']):.6f}`; effective-rank median: `{float(row['pca64_effective_rank_median']):.6f}`.
- Median PCA64 n/d: `{float(row['pca64_n_over_d_median']):.6f}`.
- Train-only covariance bootstrap relative instability: `{float(row['covariance_bootstrap_frobenius_relative_mean']):.6f}`.
- Known-Validation Full-vs-Spherical DeltaNLL: `{float(row['validation_delta_nll_full_mean']):.6f}`.
- Known-Validation Full-K1 minus centroid Macro-F1: `{float(row['validation_delta_macro_f1_full_mean']):.6f}`.
- R1/R2 coverage variance: `{float(row['coverage_r1_variance_mean']):.6f}` / `{float(row['coverage_r2_variance_mean']):.6f}`.

## Post-hoc outcome

- Stage11B R2-R1 AUROC: `{float(row['outcome_r2_minus_r1_auroc_mean']):.6f}` (`OUTCOME_ONLY`).

## Interpretation

{interpretation}. This is an exploratory mechanism inference, not causal proof. Overall Stage11C gate: `{gate}`.
"""


def main() -> None:
    expected = [(scenario, fold, seed) for scenario in SCENARIOS for fold, seed in enumerate(SEEDS)]
    features: list[dict] = []
    outcomes: list[dict] = []
    geometry: list[dict] = []
    stability: list[dict] = []
    fit: list[dict] = []
    classification: list[dict] = []
    coverage: list[dict] = []
    for scenario, fold, seed in expected:
        run = output_run_dir(scenario, fold, seed)
        if not (run / "SUCCESS").is_file():
            raise RuntimeError(f"Incomplete Stage11C run: {run}")
        features.append(read_json(run / "known_only_features.json"))
        outcomes.append(read_json(run / "outcome_only.json"))
        geometry.extend(read_csv(run / "class_level_geometry.csv"))
        stability.extend(read_csv(run / "covariance_stability.csv"))
        fit.extend(read_csv(run / "validation_fit_comparison.csv"))
        classification.extend(read_csv(run / "validation_classification_comparison.csv"))
        coverage.extend(read_csv(run / "coverage_stability.csv"))
    if any(row.get("features_use_unknown") or row.get("features_use_known_test") or row.get("features_use_unknown_test") for row in features):
        raise RuntimeError("Forbidden feature source detected")

    correlations = build_correlations(features, outcomes)
    rule_rows, loso_rows, stability_thresholds = build_rules(features, outcomes)
    evaluations, gate, mechanism = gate_rules(rule_rows)
    scenario_rows = scenario_known_summary(features, outcomes)

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY_ROOT / "known_only_features.csv", features)
    write_csv(SUMMARY_ROOT / "outcomes_only.csv", outcomes)
    write_csv(SUMMARY_ROOT / "class_level_geometry.csv", geometry)
    write_csv(SUMMARY_ROOT / "covariance_stability.csv", stability)
    write_csv(SUMMARY_ROOT / "validation_fit_comparison.csv", fit)
    write_csv(SUMMARY_ROOT / "validation_classification_comparison.csv", classification)
    write_csv(SUMMARY_ROOT / "coverage_stability.csv", coverage)
    write_csv(SUMMARY_ROOT / "feature_outcome_correlations.csv", correlations)
    write_csv(SUMMARY_ROOT / "candidate_rule_results.csv", rule_rows)
    write_csv(SUMMARY_ROOT / "loso_analysis.csv", loso_rows)
    write_csv(SUMMARY_ROOT / "scenario_known_only_summary.csv", scenario_rows)
    for scenario in SCENARIOS:
        scenario_dir = STAGE_ROOT / "outputs" / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        write_csv(scenario_dir / "known_only_features.csv", [row for row in features if row["scenario"] == scenario])
        write_csv(scenario_dir / "candidate_rule_results.csv", [row for row in rule_rows if row["scenario"] == scenario])

    highest = correlations[0]
    gate_result = {
        "final_gate": gate,
        "mechanism": mechanism,
        "rule_evaluations": evaluations,
        "stability_loso_training_medians": stability_thresholds,
        "highest_absolute_spearman_feature": highest,
        "unknown_feature_used_for_criterion": False,
        "known_test_feature_used_for_criterion": False,
        "cipherspectrum_stage9_sample_level_test_used": False,
        "new_training": False,
        "new_detector_fitting": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
    }
    write_json(SUMMARY_ROOT / "final_gate.json", gate_result)
    (SUMMARY_ROOT / "mechanism_case.md").write_text(f"# Stage 11C mechanism\n\n**{mechanism}**\n\nThe fixed Known-only analyses produce `{gate}`. Correlations remain `EXPLORATORY_ONLY`; no causal or external-validity claim is made.\n", encoding="utf-8")
    (SUMMARY_ROOT / "final_gate.md").write_text(f"# Stage 11C final gate\n\n**{gate}**\n\nMechanism: **{mechanism}**. No new training, detector fitting, threshold search, Unknown-derived criterion, CipherSpectrum Test access, or next-stage execution occurred.\n", encoding="utf-8")
    by_scenario = {row["scenario"]: row for row in scenario_rows}
    (SUMMARY_ROOT / "a1_failure_diagnosis.md").write_text(diagnosis_markdown("a1", by_scenario["a1"], gate), encoding="utf-8")
    (SUMMARY_ROOT / "a2_covariance_benefit_diagnosis.md").write_text(diagnosis_markdown("a2", by_scenario["a2"], gate), encoding="utf-8")
    (SUMMARY_ROOT / "a3_diagnosis.md").write_text(diagnosis_markdown("a3", by_scenario["a3"], gate), encoding="utf-8")
    print({"status": "PASS", "runs": len(features), "gate": gate, "mechanism": mechanism, "top_feature": highest["feature"], "top_spearman": highest["spearman_rho"]}, flush=True)


if __name__ == "__main__":
    main()
