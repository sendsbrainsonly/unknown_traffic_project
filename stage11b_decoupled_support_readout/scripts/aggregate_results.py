#!/usr/bin/env python3
"""Aggregate all 15 frozen Stage 11B evaluations and apply the fixed gate."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from stage11b_common import CONFIG_PATH, METHODS, STAGE_ROOT, read_json, write_csv, write_json


SUMMARY = STAGE_ROOT / "outputs" / "summary"
METRICS = ("auroc", "auprc", "binary_f1", "ufar", "known_frr", "known_macro_f1")
COMPARISONS = {
    "R1-R0": ("R1", "R0"),
    "R2-R0": ("R2", "R0"),
    "R2-R1": ("R2", "R1"),
    "R3-R2": ("R3", "R2"),
}


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def detector_row(result: dict[str, object], method: str) -> dict[str, object]:
    detector = result["evaluation"]["detectors"][method]
    detection = detector["combined_balanced_1to1"]
    return {
        "evaluation_label": result["evaluation_label"],
        "external_validation_label": result["external_validation_label"],
        "scenario": result["scenario"],
        "fold": result["fold"],
        "seed": result["seed"],
        "method": method,
        "checkpoint_sha256": result["frozen_encoder"]["checkpoint_sha256_before"],
        "threshold": detection["threshold"],
        "threshold_source": detection["threshold_source"],
        "validation_acceptance": detector["validation_known"]["acceptance"],
        "validation_frr": detector["validation_known"]["frr"],
        "known_test_accuracy": detector["known_test"]["accuracy"],
        "known_macro_f1": detector["known_test"]["macro_f1"],
        "known_test_acceptance_all": detector["known_test"]["acceptance"],
        "known_test_frr_all": detector["known_test"]["frr"],
        "unknown_rejection_all": detector["unknown_test"]["rejection"],
        "ufar_all": detector["unknown_test"]["ufar"],
        "accuracy": detection["accuracy"],
        "binary_f1": detection["binary_f1"],
        "auroc": detection["auroc"],
        "auprc": detection["auprc"],
        "known_frr": detection["known_frr"],
        "ufar": detection["ufar"],
        "sample_count_known": detection["sample_count_known"],
        "sample_count_unknown": detection["sample_count_unknown"],
        "new_encoder_training": False,
    }


def paired_rows(run_rows: list[dict[str, object]], name: str) -> list[dict[str, object]]:
    lhs, rhs = COMPARISONS[name]
    indexed = {
        (row["scenario"], int(row["fold"]), int(row["seed"]), row["method"]): row
        for row in run_rows
    }
    rows: list[dict[str, object]] = []
    identities = sorted({key[:3] for key in indexed})
    for scenario, fold, seed in identities:
        left = indexed[(scenario, fold, seed, lhs)]
        right = indexed[(scenario, fold, seed, rhs)]
        row: dict[str, object] = {
            "scenario": scenario,
            "fold": fold,
            "seed": seed,
            "comparison": name,
            "lhs": lhs,
            "rhs": rhs,
        }
        for metric in METRICS:
            row[f"delta_{metric}"] = float(left[metric]) - float(right[metric])
        rows.append(row)
    return rows


def bootstrap_rows(comparison_rows: dict[str, list[dict[str, object]]], config: dict[str, object]) -> list[dict[str, object]]:
    iterations = int(config["bootstrap"]["iterations"])
    seed = int(config["bootstrap"]["seed"])
    alpha = (100.0 - float(config["bootstrap"]["ci_percent"])) / 2.0
    rows: list[dict[str, object]] = []
    for comparison in ("R2-R0", "R1-R0", "R3-R2"):
        for scenario in ("a1", "a2", "a3"):
            source = [row for row in comparison_rows[comparison] if row["scenario"] == scenario]
            if len(source) != 5:
                raise AssertionError(f"Bootstrap requires five paired seeds: {comparison}/{scenario}")
            for metric in ("auroc", "binary_f1", "ufar", "known_frr"):
                values = np.asarray([row[f"delta_{metric}"] for row in source], dtype=np.float64)
                rng = np.random.default_rng(seed)
                indices = rng.integers(0, len(values), size=(iterations, len(values)))
                means = values[indices].mean(axis=1)
                rows.append(
                    {
                        "comparison": comparison,
                        "scenario": scenario,
                        "metric": f"delta_{metric}",
                        "paired_unit": "seed",
                        "paired_seeds": len(values),
                        "iterations": iterations,
                        "seed": seed,
                        "observed_mean": float(values.mean()),
                        "ci_low": float(np.percentile(means, alpha)),
                        "ci_high": float(np.percentile(means, 100.0 - alpha)),
                    }
                )
    return rows


def add_class_deltas(rows: list[dict[str, object]]) -> None:
    for row in rows:
        for method in ("r1", "r2", "r3"):
            row[f"{method}_minus_r0_known_f1"] = float(row[f"{method}_known_test_f1"]) - float(
                row["r0_known_test_f1"]
            )
            row[f"r0_minus_{method}_absorption_rate"] = float(
                row["r0_unknown_absorption_rate_all_unknown"]
            ) - float(row[f"{method}_unknown_absorption_rate_all_unknown"])


def correlation_row(
    scope: str,
    scenario: str,
    level: str,
    outcome: str,
    x: list[float],
    y: list[float],
) -> dict[str, object]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        rho, pvalue = float("nan"), float("nan")
    else:
        result = spearmanr(x, y)
        rho, pvalue = float(result.statistic), float(result.pvalue)
    return {
        "scope": scope,
        "scenario": scenario,
        "level": level,
        "predictor": "frozen_B0_normalized_prototype_gap",
        "outcome": outcome,
        "n": len(x),
        "spearman_rho": rho,
        "p_value": pvalue,
        "used_for_tuning": False,
    }


def gap_predictiveness(
    results: list[dict[str, object]],
    comparison_rows: dict[str, list[dict[str, object]]],
    class_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    gap_by_run = {
        (r["scenario"], int(r["fold"]), int(r["seed"])): float(
            r["evaluation"]["prototype_alignment"]["normalized_gap_mean"]
        )
        for r in results
    }
    rows: list[dict[str, object]] = []
    for scenario in ("a1", "a2", "a3", "all"):
        for comparison in ("R1-R0", "R2-R0"):
            source = comparison_rows[comparison]
            if scenario != "all":
                source = [row for row in source if row["scenario"] == scenario]
            x = [gap_by_run[(row["scenario"], int(row["fold"]), int(row["seed"]))] for row in source]
            y = [float(row["delta_auroc"]) for row in source]
            rows.append(correlation_row("scenario" if scenario != "all" else "pooled", scenario, "run", f"{comparison}_delta_auroc", x, y))
    for scenario in ("a1", "a2", "a3", "all"):
        source = class_rows if scenario == "all" else [row for row in class_rows if row["scenario"] == scenario]
        x = [float(row["normalized_gap"]) for row in source]
        for outcome in (
            "r1_minus_r0_known_f1",
            "r2_minus_r0_known_f1",
            "r0_minus_r1_absorption_rate",
            "r0_minus_r2_absorption_rate",
        ):
            y = [float(row[outcome]) for row in source]
            rows.append(correlation_row("scenario" if scenario != "all" else "pooled", scenario, "class", outcome, x, y))
    return rows


def absorption_hubs(absorption_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in absorption_rows:
        groups[(row["scenario"], int(row["fold"]), int(row["seed"]), row["method"])].append(row)
    run_rows: list[dict[str, object]] = []
    for (scenario, fold, seed, method), source in sorted(groups.items()):
        counts: dict[int, int] = defaultdict(int)
        for row in source:
            counts[int(row["predicted_known_class"])] += int(row["accepted_count"])
        total = sum(counts.values())
        top_class, top_count = max(counts.items(), key=lambda item: item[1])
        run_rows.append(
            {
                "row_type": "run",
                "scenario": scenario,
                "fold": fold,
                "seed": seed,
                "method": method,
                "accepted_unknown_total": total,
                "top_absorption_hub_original_class": top_class,
                "top_absorption_hub_count": top_count,
                "top_absorption_hub_share": top_count / total if total else 0.0,
                "active_absorption_hubs": sum(value > 0 for value in counts.values()),
            }
        )
    summary: list[dict[str, object]] = []
    for scenario in ("a1", "a2", "a3"):
        for method in METHODS:
            source = [row for row in run_rows if row["scenario"] == scenario and row["method"] == method]
            summary.append(
                {
                    "row_type": "scenario_summary",
                    "scenario": scenario,
                    "fold": "",
                    "seed": "",
                    "method": method,
                    "accepted_unknown_total": float(np.mean([row["accepted_unknown_total"] for row in source])),
                    "top_absorption_hub_original_class": "",
                    "top_absorption_hub_count": float(np.mean([row["top_absorption_hub_count"] for row in source])),
                    "top_absorption_hub_share": float(np.mean([row["top_absorption_hub_share"] for row in source])),
                    "active_absorption_hubs": float(np.mean([row["active_absorption_hubs"] for row in source])),
                }
            )
    return run_rows + summary


def main() -> None:
    config = read_json(CONFIG_PATH)
    results: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    absorption_rows: list[dict[str, object]] = []
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            if not (run_dir / "SUCCESS").is_file():
                raise RuntimeError(f"Incomplete formal run: {run_dir}")
            result = read_json(run_dir / "results.json")
            if result["representation_parity"]["status"] != "PASS":
                raise RuntimeError(f"Representation parity failed: {run_dir}")
            if result["frozen_encoder"]["new_encoder_training"]:
                raise RuntimeError(f"Unexpected training flag: {run_dir}")
            results.append(result)
            import csv

            with (run_dir / "per_class_support_analysis.csv").open(encoding="utf-8") as handle:
                class_rows.extend(list(csv.DictReader(handle)))
            with (run_dir / "unknown_absorption_analysis.csv").open(encoding="utf-8") as handle:
                absorption_rows.extend(list(csv.DictReader(handle)))
    if len(results) != 15:
        raise AssertionError("Expected exactly 15 formal frozen evaluations")
    add_class_deltas(class_rows)
    run_rows = [detector_row(result, method) for result in results for method in METHODS]
    comparison_rows = {name: paired_rows(run_rows, name) for name in COMPARISONS}
    bootstrap = bootstrap_rows(comparison_rows, config)
    correlations = gap_predictiveness(results, comparison_rows, class_rows)
    hub_rows = absorption_hubs(absorption_rows)

    SUMMARY.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY / "run_level_results.csv", run_rows)
    write_csv(SUMMARY / "primary_r2_vs_r0.csv", comparison_rows["R2-R0"])
    write_csv(SUMMARY / "r1_vs_r0.csv", comparison_rows["R1-R0"])
    write_csv(SUMMARY / "r2_vs_r1.csv", comparison_rows["R2-R1"])
    write_csv(SUMMARY / "r3_vs_r2.csv", comparison_rows["R3-R2"])
    write_csv(SUMMARY / "per_class_support_analysis.csv", class_rows)
    write_csv(SUMMARY / "unknown_absorption_analysis.csv", absorption_rows)
    write_csv(SUMMARY / "unknown_absorption_hub_summary.csv", hub_rows)
    write_csv(SUMMARY / "bootstrap_summary.csv", bootstrap)
    write_csv(SUMMARY / "gap_predictiveness.csv", correlations)

    scenario_summary: list[dict[str, object]] = []
    for scenario in ("a1", "a2", "a3"):
        for method in METHODS:
            source = [row for row in run_rows if row["scenario"] == scenario and row["method"] == method]
            for metric in METRICS:
                scenario_summary.append(
                    {
                        "row_type": "method_metric",
                        "scenario": scenario,
                        "method": method,
                        "comparison": "",
                        "metric": metric,
                        **stats([float(row[metric]) for row in source]),
                        "positive_count": "",
                        "negative_count": "",
                    }
                )
        for comparison, rows in comparison_rows.items():
            source = [row for row in rows if row["scenario"] == scenario]
            for metric in METRICS:
                values = [float(row[f"delta_{metric}"]) for row in source]
                scenario_summary.append(
                    {
                        "row_type": "paired_delta",
                        "scenario": scenario,
                        "method": "",
                        "comparison": comparison,
                        "metric": f"delta_{metric}",
                        **stats(values),
                        "positive_count": sum(value > 0 for value in values),
                        "negative_count": sum(value < 0 for value in values),
                    }
                )
    write_csv(SUMMARY / "scenario_summary.csv", scenario_summary)

    def delta_mean(comparison: str, scenario: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    row[f"delta_{metric}"]
                    for row in comparison_rows[comparison]
                    if row["scenario"] == scenario
                ]
            )
        )

    def better_count(comparison: str, scenario: str, metric: str = "auroc") -> int:
        return sum(
            float(row[f"delta_{metric}"]) > 0
            for row in comparison_rows[comparison]
            if row["scenario"] == scenario
        )

    scenarios = ("a1", "a2", "a3")
    gate_cfg = config["gate"]
    r2_deltas = {s: delta_mean("R2-R0", s, "auroc") for s in scenarios}
    r1_deltas = {s: delta_mean("R1-R0", s, "auroc") for s in scenarios}
    r2_r1_deltas = {s: delta_mean("R2-R1", s, "auroc") for s in scenarios}
    r3_r2_deltas = {s: delta_mean("R3-R2", s, "auroc") for s in scenarios}
    r2_better = {s: better_count("R2-R0", s) for s in scenarios}
    r1_better = {s: better_count("R1-R0", s) for s in scenarios}
    r2_r1_better = {s: better_count("R2-R1", s) for s in scenarios}
    r3_r2_better = {s: better_count("R3-R2", s) for s in scenarios}
    r2_nonnegative_all = all(value >= 0 for value in r2_deltas.values())
    r2_material_count = sum(value >= float(gate_cfg["material_auroc_delta"]) for value in r2_deltas.values())
    no_clear_harm = all(value > float(gate_cfg["clear_harm_auroc_delta"]) for value in r2_deltas.values())
    known_f1_deltas = {s: delta_mean("R2-R0", s, "known_macro_f1") for s in scenarios}
    no_known_f1_harm = all(value > float(gate_cfg["known_macro_f1_drop"]) for value in known_f1_deltas.values())
    known_frr_deltas = {s: delta_mean("R2-R0", s, "known_frr") for s in scenarios}
    no_frr_cost = all(value < float(gate_cfg["known_frr_increase"]) for value in known_frr_deltas.values())
    majority_r2_all = all(value >= int(gate_cfg["majority_seed_count"]) for value in r2_better.values())
    if r2_nonnegative_all and r2_material_count >= 2 and no_known_f1_harm and no_frr_cost and majority_r2_all:
        final_gate = "GO"
    elif r2_material_count >= 2 and no_clear_harm and no_known_f1_harm and no_frr_cost:
        final_gate = "CONDITIONAL_GO"
    else:
        final_gate = "NO_GO"

    r1_stable = all(r1_deltas[s] > 0 and r1_better[s] >= 3 for s in scenarios)
    r2_stable = all(r2_deltas[s] > 0 and r2_better[s] >= 3 for s in scenarios)
    covariance_supported = (
        all(r2_r1_deltas[s] > 0 and r2_r1_better[s] >= 3 for s in scenarios)
        and sum(value >= float(gate_cfg["material_auroc_delta"]) for value in r2_r1_deltas.values()) >= 2
    )
    k2_material = (
        all(r3_r2_deltas[s] > 0 and r3_r2_better[s] >= 3 for s in scenarios)
        and sum(value >= float(gate_cfg["material_auroc_delta"]) for value in r3_r2_deltas.values()) >= 2
    )
    if k2_material and r2_stable:
        mechanism = "D4"
    elif r2_stable and covariance_supported and not r1_stable:
        mechanism = "D3"
    elif r1_stable and sum(value >= 0.02 for value in r2_r1_deltas.values()) == 0:
        mechanism = "D2"
    elif r1_stable or r2_stable:
        mechanism = "D1"
    elif all(value <= 0 for value in r1_deltas.values()) and all(value <= 0 for value in r2_deltas.values()):
        mechanism = "D6"
    else:
        mechanism = "D5"
    covariance_verdict = (
        "COVARIANCE_AWARE_SUPPORT_SUPPORTED"
        if covariance_supported
        else "COVARIANCE_BENEFIT_CONTEXT_DEPENDENT"
    )
    multicomponent_verdict = (
        "MULTICOMPONENT_REQUIRED"
        if k2_material
        else "MULTICOMPONENT_SECONDARY"
        if max(r3_r2_deltas.values()) < 0.02
        else "MULTICOMPONENT_GAIN_CONTEXT_DEPENDENT"
    )

    hub_summary = [row for row in hub_rows if row["row_type"] == "scenario_summary"]
    hub_index = {(row["scenario"], row["method"]): row for row in hub_summary}
    hub_reduction = {
        method: {
            scenario: {
                "accepted_unknown_delta": float(hub_index[(scenario, method)]["accepted_unknown_total"])
                - float(hub_index[(scenario, "R0")]["accepted_unknown_total"]),
                "top_hub_count_delta": float(hub_index[(scenario, method)]["top_absorption_hub_count"])
                - float(hub_index[(scenario, "R0")]["top_absorption_hub_count"]),
                "top_hub_share_delta": float(hub_index[(scenario, method)]["top_absorption_hub_share"])
                - float(hub_index[(scenario, "R0")]["top_absorption_hub_share"]),
            }
            for scenario in scenarios
        }
        for method in ("R1", "R2")
    }
    result = {
        "final_gate": final_gate,
        "mechanism": mechanism,
        "r1_stable_over_r0": r1_stable,
        "r2_stable_over_r0": r2_stable,
        "covariance_verdict": covariance_verdict,
        "multicomponent_verdict": multicomponent_verdict,
        "mean_delta_auroc": {
            "R1-R0": r1_deltas,
            "R2-R0": r2_deltas,
            "R2-R1": r2_r1_deltas,
            "R3-R2": r3_r2_deltas,
        },
        "better_seed_counts": {
            "R1-R0": r1_better,
            "R2-R0": r2_better,
            "R2-R1": r2_r1_better,
            "R3-R2": r3_r2_better,
        },
        "r2_minus_r0_known_macro_f1": known_f1_deltas,
        "r2_minus_r0_known_frr": known_frr_deltas,
        "gate_conditions": {
            "r2_nonnegative_all_scenarios": r2_nonnegative_all,
            "r2_material_scenario_count": r2_material_count,
            "no_clear_auroc_harm": no_clear_harm,
            "no_known_macro_f1_harm": no_known_f1_harm,
            "no_known_frr_cost": no_frr_cost,
            "majority_r2_better_each_scenario": majority_r2_all,
        },
        "absorption_hub_change_vs_r0": hub_reduction,
        "new_encoder_training": False,
        "cipherspectrum_stage9_sample_level_test_used": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
    }
    write_json(SUMMARY / "final_gate.json", result)
    lines = [
        "# Stage 11B Final Gate",
        "",
        "## Decision",
        "",
        f"**{final_gate}**",
        "",
        "## Primary R2 versus R0",
        "",
        "| Scenario | Mean ΔAUROC | R2 better seeds | Mean ΔKnown Macro-F1 | Mean ΔKnown FRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in scenarios:
        lines.append(
            f"| {scenario.upper()} | {r2_deltas[scenario]:.6f} | {r2_better[scenario]}/5 | "
            f"{known_f1_deltas[scenario]:.6f} | {known_frr_deltas[scenario]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed interpretations",
            "",
            f"- Mechanism: **{mechanism}**.",
            f"- R1 stable over R0: `{r1_stable}`.",
            f"- R2 stable over R0: `{r2_stable}`.",
            f"- Covariance: **{covariance_verdict}**.",
            f"- K2: **{multicomponent_verdict}**.",
            "",
            "## Scope",
            "",
            "`DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. No encoder training, "
            "no CipherSpectrum Stage9 sample-level Test access, and no next stage.",
        ]
    )
    (SUMMARY / "final_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mechanism_text = (
        "# Stage 11B Mechanism\n\n"
        f"- Case: **{mechanism}**\n"
        f"- R1 stable over R0: `{r1_stable}`\n"
        f"- R2 stable over R0: `{r2_stable}`\n"
        f"- Covariance verdict: **{covariance_verdict}**\n"
        f"- Multi-component verdict: **{multicomponent_verdict}**\n\n"
        "This is a USTC development diagnosis, not independent external validation.\n"
    )
    (SUMMARY / "mechanism_case.md").write_text(mechanism_text, encoding="utf-8")

    for scenario in scenarios:
        destination = STAGE_ROOT / "outputs" / scenario
        write_csv(destination / "run_level_results.csv", [row for row in run_rows if row["scenario"] == scenario])
        write_csv(destination / "per_class_support_analysis.csv", [row for row in class_rows if row["scenario"] == scenario])
        write_csv(destination / "unknown_absorption_analysis.csv", [row for row in absorption_rows if row["scenario"] == scenario])
        for filename, comparison in (
            ("primary_r2_vs_r0.csv", "R2-R0"),
            ("r1_vs_r0.csv", "R1-R0"),
            ("r2_vs_r1.csv", "R2-R1"),
            ("r3_vs_r2.csv", "R3-R2"),
        ):
            write_csv(destination / filename, [row for row in comparison_rows[comparison] if row["scenario"] == scenario])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
