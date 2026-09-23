#!/usr/bin/env python3
"""Aggregate all paired Stage 11A B0/DAP results and apply the fixed gate."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage11a_common import CONFIG_PATH, STAGE_ROOT, read_json, write_csv, write_json


SUMMARY = STAGE_ROOT / "outputs" / "summary"


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def result_path(method: str, scenario: str, fold: int, seed: int) -> Path:
    suffix = f"fold{fold}_seed{seed}"
    if method == "DAP":
        return STAGE_ROOT / "runs" / scenario / suffix / "results.json"
    return STAGE_ROOT / "artifacts" / "b0" / scenario / suffix / "results.json"


def require_complete(config: dict) -> dict[tuple[str, int, str], dict]:
    results: dict[tuple[str, int, str], dict] = {}
    missing: list[str] = []
    for scenario in config["scenarios"]:
        for fold, seed in enumerate(config["seeds"]):
            for method in ("B0", "DAP"):
                path = result_path(method, scenario, fold, seed)
                if not path.is_file():
                    missing.append(str(path))
                else:
                    results[(scenario, seed, method)] = read_json(path)
    if missing:
        raise FileNotFoundError("Incomplete Stage11A matrix:\n" + "\n".join(missing))
    return results


def flat_detection_row(result: dict, detector: str) -> dict[str, object]:
    if detector == "Native":
        method_result = result["native"]
    else:
        method_result = result["density_diagnostics"]["methods"][detector]
    detection = method_result["detection"]
    return {
        "evaluation_label": "DEVELOPMENT_RESULT",
        "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
        "scenario": result["scenario"],
        "fold": result["fold"],
        "seed": result["seed"],
        "training_method": result["method"],
        "detector": detector,
        "threshold": detection["threshold"],
        "threshold_source": detection["threshold_source"],
        "accuracy": detection["accuracy"],
        "binary_f1": detection["binary_f1"],
        "auroc": detection["auroc"],
        "auprc": detection["auprc"],
        "known_frr": detection["known_frr"],
        "ufar": detection["ufar"],
        "validation_known_acceptance": detection["validation_known_acceptance"],
        "sample_count_known": detection["sample_count_known"],
        "sample_count_unknown": detection["sample_count_unknown"],
        "validation_accuracy": method_result["validation_known"]["accuracy"],
        "validation_macro_f1": method_result["validation_known"]["macro_f1"],
        "known_test_accuracy": method_result["known_test"]["accuracy"],
        "known_test_macro_f1": method_result["known_test"]["macro_f1"],
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    config = read_json(CONFIG_PATH)
    results = require_complete(config)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for scenario in config["scenarios"]:
        for fold, seed in enumerate(config["seeds"]):
            for method in ("B0", "DAP"):
                result = results[(scenario, seed, method)]
                if method == "DAP":
                    training = result["training"]
                    best_epoch = training["best_epoch"]
                    stop_epoch = training["stop_epoch"]
                    best_val_accuracy = training["best_validation_accuracy"]
                    prototype_in_optimizer = training["prototype_in_optimizer"]
                    prototype_update = training["prototype_update"]
                    unknown_for_selection = training["unknown_used_for_training_or_selection"]
                else:
                    source = result["frozen_b0"]
                    frozen_metrics = read_json(Path(source["source_run"]) / "test_metrics.json")
                    best_epoch = source["best_epoch"]
                    stop_epoch = frozen_metrics["training"]["completed_epochs"]
                    best_val_accuracy = source["validation_accuracy"]
                    prototype_in_optimizer = True
                    prototype_update = "frozen v6 corrected reset logic"
                    unknown_for_selection = False
                training_rows.append({
                    "scenario": scenario,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "best_epoch": best_epoch,
                    "stop_epoch": stop_epoch,
                    "best_validation_accuracy": best_val_accuracy,
                    "frozen_checkpoint_validation_accuracy_recomputed": result["native"]["validation_known"]["accuracy"],
                    "frozen_checkpoint_validation_macro_f1": result["native"]["validation_known"]["macro_f1"],
                    "known_test_accuracy": result["native"]["known_test"]["accuracy"],
                    "known_test_macro_f1": result["native"]["known_test"]["macro_f1"],
                    "prototype_in_optimizer": prototype_in_optimizer,
                    "prototype_update": prototype_update,
                    "unknown_used_for_training_or_checkpoint_selection": unknown_for_selection,
                    "checkpoint_sha256": result.get("checkpoint_sha256", result.get("frozen_b0", {}).get("checkpoint_sha256")),
                })
                for class_row in result["prototype_alignment_by_class"]:
                    alignment_rows.append({
                        "row_type": "class",
                        "scenario": scenario,
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        **class_row,
                        "normalized_gap_mean": "",
                        "normalized_gap_median": "",
                        "normalized_gap_max": "",
                    })
                alignment_rows.append({
                    "row_type": "run_summary",
                    "scenario": scenario,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "class_index": "",
                    "original_class_id": "",
                    "train_samples": result["sample_counts"]["train_known"],
                    "gap": "",
                    "median_within_class_radius": "",
                    "normalized_gap": "",
                    **result["prototype_alignment"],
                    "centroid_source": "KNOWN_TRAIN_DETERMINISTIC_MU_ONLY",
                })
                native_rows.append(flat_detection_row(result, "Native"))
                for detector in ("K1", "K2"):
                    row = flat_detection_row(result, detector)
                    density_method = result["density_diagnostics"]["methods"][detector]
                    row.update({
                        "representation": result["density_diagnostics"]["representation"],
                        "pca_explained_variance": result["density_diagnostics"]["pca_explained_variance"],
                        "components": density_method["components"],
                        "covariance_type": density_method["covariance_type"],
                        "reg_covar": density_method["reg_covar"],
                        "n_init": density_method["n_init"],
                        "max_iter": density_method["max_iter"],
                        "random_state": density_method["random_state"],
                        "all_converged": density_method["all_converged"],
                        "warning_count": sum(
                            1 for warning in result["density_diagnostics"]["warnings"]
                            if warning["method"] == detector
                        ),
                    })
                    density_rows.append(row)

            b0 = results[(scenario, seed, "B0")]
            dap = results[(scenario, seed, "DAP")]
            b0_native = b0["native"]
            dap_native = dap["native"]
            paired_rows.append({
                "scenario": scenario,
                "fold": fold,
                "seed": seed,
                "delta_auroc": dap_native["detection"]["auroc"] - b0_native["detection"]["auroc"],
                "delta_binary_f1": dap_native["detection"]["binary_f1"] - b0_native["detection"]["binary_f1"],
                "delta_ufar": dap_native["detection"]["ufar"] - b0_native["detection"]["ufar"],
                "delta_known_frr": dap_native["detection"]["known_frr"] - b0_native["detection"]["known_frr"],
                "delta_known_test_accuracy": dap_native["known_test"]["accuracy"] - b0_native["known_test"]["accuracy"],
                "delta_known_test_macro_f1": dap_native["known_test"]["macro_f1"] - b0_native["known_test"]["macro_f1"],
                "delta_normalized_prototype_gap_mean": dap["prototype_alignment"]["normalized_gap_mean"] - b0["prototype_alignment"]["normalized_gap_mean"],
                "dap_better_auroc": dap_native["detection"]["auroc"] > b0_native["detection"]["auroc"],
                "dap_lower_prototype_gap": dap["prototype_alignment"]["normalized_gap_mean"] < b0["prototype_alignment"]["normalized_gap_mean"],
                "dap_k1_minus_native_auroc": dap["density_diagnostics"]["methods"]["K1"]["detection"]["auroc"] - dap_native["detection"]["auroc"],
                "dap_k2_minus_k1_auroc": dap["density_diagnostics"]["methods"]["K2"]["detection"]["auroc"] - dap["density_diagnostics"]["methods"]["K1"]["detection"]["auroc"],
            })

    scenario_rows: list[dict[str, object]] = []
    for scenario in config["scenarios"]:
        scenario_pairs = [row for row in paired_rows if row["scenario"] == scenario]
        for metric in (
            "delta_auroc",
            "delta_binary_f1",
            "delta_ufar",
            "delta_known_frr",
            "delta_known_test_accuracy",
            "delta_known_test_macro_f1",
            "delta_normalized_prototype_gap_mean",
            "dap_k1_minus_native_auroc",
            "dap_k2_minus_k1_auroc",
        ):
            values = [float(row[metric]) for row in scenario_pairs]
            scenario_rows.append({
                "row_type": "paired_delta",
                "scenario": scenario,
                "training_method": "DAP-B0" if not metric.startswith("dap_k") else "DAP",
                "detector": "Native" if not metric.startswith("dap_k") else "K1_K2_GAP",
                "metric": metric,
                **stats(values),
                "positive_count": sum(value > 0 for value in values),
                "negative_count": sum(value < 0 for value in values),
                "zero_count": sum(value == 0 for value in values),
            })
        for method in ("B0", "DAP"):
            method_result = [results[(scenario, seed, method)] for seed in config["seeds"]]
            for detector in ("Native", "K1", "K2"):
                for metric in ("auroc", "binary_f1", "ufar", "known_frr"):
                    if detector == "Native":
                        values = [item["native"]["detection"][metric] for item in method_result]
                    else:
                        values = [item["density_diagnostics"]["methods"][detector]["detection"][metric] for item in method_result]
                    scenario_rows.append({
                        "row_type": "method_metric",
                        "scenario": scenario,
                        "training_method": method,
                        "detector": detector,
                        "metric": metric,
                        **stats(list(map(float, values))),
                        "positive_count": "",
                        "negative_count": "",
                        "zero_count": "",
                    })
            for metric, path in (
                ("known_test_accuracy", ("native", "known_test", "accuracy")),
                ("known_test_macro_f1", ("native", "known_test", "macro_f1")),
                ("normalized_prototype_gap", ("prototype_alignment", "normalized_gap_mean")),
            ):
                values: list[float] = []
                for item in method_result:
                    value: object = item
                    for key in path:
                        value = value[key]
                    values.append(float(value))
                scenario_rows.append({
                    "row_type": "method_metric",
                    "scenario": scenario,
                    "training_method": method,
                    "detector": "Native",
                    "metric": metric,
                    **stats(values),
                    "positive_count": "",
                    "negative_count": "",
                    "zero_count": "",
                })

        for method in ("B0", "DAP"):
            class_gaps = [
                float(row["normalized_gap"])
                for row in alignment_rows
                if row["row_type"] == "class" and row["scenario"] == scenario and row["method"] == method
            ]
            gap_stats = stats(class_gaps)
            alignment_rows.append({
                "row_type": "scenario_all_seed_class_summary",
                "scenario": scenario,
                "fold": "",
                "seed": "",
                "method": method,
                "class_index": "",
                "original_class_id": "",
                "train_samples": "",
                "gap": "",
                "median_within_class_radius": "",
                "normalized_gap": "",
                "normalized_gap_mean": gap_stats["mean"],
                "normalized_gap_median": gap_stats["median"],
                "normalized_gap_max": gap_stats["max"],
                "centroid_source": "KNOWN_TRAIN_DETERMINISTIC_MU_ONLY",
            })

    write_csv(SUMMARY / "training_summary.csv", training_rows)
    write_csv(SUMMARY / "prototype_alignment.csv", alignment_rows)
    write_csv(SUMMARY / "native_detection_results.csv", native_rows)
    write_csv(SUMMARY / "k1_k2_diagnostics.csv", density_rows)
    write_csv(SUMMARY / "paired_seed_comparison.csv", paired_rows)
    write_csv(SUMMARY / "scenario_summary.csv", scenario_rows)
    for scenario in config["scenarios"]:
        output = STAGE_ROOT / "outputs" / scenario
        write_csv(output / "native_detection_results.csv", [row for row in native_rows if row["scenario"] == scenario])
        write_csv(output / "k1_k2_diagnostics.csv", [row for row in density_rows if row["scenario"] == scenario])
        write_csv(output / "paired_seed_comparison.csv", [row for row in paired_rows if row["scenario"] == scenario])
        write_csv(output / "prototype_alignment.csv", [row for row in alignment_rows if row["scenario"] == scenario])

    scenario_delta = {
        scenario: stats([float(row["delta_auroc"]) for row in paired_rows if row["scenario"] == scenario])
        for scenario in config["scenarios"]
    }
    auroc_better_counts = {
        scenario: sum(bool(row["dap_better_auroc"]) for row in paired_rows if row["scenario"] == scenario)
        for scenario in config["scenarios"]
    }
    gap_lower_counts = {
        scenario: sum(bool(row["dap_lower_prototype_gap"]) for row in paired_rows if row["scenario"] == scenario)
        for scenario in config["scenarios"]
    }
    mean_macro_delta = {
        scenario: stats([float(row["delta_known_test_macro_f1"]) for row in paired_rows if row["scenario"] == scenario])["mean"]
        for scenario in config["scenarios"]
    }
    mean_frr_delta = {
        scenario: stats([float(row["delta_known_frr"]) for row in paired_rows if row["scenario"] == scenario])["mean"]
        for scenario in config["scenarios"]
    }
    alignment_clear = all(count == 5 for count in gap_lower_counts.values())
    all_nonnegative = all(value["mean"] >= 0 for value in scenario_delta.values())
    material_count = sum(value["mean"] >= config["material_auroc_delta"] for value in scenario_delta.values())
    known_macro_stable = all(value > -0.02 for value in mean_macro_delta.values())
    no_frr_cost = all(value < 0.02 for value in mean_frr_delta.values())
    direction_stable = all(count >= 4 for count in auroc_better_counts.values())
    if alignment_clear and all_nonnegative and material_count >= 2 and known_macro_stable and no_frr_cost and direction_stable:
        gate = "GO"
    elif (
        not alignment_clear
        or any(value["mean"] < -config["material_auroc_delta"] for value in scenario_delta.values())
        or not known_macro_stable
    ):
        gate = "NO_GO"
    else:
        gate = "CONDITIONAL_GO"

    overall_delta = float(np.mean([row["delta_auroc"] for row in paired_rows]))
    if not alignment_clear:
        mechanism = "P4"
    elif overall_delta >= config["material_auroc_delta"] and material_count >= 2:
        mechanism = "P1"
    elif overall_delta <= -config["material_auroc_delta"]:
        mechanism = "P3"
    else:
        mechanism = "P2"

    k1_gap = {
        scenario: stats([float(row["dap_k1_minus_native_auroc"]) for row in paired_rows if row["scenario"] == scenario])
        for scenario in config["scenarios"]
    }
    k2_gap = {
        scenario: stats([float(row["dap_k2_minus_k1_auroc"]) for row in paired_rows if row["scenario"] == scenario])
        for scenario in config["scenarios"]
    }
    covariance_required = all(value["mean"] > config["covariance_required_delta"] for value in k1_gap.values())
    multicomponent_secondary = all(value["mean"] < config["multicomponent_secondary_delta"] for value in k2_gap.values())
    mechanism_text = f"""# Stage 11A Mechanism Case

- Case: **{mechanism}**
- Prototype gap decreased in all five paired seeds per scenario: **{alignment_clear}** (`{gap_lower_counts}`).
- Mean paired Native AUROC delta over all 15 runs: **{overall_delta:.6f}**.
- Scenario mean AUROC deltas: {', '.join(f'{key.upper()}={value["mean"]:.6f}' for key, value in scenario_delta.items())}.
- Interpretation: **{'prototype mismatch has prospective causal support' if mechanism == 'P1' else 'hard anchoring alone is insufficient or harmful' if mechanism in {'P2', 'P3'} else 'DAP alignment mechanism failed'}**.
- Covariance verdict: **{'COVARIANCE_MODELING_STILL_REQUIRED' if covariance_required else 'COVARIANCE_REQUIREMENT_NOT_STABLE'}**.
- Multi-component verdict: **{'MULTICOMPONENT_SECONDARY' if multicomponent_secondary else 'K2_GAIN_NOT_SECONDARY'}**.

All conclusions are `DEVELOPMENT_RESULT` and `NOT_INDEPENDENT_EXTERNAL_VALIDATION`.
"""
    (SUMMARY / "mechanism_case.md").write_text(mechanism_text, encoding="utf-8")

    gate_payload = {
        "final_gate": gate,
        "mechanism_case": mechanism,
        "conditions": {
            "alignment_clear_5of5_each_scenario": alignment_clear,
            "scenario_mean_auroc_all_nonnegative": all_nonnegative,
            "scenario_material_gain_count": material_count,
            "known_macro_f1_stable_no_mean_drop_ge_0.02": known_macro_stable,
            "no_mean_known_frr_cost_ge_0.02": no_frr_cost,
            "auroc_direction_stable_at_least_4of5_each_scenario": direction_stable,
        },
        "scenario_delta_auroc": scenario_delta,
        "auroc_better_counts": auroc_better_counts,
        "prototype_gap_lower_counts": gap_lower_counts,
        "mean_known_macro_f1_delta": mean_macro_delta,
        "mean_known_frr_delta": mean_frr_delta,
        "dap_k1_minus_native_auroc": k1_gap,
        "dap_k2_minus_k1_auroc": k2_gap,
        "covariance_verdict": "COVARIANCE_MODELING_STILL_REQUIRED" if covariance_required else "COVARIANCE_REQUIREMENT_NOT_STABLE",
        "multicomponent_verdict": "MULTICOMPONENT_SECONDARY" if multicomponent_secondary else "K2_GAIN_NOT_SECONDARY",
        "cipherspectrum_stage9_test_used_for_development": False,
        "next_stage_started": False,
    }
    write_json(SUMMARY / "final_gate.json", gate_payload)
    gate_text = f"""# Stage 11A Final Gate

## Decision

**{gate}**

## Primary evidence

| Scenario | Mean ΔAUROC | Median ΔAUROC | DAP better seeds | Gap-lower seeds | Mean ΔKnown Macro-F1 | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|---:|---:|
"""
    for scenario in config["scenarios"]:
        gate_text += (
            f"| {scenario.upper()} | {scenario_delta[scenario]['mean']:.6f} | "
            f"{scenario_delta[scenario]['median']:.6f} | {auroc_better_counts[scenario]}/5 | "
            f"{gap_lower_counts[scenario]}/5 | {mean_macro_delta[scenario]:.6f} | "
            f"{mean_frr_delta[scenario]:.6f} |\n"
        )
    gate_text += f"""

## Fixed gate checks

- Prototype alignment decreases 5/5 in every scenario: `{alignment_clear}`.
- DAP-Native scenario mean AUROC is non-negative versus B0 in all scenarios: `{all_nonnegative}`.
- Scenarios meeting material `+0.02` AUROC: `{material_count}`.
- Known Test Macro-F1 has no scenario mean drop of 0.02 or more: `{known_macro_stable}`.
- Mean Known FRR does not rise by 0.02 or more: `{no_frr_cost}`.
- DAP AUROC improves in at least 4/5 paired seeds per scenario: `{direction_stable}`.

## Covariance diagnostics

- DAP K1 minus Native mean AUROC: {', '.join(f'{key.upper()}={value["mean"]:.6f}' for key, value in k1_gap.items())}.
- Verdict: **{gate_payload['covariance_verdict']}**.
- DAP K2 minus K1 mean AUROC: {', '.join(f'{key.upper()}={value["mean"]:.6f}' for key, value in k2_gap.items())}.
- Verdict: **{gate_payload['multicomponent_verdict']}**.

## Scope boundary

`DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. CipherSpectrum Stage 9 sample-level Test data were not opened, read, or rescored. No next stage was started.
"""
    (SUMMARY / "final_gate.md").write_text(gate_text, encoding="utf-8")

    readme = (STAGE_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "\n## Recorded Final Results\n"
    if marker in readme:
        readme = readme.split(marker, 1)[0]
    readme += marker + "\n" + gate_text.replace("# Stage 11A Final Gate\n", "")
    (STAGE_ROOT / "README.md").write_text(readme, encoding="utf-8")

    results_md = f"""# Experiment results: stage11a-data-anchored-prototype-20260914-v1

- Status: `success`
- Experiment type: `method-development`
- Claim scope: `diagnostic`
- Objective: Paired USTC-only feasibility study of every-epoch data-anchored Open-Detect prototypes.

## Data and split

- Frozen v6 USTC A1/A2/A3, five paired seeds 2022–2026.
- Repeated grouped-image-disjoint 8:1:1 local splits; not author-exact folds.
- Known Train only for anchoring and density fitting; Known Validation only for checkpoint selection and P95 thresholds.

## Configuration and execution

- B0 reused 15 frozen checkpoints; DAP trained 15 paired runs.
- DAP prototypes are optimizer-free registered buffers and are anchored before epoch 1 and after every epoch.
- Native and fixed K1/K2 diagnostics only; no threshold or K search.

## Core results

- Final gate: **{gate}**; mechanism: **{mechanism}**.
- Scenario mean DAP-Native minus B0-Native AUROC: {', '.join(f'{key.upper()}={value["mean"]:.6f}' for key, value in scenario_delta.items())}.
- DAP-better directions: {', '.join(f'{key.upper()}={value}/5' for key, value in auroc_better_counts.items())}.
- Covariance verdict: **{gate_payload['covariance_verdict']}**.
- Multi-component verdict: **{gate_payload['multicomponent_verdict']}**.

## Preserved evidence

- `outputs/summary/*.csv`, `mechanism_case.md`, `final_gate.md`, and `provenance_verification.json`.
- Per-run configs, logs, drift history, frozen best checkpoints/hashes, results, density models, RESULTS, and manifests.

## Limitations

- `DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`.
- CipherSpectrum Stage 9 sample-level Test was not used. USTC scenarios are development scenarios.

## Conclusion and next step

Stage 11A terminates at **{gate}**. No subsequent stage or final covariance-aware method was started.
"""
    (STAGE_ROOT / "RESULTS.md").write_text(results_md, encoding="utf-8")
    print(json.dumps(gate_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
