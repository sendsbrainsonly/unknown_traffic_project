#!/usr/bin/env python3
"""Aggregate all three primary settings and apply the predeclared Stage 3 Gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, STAGE3_ROOT, VALID_SETTINGS, load_fold, verify_frozen_inputs


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_map(setting: str) -> dict[str, dict[str, float]]:
    rows = read_csv(STAGE3_ROOT / "outputs" / setting / "detector_metrics.csv")
    return {
        row["detector"]: {
            key: float(value)
            for key, value in row.items()
            if key not in {"setting", "detector", "score_semantics"}
        }
        for row in rows
    }


def ci_map(setting: str) -> dict[str, dict[str, float]]:
    rows = read_csv(STAGE3_ROOT / "outputs" / setting / "paired_bootstrap_ci.csv")
    return {
        row["metric"]: {
            "lower": float(row["ci_2_5pct"]),
            "upper": float(row["ci_97_5pct"]),
            "mean": float(row["bootstrap_mean"]),
        }
        for row in rows
    }


def determine_gate(rows: list[dict[str, object]], cis: dict[str, dict[str, dict[str, float]]]) -> tuple[str, str]:
    deltas = np.asarray([float(row["delta_ufar_multi_minus_single"]) for row in rows])
    ci_bounds = [cis[str(row["setting"])]["delta_unknown_far_multi_minus_single"] for row in rows]
    if np.all(deltas < 0) and all(bound["upper"] < 0 for bound in ci_bounds):
        return "A", "TASK UTILITY CONFIRMED"
    if np.any(deltas < 0) and np.any(deltas > 0):
        return "D", "MIXED"
    if np.all(deltas > 0) and sum(bound["lower"] > 0 for bound in ci_bounds) >= 2:
        return "C", "MULTI HURTS"
    if np.all(np.abs(deltas) <= 0.005):
        return "B", "DENSITY ONLY"
    return "D", "MIXED"


def main() -> None:
    verify_frozen_inputs()
    summary_dir = STAGE3_ROOT / "outputs/summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cis: dict[str, dict[str, dict[str, float]]] = {}
    for setting in VALID_SETTINGS:
        fold = load_fold(setting)
        metrics = metric_map(setting)
        cis[setting] = ci_map(setting)
        native = metrics["Native"]
        single = metrics["Single-Full"]
        multi = metrics["Multi-Full-K2"]
        ci = cis[setting]["delta_unknown_far_multi_minus_single"]
        rows.append(
            {
                "setting": setting,
                "num_known_classes": fold["known_count"],
                "num_unknown_classes": fold["unknown_count"],
                "native_ufar": native["unknown_false_acceptance_rate"],
                "single_ufar": single["unknown_false_acceptance_rate"],
                "multi_ufar": multi["unknown_false_acceptance_rate"],
                "delta_ufar_multi_minus_single": multi["unknown_false_acceptance_rate"] - single["unknown_false_acceptance_rate"],
                "native_auroc": native["auroc_unknown_positive"],
                "single_auroc": single["auroc_unknown_positive"],
                "multi_auroc": multi["auroc_unknown_positive"],
                "native_auprc": native["auprc_unknown_positive"],
                "single_auprc": single["auprc_unknown_positive"],
                "multi_auprc": multi["auprc_unknown_positive"],
                "single_known_frr": single["known_false_rejection_rate"],
                "multi_known_frr": multi["known_false_rejection_rate"],
                "delta_ufar_ci_2_5pct": ci["lower"],
                "delta_ufar_ci_97_5pct": ci["upper"],
            }
        )
    numeric_fields = [key for key in rows[0] if key != "setting"]
    for aggregate, reducer in (("MEAN", np.mean), ("STD", lambda values: np.std(values, ddof=0))):
        rows.append(
            {
                "setting": aggregate,
                **{field: float(reducer([float(row[field]) for row in rows[:3]])) for field in numeric_fields},
            }
        )
    write_csv(summary_dir / "cross_setting_results.csv", rows)
    primary_rows = rows[:3]
    gate, gate_name = determine_gate(primary_rows, cis)

    single_values = [float(row["single_ufar"]) for row in primary_rows]
    multi_values = [float(row["multi_ufar"]) for row in primary_rows]
    gap_values = [float(row["delta_ufar_multi_minus_single"]) for row in primary_rows]
    def movement(values: list[float]) -> str:
        if values[0] < values[1] < values[2]:
            return "monotonically increases"
        if values[0] > values[1] > values[2]:
            return "monotonically decreases"
        return "non-monotonic / unstable"
    openness = [
        "# Stage 3 Openness Trend",
        "",
        "This is a three-point descriptive comparison only; no trend model is fitted.",
        "",
        "| Setting | Known/Unknown classes | Single UFAR | Multi UFAR | Multi-Single gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        openness.append(
            f"| {row['setting']} | {row['num_known_classes']}/{row['num_unknown_classes']} | {float(row['single_ufar']):.6f} | {float(row['multi_ufar']):.6f} | {float(row['delta_ufar_multi_minus_single']):+.6f} |"
        )
    openness += [
        "",
        f"- Single UFAR: {movement(single_values)} from A-1 to A-3.",
        f"- Multi UFAR: {movement(multi_values)} from A-1 to A-3.",
        f"- Multi-Single gap: {movement(gap_values)}; negative values favor Multi.",
        "- Because the held-out class composition changes together with openness, these points do not identify a causal openness effect.",
    ]
    (summary_dir / "openness_trend.md").write_text("\n".join(openness) + "\n", encoding="utf-8")

    decision = {
        "gate": gate,
        "gate_name": gate_name,
        "utility_proven": gate == "A",
        "primary_settings_all_reported": True,
        "rule": "Gate A requires Delta UFAR < 0 with a paired-bootstrap upper 95% CI bound < 0 in all three primary settings; inconsistent directions force Gate D.",
    }
    (summary_dir / "final_gate.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate_lines = [
        "# Stage 3 Final Gate",
        "",
        f"## Gate {gate} — {gate_name}",
        "",
        f"Multi local support practical Unknown Detection utility proven: **{'yes' if gate == 'A' else 'no'}**.",
        "",
        "The gate uses all three primary settings. No setting was selected or omitted based on its result.",
        "",
        "| Setting | Delta UFAR | Paired bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for row in primary_rows:
        gate_lines.append(
            f"| {row['setting']} | {float(row['delta_ufar_multi_minus_single']):+.6f} | [{float(row['delta_ufar_ci_2_5pct']):+.6f}, {float(row['delta_ufar_ci_97_5pct']):+.6f}] |"
        )
    (summary_dir / "final_gate.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    readme = [
        "# Stage 3 Strict Unknown-Free Utility Test",
        "",
        "Core question: Does multi-local-support modeling reduce unknown false acceptance compared with a single global Gaussian, under strict unknown-free evaluation?",
        "",
        "## Frozen Execution Plan",
        "",
        "A-1, A-2, and A-3 are all primary benchmarks and were executed in that order. The execution supplement did not change any frozen class list or flow split.",
    ]
    for setting in VALID_SETTINGS:
        fold = load_fold(setting)
        row = next(item for item in primary_rows if item["setting"] == setting)
        readme += [
            "",
            f"## {setting}",
            "",
            f"Known ({fold['known_count']}): {', '.join(fold['known_classes'])}",
            f"Unknown ({fold['unknown_count']}): {', '.join(fold['unknown_classes'])}",
            f"Single/Multi UFAR: {float(row['single_ufar']):.6f} / {float(row['multi_ufar']):.6f}; delta {float(row['delta_ufar_multi_minus_single']):+.6f}.",
        ]
    readme += [
        "",
        "## Strict Unknown-Free Audit",
        "",
        "All three manifests passed the zero-Unknown assertions for encoder train/validation, scaler/PCA/density fitting, threshold calibration, and checkpoint selection. Unknown rows were evaluated only from the original test split.",
        "",
        "## Single vs Multi Results",
        "",
        "See `outputs/summary/cross_setting_results.csv` and each setting's `detector_metrics.csv`.",
        "",
        "## Per-Unknown-Class Results",
        "",
        "See each setting's `per_unknown_class_results.csv`, `decision_transition_matrix.csv`, and `unknown_absorption_by_known_class.csv`.",
        "",
        "## Openness Trend",
        "",
        "See `outputs/summary/openness_trend.md`; the three points are descriptive and no trend is forced.",
        "",
        "## Final Gate",
        "",
        f"Gate {gate} — {gate_name}. Multi local support utility proven: {'yes' if gate == 'A' else 'no'}.",
        "",
        "## Limitations",
        "",
        "The official Open-Detect sample-fold identities are unavailable. Results use official class-holdout lists with the frozen local 80/10/10 flow split and are not author-fold equivalent. Fixed K2 components are support-model parameters, not semantic subgroups.",
        "",
        "## Next Step",
        "",
        "This task stops here. Adaptive K, prototype-specific boundaries, Unknown discovery, CSTNET, and CipherSpectrum are explicitly outside this execution.",
    ]
    (STAGE3_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    index_path = PROJECT_ROOT / "EXPERIMENT_RESULTS.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Experiment Results\n\n"
    additions = []
    for row in primary_rows:
        marker = f"stage3-ustc-{str(row['setting']).lower().replace('-', '')}-primary-20260912"
        if marker not in existing:
            additions.append(
                f"- `{marker}`: complete; Single/Multi UFAR {float(row['single_ufar']):.6f}/{float(row['multi_ufar']):.6f}; [bundle](stage3_unknown_utility/outputs/{row['setting']}/)."
            )
    if additions:
        index_path.write_text(existing.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
