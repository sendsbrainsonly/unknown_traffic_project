#!/usr/bin/env python3
"""Aggregate all 15 fixed fusion evaluations and apply the preregistered gate."""

from __future__ import annotations

import json

from stage13a2_common import (
    CONFIG_PATH,
    METHODS,
    METRICS,
    STAGE_ROOT,
    SUMMARY,
    apply_gate,
    delta_row,
    read_json,
    summary_stats,
    write_csv,
    write_json,
)


def run_row(result: dict[str, object], method: str) -> dict[str, object]:
    metrics = result["methods"][method]
    return {
        "evaluation_label": result["evaluation_label"],
        "external_validation_label": result["external_validation_label"],
        "scenario": result["scenario"],
        "fold": result["fold"],
        "seed": result["seed"],
        "method": method,
        "checkpoint_sha256": result["frozen_inputs"]["checkpoint_sha256"],
        **{metric: metrics[metric] for metric in METRICS},
        "known_accuracy": metrics["known_accuracy"],
        "threshold": metrics["threshold"],
        "validation_known_acceptance": metrics["validation_known_acceptance"],
        "all_known_test_frr": metrics["all_known_test_frr"],
        "all_unknown_test_ufar": metrics["all_unknown_test_ufar"],
        "new_encoder_training": False,
    }


def main() -> None:
    config = read_json(CONFIG_PATH)
    results: list[dict[str, object]] = []
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            if not (run_dir / "SUCCESS").is_file():
                raise RuntimeError(f"Incomplete Stage13A-2 run: {run_dir}")
            results.append(read_json(run_dir / "results.json"))
    if len(results) != 15:
        raise AssertionError("Expected exactly 15 successful paired runs")

    run_rows = [run_row(result, method) for result in results for method in METHODS]
    comparison_rows = [delta_row(result) for result in results]
    scenario_rows: list[dict[str, object]] = []
    for scenario in ("a1", "a2", "a3"):
        for method in METHODS:
            source = [row for row in run_rows if row["scenario"] == scenario and row["method"] == method]
            row: dict[str, object] = {"scenario": scenario, "method": method, "seed_count": len(source)}
            for metric in METRICS:
                for name, value in summary_stats(float(item[metric]) for item in source).items():
                    row[f"{metric}_{name}"] = value
            scenario_rows.append(row)

    gate = apply_gate(comparison_rows, config)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY / "run_level_results.csv", run_rows)
    write_csv(SUMMARY / "scenario_summary.csv", scenario_rows)
    write_csv(SUMMARY / "gl_vs_centroid.csv", comparison_rows)
    write_json(SUMMARY / "final_gate.json", gate)
    lines = [
        "# Stage 13A-2 Final Gate",
        "",
        f"## Decision: {gate['final_gate']}",
        "",
        "## GL-Fusion versus centroid",
        "",
        "| Scenario | Mean ΔAUROC | Positive seeds | Mean ΔUFAR | Mean ΔKnown FRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in ("a1", "a2", "a3"):
        value = gate["scenario_characterization"][scenario]
        lines.append(
            f"| {scenario.upper()} | {value['mean_delta_auroc']:+.6f} | "
            f"{value['positive_seed_count']}/5 | {value['mean_delta_ufar']:+.6f} | "
            f"{value['mean_delta_known_frr']:+.6f} |"
        )
    lines.extend([
        "",
        f"- Overall positive paired seeds: `{gate['positive_seed_count']}/15`.",
        "",
        "## GO conditions",
        "",
        *[f"- {name}: `{'PASS' if passed else 'FAIL'}`" for name, passed in gate["conditions"].items()],
        "",
        "## Conditional conditions",
        "",
        *[f"- {name}: `{'PASS' if passed else 'FAIL'}`" for name, passed in gate["conditional_conditions"].items()],
        "",
        "- New encoder training: `NO`.",
        "- External Test datasets read: `NO`.",
        "- Frozen experiment modified: `NO`.",
    ])
    (SUMMARY / "final_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

