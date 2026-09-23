#!/usr/bin/env python3
"""Write canonical Stage 13A-2 run/root documentation after aggregation."""

from __future__ import annotations

import csv
import datetime as dt
import json

from stage13a2_common import CONFIG_PATH, METHODS, STAGE_ROOT, SUMMARY, UNKNOWN_ROOT, read_json, write_json


def format_metrics(method: str, values: dict[str, object]) -> str:
    return (
        f"- {method}: AUROC `{values['auroc']:.6f}`, AUPRC `{values['auprc']:.6f}`, "
        f"UFAR `{values['ufar']:.6f}`, Known FRR `{values['known_frr']:.6f}`, "
        f"Binary F1 `{values['binary_f1']:.6f}`, Known Macro-F1 `{values['known_macro_f1']:.6f}`."
    )


def update_run(run_dir, result: dict[str, object]) -> None:
    lines = [
        f"# Experiment results: stage13a2-{result['scenario']}-fold{result['fold']}-seed{result['seed']}",
        "",
        "## Data and split",
        "",
        f"Frozen Stage13A-1 USTC `{result['scenario'].upper()}` fold `{result['fold']}` seed `{result['seed']}` centroid and kNN-10 scores; all source assets are read-only.",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Experiment type: `method-development-evaluation`",
        "- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`",
        "- Robust statistics: median and unscaled MAD fitted on Known Validation only",
        "- Fusion: `0.5 * Zc + 0.5 * Zk`; no weight search",
        "- Thresholds: separate Known Validation P95 only",
        "- New encoder training/inference: `NO`",
        "",
        "## Core results",
        "",
        *[format_metrics(method, result["methods"][method]) for method in METHODS],
        f"- GL_FUSION-CENTROID ΔAUROC: `{result['deltas']['GL_FUSION']['auroc']:+.6f}`.",
        "",
        "## Preserved evidence",
        "",
        "`config.json`, `input_hashes.json`, `normalization_stats.json`, `thresholds.json`, `score_arrays.npz`, `results.json`, `SUCCESS`, and `manifest.json`.",
        "",
        "## Limitations",
        "",
        "USTC development result; not independent external validation. Cached Stage13A-1 scores are reused without encoder execution.",
        "",
        "## Conclusion and next step",
        "",
        "This run contributes to the preregistered five-seed scenario aggregate. No subsequent stage was started.",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = read_json(run_dir / "manifest.json")
    manifest.update({
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "success",
        "experiment_type": "method-development-evaluation",
        "claim_scope": "diagnostic",
        "objective": "Compare fixed 0.5/0.5 robust-normalized global-local fusion with frozen centroid and kNN-10 scores",
        "inputs": [
            {"path": result["frozen_inputs"]["stage13a1_run"], "role": "frozen Stage13A-1 score bundle", "access": "read_only"},
            {"path": result["frozen_inputs"]["checkpoint_path"], "role": "frozen B0 checkpoint verified by hash", "access": "read_only"},
        ],
        "code": {"entrypoint": "scripts/evaluate_run.py", "config": str(CONFIG_PATH)},
        "execution": {"new_encoder_training": False, "encoder_inference": False, "optimizer": None, "backward": False},
        "configuration": {
            "scenario": result["scenario"],
            "fold": result["fold"],
            "seed": result["seed"],
            "methods": list(METHODS),
            "normalization": "Known Validation median/MAD",
            "fusion_weights": [0.5, 0.5],
            "threshold": "Known Validation P95",
        },
        "core_results": [
            {"name": f"{method}_auroc", "value": result["methods"][method]["auroc"]}
            for method in METHODS
        ] + [{"name": "GL_FUSION_minus_CENTROID_auroc", "value": result["deltas"]["GL_FUSION"]["auroc"]}],
        "limitations": ["USTC development result", "Frozen Stage13A-1 scores reused", "No external Test dataset used"],
        "next_step": "No subsequent stage started",
    })
    write_json(run_dir / "manifest.json", manifest)


def scenario_value(rows: list[dict[str, str]], scenario: str, method: str, metric: str) -> float:
    row = next(item for item in rows if item["scenario"] == scenario and item["method"] == method)
    return float(row[f"{metric}_mean"])


def main() -> None:
    config = read_json(CONFIG_PATH)
    gate = read_json(SUMMARY / "final_gate.json")
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            update_run(run_dir, read_json(run_dir / "results.json"))

    with (SUMMARY / "scenario_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    root_lines = [
        "# Experiment results: stage13a2-global-local-fusion-20260916-v1",
        "",
        "## Data and split",
        "",
        "Fifteen frozen Stage13A-1 USTC score bundles: A1/A2/A3 crossed with seeds 2022–2026. Known Validation alone fits robust normalization and P95 thresholds; USTC Test is evaluation-only.",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Experiment type: `method-development-evaluation`",
        "- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`",
        "- Fusion: fixed `0.5 * Zc + 0.5 * Zk`",
        "- New encoder training/inference: `NO`",
        "- External Test datasets read: `NO`",
        "- Frozen experiments modified: `NO`",
        "",
        "## Core results",
        "",
        f"- Final gate: **{gate['final_gate']}**",
    ]
    for scenario in ("a1", "a2", "a3"):
        root_lines.append(
            f"- {scenario.upper()}: centroid/kNN-10/GL-Fusion AUROC "
            f"`{scenario_value(rows, scenario, 'CENTROID', 'auroc'):.6f}` / "
            f"`{scenario_value(rows, scenario, 'KNN_10', 'auroc'):.6f}` / "
            f"`{scenario_value(rows, scenario, 'GL_FUSION', 'auroc'):.6f}`."
        )
    root_lines.extend([
        "",
        "## Preserved evidence",
        "",
        "Per-run scores, normalization statistics, metrics, thresholds, hashes, RESULTS and manifests are under `artifacts/`; aggregate evidence is under `outputs/summary/`.",
        "",
        "## Limitations",
        "",
        "This is USTC method development, not independent external validation. It reuses cached scores from Stage13A-1.",
        "",
        "## Conclusion and next step",
        "",
        f"The preregistered decision is {gate['final_gate']}. No next stage was started.",
    ])
    (STAGE_ROOT / "RESULTS.md").write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    readme = (STAGE_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "\n## Recorded Final Result\n"
    if marker in readme:
        readme = readme.split(marker)[0].rstrip() + "\n"
    (STAGE_ROOT / "README.md").write_text(
        readme.rstrip() + marker + "\n" + (SUMMARY / "final_gate.md").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    manifest = read_json(STAGE_ROOT / "manifest.json")
    manifest.update({
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "success",
        "experiment_type": "method-development-evaluation",
        "claim_scope": "diagnostic",
        "objective": "Test fixed robust-normalized centroid and kNN-10 support fusion on 15 frozen USTC development runs",
        "inputs": [{"path": config["stage13a1_root"], "role": "15 frozen Stage13A-1 score bundles", "access": "read_only"}],
        "code": {"entrypoint": "scripts/evaluate_run.py", "aggregator": "scripts/aggregate_results.py"},
        "execution": {"formal_runs": 15, "methods_per_run": 3, "new_encoder_training": False, "encoder_inference": False, "external_test_datasets_read": []},
        "configuration": config,
        "core_results": [{"name": "final_gate", "value": gate["final_gate"]}, {"name": "positive_seed_count", "value": gate["positive_seed_count"]}, {"name": "formal_runs", "value": 15}],
        "limitations": ["USTC development result", "Frozen Stage13A-1 scores reused", "No external Test dataset used"],
        "next_step": "No subsequent stage started",
    })
    write_json(STAGE_ROOT / "manifest.json", manifest)

    commands = """# Payloads run through the fixed Conda environment and workspace tmux helper.
python -B scripts/verify_provenance.py --phase before
python -B -m pytest -q tests
python -B scripts/run_queue.py --queue-index 0 --queue-count 1
python -B scripts/aggregate_results.py
python -B scripts/verify_provenance.py --phase after
python -B scripts/finalize_bundles.py
python -B scripts/verify_stage13a2.py
"""
    (SUMMARY / "execution_commands.txt").write_text(commands, encoding="utf-8")
    index = UNKNOWN_ROOT / "EXPERIMENT_RESULTS.md"
    existing = index.read_text(encoding="utf-8")
    experiment_id = "stage13a2-global-local-fusion-20260916-v1"
    entry = (
        f"- `{experiment_id}`: `success / {gate['final_gate']}`; 15 frozen Stage13A-1 runs, fixed 0.5/0.5 robust global-local fusion; "
        "no encoder execution and no external Test access; [bundle](stage13a2_global_local_fusion/).\n"
    )
    if experiment_id not in existing:
        index.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")
    print(json.dumps({"status": "PASS", "run_manifests": 15, "final_gate": gate["final_gate"]}, indent=2))


if __name__ == "__main__":
    main()

