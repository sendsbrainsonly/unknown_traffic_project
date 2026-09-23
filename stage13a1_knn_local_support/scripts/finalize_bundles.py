#!/usr/bin/env python3
"""Write canonical Stage13A-1 run/root documentation after aggregation."""

from __future__ import annotations

import csv
import datetime as dt
import json

from stage13_common import CONFIG_PATH, METHODS, STAGE_ROOT, SUMMARY, UNKNOWN_ROOT, read_json, write_json


def format_metrics(method: str, values: dict[str, object]) -> str:
    return (
        f"- {method}: AUROC `{values['auroc']:.6f}`, AUPRC `{values['auprc']:.6f}`, "
        f"UFAR `{values['ufar']:.6f}`, Known FRR `{values['known_frr']:.6f}`, "
        f"Binary F1 `{values['binary_f1']:.6f}`, Known Macro-F1 `{values['known_macro_f1']:.6f}`."
    )


def update_run(run_dir, result: dict[str, object]) -> None:
    lines = [
        f"# Experiment results: stage13a1-{result['scenario']}-fold{result['fold']}-seed{result['seed']}",
        "",
        "## Data and split",
        "",
        f"Frozen Stage11B USTC `{result['scenario'].upper()}` fold `{result['fold']}` seed `{result['seed']}` deterministic representations; source embeddings and checkpoint are read-only.",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Experiment type: `method-development-evaluation`",
        "- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`",
        "- Methods: centroid, class-conditional kNN-5, class-conditional kNN-10",
        "- Thresholds: separate Known Validation P95 only",
        "- New encoder training: `NO`",
        "- Train self-neighbour exclusion: `YES`",
        "",
        "## Core results",
        "",
        *[format_metrics(method, result["methods"][method]) for method in METHODS],
        f"- KNN_10-CENTROID ΔAUROC: `{result['deltas']['KNN_10']['auroc']:+.6f}`.",
        "",
        "## Preserved evidence",
        "",
        "`config.json`, `input_hashes.json`, `thresholds.json`, `score_arrays.npz`, `results.json`, `SUCCESS`, and `manifest.json`.",
        "",
        "## Limitations",
        "",
        "USTC development result; not independent external validation. Frozen Stage11B embeddings are reused rather than re-extracted.",
        "",
        "## Conclusion and next step",
        "",
        "This run contributes to the preregistered five-seed scenario aggregate. No subsequent method was started.",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = read_json(run_dir / "manifest.json")
    manifest.update({
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "success",
        "experiment_type": "method-development-evaluation",
        "claim_scope": "diagnostic",
        "objective": "Compare class-conditional kNN-5/kNN-10 support with DES-v0 centroid on one frozen USTC B0 representation",
        "inputs": [
            {"path": result["frozen_inputs"]["stage11b_run"], "role": "frozen Stage11B representation bundle", "access": "read_only"},
            {"path": result["frozen_inputs"]["checkpoint_path"], "role": "frozen B0 checkpoint verified by hash", "access": "read_only"},
        ],
        "code": {"entrypoint": "scripts/evaluate_run.py", "config": str(CONFIG_PATH)},
        "execution": {"new_encoder_training": False, "optimizer": None, "backward": False},
        "configuration": {"scenario": result["scenario"], "fold": result["fold"], "seed": result["seed"], "methods": list(METHODS), "threshold": "Known Validation P95"},
        "core_results": [
            {"name": f"{method}_auroc", "value": result["methods"][method]["auroc"]}
            for method in METHODS
        ] + [{"name": "KNN_10_minus_CENTROID_auroc", "value": result["deltas"]["KNN_10"]["auroc"]}],
        "limitations": ["USTC development result", "Frozen Stage11B embeddings reused rather than re-extracted"],
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
        "# Experiment results: stage13a1-knn-local-support-20260916-v1",
        "",
        "## Data and split",
        "",
        "Fifteen frozen Stage11B USTC B0 representation bundles: A1/A2/A3 crossed with seeds 2022–2026. Known Train supplies support, Known Validation supplies P95 thresholds, and USTC Test is evaluation-only.",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Experiment type: `method-development-evaluation`",
        "- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`",
        "- New encoder training: `NO`",
        "- External Test datasets read: `NO`",
        "- Frozen experiments modified: `NO`",
        "",
        "## Core results",
        "",
        f"- Final gate: **{gate['final_gate']}**",
    ]
    for scenario in ("a1", "a2", "a3"):
        root_lines.append(
            f"- {scenario.upper()}: centroid/kNN-5/kNN-10 AUROC "
            f"`{scenario_value(rows, scenario, 'CENTROID', 'auroc'):.6f}` / "
            f"`{scenario_value(rows, scenario, 'KNN_5', 'auroc'):.6f}` / "
            f"`{scenario_value(rows, scenario, 'KNN_10', 'auroc'):.6f}`."
        )
    root_lines.extend([
        "",
        "## Preserved evidence",
        "",
        "Per-run scores, metrics, thresholds, hashes, RESULTS and manifests are under `artifacts/`; aggregate CSV/JSON/Markdown evidence is under `outputs/summary/`.",
        "",
        "## Limitations",
        "",
        "This is USTC method development, not independent external validation. It reuses cached deterministic representations from Stage11B.",
        "",
        "## Conclusion and next step",
        "",
        f"The preregistered decision is {gate['final_gate']}. Fusion research recommendation is recorded, but no next stage was started.",
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
        "objective": "Diagnose class-conditional kNN local support against DES-v0 centroid on 15 frozen USTC B0 representations",
        "inputs": [{"path": str(config["stage11b_root"]), "role": "15 frozen Stage11B representation bundles", "access": "read_only"}],
        "code": {"entrypoint": "scripts/evaluate_run.py", "aggregator": "scripts/aggregate_results.py"},
        "execution": {"formal_runs": 15, "methods_per_run": 3, "new_encoder_training": False, "external_test_datasets_read": []},
        "configuration": config,
        "core_results": [{"name": "final_gate", "value": gate["final_gate"]}, {"name": "more_stable_k", "value": gate["more_stable_k"]}, {"name": "formal_runs", "value": 15}],
        "limitations": ["USTC development result", "Frozen Stage11B embeddings reused", "No external Test dataset used"],
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
python -B scripts/verify_stage13a1.py
"""
    (SUMMARY / "execution_commands.txt").write_text(commands, encoding="utf-8")
    index = UNKNOWN_ROOT / "EXPERIMENT_RESULTS.md"
    existing = index.read_text(encoding="utf-8")
    experiment_id = "stage13a1-knn-local-support-20260916-v1"
    entry = (
        f"- `{experiment_id}`: `success / {gate['final_gate']}`; 15 frozen Stage11B B0 representations, centroid vs kNN-5/kNN-10; "
        "no encoder training and no external Test access; [bundle](stage13a1_knn_local_support/).\n"
    )
    if experiment_id not in existing:
        index.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")
    print(json.dumps({"status": "PASS", "run_manifests": 15, "final_gate": gate["final_gate"]}, indent=2))


if __name__ == "__main__":
    main()

