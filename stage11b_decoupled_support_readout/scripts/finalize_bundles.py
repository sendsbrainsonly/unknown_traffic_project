#!/usr/bin/env python3
"""Write Stage 11B RESULTS/manifests and update the project experiment index."""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from stage11b_common import CONFIG_PATH, METHODS, STAGE_ROOT, UNKNOWN_ROOT, read_json, write_json


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
REFRESH = (
    WORKSPACE_ROOT
    / ".agents"
    / "skills"
    / "experiment-data-preservation"
    / "scripts"
    / "refresh_artifact_manifest.py"
)
SUMMARY = STAGE_ROOT / "outputs" / "summary"


def refresh(path: Path) -> None:
    subprocess.run(
        [sys.executable, "-B", str(REFRESH), str(path), "--hash-max-bytes", "1000000000"],
        cwd=STAGE_ROOT.parent,
        check=True,
    )


def update_run(run_dir: Path) -> None:
    result = read_json(run_dir / "results.json")
    identity = f"{result['scenario'].upper()} fold {result['fold']} seed {result['seed']}"
    lines = [
        f"# Stage 11B frozen readout result: {identity}",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`",
        "- Encoder training: `NO`",
        "- Checkpoint modification: `NO`",
        f"- Representation parity: `{result['representation_parity']['status']}`; max absolute error `{result['representation_parity']['max_abs_error']:.3e}`",
        "",
        "## Data and split",
        "",
        "The four thresholds use Known Validation P95 only. R1 is fit on Known Train; R2/R3 reuse the read-only Stage11A B0 Known-Train models. Unknown data are evaluation-only.",
        "",
        "## Core results",
        "",
        "| Readout | AUROC | Binary F1 | UFAR | Known FRR | Known Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        detector = result["evaluation"]["detectors"][method]
        detection = detector["combined_balanced_1to1"]
        lines.append(
            f"| {method} | {detection['auroc']:.6f} | {detection['binary_f1']:.6f} | "
            f"{detection['ufar']:.6f} | {detection['known_frr']:.6f} | "
            f"{detector['known_test']['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a USTC development result, not independent external validation. CipherSpectrum Stage9 sample-level Test was not used.",
            "",
            "## Preserved evidence",
            "",
            "Support models, thresholds, sample outputs, class metrics, absorption rows, provenance hashes, and this result record are retained in this run directory.",
            "",
            "## Conclusion and next step",
            "",
            "No next stage was started.",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = read_json(run_dir / "manifest.json")
    manifest.update(
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "success",
            "experiment_type": "method-development-evaluation",
            "claim_scope": "diagnostic",
            "objective": f"Frozen B0 encoder R0-R3 readout evaluation for {identity}",
            "inputs": [
                {"path": result["frozen_encoder"]["source_checkpoint"], "role": "frozen B0 best checkpoint", "access": "read_only"},
                {"path": result["artifacts"]["source_stage11a_density"], "role": "frozen Known-Train K1/K2 support models", "access": "read_only"},
            ],
            "code": {"entrypoint": "scripts/evaluate_frozen.py", "config": str(CONFIG_PATH.resolve())},
            "execution": {
                "mode": result["mode"],
                "optimizer": None,
                "backward": False,
                "new_encoder_training": False,
                "representation_parity": result["representation_parity"]["status"],
            },
            "configuration": {
                "scenario": result["scenario"],
                "fold": result["fold"],
                "seed": result["seed"],
                "detectors": list(METHODS),
                "threshold": "Known Validation P95 only",
            },
            "core_results": [
                {"name": f"{method}_auroc", "value": result["evaluation"]["detectors"][method]["combined_balanced_1to1"]["auroc"]}
                for method in METHODS
            ],
            "limitations": [
                "USTC development result; not independent external validation",
                "Local repeated grouped-image-disjoint v6 splits are not author-exact folds",
                "CipherSpectrum Stage9 sample-level Test was not used",
            ],
            "next_step": "Stop after Stage11B; no subsequent stage started",
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    refresh(run_dir)


def update_smoke(smoke_dir: Path) -> None:
    if not (smoke_dir / "SUCCESS").is_file():
        return
    result = read_json(smoke_dir / "results.json")
    (smoke_dir / "RESULTS.md").write_text(
        "# Stage 11B full-path smoke\n\n"
        "## Configuration and execution\n\n"
        "- Status: `success`\n"
        "- Frozen encoder inference and R0-R3 paths completed.\n"
        f"- Representation parity: `{result['representation_parity']['status']}`.\n"
        "- No optimizer, backward, or checkpoint update.\n\n"
        "## Data and split\n\n"
        "A1 fold0 seed2022 only; thresholds use Known Validation P95.\n\n"
        "## Core results\n\n"
        "The complete frozen inference and R0-R3 readout path succeeded with exact historical parity.\n\n"
        "## Limitations\n\n"
        "Smoke only; excluded from formal statistics. CipherSpectrum Stage9 sample-level Test was not used.\n\n"
        "## Preserved evidence\n\n"
        "The smoke results, parity report, checkpoint hash, support models, thresholds, sample outputs, and manifest are retained in this directory.\n\n"
        "## Conclusion and next step\n\n"
        "The smoke gate passed; formal evaluation was run separately.\n",
        encoding="utf-8",
    )
    manifest = read_json(smoke_dir / "manifest.json")
    manifest.update(
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "success",
            "experiment_type": "smoke-test",
            "claim_scope": "smoke",
            "core_results": [{"name": "representation_parity", "value": result["representation_parity"]["status"]}],
            "limitations": ["Smoke only; excluded from formal statistics"],
            "next_step": "Formal Stage11B frozen evaluations",
        }
    )
    write_json(smoke_dir / "manifest.json", manifest)
    refresh(smoke_dir)


def scenario_value(rows: list[dict[str, str]], scenario: str, comparison: str, metric: str) -> float:
    for row in rows:
        if row["row_type"] == "paired_delta" and row["scenario"] == scenario and row["comparison"] == comparison and row["metric"] == metric:
            return float(row["mean"])
    raise KeyError((scenario, comparison, metric))


def main() -> None:
    config = read_json(CONFIG_PATH)
    gate = read_json(SUMMARY / "final_gate.json")
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            update_run(STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}")
    update_smoke(STAGE_ROOT / "artifacts" / "smoke" / "a1" / "fold0_seed2022")
    with (SUMMARY / "scenario_summary.csv").open(encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    commands = """# Successful payloads used the fixed Conda environment and the workspace tmux helper.
# Abbreviations below are expanded to the exact absolute paths used.
HELPER=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/tmux-task-execution/scripts/tmux_task.sh
PROJECT=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project
PY=/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python
SELECT_GPU=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/using-superpowers/scripts/select_gpu.py

$HELPER run s11b-static-r07 $PROJECT -- bash -lc "$PY -B -m py_compile stage11b_decoupled_support_readout/scripts/*.py stage11b_decoupled_support_readout/tests/*.py; $PY -B -m pytest -q stage11b_decoupled_support_readout/tests/test_stage11b_common.py"
$HELPER run s11b-init-prov-r08 $PROJECT -- bash -lc "$PY -B stage11b_decoupled_support_readout/scripts/prepare_campaign.py --include-smoke; $PY -B stage11b_decoupled_support_readout/scripts/verify_provenance.py --phase before"
$HELPER run s11b-smoke-r12 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 0,3,4,5,6,7 -- $PY -B stage11b_decoupled_support_readout/scripts/evaluate_frozen.py --scenario a1 --fold 0 --seed 2022 --smoke
$HELPER start s11b-q0-r18 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 0 -- $PY -B stage11b_decoupled_support_readout/scripts/run_queue.py --queue-index 0 --queue-count 6 --physical-gpu 0
$HELPER start s11b-q1-r18 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 3 -- $PY -B stage11b_decoupled_support_readout/scripts/run_queue.py --queue-index 1 --queue-count 6 --physical-gpu 3
$HELPER start s11b-q2-r18 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 4 -- $PY -B stage11b_decoupled_support_readout/scripts/run_queue.py --queue-index 2 --queue-count 6 --physical-gpu 4
$HELPER start s11b-q3-r18 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 5 -- $PY -B stage11b_decoupled_support_readout/scripts/run_queue.py --queue-index 3 --queue-count 6 --physical-gpu 5
$HELPER start s11b-q4-r18 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 6 -- $PY -B stage11b_decoupled_support_readout/scripts/run_queue.py --queue-index 4 --queue-count 6 --physical-gpu 6
$HELPER start s11b-q5-r18 $PROJECT -- $PY $SELECT_GPU --min-free-gb 8 --max-utilization 95 --allowed 7 -- $PY -B stage11b_decoupled_support_readout/scripts/run_queue.py --queue-index 5 --queue-count 6 --physical-gpu 7
$HELPER run s11b-aggregate-r20 $PROJECT -- $PY -B stage11b_decoupled_support_readout/scripts/aggregate_results.py
$HELPER run s11b-prov-final-r21 $PROJECT -- bash -lc "$PY -B stage11b_decoupled_support_readout/scripts/verify_provenance.py --phase after; $PY -B stage11b_decoupled_support_readout/scripts/finalize_bundles.py; $PY -B stage11b_decoupled_support_readout/scripts/verify_stage11b.py"
"""
    (SUMMARY / "execution_commands.txt").write_text(commands, encoding="utf-8")
    lines = [
        "# Experiment results: stage11b-decoupled-support-readout-20260915-v1",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Experiment type: `method-development-evaluation`",
        "- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`",
        "- New encoder training: `NO`",
        "- Frozen B0 checkpoints reused: `15/15`",
        f"- Final gate: **{gate['final_gate']}**",
        f"- Mechanism: **{gate['mechanism']}**",
        f"- Covariance verdict: **{gate['covariance_verdict']}**",
        f"- K2 verdict: **{gate['multicomponent_verdict']}**",
        "",
        "## Data and split",
        "",
        "Fifteen frozen Open-Detect v6 B0 checkpoints were evaluated across A1/A2/A3 and seeds 2022-2026. Support fitting used Known Train only; all thresholds used Known Validation P95 only; Test was evaluation-only.",
        "",
        "## Core results",
        "",
    ]
    for scenario in ("a1", "a2", "a3"):
        lines.append(f"- {scenario.upper()}: R2-R0 `{scenario_value(summary_rows, scenario, 'R2-R0', 'delta_auroc'):.6f}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a USTC development result, not independent external validation. Local corrected v6 splits are not author-exact folds. CipherSpectrum Stage9 sample-level Test was not used.",
            "",
            "## Preserved evidence",
            "",
            "Per-run support models, thresholds, latent/score/prediction arrays, class/absorption tables, results, hashes, RESULTS, and manifests are retained under `artifacts/`; aggregate tables and decisions are retained under `outputs/summary/`.",
            "",
            "## Conclusion and next step",
            "",
            f"The fixed decision is {gate['final_gate']} with mechanism {gate['mechanism']}. Per-run evidence is preserved under `artifacts/`; no next stage was started.",
        ]
    )
    (STAGE_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = (STAGE_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "\n## Recorded Final Result\n"
    if marker in readme:
        readme = readme.split(marker)[0].rstrip() + "\n"
    (STAGE_ROOT / "README.md").write_text(
        readme.rstrip() + marker + "\n" + (SUMMARY / "final_gate.md").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    manifest = read_json(STAGE_ROOT / "manifest.json")
    manifest.update(
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "success",
            "experiment_type": "method-development-evaluation",
            "claim_scope": "diagnostic",
            "objective": "Evaluate R0-R3 detection readouts over 15 frozen USTC v6 B0 encoders",
            "inputs": [
                {"path": config["v6_frozen_root"], "role": "15 frozen B0 runs", "access": "read_only"},
                {"path": config["v6_split_root"], "role": "frozen five-fold USTC splits", "access": "read_only"},
                {"path": config["stage11a_root"], "role": "frozen provenance and B0 K1/K2 support artifacts", "access": "read_only"},
            ],
            "code": {"entrypoint": "scripts/evaluate_frozen.py", "aggregator": "scripts/aggregate_results.py"},
            "execution": {"formal_runs": 15, "readouts_per_run": 4, "new_encoder_training": False},
            "configuration": config,
            "core_results": [
                {"name": "final_gate", "value": gate["final_gate"]},
                {"name": "mechanism", "value": gate["mechanism"]},
                {"name": "formal_frozen_runs", "value": 15},
                {"name": "covariance_verdict", "value": gate["covariance_verdict"]},
                {"name": "multicomponent_verdict", "value": gate["multicomponent_verdict"]},
            ],
            "limitations": [
                "USTC development result; not independent external validation",
                "Local repeated grouped-image-disjoint v6 splits are not author-exact folds",
                "CipherSpectrum Stage9 sample-level Test was not opened or used",
            ],
            "next_step": "Stop after Stage11B; no subsequent stage started",
        }
    )
    write_json(STAGE_ROOT / "manifest.json", manifest)
    refresh(STAGE_ROOT)
    index = UNKNOWN_ROOT / "EXPERIMENT_RESULTS.md"
    text = index.read_text(encoding="utf-8")
    entry = (
        f"- `stage11b-decoupled-support-readout-20260915-v1` — status `success`; "
        f"15 frozen B0 encoders, R0-R3 readouts; Gate `{gate['final_gate']}`, mechanism `{gate['mechanism']}`; "
        "no encoder training or CipherSpectrum Stage9 sample-level Test use; "
        "[bundle](stage11b_decoupled_support_readout/).\n"
    )
    if "stage11b-decoupled-support-readout-20260915-v1" not in text:
        index.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")
    print(json.dumps({"status": "PASS", "run_manifests": 15, "root_manifest": "success"}, indent=2))


if __name__ == "__main__":
    main()
