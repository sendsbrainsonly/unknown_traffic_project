#!/usr/bin/env python3
"""Finalize Stage 11C RESULTS, manifests, command record and project index."""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

from stage11c_common import CONFIG_PATH, SCENARIOS, SEEDS, STAGE_ROOT, SUMMARY_ROOT, UNKNOWN_ROOT, output_run_dir, read_json, write_json


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
REFRESH = WORKSPACE_ROOT / ".agents" / "skills" / "experiment-data-preservation" / "scripts" / "refresh_artifact_manifest.py"


def refresh(path: Path) -> None:
    subprocess.run([sys.executable, "-B", str(REFRESH), str(path), "--hash-max-bytes", "1000000000"], cwd=STAGE_ROOT.parent, check=True)


def update_run(run: Path) -> None:
    result = read_json(run / "results.json")
    features = read_json(run / "known_only_features.json")
    outcome = read_json(run / "outcome_only.json")
    identity = f"{result['scenario'].upper()} fold {result['fold']} seed {result['seed']}"
    lines = [
        f"# Stage 11C Known-only diagnosis: {identity}",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Claim scope: `EXPLORATORY_ONLY`; `POST_HOC_DEVELOPMENT_DIAGNOSIS`",
        "- New training/detector fitting/threshold fitting: `NO/NO/NO`",
        "- Bootstrap: `B=100`, `seed=0`, PCA64, fixed top-5/top-10 subspaces",
        "",
        "## Data and split",
        "",
        "Features use Known Train/Validation only. Stage11B Test outcomes are isolated as `OUTCOME_ONLY`; no Test representation is loaded as a criterion feature.",
        "",
        "## Core results",
        "",
        f"- Validation Full-vs-Spherical DeltaNLL: `{float(features['delta_nll_full_validation']):.6f}`",
        f"- Validation Full-K1 minus centroid Macro-F1: `{float(features['delta_validation_macro_f1_full_minus_centroid']):.6f}`",
        f"- Covariance bootstrap relative instability: `{float(features['covariance_bootstrap_frobenius_relative_mean']):.6f}`",
        f"- Stage11B R2-R1 AUROC outcome: `{float(outcome['delta_cov_auroc']):.6f}` (`OUTCOME_ONLY`)",
        "",
        "## Limitations",
        "",
        "One frozen USTC development run; correlation and mechanism interpretation require aggregation and are not external validation.",
        "",
        "## Preserved evidence",
        "",
        "Class geometry, covariance bootstrap, validation fit/classification/coverage, Known-only features, isolated outcome, source hashes, results and manifest are retained here.",
        "",
        "## Conclusion and next step",
        "",
        "Included in the fixed 15-run Stage11C aggregate. No next stage was started.",
    ]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = read_json(run / "manifest.json")
    manifest.update(
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "success",
            "experiment_type": "diagnostic",
            "claim_scope": "diagnostic",
            "objective": f"Known-only support complexity diagnosis for {identity}",
            "inputs": [{"path": str((STAGE_ROOT.parent / "stage11b_decoupled_support_readout" / "artifacts" / result["scenario"] / f"fold{result['fold']}_seed{result['seed']}").resolve()), "role": "frozen Stage11B Known arrays/support/outcome", "access": "read_only"}],
            "code": {"entrypoint": "scripts/analyze_run.py", "config": str(CONFIG_PATH.resolve())},
            "execution": {"new_training": False, "new_detector_fitting": False, "new_threshold_fitting": False, "bootstrap_iterations": 100, "bootstrap_seed": 0},
            "configuration": {"scenario": result["scenario"], "fold": result["fold"], "seed": result["seed"], "feature_data": "Known Train/Validation only"},
            "core_results": [
                {"name": "delta_nll_full_validation", "value": features["delta_nll_full_validation"]},
                {"name": "delta_validation_macro_f1_full_minus_centroid", "value": features["delta_validation_macro_f1_full_minus_centroid"]},
                {"name": "covariance_bootstrap_frobenius_relative_mean", "value": features["covariance_bootstrap_frobenius_relative_mean"]},
                {"name": "outcome_only_r2_minus_r1_auroc", "value": outcome["delta_cov_auroc"]},
            ],
            "limitations": ["USTC development diagnosis", "15 total runs", "Outcome is post-hoc only", "No CipherSpectrum Test"],
            "next_step": "Stop after Stage11C",
        }
    )
    write_json(run / "manifest.json", manifest)
    refresh(run)


def update_attempt_container(container: Path, successful_attempt: Path) -> None:
    """Record the preserved parity-check failure and the non-overwriting retry."""

    failure_path = container / "FAILURE.json"
    failure = read_json(failure_path) if failure_path.is_file() else None
    initial_status = "failed before statistics were produced" if failure else "not started because its queue stopped at an earlier failed job"
    preserved_detail = f"`FAILURE.json` retains `{failure.get('error', 'unknown error')}`." if failure else "No initial-attempt process ran in this container, so there is no synthetic failure record."
    lines = [
        "# Stage 11C run-attempt container",
        "",
        "## Configuration and execution",
        "",
        f"- Initial attempt status: `{initial_status}`.",
        "- Corrected retry: `attempt2`, preserved separately and completed successfully.",
        "",
        "## Data and split",
        "",
        "The failed check used only frozen Stage11B Known Train centroid data.",
        "",
        "## Core results",
        "",
        "The first queue launch used an overly strict float64-vs-float32 centroid parity tolerance; maximum observed recomputation difference was 1.23e-6. No research statistic was emitted by any failed initial attempt.",
        "",
        "## Limitations",
        "",
        "This container records execution history; the formal successful bundle is `attempt2/`.",
        "",
        "## Preserved evidence",
        "",
        f"{preserved_detail} `attempt2/` retains the corrected result.",
        "",
        "## Conclusion and next step",
        "",
        "The failure was preserved and not overwritten. No next stage was started.",
    ]
    (container / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = read_json(container / "manifest.json")
    manifest.update(
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "success",
            "experiment_type": "diagnostic",
            "claim_scope": "diagnostic",
            "objective": "Preserve initial failed check and reference successful non-overwriting attempt2",
            "execution": {"initial_attempt": "failed_before_statistics" if failure else "not_started_after_queue_abort", "successful_attempt": str(successful_attempt.resolve())},
            "core_results": [{"name": "initial_attempt_failure_preserved", "value": bool(failure)}, {"name": "attempt2_status", "value": "success"}],
            "limitations": ["Container only; formal statistics are in attempt2"],
            "next_step": "Stop after Stage11C",
        }
    )
    write_json(container / "manifest.json", manifest)
    refresh(container)


def main() -> None:
    config = read_json(CONFIG_PATH)
    gate = read_json(SUMMARY_ROOT / "final_gate.json")
    for scenario in SCENARIOS:
        for fold, seed in enumerate(SEEDS):
            successful = output_run_dir(scenario, fold, seed)
            update_run(successful)
            container = successful.parent
            update_attempt_container(container, successful)

    helper = "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/tmux-task-execution/scripts/tmux_task.sh"
    project = str(UNKNOWN_ROOT)
    py = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
    commands = f"""# Successful Stage11C payloads; all paths are exact.
HELPER={helper}
PROJECT={project}
PY={py}
$HELPER run s11c-static-r07 $PROJECT -- bash -c \"$PY -B -m py_compile stage11c_known_only_support_complexity/scripts/*.py stage11c_known_only_support_complexity/tests/*.py; $PY -B -m pytest -q stage11c_known_only_support_complexity/tests\"
$HELPER run s11c-init-prov-r08 $PROJECT -- bash -c \"$PY -B stage11c_known_only_support_complexity/scripts/prepare_campaign.py; $PY -B stage11c_known_only_support_complexity/scripts/verify_provenance.py --phase before\"
# s11c-q0-r10 through s11c-q5-r10 are preserved failed first attempts (strict float64/float32 centroid parity check).
$HELPER run s11c-attempt2-init-r12 $PROJECT -- bash -c \"$PY -B -m pytest -q stage11c_known_only_support_complexity/tests; $PY -B stage11c_known_only_support_complexity/scripts/prepare_campaign.py\"
$HELPER start s11c-q0-r13 $PROJECT -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B stage11c_known_only_support_complexity/scripts/run_queue.py --queue-index 0 --queue-count 6
$HELPER start s11c-q1-r13 $PROJECT -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B stage11c_known_only_support_complexity/scripts/run_queue.py --queue-index 1 --queue-count 6
$HELPER start s11c-q2-r13 $PROJECT -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B stage11c_known_only_support_complexity/scripts/run_queue.py --queue-index 2 --queue-count 6
$HELPER start s11c-q3-r13 $PROJECT -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B stage11c_known_only_support_complexity/scripts/run_queue.py --queue-index 3 --queue-count 6
$HELPER start s11c-q4-r13 $PROJECT -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B stage11c_known_only_support_complexity/scripts/run_queue.py --queue-index 4 --queue-count 6
$HELPER start s11c-q5-r13 $PROJECT -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B stage11c_known_only_support_complexity/scripts/run_queue.py --queue-index 5 --queue-count 6
$HELPER run s11c-aggregate-r15 $PROJECT -- $PY -B stage11c_known_only_support_complexity/scripts/aggregate_results.py
$HELPER run s11c-final-r17 $PROJECT -- bash -c \"$PY -B stage11c_known_only_support_complexity/scripts/verify_provenance.py --phase after; $PY -B stage11c_known_only_support_complexity/scripts/finalize_bundles.py; $PY -B stage11c_known_only_support_complexity/scripts/verify_stage11c.py\"
$HELPER run s11c-seal-r18 $PROJECT -- bash -c \"$PY -B -m pytest -q stage11c_known_only_support_complexity/tests; $PY -B stage11c_known_only_support_complexity/scripts/finalize_bundles.py; $PY -B stage11c_known_only_support_complexity/scripts/verify_stage11c.py; $PY -B /home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/experiment-data-preservation/scripts/validate_experiment_bundle.py stage11c_known_only_support_complexity --verify-hashes\"
"""
    (SUMMARY_ROOT / "execution_commands.txt").write_text(commands, encoding="utf-8")
    scenario_rows = {row["scenario"]: row for row in __import__("csv").DictReader((SUMMARY_ROOT / "scenario_known_only_summary.csv").open(encoding="utf-8"))}
    lines = [
        "# Experiment results: stage11c-known-only-support-complexity-20260915-v1",
        "",
        "## Configuration and execution",
        "",
        "- Status: `success`",
        "- Scope: `EXPLORATORY_ONLY_POST_HOC_DEVELOPMENT_DIAGNOSIS`",
        "- New training/detector fitting/threshold fitting: `NO/NO/NO`",
        "- Formal runs: `15/15`",
        f"- Final gate: **{gate['final_gate']}**",
        f"- Mechanism: **{gate['mechanism']}**",
        "",
        "## Data and split",
        "",
        "All criterion features use frozen Stage11B Known Train/Validation arrays and models. Outcomes are isolated post-hoc. No Known/Unknown Test representation is a feature.",
        "",
        "## Core results",
        "",
    ]
    for scenario in SCENARIOS:
        row = scenario_rows[scenario]
        lines.append(f"- {scenario.upper()}: DeltaNLL `{float(row['validation_delta_nll_full_mean']):.6f}`, Delta validation Macro-F1 `{float(row['validation_delta_macro_f1_full_mean']):.6f}`, instability `{float(row['covariance_bootstrap_frobenius_relative_mean']):.6f}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Only 15 USTC development runs are available. Correlations and selector outcomes are exploratory and retrospective, not external validation.",
            "",
            "## Preserved evidence",
            "",
            "All run-level inputs-by-reference, Known-only statistics, outcomes, rule decisions, LOSO results, diagnoses, commands, hashes and manifests are retained in this directory.",
            "",
            "## Conclusion and next step",
            "",
            f"Fixed result: {gate['final_gate']} / {gate['mechanism']}. No adaptive detector or next stage was started.",
        ]
    )
    (STAGE_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = (STAGE_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "\n## Recorded Final Result\n"
    if marker in readme:
        readme = readme.split(marker)[0].rstrip() + "\n"
    (STAGE_ROOT / "README.md").write_text(readme.rstrip() + marker + "\n" + (SUMMARY_ROOT / "final_gate.md").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest = read_json(STAGE_ROOT / "manifest.json")
    manifest.update(
        {
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "success",
            "experiment_type": "diagnostic",
            "claim_scope": "diagnostic",
            "objective": "Known-only support complexity diagnosis over 15 frozen Stage11B runs",
            "inputs": [{"path": config["stage11b_root"], "role": "frozen Stage11B evidence", "access": "read_only"}],
            "code": {"entrypoint": "scripts/analyze_run.py", "aggregator": "scripts/aggregate_results.py"},
            "execution": {"formal_runs": 15, "new_training": False, "new_detector_fitting": False, "unknown_feature_used": False},
            "configuration": config,
            "core_results": [{"name": "final_gate", "value": gate["final_gate"]}, {"name": "mechanism", "value": gate["mechanism"]}, {"name": "formal_runs", "value": 15}],
            "limitations": ["USTC development diagnosis", "15 samples for correlations", "Retrospective outcomes only", "No CipherSpectrum Test"],
            "next_step": "Stop after Stage11C",
        }
    )
    write_json(STAGE_ROOT / "manifest.json", manifest)
    refresh(STAGE_ROOT)

    index = UNKNOWN_ROOT / "EXPERIMENT_RESULTS.md"
    marker = "stage11c-known-only-support-complexity-20260915-v1"
    text = index.read_text(encoding="utf-8") if index.is_file() else "# Experiment results\n"
    if marker not in text:
        text += f"\n- `{marker}`: `{gate['final_gate']} / {gate['mechanism']}`; 15 frozen Known-only diagnoses. See [Stage11C](stage11c_known_only_support_complexity/RESULTS.md).\n"
        index.write_text(text, encoding="utf-8")
    print({"status": "PASS", "run_manifests": 15, "root_gate": gate["final_gate"]}, flush=True)


if __name__ == "__main__":
    main()
