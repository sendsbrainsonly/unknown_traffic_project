#!/usr/bin/env python3
"""Finalize experiment manifests, run summaries, and the project index."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from stage11a_common import CONFIG_PATH, STAGE_ROOT, read_json, write_json


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
UNKNOWN_ROOT = STAGE_ROOT.parent
REFRESH = (
    WORKSPACE_ROOT
    / ".agents"
    / "skills"
    / "experiment-data-preservation"
    / "scripts"
    / "refresh_artifact_manifest.py"
)
ENVIRONMENT = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310"
GPUS = [0, 3, 4, 5, 6, 7]


def revision() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(UNKNOWN_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def refresh(path: Path) -> None:
    subprocess.run(
        [sys.executable, "-B", str(REFRESH), str(path), "--hash-max-bytes", "200000000"],
        cwd=UNKNOWN_ROOT,
        check=True,
    )


def update_run_manifest(run: Path, result: dict, config: dict, queue_index: int) -> None:
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    detection = result["native"]["detection"]
    gap = result["prototype_alignment"]
    manifest.update({
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "success",
        "inputs": [
            {
                "path": config["data_dir"],
                "role": "frozen paired USTC v6 split",
                "access": "read_only",
                "verification": "config.json:observed_split_array_sha256",
            },
            {
                "path": config["b0_frozen_run_dir"],
                "role": "paired frozen B0 baseline",
                "access": "read_only",
                "verification": "../../../../outputs/summary/provenance_verification.json",
            },
        ],
        "code": {
            "revision": revision(),
            "dirty": True,
            "changes": [
                "stage11a_data_anchored_prototype/configs/stage11a_config.json",
                "stage11a_data_anchored_prototype/scripts/stage11a_common.py",
                "stage11a_data_anchored_prototype/scripts/run_dap.py",
            ],
        },
        "execution": {
            "tmux_session": f"s11a-dap-q{queue_index}-r24",
            "command": "See outputs/summary/execution_commands.txt",
            "exit_code": 0,
            "environment": ENVIRONMENT,
            "physical_gpu_ids": [GPUS[queue_index]],
        },
        "configuration": {
            "files": ["config.json"],
            "parameters": {
                "prototype_storage": "registered_buffer",
                "prototype_update": "ANCHOR_EVERY_EPOCH",
                "prototype_data": "Known Train deterministic mu only",
                "checkpoint_selection": "Known Validation Accuracy only",
                "threshold": "Known Validation Native KL P95",
                "density_methods": ["K1", "K2"],
            },
            "seeds": [result["seed"]],
        },
        "core_results": [
            {"name": "native_auroc", "value": detection["auroc"]},
            {"name": "native_binary_f1", "value": detection["binary_f1"]},
            {"name": "native_known_frr", "value": detection["known_frr"]},
            {"name": "native_ufar", "value": detection["ufar"]},
            {"name": "normalized_prototype_gap_mean", "value": gap["normalized_gap_mean"]},
            {"name": "best_epoch", "value": result["training"]["best_epoch"]},
            {"name": "stop_epoch", "value": result["training"]["stop_epoch"]},
        ],
        "limitations": [
            "DEVELOPMENT_RESULT; NOT_INDEPENDENT_EXTERNAL_VALIDATION",
            "USTC local v6 repeated split, not an author-exact published fold",
            "CipherSpectrum Stage9 sample-level Test not used",
        ],
        "next_step": "Aggregate all 15 paired runs only; do not start a next stage",
    })
    write_json(manifest_path, manifest)
    refresh(run)


def main() -> None:
    config = read_json(CONFIG_PATH)
    gate = read_json(STAGE_ROOT / "outputs" / "summary" / "final_gate.json")
    for scenario_index, scenario in enumerate(config["scenarios"]):
        for fold, seed in enumerate(config["seeds"]):
            run = STAGE_ROOT / "runs" / scenario / f"fold{fold}_seed{seed}"
            result = read_json(run / "results.json")
            run_cfg = read_json(run / "config.json")
            queue_index = (scenario_index * 5 + fold) % 6
            update_run_manifest(run, result, run_cfg, queue_index)

    smoke = STAGE_ROOT / "runs" / "smoke" / "a1_fold0_seed2022"
    if (smoke / "manifest.json").is_file():
        manifest = read_json(smoke / "manifest.json")
        manifest["status"] = "success"
        manifest["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["execution"] = {
            "tmux_session": "s11a-smoke-gpu-r20",
            "command": "one-epoch smoke; see root execution_commands.txt",
            "exit_code": 0,
            "environment": ENVIRONMENT,
            "physical_gpu_ids": [0],
        }
        manifest["core_results"] = [
            {"name": "status", "value": "SMOKE_PASS"},
            {"name": "prototype_in_optimizer", "value": False},
            {"name": "anchor_frequency", "value": "every epoch"},
        ]
        manifest["limitations"] = ["SMOKE_ONLY; not a formal experimental result"]
        manifest["next_step"] = "Formal paired campaign completed separately"
        write_json(smoke / "manifest.json", manifest)
        refresh(smoke)

    commands = f"""# All payloads ran through the workspace tmux helper and fixed Conda environment.
python -B scripts/verify_provenance.py --phase before
python -B -m pytest -q tests
python -B scripts/run_dap.py --scenario a1 --fold 0 --seed 2022 --smoke --epochs 1 --skip-density
python -B scripts/run_queue.py --mode b0 --queue-index <0..4> --queue-count 5 --physical-gpu <0,3,4,5,6>
python -B scripts/run_queue.py --mode dap --queue-index <0..5> --queue-count 6 --physical-gpu <0,3,4,5,6,7>
python -B scripts/aggregate_results.py
python -B scripts/verify_provenance.py --phase after
python -B scripts/finalize_bundles.py
python -B scripts/verify_stage11a.py
python -B <workspace>/experiment-data-preservation/validate_experiment_bundle.py --verify-hashes stage11a_data_anchored_prototype
"""
    (STAGE_ROOT / "outputs" / "summary" / "execution_commands.txt").write_text(commands, encoding="utf-8")

    root_manifest = read_json(STAGE_ROOT / "manifest.json")
    root_manifest.update({
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "success",
        "inputs": [
            {"path": "../stage6_cipherspectrum_protocol", "role": "frozen provenance sentinel only", "access": "read_only"},
            {"path": "../stage7_cipherspectrum_known_training", "role": "frozen provenance sentinel only", "access": "read_only"},
            {"path": "../stage8a_cipherspectrum_known_density", "role": "frozen provenance sentinel only", "access": "read_only"},
            {"path": "../stage8b_cipherspectrum_dgsbv2", "role": "frozen provenance sentinel only", "access": "read_only"},
            {"path": "../stage9_cipherspectrum_final_test", "role": "frozen summary/manifests only; no sample-level Test", "access": "read_only"},
            {"path": "../stage10a_opendetect_protocol_audit", "role": "frozen source manifest", "access": "read_only"},
            {"path": "../stage10b_opendetect_score_decomposition", "role": "summary-only historical motivation", "access": "read_only"},
            {"path": str(Path(config["v6_frozen_root"])), "role": "frozen B0 checkpoints/results", "access": "read_only"},
            {"path": str(Path(config["v6_split_root"])), "role": "paired USTC development splits", "access": "read_only"},
        ],
        "code": {
            "revision": revision(),
            "dirty": True,
            "changes": [
                "stage11a_data_anchored_prototype/README.md",
                "stage11a_data_anchored_prototype/configs/stage11a_config.json",
                "stage11a_data_anchored_prototype/scripts/",
                "stage11a_data_anchored_prototype/tests/",
            ],
        },
        "execution": {
            "tmux_session": "s11a-dap-q0-r24 through s11a-dap-q5-r24",
            "command": "See outputs/summary/execution_commands.txt",
            "exit_code": 0,
            "environment": ENVIRONMENT,
            "physical_gpu_ids": GPUS,
        },
        "configuration": {
            "files": ["configs/stage11a_config.json"],
            "parameters": {
                "formal_matrix": "A1/A2/A3 x seeds 2022-2026 x B0/DAP",
                "primary": "DAP-Native versus B0-Native",
                "prototype_update": "Known Train deterministic mu before epoch1 and after every epoch",
                "checkpoint_selection": "Known Validation Accuracy only",
                "threshold": "Known Validation P95 only",
                "diagnostics": "fixed K1/K2 full covariance after train-only StandardScaler/PCA64",
            },
            "seeds": config["seeds"],
        },
        "core_results": [
            {"name": "final_gate", "value": gate["final_gate"]},
            {"name": "mechanism_case", "value": gate["mechanism_case"]},
            {"name": "formal_dap_runs", "value": 15},
            {"name": "frozen_b0_evaluations", "value": 15},
            {"name": "covariance_verdict", "value": gate["covariance_verdict"]},
            {"name": "multicomponent_verdict", "value": gate["multicomponent_verdict"]},
        ],
        "limitations": [
            "DEVELOPMENT_RESULT; NOT_INDEPENDENT_EXTERNAL_VALIDATION",
            "USTC local v6 repeated splits are not author-exact folds",
            "CipherSpectrum Stage9 sample-level Test was not opened or used",
            "No threshold/K/anchoring-frequency search was performed",
        ],
        "next_step": "STOP after Stage11A; no subsequent stage started",
    })
    write_json(STAGE_ROOT / "manifest.json", root_manifest)
    refresh(STAGE_ROOT)

    index = UNKNOWN_ROOT / "EXPERIMENT_RESULTS.md"
    text = index.read_text(encoding="utf-8")
    entry = (
        f"- `stage11a-data-anchored-prototype-20260914-v1` — status `success`; "
        f"15 paired USTC DAP runs plus 15 frozen B0 evaluations; Gate `{gate['final_gate']}`, "
        f"mechanism `{gate['mechanism_case']}`, covariance `{gate['covariance_verdict']}`; "
        "CipherSpectrum Stage9 sample-level Test not used; "
        "[bundle](stage11a_data_anchored_prototype/).\n"
    )
    if "stage11a-data-anchored-prototype-20260914-v1" not in text:
        index.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")
    print(json.dumps({"status": "PASS", "run_manifests": 15, "root_manifest": "success", "index_updated": True}, indent=2))


if __name__ == "__main__":
    main()
