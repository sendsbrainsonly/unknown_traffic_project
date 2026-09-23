#!/usr/bin/env python3
"""Finalize one Stage 7 run bundle from its preserved input/training evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import PROJECT_ROOT, STAGE7_ROOT, VALID_SETTINGS, sha256_file


def relative(path: Path, run_dir: Path) -> str:
    return os.path.relpath(path.resolve(), run_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--input-session", required=True)
    parser.add_argument("--training-session", required=True)
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    results_path = args.run_dir / "RESULTS.md"
    if not manifest_path.is_file() or not results_path.is_file():
        raise RuntimeError("run bundle was not initialized")
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefix = "smoke" if args.mode == "smoke" else "training"
    input_dir = args.run_dir / "inputs"
    input_audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    config = json.loads((args.run_dir / f"{prefix}_config.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (args.run_dir / f"{prefix}_checkpoint_selection.json").read_text(encoding="utf-8")
    )
    with (args.run_dir / f"{prefix}_metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    if not metrics:
        raise RuntimeError("training metrics are empty")
    if input_audit["status"] != "PASS":
        raise RuntimeError("cannot finalize a failed input gate as success")
    if selection["unknown_samples_used"] != 0 or selection["known_test_samples_used"] != 0:
        raise RuntimeError("Known Test/Unknown usage boundary violated")
    checkpoint = args.run_dir / f"artifacts/{prefix}_best_checkpoint.pt"
    if selection["checkpoint_sha256"] != sha256_file(checkpoint):
        raise RuntimeError("selected checkpoint hash mismatch")

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    known_count = len(input_audit["known_classes"])
    unknown_count = len(input_audit["unknown_classes"])
    train_count = int(input_audit["selected_role_counts"]["KNOWN_TRAIN"])
    validation_count = int(input_audit["selected_role_counts"]["KNOWN_VALIDATION"])
    first_loss = float(metrics[0]["train_total"])
    last_loss = float(metrics[-1]["train_total"])
    finite = all(
        math.isfinite(float(value))
        for row in metrics
        for key, value in row.items()
        if key.startswith("train_") or key.startswith("val_")
    )
    limitations = [
        "No Known Test or Unknown evaluation was performed."
    ]
    if args.mode == "smoke":
        limitations.insert(0, "Three-epoch balanced-subset smoke test only; not a performance result.")
        next_step = "Run remaining setting smoke gates; after all pass, build formal Known-only inputs and train independently."
    else:
        limitations.insert(0, "Checkpoint selection used Known Validation only; external performance is not measured in this run.")
        next_step = "Keep the checkpoint frozen; do not access Unknown until all formal encoders and comparison choices are frozen."

    results = f"""# Experiment results: {original['experiment_id']}

- Status: `success`
- Experiment type: `{original['experiment_type']}`
- Claim scope: `{original['claim_scope']}`
- Created (UTC): `{original['created_at_utc']}`
- Completed (UTC): `{now}`
- Objective: {original['objective']}

## Data and split

- Frozen setting: {args.setting.title()}, {known_count} Known / {unknown_count} Unknown classes.
- Encoded Known Train/Validation: {train_count:,}/{validation_count:,}; failures: {input_audit['input_failures']}.
- Known Test PCAPs opened/used: {input_audit['known_test_pcap_files_opened']}/0.
- Unknown PCAPs opened/used: {input_audit['unknown_pcap_files_opened']}/0.
- Stage 6 protocol hashes: {input_audit['stage6_protocol_hashes_verified']}/16 verified before input construction and training.
- Input manifest SHA-256: `{input_audit['input_manifest_sha256']}`.

## Configuration and execution

- GPU: physical {config['physical_gpu']} ({config['cuda_device_name']}).
- Seed/epochs completed/batch: {config['seed']}/{selection['stop_epoch']}/{config['batch_size']}.
- Open-Detect: {known_count} prototypes, latent dimension {config['latent_dimension']}, released initialization from scratch.
- Checkpoint monitor: harmonic mean of Known Validation Accuracy and Macro-F1; patience {config['early_stopping_patience']}.
- Input session: `{args.input_session}`, exit 0.
- Training session: `{args.training_session}`, exit 0.

## Core results

- Training total loss: {first_loss:.9f} at epoch 1 to {last_loss:.9f} at final epoch.
- Best/final epoch: {selection['best_epoch']}/{selection['stop_epoch']}.
- Best Known Validation Accuracy/Macro-F1/composite: {selection['val_accuracy']:.9f}/{selection['val_macro_f1']:.9f}/{selection['combined_score']:.9f}.
- Recorded metrics finite: {str(finite).lower()}; stability-guard activations: {selection['stability_guard_cumulative_activations']}.
- Checkpoint SHA-256: `{selection['checkpoint_sha256']}`.
- Unknown samples used: 0; Known Test samples used: 0; Unknown inference: false.

## Preserved evidence

- `inputs/`: images, labels, sample IDs, label map, input manifest, failure ledger, and input audit.
- `artifacts/`: best and latest checkpoints.
- `{prefix}_metrics.csv`, `{prefix}_config.json`, `{prefix}_checkpoint_selection.json`.
- Project-local tmux logs/status are referenced by `manifest.json`.

## Limitations

{chr(10).join(f'- {item}' for item in limitations)}

## Conclusion and next step

- {next_step}
"""
    results_path.write_text(results, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment_id": original["experiment_id"],
        "created_at_utc": original["created_at_utc"],
        "updated_at_utc": now,
        "status": "success",
        "experiment_type": original["experiment_type"],
        "claim_scope": original["claim_scope"],
        "objective": original["objective"],
        "inputs": [
            {
                "path": relative(Path(input_audit["stage6_split_manifest"]), args.run_dir),
                "sha256": input_audit["stage6_split_manifest_sha256"],
            },
            {
                "path": relative(
                    PROJECT_ROOT / f"stage6_cipherspectrum_protocol/outputs/splits/{args.setting}_fold.json",
                    args.run_dir,
                ),
                "sha256": input_audit["stage6_fold_sha256"],
            },
            {
                "path": "inputs/input_manifest.csv",
                "sha256": input_audit["input_manifest_sha256"],
            },
        ],
        "code": {
            "revision": revision,
            "dirty": dirty,
            "changes": ["stage7_cipherspectrum_known_training"],
        },
        "execution": {
            "tmux_session": args.training_session,
            "related_sessions": [args.input_session],
            "command": (
                "select_gpu.py --allowed 0,3,4,5,6,7 --min-free-gb 10 -- "
                f"python -u stage7_cipherspectrum_known_training/scripts/train_known_encoder.py "
                f"--setting {args.setting} --mode {args.mode} --run-dir {args.run_dir}"
            ),
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": [int(config["physical_gpu"])],
            "log": relative(PROJECT_ROOT / f".tmux-task/{args.training_session}/output.log", args.run_dir),
            "exit_status": relative(PROJECT_ROOT / f".tmux-task/{args.training_session}/exit.status", args.run_dir),
        },
        "configuration": {
            "files": [
                relative(STAGE7_ROOT / "configs/training_config.json", args.run_dir),
                f"{prefix}_config.json",
            ],
            "parameters": {
                "setting": args.setting,
                "known_classes": known_count,
                "unknown_classes": unknown_count,
                "epochs_completed": int(selection["stop_epoch"]),
                "batch_size": int(config["batch_size"]),
                "early_stopping_patience": int(config["early_stopping_patience"]),
                "checkpoint_monitor": "harmonic_mean(val_accuracy,val_macro_f1)",
            },
            "seeds": [int(config["seed"])],
        },
        "core_results": [
            {"metric": "input_failures", "value": int(input_audit["input_failures"])},
            {"metric": "train_total_epoch_1", "value": first_loss},
            {"metric": "train_total_final_epoch", "value": last_loss},
            {"metric": "best_validation_accuracy", "value": selection["val_accuracy"]},
            {"metric": "best_validation_macro_f1", "value": selection["val_macro_f1"]},
            {"metric": "best_validation_composite", "value": selection["combined_score"]},
            {"metric": "unknown_samples_used", "value": 0},
            {"metric": "known_test_samples_used", "value": 0},
        ],
        "artifacts": [],
        "limitations": limitations,
        "next_step": next_step,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "setting": args.setting, "mode": args.mode}, sort_keys=True))


if __name__ == "__main__":
    main()
