#!/usr/bin/env python3
"""Train one formal Data-Anchored Prototype run on a frozen USTC v6 split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from stage11a_common import (
    CONFIG_PATH,
    DRIFT_FIELDS,
    STAGE_ROOT,
    anchor_prototypes,
    append_csv,
    array_hashes,
    assert_prototype_is_buffer,
    build_loaders,
    evaluate_checkpoint,
    initialised_dap_model,
    load_checkpoint,
    read_json,
    run_epoch,
    set_reproducible,
    sha256_file,
    write_json,
)


TRAINING_FIELDS = [
    "epoch",
    "learning_rate",
    "elapsed_seconds",
    "train_accuracy",
    "train_macro_f1",
    "train_total_loss",
    "train_rec_loss",
    "train_kld_loss",
    "train_ent_loss",
    "train_dis_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "validation_total_loss",
    "validation_rec_loss",
    "validation_kld_loss",
    "validation_ent_loss",
    "validation_dis_loss",
    "best_epoch",
    "best_validation_accuracy",
    "epochs_without_improvement",
    "prototype_anchored",
    "patience_phase_reset",
    "checkpoint_selection_data",
]


def save_last(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def rng_state(train_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "train_generator": train_generator.get_state(),
    }


def restore_rng(state: dict, train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["train_generator"])


def build_runtime_config(
    scenario: str,
    fold: int,
    seed: int,
    run_dir: Path,
    smoke: bool,
    epochs: int,
) -> dict:
    base = read_json(CONFIG_PATH)
    if scenario not in base["scenarios"]:
        raise ValueError(f"Unsupported scenario: {scenario}")
    if seed != base["seeds"][fold]:
        raise ValueError("Formal pairing requires fold index and seed to match v6")
    if not smoke:
        prepared = read_json(run_dir / "config.json")
        if prepared["scenario"] != scenario or prepared["fold"] != fold or prepared["seed"] != seed:
            raise RuntimeError("Prepared run config does not match requested run")
        return prepared
    split_root = Path(base["v6_split_root"])
    b0_root = Path(base["v6_frozen_root"])
    return {
        "evaluation_label": "SMOKE_ONLY",
        "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
        "scenario": scenario,
        "split": base["scenarios"][scenario]["split"],
        "fold": fold,
        "seed": seed,
        "method": "DAP",
        "data_dir": str((split_root / f"fold{fold}_seed{seed}").resolve()),
        "b0_frozen_run_dir": str((b0_root / scenario / f"fold{fold}_seed{seed}").resolve()),
        "stage11a_config": str(CONFIG_PATH.resolve()),
        "training": {**base["training"], "epochs": epochs},
        "density_diagnostics": base["density_diagnostics"],
        "unknown_use": "evaluation after frozen best checkpoint only",
        "checkpoint_selection": "Known Validation Accuracy only",
        "prototype_update_data": "Known Train deterministic mu only",
    }


def write_results_markdown(run_dir: Path, runtime: dict, results: dict) -> None:
    detection = results["native"]["detection"]
    alignment = results["prototype_alignment"]
    training = results["training"]
    text = f"""# Experiment results: stage11a-dap-{runtime['scenario']}-fold{runtime['fold']}-seed{runtime['seed']}

- Status: `success`
- Experiment type: `method-development-training`
- Claim scope: `diagnostic`
- Objective: Formal DAP training and frozen-checkpoint diagnostics on the paired USTC v6 development split.

## Data and split

- Scenario: `{runtime['scenario'].upper()}`; fold `{runtime['fold']}`; seed `{runtime['seed']}`.
- Prototype anchoring/density fit: Known Train deterministic mu only.
- Checkpoint selection: Known Validation Accuracy only.
- USTC Unknown Test: evaluation only after checkpoint freezing.

## Configuration and execution

- Prototype is a registered buffer and is excluded from the optimizer.
- Anchoring: before epoch 1 and after every complete training epoch.
- Loss, optimizer, scheduler, checkpoint metric, and early stopping match frozen v6 corrected B0.

## Core results

- Best/stop epoch: `{training['best_epoch']}` / `{training['stop_epoch']}`.
- Native AUROC / Binary F1: `{detection['auroc']:.9f}` / `{detection['binary_f1']:.9f}`.
- Native Known FRR / UFAR: `{detection['known_frr']:.9f}` / `{detection['ufar']:.9f}`.
- Normalized prototype gap mean/median/max: `{alignment['normalized_gap_mean']:.9g}` / `{alignment['normalized_gap_median']:.9g}` / `{alignment['normalized_gap_max']:.9g}`.

## Preserved evidence

- `config.json`, `training_log.csv`, `prototype_drift_history.csv`
- `model_best.pt`, `checkpoint.sha256`, `results.json`, `density_models.joblib`
- `manifest.json`

## Limitations

- `DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`.
- This run is paired to a local v6 reproduction, not an author-exact published fold.

## Conclusion and next step

This run is one member of the preregistered 15-run paired campaign. No next stage is started from an individual run.
"""
    (run_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a1", "a2", "a3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--skip-density", action="store_true")
    args = parser.parse_args()
    base = read_json(CONFIG_PATH)
    formal_epochs = int(base["training"]["epochs"])
    epochs = args.epochs if args.epochs is not None else formal_epochs
    if not args.smoke and epochs != formal_epochs:
        raise ValueError("Formal Stage 11A runs must use the frozen 100-epoch maximum")
    run_dir = args.output_dir or (
        STAGE_ROOT / "runs" / args.scenario / f"fold{args.fold}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime_config(args.scenario, args.fold, args.seed, run_dir, args.smoke, epochs)
    if args.smoke:
        write_json(run_dir / "config.json", runtime)
    if (run_dir / "SUCCESS").is_file():
        print(json.dumps({"status": "SKIP_VERIFIED_SUCCESS", "run_dir": str(run_dir)}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Open-Detect implementation")
    device = torch.device("cuda:0")
    set_reproducible(args.seed)
    train_cfg = runtime["training"]
    loaders = build_loaders(
        Path(runtime["data_dir"]),
        int(runtime["split"]),
        args.seed,
        int(train_cfg["batch_size"]),
        int(train_cfg["eval_batch_size"]),
        int(train_cfg["workers"]),
    )
    observed_hashes = array_hashes(loaders)
    b0_config = read_json(Path(runtime["b0_frozen_run_dir"]) / "run_config.json")
    if observed_hashes != b0_config["split_array_sha256"]:
        raise RuntimeError("DAP arrays do not match the paired frozen B0 split hashes")
    if loaders.known_classes != list(map(int, b0_config["known_classes"])):
        raise RuntimeError("DAP Known classes do not match the paired frozen B0")
    if loaders.unknown_classes != list(map(int, b0_config["unknown_classes"])):
        raise RuntimeError("DAP Unknown classes do not match the paired frozen B0")
    runtime["observed_split_array_sha256"] = observed_hashes
    runtime["known_classes"] = loaders.known_classes
    runtime["unknown_classes"] = loaders.unknown_classes
    runtime["dataset_sizes"] = b0_config["dataset_sizes"]
    runtime["physical_gpu_visible_as_cuda0"] = torch.cuda.get_device_name(device)
    write_json(run_dir / "config.json", runtime)

    model = initialised_dap_model(len(loaders.known_classes), int(train_cfg["latent_dim"]), device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["adam_betas"]),
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, train_cfg["scheduler_milestones"])),
        gamma=float(train_cfg["scheduler_gamma"]),
    )
    assert_prototype_is_buffer(model, optimizer)
    drift_path = run_dir / "prototype_drift_history.csv"
    training_path = run_dir / "training_log.csv"
    last_path = run_dir / "checkpoint_last.pt"
    best_path = run_dir / "model_best.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng(state["rng_state"], loaders.train.generator)
        start_epoch = int(state["completed_epochs"])
        best_epoch = int(state["best_epoch"])
        best_accuracy = float(state["best_accuracy"])
        patience_reference = float(state["patience_reference"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        stopped_early = bool(state["stopped_early"])
        print(f"resume completed_epochs={start_epoch}", flush=True)
    else:
        initial_rows = anchor_prototypes(
            model,
            loaders.anchor_train,
            device,
            loaders.known_classes,
            epoch=0,
            phase="PRE_EPOCH_1_INITIALIZATION",
        )
        append_csv(drift_path, initial_rows, DRIFT_FIELDS)
        assert_prototype_is_buffer(model, optimizer)
        start_epoch = 0
        best_epoch = -1
        best_accuracy = -1.0
        patience_reference = -1.0
        epochs_without_improvement = 0
        stopped_early = False

    started = time.time()
    epoch_limit = start_epoch if stopped_early else epochs
    for epoch_index in range(start_epoch, epoch_limit):
        epoch_started = time.time()
        train_metrics = run_epoch(model, loaders.train, device, float(train_cfg["lambda"]), optimizer)
        drift_rows = anchor_prototypes(
            model,
            loaders.anchor_train,
            device,
            loaders.known_classes,
            epoch=epoch_index + 1,
            phase="POST_TRAIN_EPOCH_ANCHOR",
        )
        append_csv(drift_path, drift_rows, DRIFT_FIELDS)
        assert_prototype_is_buffer(model, optimizer)
        validation_metrics = run_epoch(
            model, loaders.validation, device, float(train_cfg["lambda"]), None
        )
        scheduler.step()
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(validation_metrics["accuracy"])
            best_epoch = epoch_index + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_classes": len(loaders.known_classes),
                    "latent_dim": int(train_cfg["latent_dim"]),
                    "epoch": best_epoch,
                    "validation_accuracy": best_accuracy,
                    "validation_macro_f1": float(validation_metrics["macro_f1"]),
                    "prototype_storage": "registered_buffer",
                    "prototype_update": "KNOWN_TRAIN_ANCHOR_EVERY_EPOCH",
                },
                best_path,
            )
        if validation_metrics["accuracy"] > (
            patience_reference + float(train_cfg["early_stop_min_delta"])
        ):
            patience_reference = float(validation_metrics["accuracy"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # Preserve v6's two patience-phase boundaries without performing its
        # prototype resets: DAP's only reset mechanism is the every-epoch anchor.
        patience_phase_reset = epoch_index in (50, 80)
        if patience_phase_reset:
            patience_reference = float(validation_metrics["accuracy"])
            epochs_without_improvement = 0
        completed_epochs = epoch_index + 1
        elapsed = time.time() - epoch_started
        row = {
            "epoch": completed_epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_total_loss": train_metrics["total"],
            "train_rec_loss": train_metrics["rec"],
            "train_kld_loss": train_metrics["kld"],
            "train_ent_loss": train_metrics["ent"],
            "train_dis_loss": train_metrics["dis"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_total_loss": validation_metrics["total"],
            "validation_rec_loss": validation_metrics["rec"],
            "validation_kld_loss": validation_metrics["kld"],
            "validation_ent_loss": validation_metrics["ent"],
            "validation_dis_loss": validation_metrics["dis"],
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
            "prototype_anchored": True,
            "patience_phase_reset": patience_phase_reset,
            "checkpoint_selection_data": "KNOWN_VALIDATION_ONLY",
        }
        append_csv(training_path, [row], TRAINING_FIELDS)
        if (
            int(train_cfg["early_stop_patience"]) > 0
            and completed_epochs >= int(train_cfg["early_stop_min_epoch"])
            and epochs_without_improvement >= int(train_cfg["early_stop_patience"])
        ):
            stopped_early = True
        save_last(
            last_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "completed_epochs": completed_epochs,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "patience_reference": patience_reference,
                "epochs_without_improvement": epochs_without_improvement,
                "stopped_early": stopped_early,
                "rng_state": rng_state(loaders.train.generator),
            },
        )
        print(
            f"epoch={completed_epochs:03d}/{epochs} train_acc={train_metrics['accuracy']:.6f} "
            f"val_acc={validation_metrics['accuracy']:.6f} val_macro_f1={validation_metrics['macro_f1']:.6f} "
            f"best_epoch={best_epoch} seconds={elapsed:.1f}",
            flush=True,
        )
        if stopped_early:
            print(f"early_stop epoch={completed_epochs}", flush=True)
            break

    completed_epochs = locals().get("completed_epochs", start_epoch)
    checkpoint = load_checkpoint(model, best_path, device)
    assert_prototype_is_buffer(model, optimizer)
    density_path = run_dir / "density_models.joblib"
    if args.skip_density:
        from stage11a_common import extract_outputs, prototype_alignment, classification_metrics, detection_metrics

        train_output = extract_outputs(model, loaders.anchor_train, device)
        val_output = extract_outputs(model, loaders.validation, device)
        known_output = extract_outputs(model, loaders.known_test, device)
        unknown_output = extract_outputs(model, loaders.unknown_test, device)
        _, alignment = prototype_alignment(model, train_output, loaders.known_classes)
        results = {
            "evaluation_label": "SMOKE_ONLY",
            "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
            "scenario": args.scenario,
            "seed": args.seed,
            "fold": args.fold,
            "method": "DAP",
            "prototype_alignment": alignment,
            "native": {
                "validation_known": classification_metrics(val_output.labels, val_output.predictions),
                "known_test": classification_metrics(known_output.labels, known_output.predictions),
                "detection": detection_metrics(
                    val_output.native_scores,
                    known_output.native_scores,
                    unknown_output.native_scores,
                    int(train_cfg["eval_seed"]),
                ),
            },
            "density_diagnostics": {"status": "SKIPPED_FOR_SMOKE"},
        }
    else:
        results = evaluate_checkpoint(
            model,
            loaders,
            device,
            "DAP",
            args.scenario,
            args.seed,
            args.fold,
            density_path,
        )
    results["training"] = {
        "best_epoch": best_epoch,
        "stop_epoch": completed_epochs,
        "configured_epochs": epochs,
        "stopped_early": stopped_early,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
        "elapsed_seconds_this_invocation": time.time() - started,
        "checkpoint_selection": "KNOWN_VALIDATION_ACCURACY_ONLY",
        "unknown_used_for_training_or_selection": False,
        "prototype_in_optimizer": False,
        "prototype_update": "ANCHOR_EVERY_EPOCH",
    }
    checkpoint_hash = sha256_file(best_path)
    results["checkpoint_sha256"] = checkpoint_hash
    (run_dir / "checkpoint.sha256").write_text(
        f"{checkpoint_hash}  model_best.pt\n", encoding="utf-8"
    )
    write_json(run_dir / "results.json", results)
    write_results_markdown(run_dir, runtime, results)
    (run_dir / "SUCCESS").write_text(
        "STAGE11A_DAP_SMOKE_SUCCESS\n" if args.smoke else "STAGE11A_DAP_FORMAL_SUCCESS\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "SUCCESS", "run_dir": str(run_dir), "results": results}, indent=2))


if __name__ == "__main__":
    main()
