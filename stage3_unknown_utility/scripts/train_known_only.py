#!/usr/bin/env python3
"""Independently train one frozen-setting Known-only Open-Detect encoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from common import (
    AUDIT_ROOT,
    EXPECTED_EXECUTION_PLAN_RAW_SHA256,
    EXPECTED_PROTOCOL_CANONICAL_SHA256,
    STAGE3_ROOT,
    KnownImageDataset,
    balanced_positions,
    load_fold,
    seed_everything,
    sha256_file,
    verify_frozen_inputs,
)


sys.path.insert(0, str(AUDIT_ROOT))
from adapters.opendetect_model import released_weight_init  # noqa: E402
from stage3_model import Stage3OpenDetectNet  # noqa: E402
from scripts.train_opendetect import (  # noqa: E402
    released_reset_prototypes,
    run_epoch,
    validation_composite,
)


UPSTREAM_CODE = AUDIT_ROOT.parent.parent / "Open-Detect/code"


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_leakage_gate(setting: str) -> dict[str, object]:
    output_dir = STAGE3_ROOT / "outputs" / setting
    audit_path = output_dir / "leakage_audit.json"
    manifest_path = output_dir / "data_manifest.csv"
    if not audit_path.exists() or not manifest_path.exists():
        raise RuntimeError("leakage audit must run before any GPU training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("setting") != setting:
        raise RuntimeError("leakage gate is not PASS for this setting")
    if audit["data_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("data manifest changed after leakage audit")
    if any(int(value) for value in audit["unknown_forbidden_usage_counts"].values()):
        raise RuntimeError("Unknown leakage assertion is nonzero")
    return audit


def restore_rng(payload: dict[str, object], generator: torch.Generator) -> None:
    if "python_rng_state" in payload:
        import random

        random.setstate(payload["python_rng_state"])
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_states" in payload:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
    if "data_generator_state" in payload:
        generator.set_state(payload["data_generator_state"].cpu())


def checkpoint_payload(
    model: Stage3OpenDetectNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.MultiStepLR,
    generator: torch.Generator,
    epoch: int,
    val_metrics: dict[str, float],
    config: dict[str, object],
    best_epoch: int,
    best_score: float,
    epochs_without_improvement: int,
) -> dict[str, object]:
    import random

    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "data_generator_state": generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": config,
        "best_epoch": best_epoch,
        "best_validation_composite": best_score,
        "epochs_without_improvement": epochs_without_improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--lamda", type=float, default=0.005)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--temp-inter", type=float, default=1.0)
    parser.add_argument("--temp-intra", type=float, default=1.0)
    parser.add_argument("--smoke-train-per-class", type=int, default=32)
    parser.add_argument("--smoke-val-per-class", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    verify_frozen_inputs()
    audit = validate_leakage_gate(args.setting)
    fold = load_fold(args.setting)
    if not torch.cuda.is_available():
        raise RuntimeError("Known-only Open-Detect training requires CUDA")
    if args.early_stopping_patience != 5:
        raise RuntimeError("formal Stage 3 freezes early-stopping patience at 5")
    if args.batch_size != 512:
        raise RuntimeError("formal Stage 3 freezes batch size at 512")

    output_dir = STAGE3_ROOT / "outputs" / args.setting
    artifact_dir = STAGE3_ROOT / "artifacts" / args.setting
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_name = "smoke_metrics.csv" if args.mode == "smoke" else "training_metrics.csv"
    config_name = "smoke_training_config.json" if args.mode == "smoke" else "training_config.json"
    metrics_path = output_dir / metrics_name
    config_path = output_dir / config_name
    checkpoint_path = artifact_dir / ("smoke_checkpoint.pt" if args.mode == "smoke" else "best_checkpoint.pt")
    latest_path = artifact_dir / ("smoke_latest_checkpoint.pt" if args.mode == "smoke" else "latest_checkpoint.pt")
    if args.resume is None and any(path.exists() for path in (metrics_path, config_path, checkpoint_path, latest_path)):
        raise RuntimeError("refusing to overwrite an existing training run; use --resume for the same setting")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded")
    epochs = args.epochs if args.epochs is not None else (3 if args.mode == "smoke" else 100)
    train_base = KnownImageDataset("train", fold, augment=True)
    val_base = KnownImageDataset("val", fold, augment=False)
    expected = fold["formal_counts"]
    if len(train_base) != expected["known_train"] or len(val_base) != expected["known_validation"]:
        raise RuntimeError("Known train/validation count differs from frozen fold")
    train_set: Dataset = train_base
    val_set: Dataset = val_base
    if args.mode == "smoke":
        train_set = Subset(train_base, balanced_positions(train_base, args.smoke_train_per_class, args.seed))
        val_set = Subset(val_base, balanced_positions(val_base, args.smoke_val_per_class, args.seed + 1))

    generator = torch.Generator().manual_seed(args.seed)
    common_loader = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common_loader)
    val_loader = DataLoader(val_set, shuffle=False, **common_loader)
    num_classes = int(fold["known_count"])
    model = Stage3OpenDetectNet(
        upstream_code=UPSTREAM_CODE,
        channels=1,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        temp_inter=args.temp_inter,
        temp_intra=args.temp_intra,
    ).to(device)
    model.apply(released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    rows: list[dict[str, object]] = []
    start_epoch = 0
    best_epoch = -1
    best_score = -math.inf
    epochs_without_improvement = 0
    if args.resume is not None:
        if args.resume.resolve() != latest_path.resolve():
            raise RuntimeError("resume is allowed only from this setting's own latest checkpoint")
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload["config"]["setting"] != args.setting or payload["config"]["mode"] != args.mode:
            raise RuntimeError("cross-setting or cross-mode resume is forbidden")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_validation_composite"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        restore_rng(payload, generator)
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle) if int(row["epoch"]) <= start_epoch]

    config = {
        "setting": args.setting,
        "mode": args.mode,
        "primary_setting": True,
        "protocol_canonical_sha256": EXPECTED_PROTOCOL_CANONICAL_SHA256,
        "execution_plan_sha256": EXPECTED_EXECUTION_PLAN_RAW_SHA256,
        "leakage_manifest_sha256": audit["data_manifest_sha256"],
        "known_classes": fold["known_classes"],
        "unknown_classes_excluded": fold["unknown_classes"],
        "unknown_samples_loaded": 0,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam(beta1=0.9,beta2=0.999)",
        "scheduler": "MultiStepLR(milestones=[50,80],gamma=0.1)",
        "lambda": args.lamda,
        "latent_dimension": args.latent_dim,
        "num_classes": num_classes,
        "prototype_count": num_classes,
        "checkpoint_selection": "highest harmonic mean of Known Validation Accuracy and Macro-F1",
        "early_stopping_monitor": "Known Validation composite only",
        "early_stopping_patience": args.early_stopping_patience,
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "formal_train_samples": len(train_base),
        "formal_validation_samples": len(val_base),
        "initialization": "released_weight_init from scratch; no pretrained checkpoint",
        "numerical_stability_guard": "when raw logvar exceeds 20, clamp only its upper tail to 20; preserve all negative values",
        "generic_pretrained_backbone": None,
        "warm_start": False,
        "forbidden_20class_checkpoint_used": False,
        "physical_gpu": physical_gpu,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "upstream_code": str(UPSTREAM_CODE.resolve()),
        "upstream_commit": "f26ac911be4324110a6632912388cd19413353bd",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    start = time.time()
    stop_epoch = start_epoch
    for epoch_index in range(start_epoch, epochs):
        guard_before_train = model.stability_guard_activations
        train_metrics = run_epoch(model, train_loader, device, args.lamda, optimizer)
        guard_after_train = model.stability_guard_activations
        reset = epoch_index in (50, 80)
        if reset:
            model.prototypes = released_reset_prototypes(model, train_loader, device)
        val_metrics = run_epoch(model, val_loader, device, args.lamda, None)
        guard_after_val = model.stability_guard_activations
        scheduler.step()
        score = validation_composite(val_metrics["accuracy"], val_metrics["macro_f1"])
        row: dict[str, object] = {
            "epoch": epoch_index + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "prototype_reset": reset,
            "elapsed_seconds": time.time() - start,
            "val_composite_score": score,
            "stability_guard_train_activations": guard_after_train - guard_before_train,
            "stability_guard_val_activations": guard_after_val - guard_after_train,
            "stability_guard_cumulative_activations": guard_after_val,
        }
        for prefix, values in (("train", train_metrics), ("val", val_metrics)):
            for name, value in values.items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
        write_rows(metrics_path, rows)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            generator,
            epoch_index + 1,
            val_metrics,
            config,
            best_epoch,
            best_score,
            epochs_without_improvement,
        )
        if improved:
            torch.save(payload, checkpoint_path)
        torch.save(payload, latest_path)
        stop_epoch = epoch_index + 1
        print(
            json.dumps(
                {
                    "setting": args.setting,
                    "mode": args.mode,
                    "epoch": stop_epoch,
                    "train_total": train_metrics["total"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_total": val_metrics["total"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_composite": score,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            print(json.dumps({"event": "early_stopping", "stop_epoch": stop_epoch, "best_epoch": best_epoch}), flush=True)
            break

    best_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selection = {
        "setting": args.setting,
        "mode": args.mode,
        "selection_data": "Known Validation only",
        "unknown_samples_used": 0,
        "best_epoch": int(best_payload["epoch"]),
        "stop_epoch": stop_epoch,
        "val_accuracy": float(best_payload["val_metrics"]["accuracy"]),
        "val_macro_f1": float(best_payload["val_metrics"]["macro_f1"]),
        "combined_score": float(best_payload["best_validation_composite"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "early_stopping_patience": args.early_stopping_patience,
        "stability_guard_cumulative_activations": model.stability_guard_activations,
    }
    selection_name = "smoke_checkpoint_selection.json" if args.mode == "smoke" else "checkpoint_selection.json"
    (output_dir / selection_name).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "smoke":
        finite = all(
            np.isfinite(float(row[key]))
            for row in rows
            for key in ("train_total", "train_rec", "train_kld", "train_dis", "val_total")
        )
        prototype_ok = tuple(model.prototypes.shape) == (num_classes, args.latent_dim)
        status = "PASS" if finite and prototype_ok and audit["unknown_forbidden_usage_counts"] == {
            key: 0 for key in audit["unknown_forbidden_usage_counts"]
        } else "FAIL"
        report = f"""# {args.setting} Open-Detect Smoke Report

- Status: **{status}**
- Unknown samples loaded: 0
- Known classes / prototypes: {num_classes} / {model.prototypes.shape[0]}
- Prototype shape: `{list(model.prototypes.shape)}`
- Latent `mu`/`logvar` dimension: {args.latent_dim}
- Train / validation samples: {len(train_set)} / {len(val_set)}
- Epochs completed: {stop_epoch}
- Total and component losses finite: {str(finite).lower()}
- NaN/Inf observed: no
- Upper-logvar stability-guard activations: {model.stability_guard_activations}
- Best validation composite: {best_score:.9f}
- Physical GPU: `{physical_gpu}`

The smoke graph exercised raw-byte images, encoder `mu`/`logvar`, stochastic
latent sampling, {num_classes} learned prototypes, decoder reconstruction, and
all released loss terms. No held-out Unknown sample was fetched.
"""
        (output_dir / "smoke_report.md").write_text(report, encoding="utf-8")
        if status != "PASS":
            raise RuntimeError("smoke gate failed")
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
