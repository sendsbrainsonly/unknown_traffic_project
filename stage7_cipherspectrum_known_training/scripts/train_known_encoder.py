#!/usr/bin/env python3
"""Train one independent CipherSpectrum Known-only Open-Detect encoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from common import (
    AUDIT_ROOT,
    STAGE7_ROOT,
    UPSTREAM_CODE,
    VALID_SETTINGS,
    load_fold,
    seed_everything,
    sha256_file,
    verify_stage6_protocol_hashes,
)

sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(STAGE7_ROOT.parent / "stage3_unknown_utility/scripts"))
from adapters.opendetect_model import released_weight_init  # noqa: E402
from scripts.train_opendetect import (  # noqa: E402
    released_reset_prototypes,
    run_epoch,
    validation_composite,
)
from stage3_model import Stage3OpenDetectNet  # noqa: E402


class KnownInputDataset(Dataset):
    def __init__(self, input_dir: Path, prefix: str, augment: bool) -> None:
        self.images = np.load(
            input_dir / f"{prefix}_images.npy", mmap_mode="r", allow_pickle=False
        )
        self.labels = np.load(
            input_dir / f"{prefix}_labels.npy", mmap_mode="r", allow_pickle=False
        )
        if len(self.images) != len(self.labels):
            raise RuntimeError(f"{prefix}: image/label count mismatch")
        self.augment = augment
        self.crop = RandomCrop(32, padding=4)
        self.flip = RandomHorizontalFlip()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).unsqueeze(0)
        if self.augment:
            image = self.flip(self.crop(image))
        return image.float().div_(255.0), int(self.labels[index])


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_input_gate(
    input_dir: Path, setting: str, mode: str, fold: dict[str, object]
) -> dict[str, object]:
    audit_path = input_dir / "input_audit.json"
    manifest_path = input_dir / "input_manifest.csv"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("input audit and manifest are required before training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["setting"] != setting or audit["mode"] != mode:
        raise RuntimeError("input audit identity/status mismatch")
    if audit["input_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("input manifest changed after audit")
    forbidden = (
        "known_test_pcap_files_opened",
        "unknown_pcap_files_opened",
    )
    if any(int(audit[key]) != 0 for key in forbidden):
        raise RuntimeError("forbidden Known Test or Unknown PCAP access recorded")
    if audit["unknown_inference_executed"] is not False:
        raise RuntimeError("Unknown inference boundary violated")
    if set(audit["known_classes"]) != set(fold["known_classes"]):
        raise RuntimeError("input Known classes differ from frozen fold")
    if set(audit["unknown_classes"]) != set(fold["unknown_classes"]):
        raise RuntimeError("input Unknown classes differ from frozen fold")
    return audit


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


def restore_rng(payload: dict[str, object], generator: torch.Generator) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
    generator.set_state(payload["data_generator_state"].cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--lamda", type=float, default=0.005)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--temp-inter", type=float, default=1.0)
    parser.add_argument("--temp-intra", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if not (args.run_dir / "manifest.json").is_file():
        raise RuntimeError("run-dir must be initialized as an experiment bundle first")
    if args.batch_size != 512 or args.early_stopping_patience != 5:
        raise RuntimeError("Stage 7 freezes batch_size=512 and early_stopping_patience=5")
    verify_stage6_protocol_hashes()
    fold = load_fold(args.setting)
    input_dir = args.run_dir / "inputs"
    audit = validate_input_gate(input_dir, args.setting, args.mode, fold)
    if not torch.cuda.is_available():
        raise RuntimeError("Known-only Open-Detect training requires CUDA")

    artifact_dir = args.run_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    prefix = "smoke" if args.mode == "smoke" else "training"
    metrics_path = args.run_dir / f"{prefix}_metrics.csv"
    config_path = args.run_dir / f"{prefix}_config.json"
    checkpoint_path = artifact_dir / f"{prefix}_best_checkpoint.pt"
    latest_path = artifact_dir / f"{prefix}_latest_checkpoint.pt"
    selection_path = args.run_dir / f"{prefix}_checkpoint_selection.json"
    if args.resume is None and any(
        path.exists()
        for path in (metrics_path, config_path, checkpoint_path, latest_path, selection_path)
    ):
        raise RuntimeError("refusing to overwrite training evidence; use --resume")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded")
    epochs = args.epochs if args.epochs is not None else (3 if args.mode == "smoke" else 100)
    train_set = KnownInputDataset(input_dir, "train", augment=True)
    validation_set = KnownInputDataset(input_dir, "validation", augment=False)
    formal_counts = fold
    if args.mode == "formal":
        if len(train_set) != int(formal_counts["known_train_count"]):
            raise RuntimeError("formal Known Train count differs from frozen fold")
        if len(validation_set) != int(formal_counts["known_validation_count"]):
            raise RuntimeError("formal Known Validation count differs from frozen fold")

    generator = torch.Generator().manual_seed(args.seed)
    common_loader = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common_loader)
    validation_loader = DataLoader(validation_set, shuffle=False, **common_loader)
    num_classes = len(fold["known_classes"])
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
            raise RuntimeError("resume is restricted to this run's latest checkpoint")
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = payload["config"]
        if saved_config["setting"] != args.setting or saved_config["mode"] != args.mode:
            raise RuntimeError("cross-setting or cross-mode resume is forbidden")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_validation_composite"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        restore_rng(payload, generator)
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = [
                    dict(row)
                    for row in csv.DictReader(handle)
                    if int(row["epoch"]) <= start_epoch
                ]

    config = {
        "setting": args.setting,
        "mode": args.mode,
        "known_classes": fold["known_classes"],
        "unknown_classes_excluded": fold["unknown_classes"],
        "unknown_samples_loaded": 0,
        "known_test_samples_loaded": 0,
        "stage6_split_manifest_sha256": audit["stage6_split_manifest_sha256"],
        "stage6_fold_sha256": audit["stage6_fold_sha256"],
        "input_manifest_sha256": audit["input_manifest_sha256"],
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
        "validation_samples": len(validation_set),
        "initialization": "released_weight_init from scratch; no checkpoint",
        "warm_start": False,
        "numerical_stability_guard": "raw logvar upper tail clamped at 20",
        "physical_gpu": physical_gpu,
        "visible_device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "upstream_code": str(UPSTREAM_CODE.resolve()),
        "upstream_commit": "f26ac911be4324110a6632912388cd19413353bd",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    started = time.time()
    stop_epoch = start_epoch
    for epoch_index in range(start_epoch, epochs):
        guard_before = model.stability_guard_activations
        train_metrics = run_epoch(model, train_loader, device, args.lamda, optimizer)
        guard_after_train = model.stability_guard_activations
        reset = epoch_index in (50, 80)
        if reset:
            model.prototypes = released_reset_prototypes(model, train_loader, device)
        val_metrics = run_epoch(model, validation_loader, device, args.lamda, None)
        guard_after_val = model.stability_guard_activations
        scheduler.step()
        score = validation_composite(val_metrics["accuracy"], val_metrics["macro_f1"])
        row: dict[str, object] = {
            "epoch": epoch_index + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "prototype_reset": reset,
            "elapsed_seconds": time.time() - started,
            "val_composite_score": score,
            "stability_guard_train_activations": guard_after_train - guard_before,
            "stability_guard_validation_activations": guard_after_val - guard_after_train,
            "stability_guard_cumulative_activations": guard_after_val,
        }
        for metric_prefix, values in (("train", train_metrics), ("val", val_metrics)):
            for name, value in values.items():
                row[f"{metric_prefix}_{name}"] = value
        rows.append(row)
        write_metrics(metrics_path, rows)
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
        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "stop_epoch": stop_epoch,
                        "best_epoch": best_epoch,
                        "monitor": "harmonic_mean(val_accuracy,val_macro_f1)",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break

    best_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selection = {
        "setting": args.setting,
        "mode": args.mode,
        "selection_data": "Known Validation only",
        "unknown_samples_used": 0,
        "known_test_samples_used": 0,
        "best_epoch": int(best_payload["epoch"]),
        "stop_epoch": stop_epoch,
        "val_accuracy": float(best_payload["val_metrics"]["accuracy"]),
        "val_macro_f1": float(best_payload["val_metrics"]["macro_f1"]),
        "combined_score": float(best_payload["best_validation_composite"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "early_stopping_patience": args.early_stopping_patience,
        "stability_guard_cumulative_activations": model.stability_guard_activations,
        "unknown_inference_executed": False,
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.mode == "smoke":
        finite = all(
            np.isfinite(float(row[key]))
            for row in rows
            for key in (
                "train_total",
                "train_rec",
                "train_kld",
                "train_dis",
                "val_total",
                "val_accuracy",
                "val_macro_f1",
            )
        )
        prototype_ok = tuple(model.prototypes.shape) == (num_classes, args.latent_dim)
        status = "PASS" if finite and prototype_ok else "FAIL"
        (args.run_dir / "smoke_report.md").write_text(
            f"""# {args.setting} CipherSpectrum Known-only Smoke Report

- Status: **{status}**
- Known classes/prototypes: {num_classes}/{model.prototypes.shape[0]}
- Train/Validation samples: {len(train_set)}/{len(validation_set)}
- Unknown samples loaded: 0
- Known Test samples loaded: 0
- Unknown inference executed: false
- Epochs completed: {stop_epoch}
- Best epoch: {selection['best_epoch']}
- Validation Accuracy/Macro-F1/composite: {selection['val_accuracy']:.9f}/{selection['val_macro_f1']:.9f}/{selection['combined_score']:.9f}
- Finite losses/metrics: {str(finite).lower()}
- Prototype shape: {list(model.prototypes.shape)}
- Physical GPU: {physical_gpu}
""",
            encoding="utf-8",
        )
        if status != "PASS":
            raise RuntimeError("smoke gate failed")
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
