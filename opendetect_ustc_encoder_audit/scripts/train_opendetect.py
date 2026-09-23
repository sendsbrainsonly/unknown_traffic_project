#!/usr/bin/env python3
"""Smoke or formal fixed-split 20-class Open-Detect training."""

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
sys.path.insert(0, str(AUDIT_ROOT))

from adapters.opendetect_model import (  # noqa: E402
    AuditedOpenDetectNet,
    released_weight_init,
)


class AlignedImageDataset(Dataset):
    def __init__(self, images: Path, labels: Path, augment: bool) -> None:
        self.images = np.load(images, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(labels, mmap_mode="r", allow_pickle=False)
        if len(self.images) != len(self.labels):
            raise ValueError("image and label counts differ")
        self.augment = augment
        self.crop = RandomCrop(32, padding=4)
        self.flip = RandomHorizontalFlip()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).unsqueeze(0)
        if self.augment:
            image = self.flip(self.crop(image))
        image = image.float().div_(255.0)
        return image, int(self.labels[index])


def fixed_subset(labels_path: Path, per_class: int, seed: int) -> list[int]:
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in range(20):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < per_class:
            raise ValueError(f"class {class_id} has fewer than {per_class} samples")
        selected.extend(rng.choice(indices, size=per_class, replace=False).tolist())
    return sorted(selected)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(confusion: np.ndarray) -> float:
    scores = []
    for class_id in range(confusion.shape[0]):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(scores))


def validation_composite(accuracy: float, macro_f1: float) -> float:
    """Balanced validation score: harmonic mean of accuracy and macro-F1."""
    denominator = accuracy + macro_f1
    return 0.0 if denominator == 0.0 else float(2.0 * accuracy * macro_f1 / denominator)


def run_epoch(
    model: AuditedOpenDetectNet,
    loader: DataLoader,
    device: torch.device,
    lamda: float,
    optimizer: optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "rec", "kld", "ent", "dis")}
    confusion = np.zeros((model.n_classes, model.n_classes), dtype=np.int64)
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            _, preds, losses = model.loss(images, labels)
            total = lamda * (losses["rec"] + losses["kld"] + losses["ent"]) + (
                1.0 - lamda
            ) * losses["dis"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite total loss: {total.item()}")
            if optimizer is not None:
                total.backward()
                optimizer.step()
        batch = len(labels)
        totals["total"] += float(total.detach()) * batch
        for name in ("rec", "kld", "ent", "dis"):
            value = losses[name].detach()
            if not torch.isfinite(value):
                raise FloatingPointError(f"non-finite {name} loss")
            totals[name] += float(value) * batch
        truth = labels.detach().cpu().numpy()
        guess = preds.detach().cpu().numpy()
        np.add.at(confusion, (truth, guess), 1)
        count += batch
    result = {name: value / count for name, value in totals.items()}
    result["accuracy"] = float(np.trace(confusion) / count)
    result["macro_f1"] = macro_f1_from_confusion(confusion)
    result["samples"] = count
    return result


@torch.no_grad()
def released_reset_prototypes(
    model: AuditedOpenDetectNet, loader: DataLoader, device: torch.device
) -> nn.Parameter:
    model.eval()
    sums = torch.zeros(model.n_classes, model.latent_dim, dtype=torch.float64)
    counts = torch.zeros(model.n_classes, dtype=torch.int64)
    for images, labels in loader:
        mu, _, _ = model.encoder(images.to(device, non_blocking=True))
        mu = mu.detach().cpu().double()
        for class_id in range(model.n_classes):
            mask = labels == class_id
            if mask.any():
                sums[class_id] += mu[mask].sum(dim=0)
                counts[class_id] += int(mask.sum())
    if torch.any(counts == 0):
        raise ValueError("prototype reset encountered an empty class")
    means = (sums / counts[:, None]).float().to(device)
    # Deliberately reproduce author code's parameter replacement after the
    # optimizer has been built. This behavior is documented in protocol_diff.
    return nn.Parameter(means, requires_grad=True)


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--images-dir", type=Path, default=AUDIT_ROOT / "artifacts")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage1/modelA/embeddings",
    )
    parser.add_argument("--output-dir", type=Path, default=AUDIT_ROOT / "outputs")
    parser.add_argument("--checkpoint", type=Path, default=AUDIT_ROOT / "artifacts/best_checkpoint.pt")
    parser.add_argument("--latest-checkpoint", type=Path, default=AUDIT_ROOT / "artifacts/latest_checkpoint.pt")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs without val_accuracy improvement; 0 disables early stopping.",
    )
    parser.add_argument("--lamda", type=float, default=0.005)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--temp-inter", type=float, default=1.0)
    parser.add_argument("--temp-intra", type=float, default=1.0)
    parser.add_argument("--smoke-train-per-class", type=int, default=32)
    parser.add_argument("--smoke-val-per-class", type=int, default=16)
    args = parser.parse_args()

    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")

    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded")
    epochs = args.epochs if args.epochs is not None else (5 if args.mode == "smoke" else 100)

    train_labels = args.labels_dir / "labels_train.npy"
    val_labels = args.labels_dir / "labels_val.npy"
    train_set: Dataset = AlignedImageDataset(
        args.images_dir / "train_images.npy", train_labels, augment=True
    )
    val_set: Dataset = AlignedImageDataset(
        args.images_dir / "val_images.npy", val_labels, augment=False
    )
    if args.mode == "smoke":
        train_set = Subset(
            train_set, fixed_subset(train_labels, args.smoke_train_per_class, args.seed)
        )
        val_set = Subset(
            val_set, fixed_subset(val_labels, args.smoke_val_per_class, args.seed + 1)
        )

    generator = torch.Generator().manual_seed(args.seed)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    model = AuditedOpenDetectNet(
        upstream_code=UPSTREAM_CODE,
        channels=1,
        latent_dim=args.latent_dim,
        num_classes=20,
        temp_inter=args.temp_inter,
        temp_intra=args.temp_intra,
    ).to(device)
    model.apply(released_weight_init)
    resume_payload = None
    start_epoch = 0
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        start_epoch = int(resume_payload["epoch"])
        if "data_generator_state" in resume_payload:
            generator.set_state(resume_payload["data_generator_state"].cpu())
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optimizer_state_restored = False
    if resume_payload is not None and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        optimizer_state_restored = True
    if resume_payload is not None and "scheduler_state_dict" in resume_payload:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
    else:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 80], gamma=0.1, last_epoch=start_epoch - 1
        )

    config = {
        "mode": args.mode,
        "author_code_commit": "b50a18515a01799468c10f6c9b60c01f8a6a4e7c",
        "upstream_code": str(UPSTREAM_CODE.resolve()),
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam(beta1=0.9,beta2=0.999)",
        "scheduler": "MultiStepLR(milestones=[50,80],gamma=0.1)",
        "lambda": args.lamda,
        "gamma_paper": 1.0,
        "temp_inter_code": args.temp_inter,
        "temp_intra_code": args.temp_intra,
        "latent_dimension": args.latent_dim,
        "num_classes": 20,
        "channels": 1,
        "image_parameters": {
            "shape": [32, 32],
            "packets": 8,
            "header_bytes_per_packet": 80,
            "payload_bytes_per_packet": 48,
            "ip_anonymization": "0.0.0.0",
        },
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "checkpoint_selection": "highest harmonic mean of validation accuracy and macro-F1; validation only",
        "early_stopping_monitor": "harmonic mean of validation accuracy and macro-F1; validation only",
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": 0.0,
        "physical_gpu": physical_gpu,
        "visible_device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "released_semantics": True,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "resume_epoch": start_epoch,
        "optimizer_state_restored": optimizer_state_restored,
        "data_rng_state_restored": bool(resume_payload is not None and "data_generator_state" in resume_payload),
    }
    config_name = "smoke_config.json" if args.mode == "smoke" else "training_config.json"
    config_path = args.output_dir / config_name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    metrics_name = "smoke_metrics.csv" if args.mode == "smoke" else "training_metrics.csv"
    metrics_path = args.output_dir / metrics_name
    if start_epoch and metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle) if int(row["epoch"]) <= start_epoch]
    for historical_row in rows:
        historical_row["val_composite_score"] = validation_composite(
            float(historical_row["val_accuracy"]),
            float(historical_row["val_macro_f1"]),
        )
    selected_payload = None
    if resume_payload is not None and args.checkpoint.exists():
        selected_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if selected_payload is not None and "val_metrics" in selected_payload:
        selected_metrics = selected_payload["val_metrics"]
        best_score = validation_composite(
            float(selected_metrics["accuracy"]), float(selected_metrics["macro_f1"])
        )
        best_epoch = int(selected_payload["epoch"])
    elif rows:
        historical_best = max(rows, key=lambda row: float(row["val_composite_score"]))
        best_score = float(historical_best["val_composite_score"])
        best_epoch = int(historical_best["epoch"])
    elif resume_payload is not None:
        best_score = validation_composite(
            float(resume_payload["val_metrics"]["accuracy"]),
            float(resume_payload["val_metrics"]["macro_f1"]),
        )
        best_epoch = start_epoch
    else:
        best_score = -math.inf
        best_epoch = -1
    epochs_without_improvement = max(0, start_epoch - best_epoch)
    start = time.time()
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(model, train_loader, device, args.lamda, optimizer)
        reset = epoch in (50, 80)
        if reset:
            model.prototypes = released_reset_prototypes(model, train_loader, device)
        val_metrics = run_epoch(model, val_loader, device, args.lamda, None)
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "prototype_reset": reset,
            "elapsed_seconds": time.time() - start,
        }
        for prefix, values in (("train", train_metrics), ("val", val_metrics)):
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        val_composite_score = validation_composite(
            val_metrics["accuracy"], val_metrics["macro_f1"]
        )
        row["val_composite_score"] = val_composite_score
        rows.append(row)
        write_metrics(metrics_path, rows)
        improved = val_composite_score > best_score
        if improved:
            best_score = val_composite_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "data_generator_state": generator.get_state(),
                    "validation_composite_score": val_composite_score,
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
        args.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "config": config,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "data_generator_state": generator.get_state(),
                "best_validation_composite": best_score,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            },
            args.latest_checkpoint,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_total": train_metrics["total"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_total": val_metrics["total"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_composite_score": val_composite_score,
                    "best_epoch": best_epoch,
                    "best_validation_composite": best_score,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            ),
            flush=True,
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": epoch + 1,
                        "monitor": "harmonic_mean(val_accuracy,val_macro_f1)",
                        "patience": args.early_stopping_patience,
                        "best_epoch": best_epoch,
                        "best_validation_composite": best_score,
                    }
                ),
                flush=True,
            )
            break

    if args.mode == "smoke":
        loss_decreased = rows[-1]["train_total"] < rows[0]["train_total"]
        report = f"""# Open-Detect Smoke Report

- Status: {'PASS' if loss_decreased else 'FAIL'}
- Classes represented: 20
- Train samples: {len(train_set)}
- Validation samples: {len(val_set)}
- Epochs: {epochs}
- Prototype shape: `[20,{args.latent_dim}]`
- Latent dimension: {args.latent_dim}
- First train total loss: {rows[0]['train_total']:.9f}
- Last train total loss: {rows[-1]['train_total']:.9f}
- Loss decreased: {loss_decreased}
- NaN/Inf: none observed
- Best validation composite: {best_score:.9f}
- Best epoch: {best_epoch}
- Physical GPU: {physical_gpu}

The exercised graph included image loading and augmentation, encoder `mu` and
`logvar`, stochastic reparameterized `z`, 20 prototypes, decoder
reconstruction, MSE, target-prior KL/generative loss, released discriminative
loss, entropy term, and total objective.
"""
        (args.output_dir / "smoke_report.md").write_text(report, encoding="utf-8")
        if not loss_decreased:
            raise SystemExit("smoke gate failed: training loss did not decrease")


if __name__ == "__main__":
    main()
