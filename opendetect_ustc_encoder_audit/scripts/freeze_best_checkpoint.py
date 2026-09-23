#!/usr/bin/env python3
"""Freeze and document the unique formal checkpoint selected before latent audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import torch


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def harmonic_score(accuracy: float, macro_f1: float) -> float:
    denominator = accuracy + macro_f1
    return 0.0 if denominator == 0 else 2.0 * accuracy * macro_f1 / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", type=Path, default=AUDIT_ROOT / "outputs/training_metrics.csv"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=AUDIT_ROOT / "artifacts/best_checkpoint.pt"
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=PROJECT_ROOT
        / ".tmux-task/od_formal_composite_earlystop_gpu3/output.log",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=AUDIT_ROOT / "artifacts/latent_gaussian_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_ROOT / "outputs/latent_gaussian_audit",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    required = {"epoch", "val_accuracy", "val_macro_f1"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"training metrics missing columns: {sorted(missing)}")
    recomputed = [
        harmonic_score(float(acc), float(f1))
        for acc, f1 in zip(metrics["val_accuracy"], metrics["val_macro_f1"])
    ]
    if "val_composite_score" in metrics:
        stored = metrics["val_composite_score"].astype(float).to_numpy()
        if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(stored, recomputed)):
            raise ValueError("stored validation composite score does not match the formal rule")
    metrics = metrics.assign(val_composite_score=recomputed)
    best_score = float(metrics["val_composite_score"].max())
    best_rows = metrics[
        metrics["val_composite_score"].map(
            lambda value: math.isclose(float(value), best_score, rel_tol=0.0, abs_tol=1e-12)
        )
    ]
    if len(best_rows) != 1:
        raise ValueError(f"formal model-selection rule did not yield one best epoch: {len(best_rows)}")
    best = best_rows.iloc[0]
    best_epoch = int(best["epoch"])
    stop_epoch = int(metrics["epoch"].max())

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint["epoch"])
    checkpoint_metrics = checkpoint["val_metrics"]
    checkpoint_score = harmonic_score(
        float(checkpoint_metrics["accuracy"]), float(checkpoint_metrics["macro_f1"])
    )
    if checkpoint_epoch != best_epoch:
        raise ValueError(
            f"best checkpoint epoch {checkpoint_epoch} != metrics best epoch {best_epoch}"
        )
    checks = (
        (float(checkpoint_metrics["accuracy"]), float(best["val_accuracy"]), "accuracy"),
        (float(checkpoint_metrics["macro_f1"]), float(best["val_macro_f1"]), "macro_f1"),
        (checkpoint_score, best_score, "combined score"),
    )
    for actual, expected, name in checks:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"checkpoint {name} does not match training metrics")

    no_improvement = stop_epoch - best_epoch
    log_event = None
    if args.training_log.exists():
        for line in args.training_log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "early_stopping":
                log_event = payload
    if stop_epoch < args.max_epochs:
        if no_improvement < args.patience:
            raise ValueError(
                "training stopped before max epochs without exhausting early-stop patience"
            )
        early_stop_reason = (
            f"validation combined score did not improve for {args.patience} consecutive epochs"
        )
        if log_event is None or int(log_event.get("epoch", -1)) != stop_epoch:
            raise ValueError("early-stopping event is missing or disagrees with stop epoch")
    else:
        early_stop_reason = "maximum epoch count reached"

    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = args.artifact_dir / "best_checkpoint.pt"
    shutil.copy2(args.checkpoint, frozen)
    source_hash = sha256_file(args.checkpoint)
    frozen_hash = sha256_file(frozen)
    if source_hash != frozen_hash:
        raise ValueError("frozen checkpoint hash differs from source checkpoint")

    report = f"""# Formal Checkpoint Selection

- best_epoch: {best_epoch}
- best_val_accuracy: {float(best['val_accuracy']):.12f}
- best_val_macro_f1: {float(best['val_macro_f1']):.12f}
- best_combined_score: {best_score:.12f}
- model_selection_rule: harmonic mean of validation Accuracy and Macro-F1; strict improvement
- checkpoint_path: `{frozen.resolve()}`
- checkpoint_sha256: `{frozen_hash}`
- source_checkpoint_path: `{args.checkpoint.resolve()}`
- source_checkpoint_sha256: `{source_hash}`
- training_stop_epoch: {stop_epoch}
- early_stop_reason: {early_stop_reason}
- early_stopping_patience: {args.patience}
- consecutive_non_improving_epochs_at_stop: {no_improvement}

The checkpoint was frozen before latent extraction. No Gaussian result was used
for checkpoint selection. All downstream latent extraction must use the frozen
path and SHA-256 above.

## Known strict-reproduction limitation

At the user-requested GPU migration after epoch 5, the then-available checkpoint
did not contain optimizer or RNG state. Training resumed from model weights with
a reconstructed optimizer/scheduler state and a restarted shuffle/augmentation
stream. Later resumes restored model, optimizer, scheduler, and DataLoader RNG
state, but they cannot remove the epoch-5 discontinuity. The frozen checkpoint
is not modified to compensate for this limitation.
"""
    (args.output_dir / "checkpoint_selection.md").write_text(report, encoding="utf-8")
    selection = {
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best["val_accuracy"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "best_combined_score": best_score,
        "checkpoint_path": str(frozen.resolve()),
        "checkpoint_sha256": frozen_hash,
        "training_stop_epoch": stop_epoch,
        "early_stop_reason": early_stop_reason,
        "early_stopping_patience": args.patience,
        "consecutive_non_improving_epochs_at_stop": no_improvement,
    }
    (args.output_dir / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
