#!/usr/bin/env python3
"""Exact Stage 14C-4 primary-run replay with observation-only telemetry.

This script imports the frozen Stage 14C-4 training functions. It does not
change model, data, optimizer, scheduler, reset, loss, batch, seed, or epochs.
Extra deterministic validation passes run under a restored RNG snapshot so
they cannot perturb the next training epoch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE14C4 = PROJECT / "stage14c4_our_method_training_cleanup"
sys.path.insert(0, str(STAGE14C4 / "scripts"))
import train_cleaned as frozen_train  # noqa: E402

PROTOCOLS = ("medium_seed2025", "medium_seed2026")
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty telemetry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_group_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().double()
            total += float(torch.sum(value * value).item())
    return math.sqrt(total)


def rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


@torch.no_grad()
def per_class_validation(
    model,
    loader: DataLoader,
    device: torch.device,
    local_to_class: list[str],
) -> tuple[list[dict[str, Any]], float, float]:
    model.eval()
    confusion = np.zeros((len(local_to_class), len(local_to_class)), dtype=np.int64)
    for images, labels in loader:
        _, predictions, _ = model.loss(
            images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        )
        np.add.at(
            confusion,
            (labels.numpy(), predictions.detach().cpu().numpy()),
            1,
        )
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(local_to_class):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id].sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        f1_values.append(f1)
        rows.append({
            "class": class_name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows, float(np.trace(confusion) / confusion.sum()), float(np.mean(f1_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True, choices=PROTOCOLS)
    args = parser.parse_args()

    frozen = frozen_train.f2_common.verify_frozen_inputs()
    if frozen["freeze_hash"] != EXPECTED_FREEZE:
        raise RuntimeError("Stage 14B freeze hash changed")
    protocol = frozen_train.f2_common.stage14c_common.load_protocol(args.protocol_id)
    input_dir = (
        PROJECT / "stage14c5_feature_representation_audit" / "runs" / "f2"
        / args.protocol_id / "inputs"
    )
    input_audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if (
        input_audit["status"] != "PASS"
        or input_audit["unknown_samples_loaded"] != 0
        or input_audit["known_test_samples_loaded"] != 0
    ):
        raise RuntimeError("Known-only F2 input gate failed")

    training_seed = int(protocol["seed"])
    run_out = OUT / "telemetry_replay" / args.protocol_id
    run_out.mkdir(parents=True, exist_ok=False)
    config = {
        "audit_role": "observation-only exact-config replay",
        "protocol_id": args.protocol_id,
        "training_seed": training_seed,
        "epochs": 100,
        "batch_size": 128,
        "optimizer": "Adam(lr=0.001,betas=[0.9,0.999],weight_decay=0)",
        "scheduler": "MultiStepLR(milestones=[50,80],gamma=0.1)",
        "prototype_reset_epochs": [51, 81],
        "loss": "0.005*(rec+kld+ent)+0.995*dis",
        "stage14c4_training_code_sha256": sha256_file(STAGE14C4 / "scripts" / "train_cleaned.py"),
        "stage14c4_source_config_sha256": sha256_file(
            STAGE14C4 / "runs" / f"{args.protocol_id}_protocol" / "training_config.json"
        ),
        "input_manifest_sha256": sha256_file(input_dir / "input_manifest.csv"),
        "freeze_hash": frozen["freeze_hash"],
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
    }
    (run_out / "replay_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for exact-config replay")
    frozen_train.f2_common.stage14c_common.seed_everything(training_seed)
    device = torch.device("cuda:0")
    train_ds = frozen_train.KnownDataset(input_dir, "train", augment=True)
    val_ds = frozen_train.KnownDataset(input_dir, "validation", augment=False)
    generator = torch.Generator().manual_seed(training_seed)
    loader_kwargs = dict(
        batch_size=128,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    label_map = json.loads((input_dir / "label_map.json").read_text(encoding="utf-8"))
    local_to_class = [
        name for name, _ in sorted(label_map["class_to_local"].items(), key=lambda item: item[1])
    ]
    model = frozen_train.make_model(len(local_to_class), device)
    model.apply(frozen_train.released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    proto_grad_norms: list[float] = []
    encoder_grad_norms: list[float] = []
    original_step = optimizer.step

    def observed_step(*step_args, **step_kwargs):
        proto_grad_norms.append(tensor_group_norm([model.prototypes]))
        encoder_grad_norms.append(tensor_group_norm(model.encoder.parameters()))
        return original_step(*step_args, **step_kwargs)

    optimizer.step = observed_step  # type: ignore[method-assign]
    epoch_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for epoch_index in range(100):
        epoch = epoch_index + 1
        proto_grad_norms.clear()
        encoder_grad_norms.clear()
        lr_used = float(optimizer.param_groups[0]["lr"])
        prototype_norm_start = float(torch.linalg.vector_norm(model.prototypes.detach()).item())
        train_metrics = frozen_train.run_epoch(model, train_loader, device, optimizer)
        prototype_norm_after_train = float(torch.linalg.vector_norm(model.prototypes.detach()).item())
        reset = epoch_index in (50, 80)
        if reset:
            reset_audit = frozen_train.reset_prototypes_in_place(
                model, train_loader, device, optimizer
            )
            if not all(reset_audit.values()):
                raise RuntimeError(f"prototype reset audit failed at epoch {epoch}")
        prototype_norm_after_reset = float(torch.linalg.vector_norm(model.prototypes.detach()).item())
        val_metrics = frozen_train.run_epoch(model, val_loader, device, None)

        state = rng_snapshot()
        class_rows, replay_accuracy, replay_macro = per_class_validation(
            model, val_loader, device, local_to_class
        )
        restore_rng(state)
        if abs(replay_accuracy - val_metrics["accuracy"]) > 1e-12:
            raise RuntimeError(f"extra validation accuracy mismatch at epoch {epoch}")
        if abs(replay_macro - val_metrics["macro_f1"]) > 1e-12:
            raise RuntimeError(f"extra validation Macro-F1 mismatch at epoch {epoch}")

        row = {
            "protocol_id": args.protocol_id,
            "training_seed": training_seed,
            "epoch": epoch,
            "learning_rate_used": lr_used,
            "prototype_reset": reset,
            "prototype_norm_start": prototype_norm_start,
            "prototype_norm_after_train": prototype_norm_after_train,
            "prototype_norm_after_reset": prototype_norm_after_reset,
            "prototype_norm_reset_delta": prototype_norm_after_reset - prototype_norm_after_train,
            "prototype_gradient_norm_mean": float(np.mean(proto_grad_norms)),
            "prototype_gradient_norm_max": float(np.max(proto_grad_norms)),
            "encoder_gradient_norm_mean": float(np.mean(encoder_grad_norms)),
            "encoder_gradient_norm_max": float(np.max(encoder_grad_norms)),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        epoch_rows.append(row)
        for class_row in class_rows:
            recall_rows.append({
                "protocol_id": args.protocol_id,
                "training_seed": training_seed,
                "epoch": epoch,
                "prototype_reset": reset,
                **class_row,
            })
        write_csv(run_out / "epoch_telemetry.csv", epoch_rows)
        write_csv(run_out / "per_class_validation.csv", recall_rows)
        scheduler.step()
        print(json.dumps({
            "protocol_id": args.protocol_id,
            "epoch": epoch,
            "val_macro_f1": val_metrics["macro_f1"],
            "prototype_reset": reset,
            "prototype_norm": prototype_norm_after_reset,
            "prototype_grad_mean": row["prototype_gradient_norm_mean"],
            "encoder_grad_mean": row["encoder_gradient_norm_mean"],
        }), flush=True)

    original = pd.read_csv(STAGE14C4 / "training_dynamics.csv")
    original = original[
        (original["protocol_id"] == args.protocol_id)
        & (original["seed_mode"] == "protocol")
    ].sort_values("epoch")
    replay = pd.DataFrame(epoch_rows).sort_values("epoch")
    comparisons = {
        "learning_rate": "learning_rate_used",
        "train_total": "train_total",
        "val_total": "val_total",
        "train_accuracy": "train_accuracy",
        "val_accuracy": "val_accuracy",
        "train_macro_f1": "train_macro_f1",
        "val_macro_f1": "val_macro_f1",
        "train_weighted_f1": "train_weighted_f1",
        "val_weighted_f1": "val_weighted_f1",
    }
    parity_rows: list[dict[str, Any]] = []
    for original_col, replay_col in comparisons.items():
        differences = np.abs(
            original[original_col].to_numpy(dtype=float)
            - replay[replay_col].to_numpy(dtype=float)
        )
        parity_rows.append({
            "protocol_id": args.protocol_id,
            "metric": original_col,
            "max_absolute_difference": float(differences.max()),
            "mean_absolute_difference": float(differences.mean()),
            "exact_within_1e_10": bool(differences.max() <= 1e-10),
        })
    write_csv(run_out / "parity.csv", parity_rows)
    parity_pass = all(row["exact_within_1e_10"] for row in parity_rows)
    status = {
        "status": "PASS" if parity_pass else "FAIL_PARITY",
        "protocol_id": args.protocol_id,
        "epochs": 100,
        "parity_pass": parity_pass,
        "max_absolute_difference": max(row["max_absolute_difference"] for row in parity_rows),
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
        "freeze_hash": frozen["freeze_hash"],
    }
    (run_out / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if not parity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
