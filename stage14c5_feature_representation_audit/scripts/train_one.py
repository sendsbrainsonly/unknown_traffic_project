#!/usr/bin/env python3
"""Train one F1/F2/F3 run with the exact Stage 14C policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from common import CHECKPOINT_ROOT, NEW_FEATURES, ROOT, RUNS_ROOT, json_dump, sha256_file, stage14c_common, verify_frozen_inputs
from model_utils import KnownDataset, evaluate_predictions, make_model, write_diagnostics

AUDIT_ROOT = ROOT.parent / "opendetect_ustc_encoder_audit"
sys.path.insert(0, str(AUDIT_ROOT))
from adapters.opendetect_model import released_weight_init  # noqa: E402
from scripts.train_opendetect import released_reset_prototypes, run_epoch, validation_composite, write_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=NEW_FEATURES, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if (args.epochs, args.batch_size, args.workers, args.learning_rate, args.patience) != (100, 512, 0, 1e-3, 5):
        raise RuntimeError("Stage 14C training policy changed")
    frozen = verify_frozen_inputs()
    protocol = stage14c_common.load_protocol(args.protocol_id)
    run_dir = RUNS_ROOT / args.feature / args.protocol_id
    input_dir = run_dir / "inputs"
    input_audit = json.loads((input_dir / "input_audit.json").read_text())
    if input_audit["status"] != "PASS" or input_audit["unknown_samples_loaded"] != 0 or input_audit["known_test_samples_loaded"] != 0:
        raise RuntimeError("Known-only input gate failed")
    metrics_path = run_dir / "training_metrics.csv"
    config_path = run_dir / "training_config.json"
    selection_path = run_dir / "checkpoint_selection.json"
    result_path = run_dir / "result.json"
    latest_path = run_dir / "latest_checkpoint.pt"
    CHECKPOINT_ROOT.mkdir(exist_ok=True)
    best_path = CHECKPOINT_ROOT / f"{args.feature}_{args.protocol_id}_best.pt"
    if any(path.exists() for path in (metrics_path, config_path, selection_path, result_path, latest_path, best_path)):
        raise RuntimeError("refusing to overwrite run evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    seed = int(protocol["seed"])
    stage14c_common.seed_everything(seed)
    device = torch.device("cuda:0")
    train_ds = KnownDataset(input_dir, "train", augment=True)
    val_ds = KnownDataset(input_dir, "validation", augment=False)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(batch_size=512, num_workers=0, pin_memory=True, persistent_workers=False)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    model = make_model(len(protocol["known_applications"]), device)
    model.apply(released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    config = {
        "feature": args.feature, "protocol_id": args.protocol_id,
        "setting": protocol["setting"], "seed": seed,
        "known_classes": protocol["known_applications"], "unknown_classes_excluded": protocol["unknown_applications"],
        "unknown_samples_loaded": 0, "known_test_samples_loaded": 0,
        "freeze_hash": frozen["freeze_hash"], "stage14c_summary_sha256": frozen["stage14c_summary_sha256"],
        "epochs": 100, "batch_size": 512, "workers": 0, "learning_rate": 1e-3,
        "optimizer": "Adam(beta1=0.9,beta2=0.999)", "scheduler": "MultiStepLR(milestones=[50,80],gamma=0.1)",
        "lambda": 0.005, "latent_dimension": 128, "channels": 1, "image_shape": [32, 32],
        "checkpoint_selection": "highest harmonic mean of Known Validation Accuracy and Macro-F1",
        "early_stopping_monitor": "Known Validation composite only", "early_stopping_patience": 5,
        "initialization": "released_weight_init from scratch; no checkpoint",
        "numerical_stability_guard": "raw logvar upper tail clamped at 20",
        "train_samples": len(train_ds), "validation_samples": len(val_ds),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
        "cuda_device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    json_dump(config_path, config)

    rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_epoch = -1
    no_improve = 0
    started = time.time()
    for epoch_index in range(100):
        guard0 = model.stability_guard_activations
        train = run_epoch(model, train_loader, device, 0.005, optimizer)
        guard1 = model.stability_guard_activations
        reset = epoch_index in (50, 80)
        if reset:
            model.prototypes = released_reset_prototypes(model, train_loader, device)
        val = run_epoch(model, val_loader, device, 0.005, None)
        guard2 = model.stability_guard_activations
        scheduler.step()
        score = validation_composite(val["accuracy"], val["macro_f1"])
        if not all(math.isfinite(float(v)) for v in list(train.values()) + list(val.values()) + [score]):
            raise RuntimeError("non-finite training metric")
        row: dict[str, object] = {
            "epoch": epoch_index + 1, "learning_rate": optimizer.param_groups[0]["lr"], "prototype_reset": reset,
            "elapsed_seconds": time.time() - started, "val_composite_score": score,
            "stability_guard_train_activations": guard1 - guard0,
            "stability_guard_validation_activations": guard2 - guard1,
            "stability_guard_cumulative_activations": guard2,
        }
        for prefix, values in (("train", train), ("val", val)):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
        write_metrics(metrics_path, rows)
        improved = score > best_score
        if improved:
            best_score, best_epoch, no_improve = score, epoch_index + 1, 0
        else:
            no_improve += 1
        payload = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch_index + 1, "train_metrics": train, "val_metrics": val, "config": config,
            "best_epoch": best_epoch, "best_validation_composite": best_score, "epochs_without_improvement": no_improve,
        }
        if improved:
            torch.save(payload, best_path)
        torch.save(payload, latest_path)
        print(json.dumps({"feature": args.feature, "protocol_id": args.protocol_id, "epoch": epoch_index + 1, "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "best_epoch": best_epoch, "no_improve": no_improve}), flush=True)
        if no_improve >= 5:
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    truth, predictions = evaluate_predictions(model, input_dir, device)
    diagnostics = write_diagnostics(args.feature, args.protocol_id, run_dir, truth, predictions, input_dir)
    if abs(diagnostics["val_accuracy_recomputed"] - float(best["val_metrics"]["accuracy"])) > 1e-12 or abs(diagnostics["val_macro_f1_recomputed"] - float(best["val_metrics"]["macro_f1"])) > 1e-12:
        raise RuntimeError("best-checkpoint diagnostic metrics do not reproduce selection metrics")
    selection = {
        "feature": args.feature, "protocol_id": args.protocol_id, "selection_data": "Known Validation only",
        "unknown_samples_used": 0, "known_test_samples_used": 0,
        "best_epoch": int(best["epoch"]), "stop_epoch": int(rows[-1]["epoch"]),
        "train_loss": float(best["train_metrics"]["total"]), "val_loss": float(best["val_metrics"]["total"]),
        "val_accuracy": float(best["val_metrics"]["accuracy"]), "val_macro_f1": float(best["val_metrics"]["macro_f1"]),
        "combined_score": float(best["best_validation_composite"]), "checkpoint_path": str(best_path),
        "checkpoint_sha256": sha256_file(best_path), "freeze_hash": frozen["freeze_hash"], "converged_without_nan": True,
    }
    json_dump(selection_path, selection)
    json_dump(result_path, {**selection, **diagnostics, "setting": protocol["setting"], "seed": seed, "status": "success"})
    print(json.dumps({"event": "run_complete", **selection, **diagnostics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
