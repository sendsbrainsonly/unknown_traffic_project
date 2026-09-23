#!/usr/bin/env python3
"""Train one frozen VNAT protocol with the released Open-Detect behavior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing.util
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from common import (
    NATIVE_CONFIG, OPENDETECT_ROOT, RUNS_ROOT, TRAINING_SEED, load_protocols,
    run_input_dir, sha256_file, verify_freeze,
)

# Import the existing reproduction adapter and released author model read-only.
sys.path.insert(0, str(OPENDETECT_ROOT / "reproduction"))
sys.path.insert(0, str(OPENDETECT_ROOT / "code"))
from model import OpenDetectNet  # noqa: E402
from utils import weight_init  # noqa: E402
from run_reproduction import (  # noqa: E402
    TrafficImages, reset_prototypes_official, run_epoch, seed_everything,
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    labels_out: list[np.ndarray] = []
    predictions_out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, _, predictions, _ = model.loss(images, labels)
            labels_out.append(labels.cpu().numpy())
            predictions_out.append(predictions.cpu().numpy())
    return np.concatenate(labels_out), np.concatenate(predictions_out)


def install_project_local_short_tmp(run_dir: Path) -> tuple[str, int]:
    """Use a short procfs alias while keeping all temporary bytes project-local."""
    actual = run_dir / "runtime_tmp"
    actual.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(actual, os.O_RDONLY | os.O_DIRECTORY)
    os.set_inheritable(directory_fd, True)
    alias = f"/proc/self/fd/{directory_fd}"
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = alias
    # The tmux wrapper has already initialized tempfile's cache. Clear both
    # caches before DataLoader multiprocessing asks for its listener directory.
    tempfile.tempdir = None
    multiprocessing.util._tempdir = None
    return alias, directory_fd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.time()
    freeze_before = verify_freeze()
    protocols = load_protocols()
    if args.protocol_id not in protocols:
        raise KeyError(args.protocol_id)
    protocol = protocols[args.protocol_id]
    run_dir = RUNS_ROOT / args.protocol_id
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "COMPLETED"
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if completed.exists():
        print(json.dumps({"protocol_id": args.protocol_id, "status": "already_complete"}))
        return
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite partial run: {run_dir}")

    input_dir = run_input_dir(args.protocol_id)
    source_audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if source_audit.get("status") != "PASS":
        raise RuntimeError("source input audit did not pass")
    for field in (
        "unknown_samples_used_in_training", "unknown_samples_used_in_validation",
        "known_test_samples_used", "train_validation_flow_overlap",
    ):
        if int(source_audit.get(field, -1)) != 0:
            raise RuntimeError(f"strict input gate failed: {field}={source_audit.get(field)}")

    train_x = np.load(input_dir / "train_images.npy", allow_pickle=False)
    train_y = np.load(input_dir / "train_labels.npy", allow_pickle=False)
    val_x = np.load(input_dir / "validation_images.npy", allow_pickle=False)
    val_y = np.load(input_dir / "validation_labels.npy", allow_pickle=False)
    label_map = json.loads((input_dir / "label_map.json").read_text(encoding="utf-8"))
    known_classes = list(map(str, protocol["known_applications"]))
    unknown_classes = list(map(str, protocol["unknown_applications"]))
    if set(known_classes) & set(unknown_classes):
        raise RuntimeError("Known/Unknown class overlap")
    num_classes = len(known_classes)
    selected_classes = list(range(num_classes))

    seed_everything(TRAINING_SEED)
    temp_alias, temp_directory_fd = install_project_local_short_tmp(run_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native Stage 14C requires an assigned CUDA device")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
    train_set = TrafficImages(train_x, train_y, selected_classes, train=True, reindex=True)
    val_set = TrafficImages(val_x, val_y, selected_classes, train=False, reindex=True)
    train_loader = DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4,
        drop_last=False, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=64, shuffle=False, num_workers=4,
        drop_last=False, pin_memory=True,
    )
    model = OpenDetectNet("resnet18", 1, 128, num_classes, 1.0, 1.0).to(device)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    run_config = {
        "protocol_id": args.protocol_id,
        "setting": protocol["setting"],
        "protocol_seed": int(protocol["seed"]),
        "training_seed": TRAINING_SEED,
        "physical_gpu_id": physical_gpu,
        "project_local_tmp_alias": temp_alias,
        "project_local_tmp_directory_fd": temp_directory_fd,
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "native_config": NATIVE_CONFIG,
        "freeze_before": freeze_before,
        "input_audit_sha256": sha256_file(input_dir / "input_audit.json"),
        "model_visible_files": [
            "train_images.npy", "train_labels.npy",
            "validation_images.npy", "validation_labels.npy",
        ],
        "unknown_samples_used_in_training": 0,
        "unknown_samples_used_in_validation": 0,
        "known_test_samples_used": 0,
    }
    atomic_json(run_dir / "run_config.json", run_config)

    history_path = run_dir / "history.csv"
    fields = [
        "epoch", "learning_rate", "prototype_reset", "train_accuracy",
        "train_total", "train_rec", "train_kld", "train_ent", "train_dis",
        "val_accuracy", "val_total", "val_rec", "val_kld", "val_ent", "val_dis",
        "is_best", "elapsed_seconds",
    ]
    best_accuracy = 0.0
    best_epoch = 0
    status = "running"
    anomaly = ""
    try:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch_index in range(100):
                lr = float(optimizer.param_groups[0]["lr"])
                train_metrics = run_epoch(model, train_loader, device, 0.005, optimizer)
                prototype_reset = epoch_index in (50, 80)
                if prototype_reset:
                    reset_prototypes_official(model, train_loader, device)
                val_metrics = run_epoch(model, val_loader, device, 0.005, None)
                scheduler.step()
                all_values = list(train_metrics.values()) + list(val_metrics.values())
                if not all(math.isfinite(value) for value in all_values):
                    anomaly = f"non-finite metric at epoch {epoch_index + 1}"
                    raise FloatingPointError(anomaly)
                improved = float(val_metrics["accuracy"]) > best_accuracy
                if improved:
                    best_accuracy = float(val_metrics["accuracy"])
                    best_epoch = epoch_index + 1
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "epoch": best_epoch,
                        "validation_accuracy": best_accuracy,
                        "protocol_id": args.protocol_id,
                        "training_seed": TRAINING_SEED,
                        "protocol_hash": freeze_before["freeze_hash"],
                        "known_classes": known_classes,
                        "unknown_classes": unknown_classes,
                        "native_config": NATIVE_CONFIG,
                    }, checkpoint_path)
                row = {
                    "epoch": epoch_index + 1,
                    "learning_rate": lr,
                    "prototype_reset": int(prototype_reset),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    "is_best": int(improved),
                    "elapsed_seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                print(json.dumps({
                    "protocol_id": args.protocol_id,
                    "epoch": epoch_index + 1,
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_accuracy,
                    "prototype_reset": prototype_reset,
                }), flush=True)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        true, predicted = collect_predictions(model, val_loader, device)
        precision, recall, f1, support = precision_recall_fscore_support(
            true, predicted, labels=selected_classes, zero_division=0
        )
        accuracy = float(accuracy_score(true, predicted))
        macro_f1 = float(np.mean(f1))
        weighted_f1 = float(np.average(f1, weights=support))
        matrix = confusion_matrix(true, predicted, labels=selected_classes)
        np.save(run_dir / "validation_confusion_matrix.npy", matrix, allow_pickle=False)
        per_class = []
        for local_label in selected_classes:
            per_class.append({
                "protocol_id": args.protocol_id,
                "setting": protocol["setting"],
                "protocol_seed": int(protocol["seed"]),
                "local_label": local_label,
                "application": label_map["local_to_class"][str(local_label)],
                "precision": float(precision[local_label]),
                "recall": float(recall[local_label]),
                "f1": float(f1[local_label]),
                "support": int(support[local_label]),
            })
        atomic_json(run_dir / "per_class_metrics.json", per_class)
        freeze_after = verify_freeze()
        if freeze_after != freeze_before:
            raise RuntimeError("Stage 14B freeze changed during training")
        result = {
            "status": "success",
            "anomaly": anomaly,
            "protocol_id": args.protocol_id,
            "setting": protocol["setting"],
            "protocol_seed": int(protocol["seed"]),
            "training_seed": TRAINING_SEED,
            "physical_gpu_id": physical_gpu,
            "num_known_classes": num_classes,
            "num_unknown_classes": len(unknown_classes),
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "train_samples": len(train_set),
            "validation_samples": len(val_set),
            "best_epoch": best_epoch,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "history_sha256": sha256_file(history_path),
            "freeze_before": freeze_before,
            "freeze_after": freeze_after,
            "unknown_samples_used_in_training": 0,
            "unknown_samples_used_in_validation": 0,
            "known_test_samples_used": 0,
            "epochs_configured": 100,
            "epochs_completed": 100,
            "duration_seconds": time.time() - started,
        }
        atomic_json(result_path, result)
        completed.write_text("success\n", encoding="utf-8")
        status = "success"
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "protocol_id": args.protocol_id,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "anomaly": anomaly,
            "duration_seconds": time.time() - started,
        }
        atomic_json(run_dir / "failure.json", failure)
        status = "failed"
        raise
    finally:
        atomic_json(run_dir / "terminal_status.json", {
            "status": status,
            "protocol_id": args.protocol_id,
            "updated_at_unix": time.time(),
        })
        os.close(temp_directory_fd)


if __name__ == "__main__":
    main()
