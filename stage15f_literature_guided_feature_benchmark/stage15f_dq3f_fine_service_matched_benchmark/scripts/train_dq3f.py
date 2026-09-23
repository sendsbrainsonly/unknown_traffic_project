#!/usr/bin/env python3
"""Train one DQ-3F Fine or Service model with frozen matched inputs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dq3f_common import (
    OPEN_DETECT, OUT, SEEDS, TRAINING_CONFIG, array_digest, encoded_targets,
    label_space, load_matched_split, read_json, seed_everything, sha256_file,
    write_csv, write_json,
)


class TrafficImages(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, train: bool) -> None:
        self.data = data
        self.targets = targets.astype(np.int64, copy=False)
        self.transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            if train else [transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.transform(Image.fromarray(self.data[index], mode="L")), int(self.targets[index])


def import_native():
    vendor = OPEN_DETECT / "code"
    reproduction = OPEN_DETECT / "reproduction"
    for path in (vendor, reproduction):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from corrected_model import CorrectedOpenDetectNet
    from run_reproduction import reset_prototypes_in_place, run_epoch
    from utils import weight_init
    return CorrectedOpenDetectNet, reset_prototypes_in_place, run_epoch, weight_init


def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, target in loader:
            mean, _, _ = model.encoder(images.to(device, non_blocking=True))
            class_kl = 0.5 * model.distance(mean, model.prototypes)
            predictions.append(torch.argmin(class_kl, dim=1).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(labels), np.concatenate(predictions)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {
                "class_index": index, "class_name": name, "precision": float(precision[index]),
                "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index]),
                "correct": int(np.sum((y_true == index) & (y_pred == index))),
                "errors": int(np.sum(y_true == index) - np.sum((y_true == index) & (y_pred == index))),
            }
            for index, name in enumerate(class_names)
        ],
    }


def run(task: str, seed: int) -> dict:
    if task not in {"fine", "service"} or seed not in SEEDS:
        raise ValueError((task, seed))
    preflight = read_json(OUT / "preflight_status.json")
    if preflight["status"] != "PASS" or preflight["known_test_feature_values_used"] != 0:
        raise RuntimeError("DQ-3F preflight Gate is not PASS")
    output = OUT / "runs" / task / f"seed{seed}"
    if (output / "SUCCESS").is_file():
        return {"status": "SKIP_SUCCESS", "task": task, "seed": seed, "output": str(output)}
    if output.exists():
        raise FileExistsError(f"preserved partial run; refusing overwrite: {output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DQ-3F training")

    train = load_matched_split("known_train")
    validation = load_matched_split("known_validation")
    class_names, mapping = label_space(train, validation, task)
    train_y = encoded_targets(train, task, mapping)
    val_y = encoded_targets(validation, task, mapping)
    training = read_json(TRAINING_CONFIG)["training"]

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "num_workers": int(training["workers"]), "pin_memory": True, "persistent_workers": False,
    }
    train_loader = DataLoader(
        TrafficImages(train["data"], train_y, train=True), batch_size=int(training["batch_size"]),
        shuffle=True, generator=generator, drop_last=False, **loader_args,
    )
    stable_train_loader = DataLoader(
        TrafficImages(train["data"], train_y, train=False), batch_size=int(training["eval_batch_size"]),
        shuffle=False, **loader_args,
    )
    val_loader = DataLoader(
        TrafficImages(validation["data"], val_y, train=False), batch_size=int(training["eval_batch_size"]),
        shuffle=False, **loader_args,
    )
    CorrectedOpenDetectNet, reset_prototypes, run_epoch, weight_init = import_native()
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        training["architecture"], int(training["channels"]), int(training["latent_dim"]),
        len(class_names), 1, 1,
    ).to(device)
    model.apply(weight_init)
    optimizer = optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]),
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(map(int, training["scheduler_milestones"])),
        gamma=float(training["scheduler_gamma"]),
    )
    config = {
        "task": task, "seed": seed, "classes": class_names, "class_to_index": mapping,
        "train_samples": len(train_y), "validation_samples": len(val_y),
        "train_flow_id_sha256": sha256_file(OUT / "dq3f_manifest_parity.csv"),
        "train_subset_array_digest": array_digest(train["data"], train_y),
        "validation_subset_array_digest": array_digest(validation["data"], val_y),
        "training": training,
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
        "metadata_as_model_input": False,
    }
    write_json(output / "config.json", config)

    best_accuracy = -1.0
    best_epoch = -1
    patience_reference = -1.0
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False
    started = time.time()
    fields = [
        "epoch", "lr", "elapsed_seconds", "train_accuracy", "train_total", "train_rec",
        "train_kld", "train_ent", "train_dis", "validation_accuracy", "validation_total",
        "validation_rec", "validation_kld", "validation_ent", "validation_dis", "best_epoch",
        "best_validation_accuracy", "prototype_reset", "epochs_without_improvement",
    ]
    with (
        (output / "training_log.jsonl").open("x", encoding="utf-8") as json_handle,
        (output / "training_log.csv").open("x", encoding="utf-8", newline="") as csv_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(int(training["epochs"])):
            epoch_started = time.time()
            train_metrics = run_epoch(model, train_loader, device, float(training["lambda"]), optimizer)
            reset = epoch in set(map(int, training["prototype_reset_zero_based_epochs"]))
            if reset:
                reset_prototypes(model, train_loader, device)
            val_metrics = run_epoch(model, val_loader, device, float(training["lambda"]), None)
            scheduler.step()
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = float(val_metrics["accuracy"])
                best_epoch = epoch + 1
                torch.save({
                    "model_state_dict": model.state_dict(), "task": task, "seed": seed,
                    "n_classes": len(class_names), "latent_dim": int(training["latent_dim"]),
                    "epoch": best_epoch, "validation_accuracy": best_accuracy,
                    "classes": class_names,
                }, output / "model_best.pt")
            if val_metrics["accuracy"] > patience_reference + float(training["early_stop_min_delta"]):
                patience_reference = float(val_metrics["accuracy"])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if reset:
                patience_reference = float(val_metrics["accuracy"])
                epochs_without_improvement = 0
            completed_epochs = epoch + 1
            record = {
                "epoch": completed_epochs, "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - epoch_started, "train": train_metrics,
                "validation": val_metrics, "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy, "prototype_reset": reset,
                "epochs_without_improvement": epochs_without_improvement,
            }
            json_handle.write(json.dumps(record) + "\n")
            json_handle.flush()
            writer.writerow({
                "epoch": completed_epochs, "lr": record["lr"], "elapsed_seconds": record["elapsed_seconds"],
                **{f"train_{name}": train_metrics[name] for name in ("accuracy", "total", "rec", "kld", "ent", "dis")},
                **{f"validation_{name}": val_metrics[name] for name in ("accuracy", "total", "rec", "kld", "ent", "dis")},
                "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                "prototype_reset": reset, "epochs_without_improvement": epochs_without_improvement,
            })
            csv_handle.flush()
            print(
                f"task={task} seed={seed} epoch={completed_epochs:03d}/{training['epochs']} "
                f"train_acc={train_metrics['accuracy']:.6f} val_acc={val_metrics['accuracy']:.6f} "
                f"seconds={record['elapsed_seconds']:.1f}", flush=True,
            )
            if (
                completed_epochs >= int(training["early_stop_min_epoch"])
                and epochs_without_improvement >= int(training["early_stop_patience"])
            ):
                stopped_early = True
                break

    checkpoint = torch.load(output / "model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    stable_train_metrics = run_epoch(model, stable_train_loader, device, float(training["lambda"]), None)
    stable_val_metrics = run_epoch(model, val_loader, device, float(training["lambda"]), None)
    val_true, val_pred = predict(model, val_loader, device)
    if not np.array_equal(val_true, val_y):
        raise RuntimeError("validation prediction order mismatch")
    metrics = metric_bundle(val_true, val_pred, class_names)
    prediction_rows = []
    for index, (true_index, pred_index) in enumerate(zip(val_true, val_pred)):
        meta = validation["metadata"][index]
        prediction_rows.append({
            "task": task, "seed": seed, "row_index": index, "flow_id": meta["flow_id"],
            "capture_id": meta["capture_id"], "source_file": meta["source_file"],
            "application_label": meta["application_label"], "service_label": meta["service_label"],
            "true_index": int(true_index), "predicted_index": int(pred_index),
            "true_label": class_names[int(true_index)], "predicted_label": class_names[int(pred_index)],
            "correct": int(true_index == pred_index),
        })
    write_csv(output / "validation_predictions.csv", prediction_rows)
    write_json(output / "classification_metrics.json", metrics)
    checkpoint_sha = sha256_file(output / "model_best.pt")
    (output / "checkpoint.sha256").write_text(checkpoint_sha + "  model_best.pt\n", encoding="utf-8")
    try:
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    except RuntimeError:
        peak_gpu_memory_bytes = None
    result = {
        "status": "SUCCESS", "task": task, "seed": seed, "classes": class_names,
        "train_samples": len(train_y), "validation_samples": len(val_y),
        "best_epoch": best_epoch, "completed_epochs": completed_epochs, "stopped_early": stopped_early,
        "best_validation_accuracy": best_accuracy,
        "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
        "validation_weighted_f1": metrics["weighted_f1"],
        "train_loss": stable_train_metrics["total"], "validation_loss": stable_val_metrics["total"],
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "checkpoint_path": str(output / "model_best.pt"), "checkpoint_sha256": checkpoint_sha,
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
    }
    write_json(output / "result.json", result)
    (output / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("fine", "service"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    output = OUT / "runs" / args.task / f"seed{args.seed}"
    try:
        result = run(args.task, args.seed)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILURE.json", {
            "status": "FAILED", "task": args.task, "seed": args.seed,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_and_do_not_overwrite": True, "known_test_feature_values_used": 0,
            "unknown_test_feature_values_used": 0,
        })
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
