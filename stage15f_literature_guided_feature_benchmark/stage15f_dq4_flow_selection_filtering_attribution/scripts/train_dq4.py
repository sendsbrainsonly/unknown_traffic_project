#!/usr/bin/env python3
"""Train one frozen DQ-4 Service model on B, C_FINAL, or D."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dq4_common import (
    DQ3F, OUT, SEEDS, SERVICES, TRAINED_MODELS, array_digest, flow_id_digest,
    read_json, sha256_file, subset_data, write_csv, write_json,
)

sys.path.insert(0, str(DQ3F / "scripts"))
from dq3f_common import seed_everything  # noqa: E402
from train_dq3f import TrafficImages, import_native, metric_bundle, predict  # noqa: E402


def make_loader(split: dict, batch_size: int, workers: int, train: bool, seed: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        TrafficImages(split["data"], split["target"], train=train),
        batch_size=batch_size, shuffle=train, generator=generator, drop_last=False,
        num_workers=workers, pin_memory=True, persistent_workers=False,
    )


def run(model_name: str, seed: int) -> dict:
    if model_name not in {"M-B", "M-C", "M-D"} or seed not in SEEDS:
        raise ValueError((model_name, seed))
    preflight = read_json(OUT / "preflight_status.json")
    if preflight["status"] != "PASS" or not preflight["training_allowed"]:
        raise RuntimeError("DQ-4 preflight Gate does not permit training")
    if preflight["known_test_feature_values_used"] != 0 or preflight["unknown_test_feature_values_used"] != 0:
        raise RuntimeError("Test leakage marker is non-zero")
    output = OUT / "runs" / model_name / f"seed{seed}"
    if (output / "SUCCESS").is_file():
        return {"status": "SKIP_SUCCESS", "model": model_name, "seed": seed, "output": str(output)}
    if output.exists():
        raise FileExistsError(f"preserved partial run; refusing overwrite: {output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DQ-4 training")

    subset = TRAINED_MODELS[model_name]
    train = subset_data("known_train", subset)
    validation = subset_data("known_validation", subset)
    validation_a = subset_data("known_validation", "A")
    if sorted(set(row["service_label"] for row in train["metadata"])) != SERVICES:
        raise RuntimeError(f"{model_name} Train does not contain all six Services")
    if sorted(set(row["service_label"] for row in validation["metadata"])) != SERVICES:
        raise RuntimeError(f"{model_name} Validation does not contain all six Services")
    training = read_json(OUT / "dq4_training_configs.json")["training"]

    seed_everything(seed)
    workers = int(training["workers"])
    train_loader = make_loader(train, int(training["batch_size"]), workers, True, seed)
    stable_train_loader = make_loader(train, int(training["eval_batch_size"]), workers, False)
    val_loader = make_loader(validation, int(training["eval_batch_size"]), workers, False)
    val_a_loader = make_loader(validation_a, int(training["eval_batch_size"]), workers, False)
    CorrectedOpenDetectNet, reset_prototypes, run_epoch, weight_init = import_native()
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        training["architecture"], int(training["channels"]), int(training["latent_dim"]),
        len(SERVICES), 1, 1,
    ).to(device)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(map(int, training["scheduler_milestones"])), gamma=float(training["scheduler_gamma"]),
    )
    steps_per_epoch = math.ceil(len(train["target"]) / int(training["batch_size"]))
    config = {
        "model": model_name, "training_subset": subset, "seed": seed,
        "classes": SERVICES, "class_to_index": {name: index for index, name in enumerate(SERVICES)},
        "train_samples": len(train["target"]), "validation_samples": len(validation["target"]),
        "train_flow_id_sha256": flow_id_digest(train["metadata"]),
        "validation_flow_id_sha256": flow_id_digest(validation["metadata"]),
        "train_subset_array_digest": array_digest(train["data"], train["target"]),
        "validation_subset_array_digest": array_digest(validation["data"], validation["target"]),
        "steps_per_epoch": steps_per_epoch, "training": training,
        "gpu": torch.cuda.get_device_name(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
        "metadata_as_model_input": False, "weak_capture_labels": True,
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
                    "model_state_dict": model.state_dict(), "model": model_name, "training_subset": subset,
                    "task": "service", "seed": seed, "n_classes": len(SERVICES),
                    "latent_dim": int(training["latent_dim"]), "epoch": best_epoch,
                    "validation_accuracy": best_accuracy, "classes": SERVICES,
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
                f"model={model_name} seed={seed} epoch={completed_epochs:03d}/{training['epochs']} "
                f"train_acc={train_metrics['accuracy']:.6f} val_acc={val_metrics['accuracy']:.6f} "
                f"seconds={record['elapsed_seconds']:.1f}", flush=True,
            )
            if completed_epochs >= int(training["early_stop_min_epoch"]) and epochs_without_improvement >= int(training["early_stop_patience"]):
                stopped_early = True
                break

    checkpoint = torch.load(output / "model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    stable_train_metrics = run_epoch(model, stable_train_loader, device, float(training["lambda"]), None)
    stable_val_metrics = run_epoch(model, val_loader, device, float(training["lambda"]), None)
    own_true, own_pred = predict(model, val_loader, device)
    a_true, a_pred = predict(model, val_a_loader, device)
    if not np.array_equal(own_true, validation["target"]) or not np.array_equal(a_true, validation_a["target"]):
        raise RuntimeError("validation prediction order mismatch")
    own_metrics = metric_bundle(own_true, own_pred, SERVICES)
    a_metrics = metric_bundle(a_true, a_pred, SERVICES)
    prediction_rows = []
    for index, (true_index, pred_index) in enumerate(zip(a_true, a_pred)):
        meta = validation_a["metadata"][index]
        prediction_rows.append({
            "model": model_name, "training_subset": subset, "seed": seed, "row_index": index,
            "flow_id": meta["flow_id"], "capture_id": meta["capture_id"], "source_file": meta["source_file"],
            "application_label": meta["application_label"], "service_label": meta["service_label"],
            "true_index": int(true_index), "predicted_index": int(pred_index),
            "true_label": SERVICES[int(true_index)], "predicted_label": SERVICES[int(pred_index)],
            "correct": int(true_index == pred_index),
            **{name: int(bool(meta[name])) for name in ("A", "B", "C_PARENT", "C_FINAL", "D")},
        })
    write_csv(output / "all_a_validation_predictions.csv", prediction_rows)
    write_json(output / "own_validation_metrics.json", own_metrics)
    write_json(output / "all_a_validation_metrics.json", a_metrics)
    checkpoint_sha = sha256_file(output / "model_best.pt")
    (output / "checkpoint.sha256").write_text(checkpoint_sha + "  model_best.pt\n", encoding="utf-8")
    result = {
        "status": "SUCCESS", "model": model_name, "training_subset": subset, "seed": seed,
        "classes": SERVICES, "train_samples": len(train["target"]), "validation_samples": len(validation["target"]),
        "best_epoch": best_epoch, "completed_epochs": completed_epochs, "stopped_early": stopped_early,
        "steps_per_epoch": steps_per_epoch, "completed_optimizer_updates": completed_epochs * steps_per_epoch,
        "best_validation_accuracy": best_accuracy,
        "validation_accuracy": own_metrics["accuracy"], "validation_macro_f1": own_metrics["macro_f1"],
        "validation_weighted_f1": own_metrics["weighted_f1"],
        "all_a_validation_accuracy": a_metrics["accuracy"], "all_a_validation_macro_f1": a_metrics["macro_f1"],
        "all_a_validation_weighted_f1": a_metrics["weighted_f1"],
        "train_loss": stable_train_metrics["total"], "validation_loss": stable_val_metrics["total"],
        "runtime_seconds": time.time() - started,
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
    parser.add_argument("--model", choices=("M-B", "M-C", "M-D"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    output = OUT / "runs" / args.model / f"seed{args.seed}"
    try:
        result = run(args.model, args.seed)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILURE.json", {
            "status": "FAILED", "model": args.model, "seed": args.seed,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_and_do_not_overwrite": True, "known_test_feature_values_used": 0,
            "unknown_test_feature_values_used": 0,
        })
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
