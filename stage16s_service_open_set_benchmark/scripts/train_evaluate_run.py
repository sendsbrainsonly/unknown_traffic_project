#!/usr/bin/env python3
"""Train one strict Unknown-Free five-Service OD model and evaluate four scores."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from stage16s_common import (
    CONFIG, DQ3F, METHODS, OPEN_DETECT, OUT, RUNS, SEEDS, SERVICES, array_hash,
    known_classes, load_role, protocol_id, protocol_rows, read_json, seed_everything,
    sha256_file, write_csv, write_json,
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


def import_open_detect():
    for path in (OPEN_DETECT / "code", OPEN_DETECT / "reproduction"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from corrected_model import CorrectedOpenDetectNet
    from run_reproduction import reset_prototypes_in_place, run_epoch
    from utils import weight_init
    return CorrectedOpenDetectNet, reset_prototypes_in_place, run_epoch, weight_init


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract(model, role: dict, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    mus, logvars, scores, predictions = [], [], [], []
    # Match checkpoint selection exactly. A manual uint8 -> float/255 path can
    # differ from PIL/ToTensor by one float32 ulp; that was enough to flip one
    # near-tied prediction in the collapsed Streaming-2023 run.
    loader = DataLoader(
        TrafficImages(role["data"], role["labels"], False),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device=device, non_blocking=True)
            mu, raw_logvar, _ = model.encoder(batch)
            logvar = torch.clamp(raw_logvar, min=-30.0, max=20.0)
            class_kl = 0.5 * model.distance(mu, model.prototypes)
            variance_kl = 0.5 * torch.sum(logvar.exp() - logvar - 1, dim=1)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            scores.append((class_kl.min(dim=1).values + variance_kl).cpu().numpy())
            predictions.append(class_kl.argmin(dim=1).cpu().numpy())
    return {
        "mu": np.concatenate(mus).astype(np.float64),
        "logvar": np.concatenate(logvars).astype(np.float64),
        "od_score": np.concatenate(scores).astype(np.float64),
        "native_prediction": np.concatenate(predictions).astype(np.int64),
    }


def closed_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def detection_metrics(validation: np.ndarray, known: np.ndarray, unknown: np.ndarray,
                      known_labels: np.ndarray, native_known_predictions: np.ndarray,
                      native_unknown_predictions: np.ndarray, class_count: int,
                      threshold_function) -> dict[str, float | int]:
    threshold = float(threshold_function(validation))
    y_true_binary = np.concatenate([np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)])
    y_score = np.concatenate([known, unknown])
    known_open_prediction = native_known_predictions.copy()
    known_open_prediction[known >= threshold] = class_count
    unknown_open_prediction = native_unknown_predictions.copy()
    unknown_open_prediction[unknown >= threshold] = class_count
    y_true_open = np.concatenate([known_labels, np.full(len(unknown), class_count, dtype=int)])
    y_pred_open = np.concatenate([known_open_prediction, unknown_open_prediction])
    return {
        "threshold": threshold,
        "auroc": float(roc_auc_score(y_true_binary, y_score)),
        "auprc": float(average_precision_score(y_true_binary, y_score)),
        "ufar": float(np.mean(unknown < threshold)),
        "known_frr": float(np.mean(known >= threshold)),
        "validation_frr": float(np.mean(validation >= threshold)),
        "open_macro_f1": float(f1_score(y_true_open, y_pred_open, average="macro", zero_division=0)),
        "open_weighted_f1": float(f1_score(y_true_open, y_pred_open, average="weighted", zero_division=0)),
        "known_test_samples": int(len(known)),
        "unknown_test_samples": int(len(unknown)),
        "unknown_prevalence": float(len(unknown) / (len(known) + len(unknown))),
    }


def run(selected_protocol: str, seed: int) -> dict:
    config = read_json(CONFIG)
    if config["status"] != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("training config is not frozen")
    if seed not in SEEDS or selected_protocol not in {protocol_id(service) for service in SERVICES}:
        raise ValueError((selected_protocol, seed))
    output = RUNS / selected_protocol / f"seed{seed}"
    if (output / "SUCCESS").is_file():
        return {"status": "SKIP_SUCCESS", "output": str(output)}
    if output.exists():
        raise FileExistsError(f"preserved partial run; refusing overwrite: {output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    roles = {role: load_role(selected_protocol, role) for role in ("known_train", "known_validation", "known_test", "unknown_test")}
    classes = known_classes(selected_protocol)
    unknown_service = next(row["unknown_service"] for row in protocol_rows(selected_protocol))
    for role in ("known_train", "known_validation", "known_test"):
        if np.any(roles[role]["labels"] < 0):
            raise RuntimeError(f"negative Known label in {role}")
        observed = {row["service_label"] for row in roles[role]["metadata"]}
        if observed != set(classes):
            raise RuntimeError(f"Known class mismatch in {role}: {observed}")
    if {row["service_label"] for row in roles["unknown_test"]["metadata"]} != {unknown_service}:
        raise RuntimeError("Unknown Test contains a non-held-out Service")
    train_counts = np.bincount(roles["known_train"]["labels"], minlength=len(classes))
    if np.any(train_counts < 10):
        raise RuntimeError(f"kNN-10 support failure: {train_counts.tolist()}")

    training = config["training"]
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {"num_workers": int(training["workers"]), "pin_memory": True, "persistent_workers": False}
    train_loader = DataLoader(
        TrafficImages(roles["known_train"]["data"], roles["known_train"]["labels"], True),
        batch_size=int(training["batch_size"]), shuffle=True, generator=generator, drop_last=False, **loader_args,
    )
    stable_train_loader = DataLoader(
        TrafficImages(roles["known_train"]["data"], roles["known_train"]["labels"], False),
        batch_size=int(training["eval_batch_size"]), shuffle=False, **loader_args,
    )
    val_loader = DataLoader(
        TrafficImages(roles["known_validation"]["data"], roles["known_validation"]["labels"], False),
        batch_size=int(training["eval_batch_size"]), shuffle=False, **loader_args,
    )
    CorrectedOpenDetectNet, reset_prototypes, run_epoch, weight_init = import_open_detect()
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        training["architecture"], int(training["channels"]), int(training["latent_dim"]), len(classes), 1, 1,
    ).to(device)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(map(int, training["scheduler_milestones"])), gamma=float(training["scheduler_gamma"]),
    )
    run_config = {
        "protocol_id": selected_protocol, "unknown_service": unknown_service, "known_services": classes,
        "seed": seed, "training": training, "train_counts": train_counts.tolist(),
        "role_counts": {role: len(value["data"]) for role, value in roles.items()},
        "protocol_manifest_sha256": config["manifest_sha256"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": torch.cuda.get_device_name(device), "unknown_fit_samples": 0,
        "known_test_selection_samples": 0, "pretrained_checkpoint": None,
    }
    write_json(output / "config.json", run_config)

    best_accuracy, best_epoch = -1.0, -1
    patience_reference, epochs_without_improvement = -1.0, 0
    completed_epochs, stopped_early = 0, False
    started = time.time()
    log_fields = [
        "epoch", "lr", "elapsed_seconds", "train_accuracy", "train_total", "train_rec", "train_kld",
        "train_ent", "train_dis", "validation_accuracy", "validation_total", "validation_rec",
        "validation_kld", "validation_ent", "validation_dis", "best_epoch", "best_validation_accuracy",
        "prototype_reset", "epochs_without_improvement",
    ]
    with (output / "training_log.jsonl").open("x", encoding="utf-8") as json_handle, (output / "training_log.csv").open("x", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=log_fields)
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
                    "model_state_dict": model.state_dict(), "protocol_id": selected_protocol, "seed": seed,
                    "n_classes": len(classes), "latent_dim": int(training["latent_dim"]), "epoch": best_epoch,
                    "validation_accuracy": best_accuracy, "classes": classes, "unknown_service": unknown_service,
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
                "validation": val_metrics, "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                "prototype_reset": reset, "epochs_without_improvement": epochs_without_improvement,
            }
            json_handle.write(json.dumps(record) + "\n"); json_handle.flush()
            writer.writerow({
                "epoch": completed_epochs, "lr": record["lr"], "elapsed_seconds": record["elapsed_seconds"],
                **{f"train_{name}": train_metrics[name] for name in ("accuracy", "total", "rec", "kld", "ent", "dis")},
                **{f"validation_{name}": val_metrics[name] for name in ("accuracy", "total", "rec", "kld", "ent", "dis")},
                "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                "prototype_reset": reset, "epochs_without_improvement": epochs_without_improvement,
            }); csv_handle.flush()
            print(f"protocol={selected_protocol} seed={seed} epoch={completed_epochs:03d} train={train_metrics['accuracy']:.6f} val={val_metrics['accuracy']:.6f}", flush=True)
            if completed_epochs >= int(training["early_stop_min_epoch"]) and epochs_without_improvement >= int(training["early_stop_patience"]):
                stopped_early = True
                break

    checkpoint = torch.load(output / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    stable_train = run_epoch(model, stable_train_loader, device, float(training["lambda"]), None)
    stable_val = run_epoch(model, val_loader, device, float(training["lambda"]), None)
    latent = {role: extract(model, value, device, int(training["eval_batch_size"])) for role, value in roles.items()}
    native_val_metrics = closed_metrics(roles["known_validation"]["labels"], latent["known_validation"]["native_prediction"])
    if abs(native_val_metrics["accuracy"] - best_accuracy) > 1e-12:
        raise RuntimeError("best-checkpoint validation parity failed")

    stage14d = import_module("stage16s_stage14d_common", OUT.parent / "stage14d_vnat_frozen_open_set_evaluation" / "scripts" / "stage14d_common.py")
    stage15b = import_module("stage16s_stage15b", OUT.parent / "stage15b_known_only_hybrid_detector" / "scripts" / "run_stage15b.py")
    centroids = stage14d.empirical_centroids(latent["known_train"]["mu"], roles["known_train"]["labels"])
    des0, centroid_predictions, local = {}, {}, {}
    for role in ("known_validation", "known_test", "unknown_test"):
        des0[role], centroid_predictions[role] = stage14d.centroid_scores(latent[role]["mu"], centroids)
        local[role] = stage14d.local_knn10_scores(
            latent[role]["mu"], centroid_predictions[role], latent["known_train"]["mu"], roles["known_train"]["labels"],
        )
    epsilon = float(config["des_v1"]["epsilon"])
    global_parameters = stage14d.robust_parameters(des0["known_validation"], epsilon)
    local_parameters = stage14d.robust_parameters(local["known_validation"], epsilon)
    des1 = {
        role: 0.5 * stage14d.robust_normalize(des0[role], global_parameters)
        + 0.5 * stage14d.robust_normalize(local[role], local_parameters)
        for role in ("known_validation", "known_test", "unknown_test")
    }
    od = {role: latent[role]["od_score"] for role in ("known_validation", "known_test", "unknown_test")}
    od_percentile = {role: stage15b.empirical_percentile(od["known_validation"], od[role]) for role in od}
    des0_percentile = {role: stage15b.empirical_percentile(des0["known_validation"], des0[role]) for role in des0}
    h1 = {role: np.maximum(od_percentile[role], des0_percentile[role]) for role in od}
    method_scores = {"OD-Native": od, "DES-v0": des0, "DES-v1": des1, "H1": h1}

    known_closed = closed_metrics(roles["known_test"]["labels"], latent["known_test"]["native_prediction"])
    method_metrics = {}
    thresholds = {}
    for method, scores in method_scores.items():
        metric = detection_metrics(
            scores["known_validation"], scores["known_test"], scores["unknown_test"],
            roles["known_test"]["labels"], latent["known_test"]["native_prediction"],
            latent["unknown_test"]["native_prediction"], len(classes), stage14d.threshold_p95,
        )
        method_metrics[method] = {**metric, **{f"known_closed_{key}": value for key, value in known_closed.items()}}
        thresholds[method] = {
            "threshold": metric["threshold"], "source": "Known Validation P95 only",
            "numpy_method": "higher", "unknown_if": "score >= threshold",
            "validation_samples": len(scores["known_validation"]), "validation_rejection_rate": metric["validation_frr"],
            "score_direction": "larger_is_more_unknown",
        }

    np.savez_compressed(
        output / "latent_outputs.npz",
        **{f"{role}_{name}": values[name] for role, values in latent.items() for name in ("mu", "logvar", "native_prediction")},
        **{f"{role}_labels": roles[role]["labels"] for role in roles},
        **{f"{role}_flow_ids": np.asarray([row["flow_id"] for row in roles[role]["metadata"]]).astype(str) for role in roles},
    )
    np.savez_compressed(
        output / "score_arrays.npz",
        **{f"{role}_{method.lower().replace('-', '_')}": scores[role] for method, scores in method_scores.items() for role in scores},
        **{f"threshold_{method.lower().replace('-', '_')}": np.asarray(thresholds[method]["threshold"]) for method in METHODS},
        validation_A_OD=od_percentile["known_validation"], validation_A_DES0=des0_percentile["known_validation"],
        known_test_A_OD=od_percentile["known_test"], known_test_A_DES0=des0_percentile["known_test"],
        unknown_test_A_OD=od_percentile["unknown_test"], unknown_test_A_DES0=des0_percentile["unknown_test"],
    )
    score_rows = []
    for role in ("known_validation", "known_test", "unknown_test"):
        for index, meta in enumerate(roles[role]["metadata"]):
            row = {
                "protocol_id": selected_protocol, "unknown_service": unknown_service, "seed": seed, "role": role,
                "flow_id": meta["flow_id"], "true_service": meta["service_label"],
                "application_label": meta["application_label"], "capture_id": meta["capture_id"],
                "local_true_label": int(roles[role]["labels"][index]),
                "native_predicted_index": int(latent[role]["native_prediction"][index]),
                "native_predicted_service": classes[int(latent[role]["native_prediction"][index])],
            }
            for method in METHODS:
                key = method.lower().replace("-", "_")
                score = float(method_scores[method][role][index])
                row[f"{key}_score"] = score
                row[f"{key}_rejected"] = int(score >= thresholds[method]["threshold"])
            score_rows.append(row)
    write_csv(output / "sample_scores.csv", score_rows)
    write_json(output / "thresholds.json", {
        "methods": thresholds, "des_v1_global": global_parameters, "des_v1_local": local_parameters,
        "h1_percentile_fit": "Known Validation only; searchsorted side=right",
        "unknown_calibration_samples": 0, "test_calibration_samples": 0,
    })
    write_json(output / "metrics.json", {
        "protocol_id": selected_protocol, "unknown_service": unknown_service, "seed": seed,
        "known_services": classes, "methods": method_metrics, "known_validation_closed": native_val_metrics,
        "known_test_closed": known_closed,
    })
    checkpoint_sha = sha256_file(output / "model_best.pt")
    (output / "checkpoint.sha256").write_text(checkpoint_sha + "  model_best.pt\n", encoding="utf-8")
    result = {
        "status": "SUCCESS", "protocol_id": selected_protocol, "unknown_service": unknown_service, "seed": seed,
        "known_services": classes, "role_counts": run_config["role_counts"], "train_counts": train_counts.tolist(),
        "best_epoch": best_epoch, "completed_epochs": completed_epochs, "stopped_early": stopped_early,
        "train_loss": stable_train["total"], "validation_loss": stable_val["total"],
        "validation_accuracy": native_val_metrics["accuracy"], "validation_macro_f1": native_val_metrics["macro_f1"],
        "validation_weighted_f1": native_val_metrics["weighted_f1"], "known_test_closed": known_closed,
        "methods": method_metrics, "runtime_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_path": str((output / "model_best.pt").resolve()), "checkpoint_sha256": checkpoint_sha,
        "score_file_sha256": sha256_file(output / "sample_scores.csv"),
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "unknown_support_samples": 0, "unknown_normalization_samples": 0,
        "unknown_threshold_samples": 0, "known_test_selection_samples": 0,
    }
    write_json(output / "result.json", result)
    (output / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    del model, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = RUNS / args.protocol_id / f"seed{args.seed}"
    try:
        result = run(args.protocol_id, args.seed)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILURE.json", {
            "status": "FAILED", "protocol_id": args.protocol_id, "seed": args.seed,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_and_do_not_overwrite": True, "unknown_fit_samples": 0,
        })
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
