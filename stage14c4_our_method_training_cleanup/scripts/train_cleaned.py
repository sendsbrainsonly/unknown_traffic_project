#!/usr/bin/env python3
"""Train one cleaned F2 run using Known Train/Validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F2_ROOT = PROJECT / "stage14c5_feature_representation_audit"
F2_SCRIPTS = F2_ROOT / "scripts"
AUDIT_ROOT = PROJECT / "opendetect_ustc_encoder_audit"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
PROTOCOLS = ("medium_seed2025", "medium_seed2026")

sys.path.insert(0, str(F2_SCRIPTS))
import common as f2_common  # noqa: E402
from model_utils import KnownDataset, make_model  # noqa: E402

sys.path.insert(0, str(AUDIT_ROOT))
from adapters.opendetect_model import released_weight_init  # noqa: E402
from scripts.train_opendetect import released_reset_prototypes, validation_composite  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics_from_confusion(confusion: np.ndarray) -> tuple[float, float, float]:
    supports = confusion.sum(axis=1)
    f1s: list[float] = []
    for class_id in range(len(confusion)):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else float(2 * tp / denominator))
    accuracy = float(np.trace(confusion) / confusion.sum())
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(np.asarray(f1s), weights=supports))
    return accuracy, macro_f1, weighted_f1


def run_epoch(model, loader, device, optimizer: optim.Optimizer | None) -> dict[str, float]:
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
            _, predictions, losses = model.loss(images, labels)
            total = 0.005 * (losses["rec"] + losses["kld"] + losses["ent"]) + 0.995 * losses["dis"]
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
        np.add.at(confusion, (labels.detach().cpu().numpy(), predictions.detach().cpu().numpy()), 1)
        count += batch
    result = {name: value / count for name, value in totals.items()}
    result["accuracy"], result["macro_f1"], result["weighted_f1"] = metrics_from_confusion(confusion)
    result["samples"] = count
    return result


def reset_prototypes_in_place(model, loader, device, optimizer: optim.Optimizer) -> dict[str, object]:
    before_id = id(model.prototypes)
    fresh = released_reset_prototypes(model, loader, device)
    with torch.no_grad():
        model.prototypes.copy_(fresh.detach())
    optimizer.state.pop(model.prototypes, None)
    after_id = id(model.prototypes)
    linked = any(parameter is model.prototypes for group in optimizer.param_groups for parameter in group["params"])
    if before_id != after_id or not linked:
        raise RuntimeError("prototype reset broke optimizer ownership")
    return {
        "prototype_parameter_id_preserved": before_id == after_id,
        "prototype_optimizer_link_preserved": linked,
        "prototype_optimizer_state_cleared": model.prototypes not in optimizer.state,
    }


@torch.no_grad()
def validation_predictions(model, input_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    dataset = KnownDataset(input_dir, "validation", augment=False)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, labels in loader:
        _, predicted, _ = model.loss(images.to(device, non_blocking=True), labels.to(device, non_blocking=True))
        truths.append(labels.numpy())
        predictions.append(predicted.detach().cpu().numpy())
    return np.concatenate(truths), np.concatenate(predictions)


def write_diagnostics(run_id: str, run_dir: Path, truth: np.ndarray, predictions: np.ndarray, input_dir: Path) -> dict[str, object]:
    label_map = json.loads((input_dir / "label_map.json").read_text(encoding="utf-8"))
    class_to_local = {str(name): int(value) for name, value in label_map["class_to_local"].items()}
    local_to_class = [name for name, _ in sorted(class_to_local.items(), key=lambda pair: pair[1])]
    train_labels = np.load(input_dir / "train_labels.npy", allow_pickle=False)
    confusion = np.zeros((len(local_to_class), len(local_to_class)), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    train_supports = np.bincount(train_labels, minlength=len(local_to_class))
    median_support = float(np.median(train_supports))
    rows: list[dict[str, object]] = []
    for class_id, name in enumerate(local_to_class):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({
            "run_id": run_id,
            "class": name,
            "local_label": class_id,
            "train_support": int(train_supports[class_id]),
            "val_support": int(confusion[class_id].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "minority_class": bool(train_supports[class_id] < median_support),
        })
    write_csv(run_dir / "per_class_metrics.csv", rows)
    confusion_dir = ROOT / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = confusion_dir / f"{run_id}.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *local_to_class])
        for name, values in zip(local_to_class, confusion):
            writer.writerow([name, *map(int, values)])
    np.save(run_dir / "validation_predictions.npy", predictions.astype(np.int64), allow_pickle=False)
    accuracy, macro_f1, weighted_f1 = metrics_from_confusion(confusion)
    minority = [float(row["recall"]) for row in rows if row["minority_class"]]
    return {
        "val_accuracy_recomputed": accuracy,
        "val_macro_f1_recomputed": macro_f1,
        "val_weighted_f1_recomputed": weighted_f1,
        "minority_recall": float(np.mean(minority)) if minority else float("nan"),
        "minority_class_count": len(minority),
        "per_class_metrics_path": str(run_dir / "per_class_metrics.csv"),
        "confusion_matrix_path": str(confusion_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", choices=PROTOCOLS, required=True)
    parser.add_argument("--seed-mode", choices=("protocol", "fixed2022"), required=True)
    args = parser.parse_args()

    frozen = f2_common.verify_frozen_inputs()
    if frozen["freeze_hash"] != EXPECTED_FREEZE:
        raise RuntimeError("Stage 14B freeze hash changed")
    protocol = f2_common.stage14c_common.load_protocol(args.protocol_id)
    input_dir = F2_ROOT / "runs" / "f2" / args.protocol_id / "inputs"
    input_audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if input_audit["status"] != "PASS" or input_audit["unknown_samples_loaded"] != 0 or input_audit["known_test_samples_loaded"] != 0:
        raise RuntimeError("Known-only F2 input gate failed")
    if input_audit["derived_manifest_sha256"] != sha256_file(input_dir / "input_manifest.csv"):
        raise RuntimeError("F2 input manifest hash changed")

    training_seed = int(protocol["seed"]) if args.seed_mode == "protocol" else 2022
    run_id = f"{args.protocol_id}_{args.seed_mode}"
    run_dir = ROOT / "runs" / run_id
    best_path = ROOT / "checkpoints" / f"{run_id}_best.pt"
    latest_path = run_dir / "latest_checkpoint.pt"
    required_absent = [run_dir, best_path]
    if any(path.exists() for path in required_absent):
        raise RuntimeError(f"refusing to overwrite evidence for {run_id}")
    run_dir.mkdir(parents=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    f2_common.stage14c_common.seed_everything(training_seed)
    device = torch.device("cuda:0")
    train_ds = KnownDataset(input_dir, "train", augment=True)
    val_ds = KnownDataset(input_dir, "validation", augment=False)
    generator = torch.Generator().manual_seed(training_seed)
    loader_kwargs = dict(batch_size=128, num_workers=0, pin_memory=True, persistent_workers=False)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = make_model(len(protocol["known_applications"]), device)
    model.apply(released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    config = {
        "method": "F2 own-method cleaned training",
        "protocol_id": args.protocol_id,
        "setting": protocol["setting"],
        "protocol_seed": int(protocol["seed"]),
        "training_seed": training_seed,
        "seed_mode": args.seed_mode,
        "run_role": "primary" if args.seed_mode == "protocol" else "seed_sensitivity_control",
        "known_classes": protocol["known_applications"],
        "unknown_classes_excluded": protocol["unknown_applications"],
        "unknown_samples_loaded": 0,
        "known_test_samples_loaded": 0,
        "freeze_hash": frozen["freeze_hash"],
        "stage14c_summary_sha256": frozen["stage14c_summary_sha256"],
        "input_manifest_sha256": sha256_file(input_dir / "input_manifest.csv"),
        "f2_scaler_sha256": sha256_file(input_dir / "scaler.json"),
        "epochs": 100,
        "batch_size": 128,
        "workers": 0,
        "learning_rate": 0.001,
        "optimizer": "Adam(beta1=0.9,beta2=0.999,weight_decay=0)",
        "scheduler": "MultiStepLR(milestones=[50,80],gamma=0.1)",
        "loss": "0.005*(rec+kld+ent)+0.995*dis",
        "checkpoint_selection": "highest harmonic mean of Known Validation Accuracy and Macro-F1",
        "early_stopping": "disabled",
        "prototype_reset": "epochs 51/81, in-place copy, prototype optimizer state cleared",
        "numerical_stability_guard": "upper logvar clamp at 20",
        "gradient_clipping": "none",
        "class_weight": "none",
        "sampler": "RandomSampler via shuffle=True",
        "warmup": "none",
        "extra_regularization": "none",
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (run_dir / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_epoch = -1
    reset_audits: list[dict[str, object]] = []
    started = time.time()
    steps_per_epoch = math.ceil(len(train_ds) / 128)
    for epoch_index in range(100):
        guard0 = model.stability_guard_activations
        train = run_epoch(model, train_loader, device, optimizer)
        guard1 = model.stability_guard_activations
        reset = epoch_index in (50, 80)
        if reset:
            reset_audits.append({"epoch": epoch_index + 1, **reset_prototypes_in_place(model, train_loader, device, optimizer)})
        val = run_epoch(model, val_loader, device, None)
        guard2 = model.stability_guard_activations
        score = validation_composite(val["accuracy"], val["macro_f1"])
        if not all(math.isfinite(float(value)) for value in list(train.values()) + list(val.values()) + [score]):
            raise RuntimeError("non-finite training metric")
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch_index + 1
        row: dict[str, object] = {
            "protocol_id": args.protocol_id,
            "seed_mode": args.seed_mode,
            "training_seed": training_seed,
            "epoch": epoch_index + 1,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "optimizer_steps_cumulative": steps_per_epoch * (epoch_index + 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "prototype_reset": reset,
            "elapsed_seconds": time.time() - started,
            "val_composite_score": score,
            "best_epoch_so_far": best_epoch,
            "stability_guard_train_activations": guard1 - guard0,
            "stability_guard_validation_activations": guard2 - guard1,
            "stability_guard_cumulative_activations": guard2,
        }
        for prefix, values in (("train", train), ("val", val)):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
        write_csv(run_dir / "training_metrics.csv", rows)
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch_index + 1,
            "train_metrics": train,
            "val_metrics": val,
            "config": config,
            "best_epoch": best_epoch,
            "best_validation_composite": best_score,
            "prototype_reset_audits": reset_audits,
        }
        if improved:
            torch.save(payload, best_path)
        torch.save(payload, latest_path)
        scheduler.step()
        print(json.dumps({
            "run_id": run_id,
            "epoch": epoch_index + 1,
            "train_loss": train["total"],
            "val_loss": val["total"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_f1": val["weighted_f1"],
            "best_epoch": best_epoch,
        }), flush=True)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    truth, predictions = validation_predictions(model, input_dir, device)
    diagnostics = write_diagnostics(run_id, run_dir, truth, predictions, input_dir)
    with (run_dir / "per_class_metrics.csv").open(encoding="utf-8") as handle:
        per_class = list(csv.DictReader(handle))
    weighted_f1 = float(np.average(
        np.asarray([float(row["f1"]) for row in per_class]),
        weights=np.asarray([int(row["val_support"]) for row in per_class]),
    ))
    best_val = best["val_metrics"]
    best_train = best["train_metrics"]
    if abs(diagnostics["val_accuracy_recomputed"] - float(best_val["accuracy"])) > 1e-12:
        raise RuntimeError("best checkpoint accuracy does not reproduce")
    if abs(diagnostics["val_macro_f1_recomputed"] - float(best_val["macro_f1"])) > 1e-12:
        raise RuntimeError("best checkpoint Macro-F1 does not reproduce")
    result = {
        "status": "success",
        "run_id": run_id,
        "protocol_id": args.protocol_id,
        "seed_mode": args.seed_mode,
        "run_role": config["run_role"],
        "protocol_seed": int(protocol["seed"]),
        "training_seed": training_seed,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "epochs_configured": 100,
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "train_loss": float(best_train["total"]),
        "train_accuracy": float(best_train["accuracy"]),
        "train_macro_f1": float(best_train["macro_f1"]),
        "train_weighted_f1": float(best_train["weighted_f1"]),
        "val_loss": float(best_val["total"]),
        "val_accuracy": float(best_val["accuracy"]),
        "val_macro_f1": float(best_val["macro_f1"]),
        "val_weighted_f1": weighted_f1,
        "best_validation_composite": float(best["best_validation_composite"]),
        "nan_or_crash": False,
        "stability_guard_activations": int(rows[-1]["stability_guard_cumulative_activations"]),
        "prototype_reset_audits": reset_audits,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "freeze_hash": frozen["freeze_hash"],
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
        **diagnostics,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "COMPLETED").write_text("success\n", encoding="utf-8")
    print(json.dumps({"event": "run_complete", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
