#!/usr/bin/env python3
"""Run one pre-registered F0 training-component ablation on Known Train/Val."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F0_ROOT = PROJECT / "stage14c_vnat_encoder_training"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
PROTOCOL_PATH = STAGE14B / "vnat_open_set_protocol.json"
SPLIT_PATH = STAGE14B / "vnat_split_manifest.csv"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_PROTOCOL_SHA = "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced"
EXPECTED_SPLIT_SHA = "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e"
ALLOWED_PROTOCOLS = ("medium_seed2025", "medium_seed2026")
TRAINED_VARIANTS = ("D1", "D4", "D5", "D6", "D7")

AUDIT_ROOT = PROJECT / "opendetect_ustc_encoder_audit"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(PROJECT / "stage3_unknown_utility" / "scripts"))
from adapters.opendetect_model import AuditedOpenDetectNet, released_weight_init  # noqa: E402
from scripts.train_opendetect import (  # noqa: E402
    released_reset_prototypes, run_epoch, validation_composite,
)
from stage3_model import Stage3OpenDetectNet  # noqa: E402


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze() -> dict[str, str]:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    actual = {
        "freeze_hash": document["freeze_hash"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_sha256": sha256_file(SPLIT_PATH),
    }
    expected = {
        "freeze_hash": EXPECTED_FREEZE,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "split_sha256": EXPECTED_SPLIT_SHA,
    }
    if actual != expected:
        raise RuntimeError(f"Stage 14B freeze mismatch: {actual}")
    return actual


def load_protocol(protocol_id: str) -> dict[str, object]:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    return next(item for item in document["protocols"] if item["protocol_id"] == protocol_id)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KnownDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool):
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, names: list[str]) -> tuple[list[dict[str, object]], np.ndarray, dict[str, float]]:
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        _, predictions, _ = model.loss(images, labels_device)
        np.add.at(confusion, (labels.numpy(), predictions.cpu().numpy()), 1)
    rows: list[dict[str, object]] = []
    for class_id, name in enumerate(names):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id].sum() - tp)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({
            "class": name, "local_label": class_id, "support": int(confusion[class_id].sum()),
            "precision": precision, "recall": recall, "f1": f1,
        })
    support = np.asarray([row["support"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    metrics = {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }
    return rows, confusion, metrics


def variant_config(variant: str, protocol_seed: int) -> dict[str, object]:
    base: dict[str, object] = {
        "batch_size": 512,
        "training_seed": protocol_seed,
        "guarded_logvar": True,
        "checkpoint_rule": "validation_composite",
        "early_stop_monitor": "validation_composite",
        "early_stop_patience": 5,
        "complete_100_epochs": False,
    }
    if variant == "D1":
        base["batch_size"] = 128
    elif variant == "D4":
        base["checkpoint_rule"] = "validation_accuracy"
    elif variant == "D5":
        base["guarded_logvar"] = False
    elif variant == "D6":
        base["training_seed"] = 2022
    elif variant == "D7":
        base["early_stop_patience"] = None
        base["complete_100_epochs"] = True
    else:
        raise KeyError(variant)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=TRAINED_VARIANTS, required=True)
    parser.add_argument("--protocol-id", choices=ALLOWED_PROTOCOLS, required=True)
    args = parser.parse_args()
    started = time.time()
    freeze_before = verify_freeze()
    protocol = load_protocol(args.protocol_id)
    config = variant_config(args.variant, int(protocol["seed"]))
    run_dir = ROOT / "ablation_runs" / args.variant / args.protocol_id
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    input_dir = F0_ROOT / "runs" / args.protocol_id / "inputs"
    input_audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if input_audit["status"] != "PASS" or any(int(input_audit[key]) != 0 for key in (
        "unknown_samples_used_in_training", "unknown_samples_used_in_validation",
        "known_test_samples_used", "train_validation_flow_overlap",
    )):
        raise RuntimeError("Known-only input gate failed")
    names = list(map(str, protocol["known_applications"]))
    seed_everything(int(config["training_seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    train_dataset = KnownDataset(input_dir, "train", True)
    validation_dataset = KnownDataset(input_dir, "validation", False)
    generator = torch.Generator().manual_seed(int(config["training_seed"]))
    train_loader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True,
        generator=generator, num_workers=0, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    model_class = Stage3OpenDetectNet if config["guarded_logvar"] else AuditedOpenDetectNet
    model = model_class(
        upstream_code=F0_ROOT / ".." / ".." / "Open-Detect" / "code",
        channels=1, latent_dim=128, num_classes=len(names), temp_inter=1.0, temp_intra=1.0,
    ).to(device)
    model.apply(released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    full_config = {
        "variant": args.variant, "protocol_id": args.protocol_id, **config,
        "known_classes": names, "unknown_classes_excluded": protocol["unknown_applications"],
        "train_samples": len(train_dataset), "validation_samples": len(validation_dataset),
        "model": model_class.__name__, "loss_lambda": 0.005,
        "optimizer": "Adam(lr=0.001,betas=[0.9,0.999],weight_decay=0)",
        "scheduler": "MultiStepLR([50,80],gamma=0.1)", "workers": 0,
        "model_visible_roles": ["known:train", "known:validation"],
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "known_test_samples": 0, "freeze_before": freeze_before,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
    }
    json_dump(run_dir / "config.json", full_config)
    history: list[dict[str, object]] = []
    best_selection = -math.inf
    best_epoch = 0
    monitor_best = -math.inf
    no_improve = 0
    checkpoint = run_dir / "best_checkpoint.pt"
    status = "running"
    try:
        for epoch_index in range(100):
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            guard_before = int(getattr(model, "stability_guard_activations", 0))
            train = run_epoch(model, train_loader, device, 0.005, optimizer)
            reset = epoch_index in (50, 80)
            if reset:
                model.prototypes = released_reset_prototypes(model, train_loader, device)
            validation = run_epoch(model, validation_loader, device, 0.005, None)
            scheduler.step()
            guard_after = int(getattr(model, "stability_guard_activations", 0))
            composite = validation_composite(validation["accuracy"], validation["macro_f1"])
            selection_value = validation["accuracy"] if config["checkpoint_rule"] == "validation_accuracy" else composite
            monitor_value = composite
            improved_selection = selection_value > best_selection
            if improved_selection:
                best_selection = float(selection_value)
                best_epoch = epoch_index + 1
                torch.save({
                    "model_state_dict": model.state_dict(), "epoch": best_epoch,
                    "train_metrics": train, "val_metrics": validation,
                    "selection_value": best_selection, "config": full_config,
                }, checkpoint)
            if monitor_value > monitor_best:
                monitor_best = float(monitor_value)
                no_improve = 0
            else:
                no_improve += 1
            row = {
                "epoch": epoch_index + 1, "learning_rate_used": learning_rate_used,
                "prototype_reset": int(reset), "train_loss": train["total"],
                "val_loss": validation["total"], "train_accuracy": train["accuracy"],
                "val_accuracy": validation["accuracy"], "train_macro_f1": train["macro_f1"],
                "val_macro_f1": validation["macro_f1"], "val_composite": composite,
                "checkpoint_selection_value": selection_value, "is_best_checkpoint": int(improved_selection),
                "early_stop_monitor_value": monitor_value, "epochs_without_monitor_improvement": no_improve,
                "stability_guard_cumulative_activations": guard_after,
                "stability_guard_epoch_activations": guard_after - guard_before,
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            write_csv(run_dir / "history.csv", history)
            print(json.dumps({
                "variant": args.variant, "protocol_id": args.protocol_id,
                "epoch": epoch_index + 1, "val_accuracy": validation["accuracy"],
                "val_macro_f1": validation["macro_f1"], "best_epoch": best_epoch,
                "no_improve": no_improve,
            }), flush=True)
            patience = config["early_stop_patience"]
            if patience is not None and no_improve >= int(patience):
                break
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        per_class, confusion, final_metrics = evaluate(model, validation_loader, device, names)
        write_csv(run_dir / "per_class_metrics.csv", per_class)
        np.save(run_dir / "confusion_matrix.npy", confusion, allow_pickle=False)
        freeze_after = verify_freeze()
        if freeze_after != freeze_before:
            raise RuntimeError("Stage 14B freeze changed")
        result = {
            "status": "success", "variant": args.variant, "protocol_id": args.protocol_id,
            "stop_epoch": len(history), "best_epoch": int(saved["epoch"]),
            "train_loss": float(saved["train_metrics"]["total"]),
            "val_loss": float(saved["val_metrics"]["total"]),
            "val_accuracy": final_metrics["accuracy"], "val_macro_f1": final_metrics["macro_f1"],
            "val_weighted_f1": final_metrics["weighted_f1"],
            "checkpoint_sha256": sha256_file(checkpoint), "freeze_before": freeze_before,
            "freeze_after": freeze_after, "unknown_training_samples": 0,
            "unknown_validation_samples": 0, "known_test_samples": 0,
            "guard_activations": int(getattr(model, "stability_guard_activations", 0)),
            "duration_seconds": time.time() - started,
        }
        json_dump(run_dir / "result.json", result)
        (run_dir / "COMPLETED").write_text("success\n", encoding="utf-8")
        status = "success"
        print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as exception:
        json_dump(run_dir / "failure.json", {
            "status": "failed", "exception": repr(exception),
            "traceback": traceback.format_exc(), "duration_seconds": time.time() - started,
        })
        status = "failed"
        raise
    finally:
        json_dump(run_dir / "terminal_status.json", {"status": status, "history_epochs": len(history)})


if __name__ == "__main__":
    main()

