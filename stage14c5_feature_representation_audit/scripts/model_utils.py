#!/usr/bin/env python3
"""Exact Stage 14C model/data helpers plus Known-Val diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from common import CONFUSION_ROOT, PROJECT_ROOT, STAGE14C_ROOT, stage14c_common

AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "stage3_unknown_utility" / "scripts"))
from stage3_model import Stage3OpenDetectNet  # noqa: E402


class KnownDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool):
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        if len(self.images) != len(self.labels):
            raise RuntimeError("image/label count mismatch")
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


def make_model(num_classes: int, device: torch.device) -> Stage3OpenDetectNet:
    return Stage3OpenDetectNet(
        upstream_code=stage14c_common.UPSTREAM_CODE,
        channels=1,
        latent_dim=128,
        num_classes=num_classes,
        temp_inter=1.0,
        temp_intra=1.0,
    ).to(device)


@torch.no_grad()
def evaluate_predictions(model: Stage3OpenDetectNet, input_dir: Path, device: torch.device, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    dataset = KnownDataset(input_dir, "validation", augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    truth_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for images, labels in loader:
        _, predictions, _ = model.loss(images.to(device, non_blocking=True), labels.to(device, non_blocking=True))
        truth_parts.append(labels.numpy())
        pred_parts.append(predictions.detach().cpu().numpy())
    return np.concatenate(truth_parts), np.concatenate(pred_parts)


def diagnostic_rows(truth: np.ndarray, predictions: np.ndarray, class_to_local: dict[str, int], train_labels: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], dict[str, float]]:
    n_classes = len(class_to_local)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    local_to_class = {value: key for key, value in class_to_local.items()}
    train_counts = np.bincount(train_labels, minlength=n_classes)
    median_support = float(np.median(train_counts))
    rows: list[dict[str, object]] = []
    for class_id in range(n_classes):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({
            "class": local_to_class[class_id], "local_label": class_id,
            "train_support": int(train_counts[class_id]), "val_support": int(confusion[class_id].sum()),
            "precision": precision, "recall": recall, "f1": f1,
            "minority_class": bool(train_counts[class_id] < median_support),
        })
    accuracy = float(np.trace(confusion) / confusion.sum())
    macro_f1 = float(np.mean([row["f1"] for row in rows]))
    minority = [float(row["recall"]) for row in rows if row["minority_class"]]
    summary = {
        "val_accuracy_recomputed": accuracy,
        "val_macro_f1_recomputed": macro_f1,
        "minority_recall": float(np.mean(minority)) if minority else float("nan"),
        "minority_class_count": len(minority),
    }
    return confusion, rows, summary


def write_diagnostics(feature: str, protocol_id: str, run_dir: Path, truth: np.ndarray, predictions: np.ndarray, input_dir: Path) -> dict[str, float]:
    label_map = json.loads((input_dir / "label_map.json").read_text())
    class_to_local = {str(k): int(v) for k, v in label_map["class_to_local"].items()}
    train_labels = np.load(input_dir / "train_labels.npy", allow_pickle=False)
    confusion, rows, summary = diagnostic_rows(truth, predictions, class_to_local, train_labels)
    for row in rows:
        row.update({"feature": feature, "protocol_id": protocol_id})
    per_class_path = run_dir / "per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    local_to_class = [name for name, _ in sorted(class_to_local.items(), key=lambda pair: pair[1])]
    confusion_dir = CONFUSION_ROOT / feature
    confusion_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = confusion_dir / f"{protocol_id}.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *local_to_class])
        for name, row in zip(local_to_class, confusion):
            writer.writerow([name, *map(int, row)])
    np.save(run_dir / "validation_predictions.npy", predictions.astype(np.int64), allow_pickle=False)
    return {**summary, "per_class_metrics_path": str(per_class_path), "confusion_matrix_path": str(confusion_path)}
