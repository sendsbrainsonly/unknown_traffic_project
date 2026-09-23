#!/usr/bin/env python3
"""Deterministically evaluate original and cleaned F2 checkpoints on Known Train/Val."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F2_ROOT = PROJECT / "stage14c5_feature_representation_audit"
F2_SCRIPTS = F2_ROOT / "scripts"
PROTOCOLS = ("medium_seed2025", "medium_seed2026")

sys.path.insert(0, str(F2_SCRIPTS))
from model_utils import KnownDataset, make_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(confusion: np.ndarray) -> tuple[float, float, float]:
    supports = confusion.sum(axis=1)
    f1s: list[float] = []
    for class_id in range(len(confusion)):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else float(2 * tp / denominator))
    return (
        float(np.trace(confusion) / confusion.sum()),
        float(np.mean(f1s)),
        float(np.average(np.asarray(f1s), weights=supports)),
    )


@torch.no_grad()
def evaluate(model, input_dir: Path, split: str, device: torch.device) -> dict[str, float]:
    dataset = KnownDataset(input_dir, split, augment=False)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    totals = {name: 0.0 for name in ("loss", "rec", "kld", "ent", "dis")}
    confusion = np.zeros((model.n_classes, model.n_classes), dtype=np.int64)
    count = 0
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)
        _, predictions, losses = model.loss(images, labels_gpu)
        total = 0.005 * (losses["rec"] + losses["kld"] + losses["ent"]) + 0.995 * losses["dis"]
        if not torch.isfinite(total):
            raise RuntimeError("non-finite deterministic evaluation loss")
        batch = len(labels)
        totals["loss"] += float(total) * batch
        for name in ("rec", "kld", "ent", "dis"):
            totals[name] += float(losses[name]) * batch
        np.add.at(confusion, (labels.numpy(), predictions.detach().cpu().numpy()), 1)
        count += batch
    accuracy, macro_f1, weighted_f1 = classification_metrics(confusion)
    return {
        **{name: value / count for name, value in totals.items()},
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "samples": count,
    }


def checkpoint_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for protocol_id in PROTOCOLS:
        original_result = json.loads((F2_ROOT / "runs" / "f2" / protocol_id / "result.json").read_text(encoding="utf-8"))
        records.append({
            "protocol_id": protocol_id,
            "version": "original_f2",
            "seed_mode": "protocol",
            "training_seed": int(protocol_id[-4:]),
            "checkpoint_path": Path(original_result["checkpoint_path"]),
            "expected_sha256": original_result["checkpoint_sha256"],
        })
        for seed_mode in ("protocol", "fixed2022"):
            run_id = f"{protocol_id}_{seed_mode}"
            result = json.loads((ROOT / "runs" / run_id / "result.json").read_text(encoding="utf-8"))
            records.append({
                "protocol_id": protocol_id,
                "version": "cleaned_primary" if seed_mode == "protocol" else "cleaned_fixed2022_control",
                "seed_mode": seed_mode,
                "training_seed": result["training_seed"],
                "checkpoint_path": Path(result["checkpoint_path"]),
                "expected_sha256": result["checkpoint_sha256"],
            })
    return records


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    for record in checkpoint_records():
        checkpoint_path = record["checkpoint_path"]
        if sha256_file(checkpoint_path) != record["expected_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
        protocol_id = str(record["protocol_id"])
        input_dir = F2_ROOT / "runs" / "f2" / protocol_id / "inputs"
        label_map = json.loads((input_dir / "label_map.json").read_text(encoding="utf-8"))
        model = make_model(len(label_map["class_to_local"]), device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        for split in ("train", "validation"):
            metrics = evaluate(model, input_dir, split, device)
            rows.append({
                "protocol_id": protocol_id,
                "version": record["version"],
                "seed_mode": record["seed_mode"],
                "training_seed": record["training_seed"],
                "split": split,
                **metrics,
                "checkpoint_sha256": record["expected_sha256"],
                "known_test_samples_used": 0,
                "unknown_test_samples_used": 0,
            })
        del model, payload
        torch.cuda.empty_cache()
    path = ROOT / "deterministic_split_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "rows": len(rows), "known_test_samples_used": 0, "unknown_test_samples_used": 0}))


if __name__ == "__main__":
    main()
