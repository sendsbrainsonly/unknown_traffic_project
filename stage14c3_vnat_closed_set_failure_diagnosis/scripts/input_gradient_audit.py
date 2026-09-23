#!/usr/bin/env python3
"""Measure input-gradient attribution on stratified Known-Train samples only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F0_ROOT = PROJECT / "stage14c_vnat_encoder_training"
F2_ROOT = PROJECT / "stage14c5_feature_representation_audit"
OD_ROOT = PROJECT / "stage14c_native_opendetect_vnat"
UPSTREAM = PROJECT.parent / "Open-Detect" / "code"
PROTOCOL_PATH = PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_open_set_protocol.json"
PROTOCOLS = ("medium_seed2025", "medium_seed2026")
SLOTS = np.asarray(
    [[packet * 128 + offset for offset in range(12, 20)] for packet in range(8)],
    dtype=np.int64,
).reshape(-1)

AUDIT_ROOT = PROJECT / "opendetect_ustc_encoder_audit"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(PROJECT / "stage3_unknown_utility" / "scripts"))
from adapters.opendetect_model import AuditedOpenDetectNet  # noqa: E402
from stage3_model import Stage3OpenDetectNet  # noqa: E402


def input_dir(method: str, protocol_id: str) -> Path:
    if method in {"od", "f0"}:
        return F0_ROOT / "runs" / protocol_id / "inputs"
    return F2_ROOT / "runs" / "f2" / protocol_id / "inputs"


def checkpoint(method: str, protocol_id: str) -> Path:
    if method == "od":
        return OD_ROOT / "runs" / protocol_id / "best_checkpoint.pt"
    if method == "f0":
        return F0_ROOT / "checkpoints" / f"{protocol_id}_best.pt"
    return F2_ROOT / "checkpoints" / f"f2_{protocol_id}_best.pt"


def stratified_indices(labels: np.ndarray, per_class: int = 32) -> np.ndarray:
    parts = [np.flatnonzero(labels == label)[:per_class] for label in sorted(map(int, np.unique(labels)))]
    return np.concatenate(parts)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_map = {item["protocol_id"]: item for item in document["protocols"]}
    rows: list[dict[str, object]] = []
    for protocol_id in PROTOCOLS:
        num_classes = len(protocol_map[protocol_id]["known_applications"])
        for method in ("od", "f0", "f2"):
            directory = input_dir(method, protocol_id)
            images = np.load(directory / "train_images.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(directory / "train_labels.npy", allow_pickle=False)
            indices = stratified_indices(labels)
            x = torch.from_numpy(np.asarray(images[indices]).copy()).unsqueeze(1).float().div_(255.0).to(device)
            x.requires_grad_(True)
            y = torch.from_numpy(labels[indices].astype(np.int64)).to(device)
            model_class = AuditedOpenDetectNet if method == "od" else Stage3OpenDetectNet
            model = model_class(
                upstream_code=UPSTREAM, channels=1, latent_dim=128,
                num_classes=num_classes, temp_inter=1.0, temp_intra=1.0,
            ).to(device)
            payload = torch.load(checkpoint(method, protocol_id), map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.train()
            _, _, losses = model.loss(x, y)
            total = 0.005 * (losses["rec"] + losses["kld"] + losses["ent"]) + 0.995 * losses["dis"]
            total.backward()
            flat_x = x.detach().reshape(len(x), -1)
            flat_grad = x.grad.detach().reshape(len(x), -1)
            nonslots = np.setdiff1d(np.arange(1024), SLOTS)
            gradient_abs = flat_grad.abs()
            attribution_abs = (flat_grad * flat_x).abs()
            input_energy = flat_x.square()
            rows.append({
                "method": method,
                "protocol_id": protocol_id,
                "known_train_samples": len(indices),
                "samples_per_class_cap": 32,
                "loss": float(total.detach()),
                "slot_dimension_fraction": len(SLOTS) / 1024.0,
                "slot_input_energy_fraction": float(input_energy[:, SLOTS].sum() / input_energy.sum()),
                "slot_absolute_gradient_fraction": float(gradient_abs[:, SLOTS].sum() / gradient_abs.sum()),
                "slot_gradient_times_input_fraction": float(attribution_abs[:, SLOTS].sum() / attribution_abs.sum()),
                "slot_mean_absolute_gradient": float(gradient_abs[:, SLOTS].mean()),
                "nonslot_mean_absolute_gradient": float(gradient_abs[:, nonslots].mean()),
                "slot_to_nonslot_mean_gradient_ratio": float(gradient_abs[:, SLOTS].mean() / gradient_abs[:, nonslots].mean()),
                "unknown_samples_read": 0,
                "known_test_samples_read": 0,
            })
            del model, x, y
            torch.cuda.empty_cache()
    write_csv(ROOT / "input_gradient_audit.csv", rows)
    print(json.dumps({"status": "PASS", "rows": len(rows), "unknown_samples_read": 0, "known_test_samples_read": 0}))


if __name__ == "__main__":
    main()

