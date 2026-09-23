#!/usr/bin/env python3
"""Diagnose the preserved Streaming-2023 checkpoint parity failure.

This script is read-only with respect to the failed run.  It writes a separate
diagnostic record and never trains or updates model parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage16s_common import CONFIG, OUT, RUNS, known_classes, load_role, read_json, write_json
from train_evaluate_run import TrafficImages, import_open_detect


def main() -> int:
    protocol_id = "loso_streaming"
    seed = 2023
    run_dir = RUNS / protocol_id / f"seed{seed}"
    output = OUT / "diagnostics" / "streaming_seed2023_parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    config = read_json(CONFIG)
    training = config["training"]
    role = load_role(protocol_id, "known_validation")
    classes = known_classes(protocol_id)
    loader = DataLoader(
        TrafficImages(role["data"], role["labels"], False),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        pin_memory=True,
        persistent_workers=False,
    )

    CorrectedOpenDetectNet, _, run_epoch, _ = import_open_detect()
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        training["architecture"], int(training["channels"]),
        int(training["latent_dim"]), len(classes), 1, 1,
    ).to(device)
    checkpoint = torch.load(run_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch_metrics = run_epoch(model, loader, device, float(training["lambda"]), None)
    loader_predictions: list[np.ndarray] = []
    loader_margins: list[np.ndarray] = []
    loader_inputs: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            _, class_kl, _, _ = model(images)
            ordered = torch.sort(class_kl, dim=1).values
            loader_predictions.append(class_kl.argmin(dim=1).cpu().numpy())
            loader_margins.append((ordered[:, 1] - ordered[:, 0]).cpu().numpy())
            loader_inputs.append(images.cpu().numpy())

    manual_predictions: list[np.ndarray] = []
    manual_margins: list[np.ndarray] = []
    manual_inputs: list[np.ndarray] = []
    batch_size = int(training["eval_batch_size"])
    with torch.no_grad():
        for start in range(0, len(role["data"]), batch_size):
            images = torch.from_numpy(role["data"][start : start + batch_size].copy()).unsqueeze(1)
            images = images.to(device=device, dtype=torch.float32).div_(255.0)
            mu, _, _ = model.encoder(images)
            class_kl = 0.5 * model.distance(mu, model.prototypes)
            ordered = torch.sort(class_kl, dim=1).values
            manual_predictions.append(class_kl.argmin(dim=1).cpu().numpy())
            manual_margins.append((ordered[:, 1] - ordered[:, 0]).cpu().numpy())
            manual_inputs.append(images.cpu().numpy())

    loader_prediction = np.concatenate(loader_predictions)
    manual_prediction = np.concatenate(manual_predictions)
    loader_margin = np.concatenate(loader_margins)
    manual_margin = np.concatenate(manual_margins)
    loader_input = np.concatenate(loader_inputs)
    manual_input = np.concatenate(manual_inputs)
    labels = role["labels"]
    mismatch = np.flatnonzero(loader_prediction != manual_prediction)

    record = {
        "status": "DIAGNOSTIC_ONLY_NO_TRAINING",
        "protocol_id": protocol_id,
        "seed": seed,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_recorded_validation_accuracy": float(checkpoint["validation_accuracy"]),
        "run_epoch_validation_accuracy": float(epoch_metrics["accuracy"]),
        "loader_prediction_accuracy": float(np.mean(loader_prediction == labels)),
        "manual_prediction_accuracy": float(np.mean(manual_prediction == labels)),
        "sample_count": int(len(labels)),
        "prediction_mismatch_count": int(len(mismatch)),
        "prediction_mismatch_indices": mismatch.tolist(),
        "max_abs_input_difference": float(np.max(np.abs(loader_input - manual_input))),
        "loader_margin_min": float(loader_margin.min()),
        "manual_margin_min": float(manual_margin.min()),
        "mismatch_loader_margins": loader_margin[mismatch].astype(float).tolist(),
        "mismatch_manual_margins": manual_margin[mismatch].astype(float).tolist(),
        "mismatch_loader_predictions": loader_prediction[mismatch].astype(int).tolist(),
        "mismatch_manual_predictions": manual_prediction[mismatch].astype(int).tolist(),
        "mismatch_labels": labels[mismatch].astype(int).tolist(),
        "unknown_samples_used": 0,
        "known_test_samples_used": 0,
    }
    write_json(output, record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
