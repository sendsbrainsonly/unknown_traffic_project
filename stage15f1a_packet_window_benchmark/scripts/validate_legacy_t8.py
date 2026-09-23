#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import ROOT, STAGE15R, pilot_specs, protocol_rows, read_json, sha256_file, write_json
from model import LegacySequenceCNN
from run_window_pilot import SequenceDataset, load_role, metrics, normalize


def legacy_dir(dataset: str, protocol_id: str) -> Path:
    return STAGE15R / "pilot_runs" / "e2" / dataset / protocol_id


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out_root = ROOT / "legacy_t8_parity"
    summaries = []
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        old = legacy_dir(dataset, protocol_id)
        old_result = read_json(old / "result.json")
        classes, train_rows, val_rows = protocol_rows(dataset, protocol_id)
        train_x, train_y, _ = load_role(dataset, train_rows, 8)
        val_x, val_y, val_ids = load_role(dataset, val_rows, 8)
        train_x, val_x, normalization = normalize(train_x, val_x)
        normalization_match = normalization == old_result["normalization"]

        checkpoint = Path(old_result["checkpoint_path"])
        if sha256_file(checkpoint) != old_result["checkpoint_sha256"]:
            raise RuntimeError(f"legacy checkpoint hash mismatch: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model = LegacySequenceCNN(len(classes)).to(device)
        model.load_state_dict(saved["model_state_dict"])
        loader = DataLoader(SequenceDataset(val_x, val_y), batch_size=256, shuffle=False)
        result = metrics(model, loader, device, nn.CrossEntropyLoss(), len(classes))

        confusion_reference = np.load(old / "validation_confusion_matrix.npy", allow_pickle=False)
        confusion_match = np.array_equal(result["confusion"], confusion_reference)
        metric_match = close(result["loss"], old_result["validation_loss"], 1e-6) and all(
            close(result[key], old_result[f"validation_{key}"], 1e-12)
            for key in ("accuracy", "macro_f1", "weighted_f1")
        )
        status = "PASS" if normalization_match and confusion_match and metric_match else "FAIL"
        dest = out_root / dataset / protocol_id
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest / "validation_predictions.npz",
            sample_ids=val_ids,
            true_labels=result["truth"],
            predicted_labels=result["pred"],
            logits=result["logits"].astype(np.float32),
        )
        payload = {
            "status": status,
            "dataset": dataset,
            "protocol_id": protocol_id,
            "known_train_samples": int(len(train_y)),
            "known_validation_samples": int(len(val_y)),
            "known_test_samples_used": 0,
            "unknown_test_samples_used": 0,
            "normalization_exact_match": normalization_match,
            "confusion_exact_match": confusion_match,
            "classification_metrics_within_1e_12_and_loss_within_1e_6": metric_match,
            "recomputed": {
                "validation_loss": float(result["loss"]),
                "validation_accuracy": float(result["accuracy"]),
                "validation_macro_f1": float(result["macro_f1"]),
                "validation_weighted_f1": float(result["weighted_f1"]),
            },
            "reference": {
                key: old_result[key]
                for key in (
                    "validation_loss",
                    "validation_accuracy",
                    "validation_macro_f1",
                    "validation_weighted_f1",
                )
            },
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": old_result["checkpoint_sha256"],
            "parity_device": str(device),
        }
        write_json(dest / "parity_result.json", payload)
        summaries.append(payload)

    write_json(out_root / "parity_summary.json", {
        "status": "PASS" if all(x["status"] == "PASS" for x in summaries) else "FAIL",
        "runs": summaries,
    })
    if any(x["status"] != "PASS" for x in summaries):
        raise SystemExit("legacy T8 parity failed")
    print(json.dumps({"status": "PASS", "runs": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
