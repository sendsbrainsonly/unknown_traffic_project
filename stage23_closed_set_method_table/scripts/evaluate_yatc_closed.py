#!/usr/bin/env python3
"""Post-selection Known Test evaluation for frozen Stage23 YaTC runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_yatc_closed import (
    CACHE, FROZEN, MANIFEST, OUT, WEIGHT, WEIGHT_HASH, MFRDataset,
    TraFormer_YaTC, evaluate, read_json, sha, write_json,
)


def main(dataset: str, seed: int, tag: str) -> None:
    run = OUT / "runs" / "yatc_stage20" / dataset / f"seed{seed}_formal"
    if not (run / "SUCCESS").is_file() or (run / "FAILURE.json").exists():
        raise RuntimeError("YaTC formal training not successful")
    metrics = read_json(run / "metrics.json")
    config = read_json(run / "config.json")
    checkpoint = run / "model_best.pt"
    if tag and not all(char.isalnum() or char in "_-" for char in tag):
        raise ValueError("invalid evaluation tag")
    suffix = f"_{tag}" if tag else ""
    output = run / f"test_predictions{suffix}.csv"
    evaluation_path = run / f"test_evaluation{suffix}.json"
    if output.exists() or evaluation_path.exists():
        raise FileExistsError("Known Test evaluation already exists; preserve prior result")
    if sha(MANIFEST) != FROZEN or sha(WEIGHT) != WEIGHT_HASH:
        raise RuntimeError("protected Stage20/YaTC input hash changed")
    if metrics["checkpoint_sha256"] != sha(checkpoint):
        raise RuntimeError("frozen YaTC checkpoint hash mismatch")
    if config["dataset"] != dataset or config["seed"] != seed:
        raise RuntimeError("run identity mismatch")
    if metrics["completed_epochs"] != 200 or metrics["known_test_features_loaded"] != 0:
        raise RuntimeError("Test-read-before-selection or incomplete training")
    cache = read_json(CACHE / dataset / "cache_audit_test.json")
    if cache["status"] != "PASS" or cache["stage20_manifest_sha256"] != FROZEN:
        raise RuntimeError("post-selection Known Test MFR cache not verified")
    services = config["services"]
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = TraFormer_YaTC(num_classes=len(services), drop_path_rate=0.1).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved["services"] != services or saved["epoch"] != metrics["best_epoch"]:
        raise RuntimeError("checkpoint identity/best epoch mismatch")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    validation = MFRDataset(dataset, "known_validation", services)
    val_metrics, _, _ = evaluate(model, DataLoader(validation, batch_size=64, shuffle=False), device, services)
    if any(abs(val_metrics[name] - metrics["validation"][name]) > 1e-12
           for name in ("accuracy", "macro_f1", "weighted_f1")):
        raise RuntimeError(
            f"Known Validation checkpoint replay mismatch on {device}: "
            f"observed={val_metrics}, expected={metrics['validation']}"
        )
    test = MFRDataset(dataset, "known_test", services)
    test_metrics, per_class, predictions = evaluate(
        model, DataLoader(test, batch_size=64, shuffle=False), device, services
    )
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "seed", "role", "method", "flow_id", "true_service",
            "predicted_service", "correct",
        ])
        writer.writeheader()
        for row, predicted in zip(test.rows, predictions):
            label = services[predicted]
            writer.writerow({
                "dataset": dataset, "seed": seed, "role": "known_test",
                "method": "YaTC-official-pretrained-Stage20", "flow_id": row["flow_id"],
                "true_service": row["service_label"], "predicted_service": label,
                "correct": int(row["service_label"] == label),
            })
    result = {
        "status": "PASS", "dataset": dataset, "seed": seed, "method": "YaTC-official-pretrained-Stage20",
        "evaluation_tag": tag, "device": str(device),
        "test": test_metrics, "test_per_class": per_class,
        "test_samples": len(test), "checkpoint_sha256_before": metrics["checkpoint_sha256"],
        "checkpoint_sha256_after": sha(checkpoint), "predictions_sha256": sha(output),
        "stage20_manifest_sha256": sha(MANIFEST), "test_mfr_sha256": cache["roles"]["known_test"]["mfr_sha256"],
        "selection": "Known Validation weighted F1; frozen before test MFR generation",
        "test_selection_samples": 0, "unknown_samples": 0,
    }
    if result["stage20_manifest_sha256"] != FROZEN or result["checkpoint_sha256_after"] != result["checkpoint_sha256_before"]:
        raise RuntimeError("protected hash changed during evaluation")
    write_json(evaluation_path, result)
    print(json.dumps({"status": "PASS", "dataset": dataset, "seed": seed,
                      "test": {key: test_metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--evaluation-tag", default="")
    cli = parser.parse_args()
    main(cli.dataset, cli.seed, cli.evaluation_tag)
