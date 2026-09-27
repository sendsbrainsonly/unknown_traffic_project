#!/usr/bin/env python3
"""Stage20-matched closed-set Open-Detect corrected-paper training.

Uses the frozen Stage16S training recipe. Only this project's Stage23 directory
is writable; the Open-Detect and Stage16S implementations are imported read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import optim
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
STAGE16S = PROJECT / "stage16s_service_open_set_benchmark"
MANIFEST = STAGE20 / "closed_service_manifest.csv"
CONFIG = STAGE16S / "training_configs.json"
STAGE22_CONFIG = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "config.json"
sys.path.insert(0, str(STAGE16S / "scripts"))
from train_evaluate_run import TrafficImages, extract, import_open_detect  # noqa: E402
from stage16s_common import seed_everything  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_role(dataset: str, role: str, services: list[str], source_cache: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["dataset"] == dataset and r["closed_role"] == role]
    rows.sort(key=lambda r: r["flow_id"])
    if not rows or len({r["flow_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"empty or duplicate flow IDs: {dataset}/{role}")
    if set(r["service_label"] for r in rows) != set(services):
        raise RuntimeError(f"class coverage mismatch: {dataset}/{role}")
    images = []
    labels = []
    for row in rows:
        source = (row["source_pool"], row["source_role"])
        if source not in source_cache:
            source_file = Path(source[0]) / f"pool_{source[1]}.npz"
            with np.load(source_file, allow_pickle=False) as archive:
                source_cache[source] = archive["data"]
        image = source_cache[source][int(row["source_index"])]
        if image.shape != (32, 32) or image.dtype != np.uint8:
            raise RuntimeError(f"image shape/dtype mismatch: {row['flow_id']}")
        if hashlib.sha256(image.tobytes()).hexdigest() != row["image_sha256"]:
            raise RuntimeError(f"image content mismatch: {row['flow_id']}")
        images.append(image)
        labels.append(services.index(row["service_label"]))
    return np.stack(images), np.asarray(labels, dtype=np.int64), rows


def evaluate_labels(labels: np.ndarray, predictions: np.ndarray, services: list[str]) -> tuple[dict, list[dict]]:
    ids = np.arange(len(services))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=ids, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
    }
    per_class = [
        {"service": name, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(services)
    ]
    return metrics, per_class


def execute(dataset: str, seed: int, preflight_only: bool) -> None:
    frozen = read_json(STAGE22_CONFIG)
    if dataset not in frozen["datasets"] or seed not in frozen["seeds"]:
        raise ValueError("dataset/seed not in frozen Stage22 paired scope")
    if read_json(OUT / "preflight_verification.json")["status"] != "PASS":
        raise RuntimeError("Stage23 sample parity preflight has not passed")
    services = frozen["datasets"][dataset]["services"]
    manifest_hash = sha256_file(MANIFEST)
    if manifest_hash != read_json(OUT / "preflight_verification.json")["stage20_manifest_sha256"]:
        raise RuntimeError("Stage20 frozen manifest changed")
    training = read_json(CONFIG)["training"]
    source_cache: dict = {}
    train_images, train_labels, train_rows = load_role(dataset, "known_train", services, source_cache)
    val_images, val_labels, val_rows = load_role(dataset, "known_validation", services, source_cache)
    expected = frozen["datasets"][dataset]
    if len(train_rows) != expected["expected_train"] or len(val_rows) != expected["expected_validation"]:
        raise RuntimeError("frozen Known Train/Validation counts changed")
    preflight = {
        "dataset": dataset, "seed": seed, "status": "PASS", "stage20_manifest_sha256": manifest_hash,
        "training_config_sha256": sha256_file(CONFIG),
        "known_train": len(train_rows), "known_validation": len(val_rows),
        "train_image_shape": list(train_images.shape), "val_image_shape": list(val_images.shape),
        "train_class_counts": dict(Counter(r["service_label"] for r in train_rows)),
        "validation_class_counts": dict(Counter(r["service_label"] for r in val_rows)),
        "known_test_feature_values_loaded": 0, "unknown_test_feature_values_loaded": 0,
    }
    preflight_path = OUT / f"opendetect_input_preflight_{dataset}_seed{seed}_{'dryrun' if preflight_only else 'formal'}.json"
    if preflight_path.exists():
        raise FileExistsError(f"preserve existing input preflight: {preflight_path}")
    write_json(preflight_path, preflight)
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    if preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback for formal training")

    run_dir = OUT / "runs" / "opendetect_corrected_paper" / dataset / f"seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"run directory exists; preserve and never overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.time()
    try:
        seed_everything(seed)
        device = torch.device("cuda:0")
        train_loader = DataLoader(
            TrafficImages(train_images, train_labels, True), batch_size=int(training["batch_size"]),
            shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            TrafficImages(val_images, val_labels, False), batch_size=int(training["eval_batch_size"]),
            shuffle=False, num_workers=0, pin_memory=True,
        )
        Network, reset_prototypes, run_epoch, weight_init = import_open_detect()
        model = Network(
            training["architecture"], int(training["channels"]), int(training["latent_dim"]),
            len(services), 1, 1,
        ).to(device)
        model.apply(weight_init)
        optimizer = optim.Adam(model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]))
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(map(int, training["scheduler_milestones"])),
            gamma=float(training["scheduler_gamma"]),
        )
        write_json(run_dir / "config.json", {
            "method": "Open-Detect corrected-paper", "dataset": dataset, "seed": seed,
            "services": services, "training": training, "stage20_manifest_sha256": manifest_hash,
            "source_training_config_sha256": sha256_file(CONFIG),
            "selection": "Known Validation accuracy only", "test_selection_samples": 0,
            "unknown_training_samples": 0, "unknown_validation_samples": 0,
        })
        best_accuracy, best_epoch = -1.0, -1
        patience_reference, without_improvement = -1.0, 0
        with (run_dir / "training_history.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(int(training["epochs"])):
                train_metric = run_epoch(model, train_loader, device, float(training["lambda"]), optimizer)
                reset = epoch in set(training["prototype_reset_zero_based_epochs"])
                if reset:
                    reset_prototypes(model, train_loader, device)
                val_metric = run_epoch(model, val_loader, device, float(training["lambda"]), None)
                scheduler.step()
                if val_metric["accuracy"] > best_accuracy:
                    best_accuracy, best_epoch = float(val_metric["accuracy"]), epoch + 1
                    torch.save({
                        "model_state_dict": model.state_dict(), "epoch": best_epoch,
                        "validation_accuracy": best_accuracy, "services": services, "seed": seed,
                        "training_config_sha256": sha256_file(CONFIG),
                    }, run_dir / "model_best.pt")
                if val_metric["accuracy"] > patience_reference + float(training["early_stop_min_delta"]):
                    patience_reference, without_improvement = float(val_metric["accuracy"]), 0
                else:
                    without_improvement += 1
                if reset:
                    patience_reference, without_improvement = float(val_metric["accuracy"]), 0
                event = {
                    "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                    "train": train_metric, "validation": val_metric,
                    "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                    "prototype_reset": reset, "epochs_without_improvement": without_improvement,
                }
                log.write(json.dumps(event) + "\n")
                log.flush()
                print(f"{dataset} seed={seed} epoch={epoch + 1} val_acc={val_metric['accuracy']:.6f} best={best_accuracy:.6f}", flush=True)
                if epoch + 1 >= int(training["early_stop_min_epoch"]) and without_improvement >= int(training["early_stop_patience"]):
                    break
        checkpoint = torch.load(run_dir / "model_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_output = extract(model, {"data": val_images, "labels": val_labels}, device, int(training["eval_batch_size"]))
        val_metrics, val_per_class = evaluate_labels(val_labels, val_output["native_prediction"], services)
        if abs(val_metrics["accuracy"] - best_accuracy) > 1e-12:
            raise RuntimeError("best-checkpoint Known Validation accuracy parity failed")

        # Load Known Test image values only after the model and checkpoint are fixed.
        test_images, test_labels, test_rows = load_role(dataset, "known_test", services, source_cache)
        if len(test_rows) != expected["expected_test"]:
            raise RuntimeError("frozen Known Test count changed")
        test_output = extract(model, {"data": test_images, "labels": test_labels}, device, int(training["eval_batch_size"]))
        test_metrics, test_per_class = evaluate_labels(test_labels, test_output["native_prediction"], services)
        with (run_dir / "predictions.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", "role", "method", "flow_id", "true_service", "predicted_service", "correct"])
            writer.writeheader()
            for role, rows, labels, predictions in (
                ("known_validation", val_rows, val_labels, val_output["native_prediction"]),
                ("known_test", test_rows, test_labels, test_output["native_prediction"]),
            ):
                for row, label, prediction in zip(rows, labels, predictions):
                    writer.writerow({
                        "dataset": dataset, "seed": seed, "role": role, "method": "Open-Detect corrected-paper",
                        "flow_id": row["flow_id"], "true_service": services[int(label)],
                        "predicted_service": services[int(prediction)], "correct": int(label == prediction),
                    })
        np.savez_compressed(
            run_dir / "representations.npz",
            validation_flow_ids=np.asarray([r["flow_id"] for r in val_rows]),
            validation_mu=val_output["mu"], test_flow_ids=np.asarray([r["flow_id"] for r in test_rows]),
            test_mu=test_output["mu"],
        )
        write_json(run_dir / "metrics.json", {
            "dataset": dataset, "seed": seed, "method": "Open-Detect corrected-paper",
            "best_epoch": best_epoch, "completed_epochs": epoch + 1,
            "validation": val_metrics, "test": test_metrics,
            "validation_per_class": val_per_class, "test_per_class": test_per_class,
            "stage20_manifest_sha256": manifest_hash,
            "checkpoint_sha256": sha256_file(run_dir / "model_best.pt"),
            "predictions_sha256": sha256_file(run_dir / "predictions.csv"),
            "runtime_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "test_selection_samples": 0, "unknown_training_samples": 0,
            "unknown_validation_samples": 0,
        })
        if sha256_file(MANIFEST) != manifest_hash:
            raise RuntimeError("Stage20 frozen manifest changed during training")
        (run_dir / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
        print(json.dumps({"status": "SUCCESS", "run_dir": str(run_dir), "test": test_metrics}), flush=True)
    except BaseException as exc:
        write_json(run_dir / "FAILURE.json", {
            "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "preserve_and_do_not_overwrite": True,
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    execute(args.dataset, args.seed, args.preflight_only)
