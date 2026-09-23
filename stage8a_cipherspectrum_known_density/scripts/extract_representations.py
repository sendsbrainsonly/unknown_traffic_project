#!/usr/bin/env python3
"""Extract deterministic Stage 7 mu_x for Known Train/Validation only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from common import (
    AccessLedger,
    CONFIG_PATH,
    EXPECTED_STAGE7,
    PROJECT_ROOT,
    STAGE8_ROOT,
    VALID_SETTINGS,
    load_fold,
    require,
    sha256_file,
    stage7_run,
    verify_all_provenance,
    write_csv,
    write_json,
)


AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
STAGE3_SCRIPTS = PROJECT_ROOT / "stage3_unknown_utility/scripts"
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(STAGE3_SCRIPTS))
from stage3_model import Stage3OpenDetectNet  # noqa: E402


class FrozenImageDataset(Dataset):
    def __init__(self, images: np.ndarray) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.asarray(self.images[index]).copy()
        return torch.from_numpy(image).unsqueeze(0).float().div_(255.0)


def read_source_manifest(path: Path, ledger: AccessLedger) -> dict[str, list[dict[str, str]]]:
    ledger.record(path, "csv", "KNOWN_TRAIN_AND_VALIDATION_METADATA")
    rows = {"KNOWN_TRAIN": [], "KNOWN_VALIDATION": []}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["role"]
            require(role in rows, f"forbidden/unexpected role in Stage 7 input manifest: {role}")
            require(row["mode"] == "formal", "non-formal Stage 7 input row")
            require(row["input_valid"] == "True", f"invalid Stage 7 input row: {row['sample_id']}")
            rows[role].append(row)
    return rows


def output_manifest(rows: list[dict[str, str]], path: Path) -> None:
    result = [
        {
            "sample_id": row["sample_id"],
            "pcap_path": row["pcap_path"],
            "class_name": row["class_name"],
            "cipher_source": row["cipher_source"],
            "split_group_id": row["split_group_id"],
            "row_index": index,
            "local_class_id": int(row["local_class_id"]),
            "source_role": row["role"],
        }
        for index, row in enumerate(rows)
    ]
    write_csv(path, result)


@torch.inference_mode()
def extract_split(
    model: Stage3OpenDetectNet,
    images: np.ndarray,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    save_native: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    loader = DataLoader(
        FrozenImageDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    mu = open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(images), 128))
    native_score = np.empty(len(images), dtype=np.float32) if save_native else None
    native_pred = np.empty(len(images), dtype=np.int64) if save_native else None
    model.eval()
    offset = 0
    for batch_index, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        batch_mu, logvar, _ = model.stable_encode(batch)
        require(batch_mu.shape[1] == 128, "checkpoint latent dimension changed")
        count = len(batch)
        mu[offset : offset + count] = batch_mu.cpu().numpy().astype(np.float32, copy=False)
        if save_native:
            kld = model.kl_div_to_prototypes(batch_mu, logvar)
            minimum, prediction = torch.min(kld, dim=1)
            native_score[offset : offset + count] = -minimum.cpu().numpy().astype(np.float32, copy=False)
            native_pred[offset : offset + count] = prediction.cpu().numpy()
        offset += count
        if (batch_index + 1) % 25 == 0:
            print(json.dumps({"event": "mu_progress", "rows": offset, "total": len(images)}), flush=True)
    require(offset == len(images), "mu extraction row count mismatch")
    mu.flush()
    return mu, native_score, native_pred


def audit_split(values: np.ndarray, labels: np.ndarray, class_names: list[str], threshold: float) -> dict[str, object]:
    finite = np.isfinite(values)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    require(bool(finite.all()), "non-finite deterministic mu_x")
    variances = np.var(values, axis=0, dtype=np.float64)
    all_zero = np.flatnonzero(np.all(values == 0, axis=0)).astype(int).tolist()
    near_zero = np.flatnonzero(variances <= threshold).astype(int).tolist()
    unique_rows = int(len(np.unique(np.asarray(values), axis=0)))
    per_class: list[dict[str, object]] = []
    for local_id, name in enumerate(class_names):
        subset = np.asarray(values[labels == local_id], dtype=np.float64)
        require(len(subset) > 0, f"empty representation class: {name}")
        centroid = subset.mean(axis=0)
        per_class.append(
            {
                "class_name": name,
                "support": len(subset),
                "centroid_norm": float(np.linalg.norm(centroid)),
                "within_class_variance": float(np.mean(np.var(subset, axis=0))),
            }
        )
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "all_zero_dimension_count": len(all_zero),
        "all_zero_dimensions": all_zero,
        "near_zero_variance_threshold": threshold,
        "near_zero_variance_dimension_count": len(near_zero),
        "near_zero_variance_dimensions": near_zero,
        "unique_rows": unique_rows,
        "duplicate_rows": len(values) - unique_rows,
        "collapse_detected": len(all_zero) > 0 or len(near_zero) == values.shape[1],
        "per_class": per_class,
    }


def write_classification(
    setting: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    selection: dict[str, object],
) -> dict[str, object]:
    ids = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=ids, zero_division=0
    )
    rows = [
        {
            "setting": setting,
            "class_name": name,
            "local_class_id": local_id,
            "support": int(support[local_id]),
            "accuracy": float(recall[local_id]),
            "precision": float(precision[local_id]),
            "recall": float(recall[local_id]),
            "f1": float(f1[local_id]),
        }
        for local_id, name in enumerate(class_names)
    ]
    write_csv(output_dir / "known_val_per_class_metrics.csv", rows)
    matrix = confusion_matrix(labels, predictions, labels=ids)
    confusion_rows = [
        {"true_class": name, **{class_names[column]: int(matrix[row, column]) for column in ids}}
        for row, name in enumerate(class_names)
    ]
    write_csv(output_dir / "known_val_confusion_matrix.csv", confusion_rows)
    accuracy = float(np.mean(labels == predictions))
    macro_f1 = float(np.mean(f1))
    require(abs(accuracy - float(selection["val_accuracy"])) <= 1e-12, "frozen checkpoint validation accuracy did not reproduce")
    require(abs(macro_f1 - float(selection["val_macro_f1"])) <= 1e-12, "frozen checkpoint validation macro-F1 did not reproduce")
    return {
        "setting": setting,
        "validation_samples": len(labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "stage7_accuracy": float(selection["val_accuracy"]),
        "stage7_macro_f1": float(selection["val_macro_f1"]),
        "parity": "PASS",
        "hardest_five_by_f1": sorted(rows, key=lambda row: (row["f1"], row["accuracy"], row["class_name"]))[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.workers >= 0, "invalid loader configuration")
    provenance_path = STAGE8_ROOT / "outputs/summary/provenance_verification.json"
    require(provenance_path.is_file(), "provenance gate must run before mu extraction")
    require(json.loads(provenance_path.read_text(encoding="utf-8"))["status"] == "PASS", "provenance gate failed")
    verify_all_provenance()

    setting = args.setting
    fold = load_fold(setting)
    class_names = [str(value) for value in fold["known_classes"]]
    run_dir = stage7_run(setting)
    input_dir = run_dir / "inputs"
    artifact_dir = STAGE8_ROOT / "artifacts" / setting
    output_dir = STAGE8_ROOT / "outputs" / setting
    require(not artifact_dir.exists() and not output_dir.exists(), f"{setting}: refusing to overwrite existing Stage 8A setting")
    artifact_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    ledger = AccessLedger(setting, output_dir / "file_access_audit.json")

    source_manifest = input_dir / "input_manifest.csv"
    rows = read_source_manifest(source_manifest, ledger)
    expected_counts = {
        "KNOWN_TRAIN": int(fold["known_train_count"]),
        "KNOWN_VALIDATION": int(fold["known_validation_count"]),
    }
    for role, expected in expected_counts.items():
        require(len(rows[role]) == expected, f"{setting}/{role}: source manifest count mismatch")
        require([int(row["array_index"]) for row in rows[role]] == list(range(expected)), f"{setting}/{role}: array indices are not contiguous")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for role, prefix in (("KNOWN_TRAIN", "train"), ("KNOWN_VALIDATION", "validation")):
        image_path = input_dir / f"{prefix}_images.npy"
        label_path = input_dir / f"{prefix}_labels.npy"
        sample_path = input_dir / f"{prefix}_sample_ids.npy"
        ledger.record(image_path, "npy", role, expected_counts[role])
        ledger.record(label_path, "npy", role, expected_counts[role])
        ledger.record(sample_path, "npy", role, expected_counts[role])
        arrays[prefix] = {
            "images": np.load(image_path, mmap_mode="r", allow_pickle=False),
            "labels": np.load(label_path, mmap_mode="r", allow_pickle=False),
            "sample_ids": np.load(sample_path, mmap_mode="r", allow_pickle=False),
        }
        require(len(arrays[prefix]["images"]) == expected_counts[role], f"{setting}/{role}: image count mismatch")
        require(len(arrays[prefix]["labels"]) == expected_counts[role], f"{setting}/{role}: label count mismatch")
        manifest_ids = np.asarray([row["sample_id"] for row in rows[role]])
        require(np.array_equal(np.asarray(arrays[prefix]["sample_ids"]), manifest_ids), f"{setting}/{role}: sample ID order mismatch")
        manifest_labels = np.asarray([int(row["local_class_id"]) for row in rows[role]], dtype=np.int64)
        require(np.array_equal(np.asarray(arrays[prefix]["labels"]), manifest_labels), f"{setting}/{role}: label order mismatch")

    checkpoint_path = run_dir / "artifacts/training_best_checkpoint.pt"
    selection_path = run_dir / "training_checkpoint_selection.json"
    ledger.record(checkpoint_path, "pt", "FROZEN_STAGE7_CHECKPOINT")
    ledger.record(selection_path, "json", "FROZEN_STAGE7_METADATA")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(sha256_file(checkpoint_path) == EXPECTED_STAGE7[setting]["checkpoint_sha256"], "best checkpoint hash changed")
    require(int(selection["best_epoch"]) == EXPECTED_STAGE7[setting]["best_epoch"], "best checkpoint epoch changed")
    require(torch.cuda.is_available(), "deterministic mu_x extraction requires CUDA")
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    require(int(checkpoint["epoch"]) == EXPECTED_STAGE7[setting]["best_epoch"], "loaded checkpoint is not the formal best epoch")
    require(int(config["latent_dimension"]) == 128, "latent dimension changed")
    require(int(config["num_classes"]) == len(class_names), "checkpoint class count changed")
    model = Stage3OpenDetectNet(
        upstream_code=UPSTREAM_CODE,
        channels=1,
        latent_dim=128,
        num_classes=len(class_names),
        temp_inter=1.0,
        temp_intra=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    mu_train, _, _ = extract_split(
        model, arrays["train"]["images"], artifact_dir / "mu_train.npy", device,
        args.batch_size, args.workers, save_native=False,
    )
    mu_val, native_scores, native_predictions = extract_split(
        model, arrays["validation"]["images"], artifact_dir / "mu_val.npy", device,
        args.batch_size, args.workers, save_native=True,
    )
    require(native_scores is not None and native_predictions is not None, "Native validation outputs missing")
    np.save(artifact_dir / "native_val_known_scores.npy", native_scores, allow_pickle=False)
    np.save(artifact_dir / "native_val_predictions.npy", native_predictions, allow_pickle=False)
    output_manifest(rows["KNOWN_TRAIN"], artifact_dir / "train_manifest.csv")
    output_manifest(rows["KNOWN_VALIDATION"], artifact_dir / "val_manifest.csv")

    threshold = float(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["representation_audit"]["near_zero_variance_threshold"])
    representation = {
        "setting": setting,
        "representation": "deterministic mu_x",
        "latent_dimension": 128,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": EXPECTED_STAGE7[setting]["checkpoint_sha256"],
        "checkpoint_best_epoch": EXPECTED_STAGE7[setting]["best_epoch"],
        "train": audit_split(mu_train, np.asarray(arrays["train"]["labels"]), class_names, threshold),
        "validation": audit_split(mu_val, np.asarray(arrays["validation"]["labels"]), class_names, threshold),
    }
    write_json(output_dir / "representation_audit.json", representation)
    classification = write_classification(
        setting,
        np.asarray(arrays["validation"]["labels"]),
        native_predictions,
        class_names,
        output_dir,
        selection,
    )
    write_json(output_dir / "known_val_classification_summary.json", classification)
    extraction = {
        "setting": setting,
        "status": "PASS",
        "representation": "deterministic mu_x from stable_encode; sampled z not used",
        "latent_dimension": 128,
        "train_samples": len(mu_train),
        "validation_samples": len(mu_val),
        "checkpoint_best_epoch": EXPECTED_STAGE7[setting]["best_epoch"],
        "checkpoint_sha256": EXPECTED_STAGE7[setting]["checkpoint_sha256"],
        "mu_train_sha256": sha256_file(artifact_dir / "mu_train.npy"),
        "mu_val_sha256": sha256_file(artifact_dir / "mu_val.npy"),
        "native_score_definition": "negative minimum KL divergence to frozen prototypes",
        "native_validation_parity": "PASS",
        "stability_guard_activations": model.stability_guard_activations,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "unknown_inference_executed": False,
    }
    write_json(output_dir / "extraction_metadata.json", extraction)
    ledger.save()
    print(json.dumps(extraction, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
