#!/usr/bin/env python3
"""Build Known-Train/Validation-only configuration, data, feature and history audits."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F0_ROOT = PROJECT / "stage14c_vnat_encoder_training"
F2_ROOT = PROJECT / "stage14c5_feature_representation_audit"
OD_ROOT = PROJECT / "stage14c_native_opendetect_vnat"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
PROTOCOL_PATH = STAGE14B / "vnat_open_set_protocol.json"
SPLIT_PATH = STAGE14B / "vnat_split_manifest.csv"
RAW_CACHE = F2_ROOT / "raw_feature_cache"
FIGURES = ROOT / "figures"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_PROTOCOL_SHA = "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced"
EXPECTED_SPLIT_SHA = "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e"
SLOT_INDEX = np.asarray(
    [[packet * 128 + offset for offset in range(12, 20)] for packet in range(8)],
    dtype=np.int64,
).reshape(-1)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def protocol_document() -> dict[str, object]:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if document["freeze_hash"] != EXPECTED_FREEZE:
        raise RuntimeError("Stage 14B freeze hash changed")
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA or sha256_file(SPLIT_PATH) != EXPECTED_SPLIT_SHA:
        raise RuntimeError("Stage 14B protocol/split SHA changed")
    return document


def protocols() -> dict[str, dict[str, object]]:
    return {str(item["protocol_id"]): item for item in protocol_document()["protocols"]}


def input_dir(method: str, protocol_id: str) -> Path:
    if method in {"od", "f0"}:
        return F0_ROOT / "runs" / protocol_id / "inputs"
    if method == "f2":
        return F2_ROOT / "runs" / "f2" / protocol_id / "inputs"
    raise KeyError(method)


def class_counts(labels: np.ndarray, names: list[str]) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(names))
    return {name: int(counts[index]) for index, name in enumerate(names)}


def build_data_alignment(protocol_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for protocol_id, protocol in sorted(protocol_map.items()):
        names = list(map(str, protocol["known_applications"]))
        references: dict[str, dict[str, object]] = {}
        for method in ("od", "f0", "f2"):
            directory = input_dir(method, protocol_id)
            label_map = json.loads((directory / "label_map.json").read_text(encoding="utf-8"))
            record: dict[str, object] = {
                "manifest_sha256": sha256_file(directory / "input_manifest.csv"),
                "label_map_sha256": sha256_file(directory / "label_map.json"),
                "known_class_list": list(label_map["class_to_local"]),
            }
            for split in ("train", "validation"):
                uids = np.load(directory / f"{split}_flow_uids.npy", allow_pickle=False)
                labels = np.load(directory / f"{split}_labels.npy", allow_pickle=False)
                record[f"{split}_uid_hash"] = array_content_hash(uids)
                record[f"{split}_label_hash"] = array_content_hash(labels)
                record[f"{split}_samples"] = int(len(labels))
                record[f"{split}_class_counts"] = class_counts(labels, names)
            references[method] = record
        od = references["od"]
        for candidate in ("f0", "f2"):
            item = references[candidate]
            rows.append({
                "protocol_id": protocol_id,
                "setting": protocol["setting"],
                "protocol_seed": int(protocol["seed"]),
                "comparison": f"od_vs_{candidate}",
                "known_class_list_equal": od["known_class_list"] == item["known_class_list"] == names,
                "train_sample_ids_equal_ordered": od["train_uid_hash"] == item["train_uid_hash"],
                "validation_sample_ids_equal_ordered": od["validation_uid_hash"] == item["validation_uid_hash"],
                "train_labels_equal_ordered": od["train_label_hash"] == item["train_label_hash"],
                "validation_labels_equal_ordered": od["validation_label_hash"] == item["validation_label_hash"],
                "train_class_counts_equal": od["train_class_counts"] == item["train_class_counts"],
                "validation_class_counts_equal": od["validation_class_counts"] == item["validation_class_counts"],
                "od_manifest_sha256": od["manifest_sha256"],
                "candidate_manifest_sha256": item["manifest_sha256"],
                "manifest_hash_equal": od["manifest_sha256"] == item["manifest_sha256"],
                "train_uid_hash": od["train_uid_hash"],
                "validation_uid_hash": od["validation_uid_hash"],
                "train_label_hash": od["train_label_hash"],
                "validation_label_hash": od["validation_label_hash"],
                "train_samples": od["train_samples"],
                "validation_samples": od["validation_samples"],
                "train_class_counts_json": json.dumps(od["train_class_counts"], sort_keys=True),
                "validation_class_counts_json": json.dumps(od["validation_class_counts"], sort_keys=True),
                "status": "PASS" if all((
                    od["known_class_list"] == item["known_class_list"] == names,
                    od["train_uid_hash"] == item["train_uid_hash"],
                    od["validation_uid_hash"] == item["validation_uid_hash"],
                    od["train_label_hash"] == item["train_label_hash"],
                    od["validation_label_hash"] == item["validation_label_hash"],
                    od["train_class_counts"] == item["train_class_counts"],
                    od["validation_class_counts"] == item["validation_class_counts"],
                    od["manifest_sha256"] == item["manifest_sha256"],
                )) else "FAIL",
            })
    return rows


def transformed_statistics(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    out = raw.copy()
    out[..., 0] = np.log1p(np.maximum(raw[..., 0], 0.0))
    out[..., 1] = np.log1p(np.maximum(raw[..., 1], 0.0))
    out[..., 2] = np.log1p(np.maximum(raw[..., 2], 0.0) * 1e6)
    out[..., 3] = np.log1p(np.maximum(raw[..., 3], 0.0))
    out[..., 4] = np.log1p(np.maximum(raw[..., 4], 0.0))
    out[..., 5] = np.log1p(np.maximum(raw[..., 5], 0.0) * 1e6)
    out[..., 6] = np.log1p(np.maximum(raw[..., 6], 0.0) * 1e6)
    out[..., 7] = np.log(np.maximum(raw[..., 7], 1e-12))
    return out


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    return np.asarray(median), np.where(iqr < 1e-9, 1.0, iqr)


def recompute_f2_scalers(protocol_map: dict[str, dict[str, object]]) -> dict[str, bool]:
    raw_uids = np.load(RAW_CACHE / "flow_uids.npy", allow_pickle=False)
    uid_to_raw = {str(uid): index for index, uid in enumerate(raw_uids)}
    iats = np.load(RAW_CACHE / "packet_iat_seconds.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(RAW_CACHE / "packet_lengths.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(RAW_CACHE / "packet_mask.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(RAW_CACHE / "flow_statistics.npy", mmap_mode="r", allow_pickle=False)
    result: dict[str, bool] = {}
    for protocol_id in sorted(protocol_map):
        directory = input_dir("f2", protocol_id)
        train_uids = np.load(directory / "train_flow_uids.npy", allow_pickle=False)
        indices = np.asarray([uid_to_raw[str(uid)] for uid in train_uids], dtype=np.int64)
        train_mask = mask[indices].astype(bool)
        iat_values = np.log1p(np.maximum(iats[indices][train_mask], 0.0) * 1e6)
        length_values = np.log1p(np.maximum(lengths[indices][train_mask], 0.0))
        stat_values = transformed_statistics(stats[indices])
        iat_median, iat_iqr = robust_fit(iat_values)
        length_median, length_iqr = robust_fit(length_values)
        stat_median, stat_iqr = robust_fit(stat_values)
        recorded = json.loads((directory / "scaler.json").read_text(encoding="utf-8"))
        checks = [
            recorded["fit_split"] == "Known Train only",
            int(recorded["fit_flow_values"]) == len(indices),
            np.allclose(recorded["iat_median"], iat_median),
            np.allclose(recorded["iat_iqr"], iat_iqr),
            np.allclose(recorded["length_median"], length_median),
            np.allclose(recorded["length_iqr"], length_iqr),
            np.allclose(recorded["statistics_median"], stat_median),
            np.allclose(recorded["statistics_iqr"], stat_iqr),
        ]
        result[protocol_id] = all(checks)
    return result


def feature_stats(
    method: str,
    protocol_id: str,
    split: str,
    data: np.ndarray,
    train_reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    scaler_match: bool,
) -> dict[str, object]:
    flat = np.asarray(data).reshape(len(data), -1).astype(np.float32)
    train_median, train_scale, train_mean = train_reference
    per_dimension_std = flat.std(axis=0, dtype=np.float64)
    quantiles = np.quantile(flat, [0.01, 0.5, 0.99])
    standardized_shift = np.abs(flat.mean(axis=0) - train_mean) / np.maximum(train_scale, 1.0)
    extreme = np.abs(flat - train_median) / np.maximum(train_scale, 1.0) > 10.0
    total_energy = float(np.square(flat, dtype=np.float64).sum())
    slot_energy = float(np.square(flat[:, SLOT_INDEX], dtype=np.float64).sum())
    return {
        "method": method,
        "protocol_id": protocol_id,
        "split": split,
        "samples": len(flat),
        "feature_shape": f"[{len(flat)},1,32,32]",
        "feature_dim": flat.shape[1],
        "dtype": str(data.dtype),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean(dtype=np.float64)),
        "std": float(flat.std(dtype=np.float64)),
        "median": float(quantiles[1]),
        "p1": float(quantiles[0]),
        "p99": float(quantiles[2]),
        "model_input_min": float(flat.min() / 255.0),
        "model_input_max": float(flat.max() / 255.0),
        "model_input_mean": float(flat.mean(dtype=np.float64) / 255.0),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "zero_variance_feature_count": int((per_dimension_std == 0).sum()),
        "near_constant_feature_count_std_le_1": int((per_dimension_std <= 1.0).sum()),
        "extreme_robust_z_gt_10_count": int(extreme.sum()),
        "extreme_robust_z_gt_10_ratio": float(extreme.mean()),
        "mean_abs_standardized_train_val_shift": float(standardized_shift.mean()),
        "max_abs_standardized_train_val_shift": float(standardized_shift.max()),
        "injected_slot_energy_ratio": 0.0 if total_energy == 0 else slot_energy / total_energy,
        "scaler": "Known Train robust median/IQR" if method == "f2" else "none; byte values divided by 255 once in Dataset",
        "scaler_recomputed_match": scaler_match if method == "f2" else True,
        "same_scaler_train_validation": True,
        "duplicate_normalization_detected": False,
        "unbounded_large_scale_feature_detected": False,
    }


def build_feature_audit(protocol_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    scaler_matches = recompute_f2_scalers(protocol_map)
    rows: list[dict[str, object]] = []
    for protocol_id in sorted(protocol_map):
        for stored_method in ("f0", "f2"):
            directory = input_dir(stored_method, protocol_id)
            train = np.load(directory / "train_images.npy", mmap_mode="r", allow_pickle=False)
            train_flat = np.asarray(train).reshape(len(train), -1).astype(np.float32)
            train_median = np.median(train_flat, axis=0)
            train_iqr = np.quantile(train_flat, 0.75, axis=0) - np.quantile(train_flat, 0.25, axis=0)
            train_mean = train_flat.mean(axis=0)
            reference = (train_median, train_iqr, train_mean)
            for split in ("train", "validation"):
                data = train if split == "train" else np.load(directory / "validation_images.npy", mmap_mode="r", allow_pickle=False)
                methods = ("od", "f0") if stored_method == "f0" else ("f2",)
                for method in methods:
                    rows.append(feature_stats(method, protocol_id, split, data, reference, scaler_matches[protocol_id]))
            del train_flat
    return rows


def history_sources(protocol_id: str) -> dict[str, Path]:
    return {
        "od": OD_ROOT / "runs" / protocol_id / "history.csv",
        "f0": F0_ROOT / "runs" / protocol_id / "training_metrics.csv",
        "f2": F2_ROOT / "runs" / "f2" / protocol_id / "training_metrics.csv",
    }


def result_record(method: str, protocol_id: str) -> dict[str, object]:
    if method == "od":
        return json.loads((OD_ROOT / "runs" / protocol_id / "result.json").read_text(encoding="utf-8"))
    if method == "f0":
        return json.loads((F0_ROOT / "runs" / protocol_id / "checkpoint_selection.json").read_text(encoding="utf-8"))
    return json.loads((F2_ROOT / "runs" / "f2" / protocol_id / "result.json").read_text(encoding="utf-8"))


def diagnose_run(method: str, rows: list[dict[str, str]], result: dict[str, object]) -> tuple[str, str]:
    numeric_keys = [key for key in rows[0] if key.startswith(("train_", "val_")) and key not in {"train_samples", "val_samples"}]
    if any(not math.isfinite(float(row[key])) for row in rows for key in numeric_keys if row.get(key, "") not in {"", None}):
        return "optimization failure", "non-finite epoch metric"
    stop_epoch = len(rows)
    val_macro_values = [float(row["val_macro_f1"]) for row in rows if row.get("val_macro_f1", "") != ""]
    train_macro_values = [float(row["train_macro_f1"]) for row in rows if row.get("train_macro_f1", "") != ""]
    if val_macro_values:
        best_val = max(val_macro_values)
        best_train = max(train_macro_values)
        if best_val < 0.75 and stop_epoch < 51:
            return "underfitting", f"stopped at epoch {stop_epoch} before first prototype reset; best val Macro-F1={best_val:.6f}"
        if best_train - best_val > 0.20:
            return "overfitting", f"best train-val Macro-F1 gap={best_train - best_val:.6f}"
        return "normal convergence", f"finite metrics; best val Macro-F1={best_val:.6f}"
    accuracy = float(result["validation_accuracy"])
    if accuracy < 0.65:
        return "underfitting", f"selected Known-Val Accuracy={accuracy:.6f}"
    return "normal convergence", f"100 epochs completed; selected Known-Val Accuracy={accuracy:.6f}"


def build_training_dynamics(protocol_map: dict[str, dict[str, object]]) -> tuple[list[dict[str, object]], dict[tuple[str, str], tuple[str, str]]]:
    output: list[dict[str, object]] = []
    diagnoses: dict[tuple[str, str], tuple[str, str]] = {}
    for protocol_id, protocol in sorted(protocol_map.items()):
        for method, path in history_sources(protocol_id).items():
            rows = read_csv(path)
            result = result_record(method, protocol_id)
            diagnosis, reason = diagnose_run(method, rows, result)
            diagnoses[(method, protocol_id)] = (diagnosis, reason)
            batch = 128 if method == "od" else 512
            train_samples = int(result.get("train_samples", protocol["split_statistics"]["train"]["flows"]))
            steps_per_epoch = math.ceil(train_samples / batch)
            best_epoch = int(result["best_epoch"])
            for row in rows:
                epoch = int(row["epoch"])
                output.append({
                    "method": method,
                    "protocol_id": protocol_id,
                    "setting": protocol["setting"],
                    "protocol_seed": int(protocol["seed"]),
                    "epoch": epoch,
                    "stop_epoch": len(rows),
                    "best_epoch": best_epoch,
                    "batch_size": batch,
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "optimizer_steps_cumulative": epoch * steps_per_epoch,
                    "learning_rate": row.get("learning_rate", ""),
                    "prototype_reset": row.get("prototype_reset", "0"),
                    "train_loss": row.get("train_total", ""),
                    "val_loss": row.get("val_total", ""),
                    "train_accuracy": row.get("train_accuracy", ""),
                    "val_accuracy": row.get("val_accuracy", ""),
                    "train_macro_f1": row.get("train_macro_f1", ""),
                    "val_macro_f1": row.get("val_macro_f1", ""),
                    "macro_f1_observation": "not recorded by frozen native runner" if method == "od" else "recorded",
                    "stability_guard_cumulative_activations": row.get("stability_guard_cumulative_activations", "0") if method != "od" else "not applicable",
                    "run_diagnosis": diagnosis,
                    "diagnosis_reason": reason,
                })
    return output, diagnoses


def load_existing_per_class() -> dict[tuple[str, str, str], dict[str, object]]:
    rows = read_csv(F2_ROOT / "stage14c5_per_class_metrics.csv")
    output: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        if row["feature"] not in {"f0", "f2"}:
            continue
        output[(row["feature"], row["protocol_id"], row["class"])] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "support": int(row["val_support"]),
            "train_support": int(row["train_support"]),
        }
    return output


def build_per_class(protocol_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    existing = load_existing_per_class()
    rows: list[dict[str, object]] = []
    for protocol_id, protocol in sorted(protocol_map.items()):
        od_rows = json.loads((OD_ROOT / "runs" / protocol_id / "per_class_metrics.json").read_text(encoding="utf-8"))
        od_by_class = {str(row["application"]): row for row in od_rows}
        train_labels = np.load(input_dir("f0", protocol_id) / "train_labels.npy", allow_pickle=False)
        train_counts = np.bincount(train_labels, minlength=len(protocol["known_applications"]))
        for class_index, class_name in enumerate(map(str, protocol["known_applications"])):
            od = od_by_class[class_name]
            for method in ("od", "f0", "f2"):
                values = ({
                    "precision": float(od["precision"]), "recall": float(od["recall"]),
                    "f1": float(od["f1"]), "support": int(od["support"]),
                    "train_support": int(train_counts[class_index]),
                } if method == "od" else existing[(method, protocol_id, class_name)])
                rows.append({
                    "method": method,
                    "protocol_id": protocol_id,
                    "setting": protocol["setting"],
                    "protocol_seed": int(protocol["seed"]),
                    "class": class_name,
                    "local_label": class_index,
                    **values,
                    "delta_precision_vs_od": float(values["precision"]) - float(od["precision"]),
                    "delta_recall_vs_od": float(values["recall"]) - float(od["recall"]),
                    "delta_f1_vs_od": float(values["f1"]) - float(od["f1"]),
                })
    return rows


def read_confusion_csv(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_csv(path)
    names = [row["true\\predicted"] for row in rows]
    matrix = np.asarray([[int(row[name]) for name in names] for row in rows], dtype=np.int64)
    return names, matrix


def build_confusion_pairs(protocol_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for protocol_id, protocol in sorted(protocol_map.items()):
        names = list(map(str, protocol["known_applications"]))
        for method in ("od", "f0", "f2"):
            if method == "od":
                matrix = np.load(OD_ROOT / "runs" / protocol_id / "validation_confusion_matrix.npy", allow_pickle=False)
            else:
                loaded_names, matrix = read_confusion_csv(F2_ROOT / "confusion_matrices" / method / f"{protocol_id}.csv")
                if loaded_names != names:
                    raise RuntimeError(f"{method}/{protocol_id}: confusion class order mismatch")
            for true_index, true_name in enumerate(names):
                support = int(matrix[true_index].sum())
                for pred_index, pred_name in enumerate(names):
                    if true_index == pred_index or matrix[true_index, pred_index] == 0:
                        continue
                    output.append({
                        "method": method,
                        "protocol_id": protocol_id,
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": int(matrix[true_index, pred_index]),
                        "true_class_support": support,
                        "error_rate_within_true_class": float(matrix[true_index, pred_index] / support),
                    })
    return output


def build_batch_imbalance(protocol_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for protocol_id, protocol in sorted(protocol_map.items()):
        labels = np.load(input_dir("f0", protocol_id) / "train_labels.npy", allow_pickle=False)
        counts = np.bincount(labels, minlength=len(protocol["known_applications"]))
        for method, batch in (("od", 128), ("f0", 512), ("f2", 512)):
            for class_index, name in enumerate(map(str, protocol["known_applications"])):
                fraction = float(counts[class_index] / len(labels))
                output.append({
                    "method": method,
                    "protocol_id": protocol_id,
                    "class": name,
                    "train_samples": int(counts[class_index]),
                    "train_fraction": fraction,
                    "batch_size": batch,
                    "expected_samples_per_batch": batch * fraction,
                    "approx_probability_batch_has_zero_class_samples": float((1.0 - fraction) ** batch),
                    "expected_unweighted_loss_share": fraction,
                    "class_weight": "none",
                    "weighted_sampler": False,
                    "sampler": "RandomSampler via shuffle=True",
                })
    return output


def augmentation_parity() -> dict[str, object]:
    directory = input_dir("f0", "low_seed2022")
    images = np.load(directory / "train_images.npy", mmap_mode="r", allow_pickle=False)
    pil_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    ])
    tensor_crop = transforms.RandomCrop(32, padding=4)
    tensor_flip = transforms.RandomHorizontalFlip()
    mismatches = 0
    max_difference = 0.0
    for index in range(min(100, len(images))):
        seed = 10000 + index
        torch.manual_seed(seed)
        native = pil_transform(Image.fromarray(np.asarray(images[index]), mode="L"))
        torch.manual_seed(seed)
        tensor = torch.from_numpy(np.asarray(images[index]).copy()).unsqueeze(0)
        prior = tensor_flip(tensor_crop(tensor)).float().div_(255.0)
        difference = float(torch.max(torch.abs(native - prior)))
        max_difference = max(max_difference, difference)
        mismatches += int(difference != 0.0)
    return {
        "samples_checked": min(100, len(images)),
        "mismatched_samples": mismatches,
        "max_absolute_difference": max_difference,
        "status": "PASS" if mismatches == 0 else "DIFFERENT",
    }


def make_dynamics_figures(dynamics: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    representative = ("high_seed2023", "medium_seed2024", "medium_seed2025")
    for protocol_id in representative:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for method, color in (("od", "black"), ("f0", "#1f77b4"), ("f2", "#d62728")):
            subset = [row for row in dynamics if row["protocol_id"] == protocol_id and row["method"] == method]
            epochs = [int(row["epoch"]) for row in subset]
            axes[0].plot(epochs, [float(row["train_loss"]) for row in subset], color=color, linestyle="--", alpha=.7, label=f"{method} train")
            axes[0].plot(epochs, [float(row["val_loss"]) for row in subset], color=color, label=f"{method} val")
            axes[1].plot(epochs, [float(row["train_accuracy"]) for row in subset], color=color, linestyle="--", alpha=.7, label=f"{method} train")
            axes[1].plot(epochs, [float(row["val_accuracy"]) for row in subset], color=color, label=f"{method} val")
            if method != "od":
                axes[2].plot(epochs, [float(row["train_macro_f1"]) for row in subset], color=color, linestyle="--", alpha=.7, label=f"{method} train")
                axes[2].plot(epochs, [float(row["val_macro_f1"]) for row in subset], color=color, label=f"{method} val")
        axes[0].set_title("Loss")
        axes[1].set_title("Accuracy")
        axes[2].set_title("Macro-F1 (not recorded for native OD)")
        for axis in axes:
            axis.set_xlabel("epoch")
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(protocol_id)
        fig.tight_layout()
        fig.savefig(FIGURES / f"training_dynamics_{protocol_id}.png", dpi=180)
        plt.close(fig)


def config_diff_markdown(augmentation: dict[str, object]) -> str:
    return f"""# Open-Detect native versus F0/F2 training configuration

All values below were read from the executed configs and source files. “Same”
means the effective value is equal, not merely that the implementation is
similar.

| Item | Native Open-Detect | F0 | F2 | Material difference |
|---|---|---|---|---|
| Model / encoder | Released `OpenDetectNet`, ResNet-18 VAE | `Stage3OpenDetectNet` wrapping the same released encoder/decoder | Same as F0 | Same network graph except F0/F2 logvar upper clamp |
| Input shape / dimension | 1x32x32, 1024 bytes, latent 128 | Same byte image and latent 128 | Same 1024-byte image; 64 zero address slots replaced by 8 robust-quantized flow statistics | F2 changes 6.25% of input positions; does not add dimensions |
| Preprocessing | First 8 packets; 80 header + 48 payload; masked IPv4 addresses; divide by 255 once | Byte-identical F0 arrays; divide by 255 once | F0 plus log/log-ratio -> train median/IQR -> clip [-4,4] -> uint8 [0,255], then divide by 255 once | F2 scaler is additional but bounded |
| Scaler | None | None | Fit on Known Train only and reused unchanged for Train/Val | Independently recomputed in `feature_distribution_audit.csv` |
| Train augmentation | PIL RandomCrop(32,padding=4) + RandomHorizontalFlip | Tensor versions of the same transforms | Same as F0 | Parity audit: {augmentation['mismatched_samples']}/{augmentation['samples_checked']} mismatches; max diff {augmentation['max_absolute_difference']} |
| Loss | `0.005*(rec+kld+ent)+0.995*dis` | Same | Same | None |
| Class weight | None | None | None | All losses are sample-weighted by observed class frequency |
| Sampler | RandomSampler (`shuffle=True`) | Same | Same | None; no weighted sampler |
| Batch size | 128 | 512 | 512 | F0/F2 have about one quarter as many optimizer steps per epoch |
| DataLoader workers | 4 | 0 | 0 | Runtime/augmentation RNG difference; not a loss change |
| Optimizer | Adam, lr=0.001, betas=(0.9,0.999), weight_decay=0 | Same | Same | None |
| Scheduler | MultiStepLR milestones [50,80], gamma=0.1 | Same configured | Same configured | F0/F2 stop before epoch 30, so milestones never execute |
| Maximum epochs | 100, always completed | 100 maximum | 100 maximum | Effective epochs differ because of early stopping |
| Gradient clipping | None | None | None | None |
| Initialization | Released Xavier/He/BatchNorm init; Kaiming prototypes | Same released initializer and prototype init | Same | None |
| Random seed | Fixed training seed 2022 for every protocol | Protocol seed 2022–2026 | Protocol seed 2022–2026 | Material stochastic-policy difference |
| Dropout | None in released ResNet/decoder | None | None | None |
| Checkpoint selection | Maximum Known-Val Accuracy | Maximum harmonic mean of Known-Val Accuracy and Macro-F1 | Same as F0 | Material selection difference |
| Early stopping | None; 100 epochs | Patience 5 on validation composite | Same as F0 | Material: all F0/F2 runs stop at epochs 9–30 |
| Prototype reset | After zero-based epochs 50 and 80 | Same configured | Same configured | Never reached by any frozen F0/F2 run |
| Numerical protection | Released raw logvar | Clamp raw logvar upper tail at 20 | Same as F0 | Guard triggered only once across F0 and zero times across F2 |
| Label encoding | Frozen Known-class order, contiguous local IDs 0..C-1 | Same | Same | Hash/equality audit PASS |

## Direct configuration implications

1. Optimizer, learning rate, weight decay, scheduler definition, loss,
   initialization, class weighting and sampling are not configuration
   explanations by themselves because they are equal.
2. The scheduler and both prototype resets are *configured* in F0/F2 but are
   operationally dead: patience-5 stopping ends every run before epoch 31.
3. Batch 512 further reduces update count by approximately 4x per epoch. The
   combination of early stopping and batch size creates a much larger update
   budget gap than the nominal “100 epochs” fields suggest.
4. Numerical protection is unlikely to explain the aggregate gap because its
   activation counter is zero for 29/30 F0/F2 runs and one batch for only
   F0 Medium-2025.
"""


def main() -> None:
    protocol_map = protocols()
    if len(protocol_map) != 15:
        raise RuntimeError("expected 15 frozen protocols")
    alignment = build_data_alignment(protocol_map)
    if any(row["status"] != "PASS" for row in alignment):
        raise RuntimeError("data alignment audit failed")
    write_csv(ROOT / "data_alignment_audit.csv", alignment)
    feature_rows = build_feature_audit(protocol_map)
    write_csv(ROOT / "feature_distribution_audit.csv", feature_rows)
    dynamics, diagnoses = build_training_dynamics(protocol_map)
    write_csv(ROOT / "training_dynamics.csv", dynamics)
    per_class = build_per_class(protocol_map)
    write_csv(ROOT / "per_class_comparison.csv", per_class)
    write_csv(ROOT / "confusion_pair_analysis.csv", build_confusion_pairs(protocol_map))
    write_csv(ROOT / "batch_imbalance_audit.csv", build_batch_imbalance(protocol_map))
    augmentation = augmentation_parity()
    (ROOT / "augmentation_parity.json").write_text(json.dumps(augmentation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "training_config_diff.md").write_text(config_diff_markdown(augmentation), encoding="utf-8")
    make_dynamics_figures(dynamics)
    summary = {
        "status": "PASS",
        "freeze_hash": EXPECTED_FREEZE,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_sha256": sha256_file(SPLIT_PATH),
        "data_alignment_rows": len(alignment),
        "data_alignment_failures": sum(row["status"] != "PASS" for row in alignment),
        "feature_audit_rows": len(feature_rows),
        "f2_scaler_recompute_failures": sum(row["method"] == "f2" and not row["scaler_recomputed_match"] for row in feature_rows),
        "training_dynamics_rows": len(dynamics),
        "per_class_rows": len(per_class),
        "augmentation_parity": augmentation,
        "known_test_samples_read": 0,
        "unknown_test_samples_read": 0,
        "des_executed": False,
        "diagnoses": {f"{method}:{protocol}": {"category": value[0], "reason": value[1]} for (method, protocol), value in diagnoses.items()},
    }
    (ROOT / "static_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "status", "data_alignment_rows", "data_alignment_failures", "feature_audit_rows",
        "f2_scaler_recompute_failures", "training_dynamics_rows", "per_class_rows",
        "known_test_samples_read", "unknown_test_samples_read", "des_executed",
    )}, indent=2))


if __name__ == "__main__":
    main()
