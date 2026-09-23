#!/usr/bin/env python3
"""Run one preregistered Stage 15R E1 or E2 Known-only pilot.

This module deliberately has no Test-data loader.  Every input path resolved
below is a frozen Known Train or Known Validation artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing.util
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import (
    CONFIG_PATH, PROJECT_ROOT, ROOT, STAGE12_ROOT, STAGE14C5_ROOT,
    STAGE14C_INPUT_ROOT, STAGE3_ROOT, USTC_AUDIT_ROOT, read_csv, read_json,
    seed_everything, sha256_file, write_csv, write_json,
)


class ImageDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, train: bool) -> None:
        self.data = data
        self.labels = labels.astype(np.int64, copy=False)
        self.transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            if train else [transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.transform(Image.fromarray(self.data[index], mode="L")), int(self.labels[index])


class SequenceDataset(Dataset):
    def __init__(self, values: np.ndarray, labels: np.ndarray) -> None:
        self.values = torch.from_numpy(values.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.values[index], self.labels[index]


class SequenceCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(3, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 2:3, :]
        hidden = self.block1(x) * mask
        hidden = self.block2(hidden) * mask
        pooled = hidden.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.head(pooled)


def _load_iscx_images(dataset: str, setting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base = STAGE12_ROOT / "artifacts" / dataset / "protocol" / setting
    protocol = read_json(base / "protocol.json")
    label_map = read_json(STAGE12_ROOT / "protocol" / dataset / "canonical_label_map.json")["canonical_class_to_label"]
    known = list(protocol["known_classes"])
    mapping = {int(label_map[name]): index for index, name in enumerate(known)}
    arrays = []
    for name in ("known_train.npz", "known_validation.npz"):
        bundle = np.load(base / name, allow_pickle=False)
        data = bundle["data"]
        original = bundle["target"].astype(np.int64, copy=False)
        if not set(map(int, np.unique(original))) <= set(mapping):
            raise RuntimeError(f"Unknown label entered {name}")
        labels = np.asarray([mapping[int(value)] for value in original], dtype=np.int64)
        arrays.append((data, labels))
    return arrays[0][0], arrays[0][1], arrays[1][0], arrays[1][1], known


def _load_vnat_images(protocol_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base = STAGE14C_INPUT_ROOT / "runs" / protocol_id / "inputs"
    label_map = read_json(base / "label_map.json")
    known = [label_map["local_to_class"][str(i)] for i in range(len(label_map["local_to_class"]))]
    return (
        np.load(base / "train_images.npy", mmap_mode="r", allow_pickle=False),
        np.load(base / "train_labels.npy", mmap_mode="r", allow_pickle=False),
        np.load(base / "validation_images.npy", mmap_mode="r", allow_pickle=False),
        np.load(base / "validation_labels.npy", mmap_mode="r", allow_pickle=False),
        known,
    )


def _ustc_known() -> tuple[list[str], dict[str, int]]:
    config = read_json(STAGE3_ROOT / "outputs" / "A-2" / "training_config.json")
    known = list(config["known_classes"])
    class_to_id = read_json(PROJECT_ROOT / "data" / "trafficformer_input" / "compatible_min1" / "label_map.json")["class_to_id"]
    return known, {name: int(class_to_id[name]) for name in known}


def _load_ustc_images() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    known, original_ids = _ustc_known()
    local = {value: index for index, value in enumerate(original_ids.values())}
    outputs = []
    for role in ("train", "val"):
        data = np.load(USTC_AUDIT_ROOT / "artifacts" / f"{role}_images.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(PROJECT_ROOT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{role}.npy", mmap_mode="r", allow_pickle=False)
        keep = np.isin(labels, list(local))
        positions = np.flatnonzero(keep)
        mapped = np.asarray([local[int(value)] for value in labels[positions]], dtype=np.int64)
        outputs.append((data[positions], mapped))
    return outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1], known


def load_images(dataset: str, protocol_id: str, setting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if dataset.startswith("iscx_"):
        return _load_iscx_images(dataset, setting)
    if dataset == "vnat":
        return _load_vnat_images(protocol_id)
    if dataset == "ustc":
        return _load_ustc_images()
    raise KeyError(dataset)


def _cache_arrays(cache: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uids = np.load(cache / "flow_uids.npy", allow_pickle=False)
    index = {str(uid): offset for offset, uid in enumerate(uids)}
    return (
        index,
        np.load(cache / "packet_lengths.npy", mmap_mode="r", allow_pickle=False),
        np.load(cache / "packet_iat_seconds.npy", mmap_mode="r", allow_pickle=False),
        np.load(cache / "packet_directions.npy", mmap_mode="r", allow_pickle=False),
        np.load(cache / "packet_mask.npy", mmap_mode="r", allow_pickle=False),
    )


def _materialize_sequence(indices: list[int], labels: list[int], arrays) -> tuple[np.ndarray, np.ndarray]:
    _, lengths, iats, directions, mask = arrays
    pos = np.asarray(indices, dtype=np.int64)
    valid = mask[pos].astype(np.float32)
    signed_length = np.sign(directions[pos]).astype(np.float32) * np.log1p(lengths[pos].astype(np.float32))
    log_iat = np.log1p(iats[pos].astype(np.float32) * 1_000_000.0)
    values = np.stack((signed_length, log_iat, valid), axis=1)
    return values, np.asarray(labels, dtype=np.int64)


def load_sequences(dataset: str, protocol_id: str, setting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if dataset.startswith("iscx_"):
        cache = ROOT / "feature_cache" / dataset
        arrays = _cache_arrays(cache)
        index = arrays[0]
        protocol = read_json(STAGE12_ROOT / "artifacts" / dataset / "protocol" / setting / "protocol.json")
        known = list(protocol["known_classes"])
        local = {name: i for i, name in enumerate(known)}
        manifest = [row for row in read_csv(STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv") if row["setting"] == setting]
        outputs = []
        for role in ("known_train", "known_validation"):
            selected = [row for row in manifest if row["role"] == role]
            if any(row["canonical_class"] not in local for row in selected):
                raise RuntimeError("Unknown class entered sequence input")
            outputs.append(_materialize_sequence([index[row["flow_id_sha256"]] for row in selected], [local[row["canonical_class"]] for row in selected], arrays))
        return outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1], known
    if dataset == "vnat":
        arrays = _cache_arrays(STAGE14C5_ROOT / "raw_feature_cache")
        index = arrays[0]
        base = STAGE14C_INPUT_ROOT / "runs" / protocol_id / "inputs"
        label_map = read_json(base / "label_map.json")
        known = [label_map["local_to_class"][str(i)] for i in range(len(label_map["local_to_class"]))]
        manifest = read_csv(base / "input_manifest.csv")
        outputs = []
        for role in ("train", "validation"):
            selected = [row for row in manifest if row["split"] == role]
            outputs.append(_materialize_sequence([index[row["flow_uid"]] for row in selected], [int(row["local_label"]) for row in selected], arrays))
        return outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1], known
    if dataset == "ustc":
        arrays = _cache_arrays(ROOT / "feature_cache" / "ustc")
        index = arrays[0]
        known, original_ids = _ustc_known()
        local = {value: i for i, value in enumerate(original_ids.values())}
        outputs = []
        for role in ("train", "val"):
            ids = np.load(PROJECT_ROOT / "outputs" / "stage1" / "modelA" / "embeddings" / f"flow_ids_{role}.npy", allow_pickle=False)
            labels = np.load(PROJECT_ROOT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{role}.npy", allow_pickle=False)
            keep = np.isin(labels, list(local))
            selected_ids = ids[keep]
            selected_labels = labels[keep]
            outputs.append(_materialize_sequence([index[str(uid)] for uid in selected_ids], [local[int(value)] for value in selected_labels], arrays))
        return outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1], known
    raise KeyError(dataset)


def normalize_sequences(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    result_train = train.copy()
    result_val = val.copy()
    mask = train[:, 2, :] > 0
    audit = {"fit_role": "Known Train only", "channels": {}}
    for channel, name in ((0, "signed_log1p_length"), (1, "log1p_iat_microseconds")):
        values = train[:, channel, :][mask]
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = float(max(q3 - q1, 1e-6))
        for output in (result_train, result_val):
            output[:, channel, :] = np.clip((output[:, channel, :] - median) / iqr, -4.0, 4.0) * output[:, 2, :]
        audit["channels"][name] = {"median": median, "iqr": iqr}
    return result_train, result_val, audit


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, n_classes: int):
    model.eval()
    losses, labels_out, predictions = [], [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            losses.append(float(criterion(logits, labels).item()) * len(labels))
            labels_out.append(labels.cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
    labels_np = np.concatenate(labels_out)
    pred_np = np.concatenate(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(labels_np, pred_np, labels=np.arange(n_classes), zero_division=0)
    weighted = float(np.average(f1, weights=support))
    return {
        "loss": float(sum(losses) / len(labels_np)), "accuracy": float(accuracy_score(labels_np, pred_np)),
        "macro_f1": float(f1.mean()), "weighted_f1": weighted,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "confusion": confusion_matrix(labels_np, pred_np, labels=np.arange(n_classes)),
    }


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    total_loss = total = correct = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
        correct += int((logits.argmax(1) == labels).sum().item())
    return float(total_loss / total), float(correct / total)


def install_project_local_short_tmp(output: Path) -> tuple[str, int]:
    """Give multiprocessing a short AF_UNIX path while bytes remain local."""
    actual = output / "runtime_tmp"
    actual.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(actual, os.O_RDONLY | os.O_DIRECTORY)
    os.set_inheritable(directory_fd, True)
    alias = f"/proc/self/fd/{directory_fd}"
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = alias
    tempfile.tempdir = None
    multiprocessing.util._tempdir = None
    return alias, directory_fd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("E1", "E2"), required=True)
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    allowed = {(row["dataset"], row["protocol_id"], row["setting"]) for row in config["pilot_protocols"]}
    if (args.dataset, args.protocol_id, args.setting) not in allowed:
        raise RuntimeError("pilot protocol is not preregistered")
    output = ROOT / "pilot_runs" / args.experiment.lower() / args.dataset / args.protocol_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preserved run: {output}")
    output.mkdir(parents=True)
    temp_alias, temp_directory_fd = install_project_local_short_tmp(output)
    started = time.time()
    seed = int(config["training"]["training_seed"])
    seed_everything(seed)
    if args.experiment == "E1":
        train_x, train_y, val_x, val_y, class_names = load_images(args.dataset, args.protocol_id, args.setting)
        train_set, val_set = ImageDataset(train_x, train_y, True), ImageDataset(val_x, val_y, False)
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        normalization = {"type": "ToTensor uint8/255", "fit_role": "none"}
    else:
        train_x, train_y, val_x, val_y, class_names = load_sequences(args.dataset, args.protocol_id, args.setting)
        train_x, val_x, normalization = normalize_sequences(train_x, val_x)
        train_set, val_set = SequenceDataset(train_x, train_y), SequenceDataset(val_x, val_y)
        model = SequenceCNN(len(class_names))
    if set(np.unique(train_y)) != set(range(len(class_names))) or set(np.unique(val_y)) != set(range(len(class_names))):
        raise RuntimeError("Known Train/Validation class coverage mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device(args.device)
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": 4, "pin_memory": True, "persistent_workers": False}
    train_loader = DataLoader(train_set, batch_size=int(training["batch_size"]), shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, batch_size=int(training["eval_batch_size"]), shuffle=False, **common)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(training["scheduler_milestones"]), gamma=float(training["scheduler_gamma"]))
    history = []
    best_accuracy, best_epoch = -1.0, -1
    checkpoint = output / "model_best.pt"
    for epoch in range(1, int(training["epochs"]) + 1):
        lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy = train_epoch(model, train_loader, device, criterion, optimizer)
        metrics = evaluate(model, val_loader, device, criterion, len(class_names))
        scheduler.step()
        improved = metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = metrics["accuracy"], epoch
            torch.save({"model_state_dict": model.state_dict(), "experiment": args.experiment, "dataset": args.dataset, "protocol_id": args.protocol_id, "class_names": class_names, "epoch": epoch}, checkpoint)
        row = {"epoch": epoch, "learning_rate": lr, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"], "validation_weighted_f1": metrics["weighted_f1"], "is_best": int(improved), "elapsed_seconds": time.time() - started}
        history.append(row)
        write_csv(output / "history.csv", history)
        print(json.dumps({"experiment": args.experiment, "dataset": args.dataset, "protocol": args.protocol_id, **row}), flush=True)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    final = evaluate(model, val_loader, device, criterion, len(class_names))
    np.save(output / "validation_confusion_matrix.npy", final["confusion"], allow_pickle=False)
    per_class = [{"experiment": args.experiment, "dataset": args.dataset, "protocol_id": args.protocol_id, "class_name": name, "precision": float(final["precision"][i]), "recall": float(final["recall"][i]), "f1": float(final["f1"][i]), "support": int(final["support"][i])} for i, name in enumerate(class_names)]
    write_csv(output / "per_class_results.csv", per_class)
    result = {
        "status": "PASS", "experiment": args.experiment, "dataset": args.dataset, "protocol_id": args.protocol_id,
        "setting": args.setting, "known_train_samples": len(train_set), "known_validation_samples": len(val_set),
        "known_test_samples_used": 0, "unknown_test_samples_used": 0, "class_names": class_names,
        "epochs_completed": int(training["epochs"]), "best_epoch": best_epoch,
        "validation_loss": final["loss"], "validation_accuracy": final["accuracy"], "validation_macro_f1": final["macro_f1"], "validation_weighted_f1": final["weighted_f1"],
        "runtime_seconds": time.time() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "normalization": normalization, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "data_loader_workers": 4, "project_local_tmp_alias": temp_alias,
    }
    write_json(output / "result.json", result)
    write_json(output / "run_config.json", {"stage15r_config": config, "invocation": vars(args), "strict_unknown_free": True, "visible_roles": ["known_train", "known_validation"], "project_local_tmp_alias": temp_alias, "project_local_tmp_directory_fd": temp_directory_fd})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
