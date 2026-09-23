#!/usr/bin/env python3
"""Train and evaluate recovered RoNeTC on one frozen protocol."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing.util
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import RONETC, ROOT, cache_dir, config, quantile_higher, read_csv, seed_everything, sha256_file, write_csv, write_json

sys.path.insert(0, str(RONETC))
from model import MobileVitClassifier, ce_loss  # noqa: E402


def install_project_local_short_tmp(output: Path) -> tuple[str, int]:
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


class FrozenByteDataset(Dataset):
    def __init__(self, views_path: Path, indices: np.ndarray, labels: np.ndarray, flow_ids: list[str]):
        self.views = np.load(views_path, mmap_mode="r", allow_pickle=False)
        self.indices = indices.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.flow_ids = flow_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        return torch.from_numpy(np.asarray(self.views[self.indices[index]], dtype=np.int64)), int(self.labels[index]), self.flow_ids[index]


def dense_infer(model: MobileVitClassifier, values: torch.Tensor):
    """Vectorized equivalent of recovered model.forward input assembly."""
    if values.ndim != 4 or values.shape[1] != 3 or values.shape[2] != model.packet_num:
        raise ValueError(f"invalid dense input {tuple(values.shape)}")
    embedded = model.embedding_layer(values.long())
    multi_view = []
    for view in range(3):
        channels = []
        for start in range(0, model.packet_num, 4):
            packets = embedded[:, view, start:start + 4]
            top = torch.cat((packets[:, 0], packets[:, 1]), dim=2)
            bottom = torch.cat((packets[:, 2], packets[:, 3]), dim=2)
            channels.append(torch.cat((top, bottom), dim=1))
        multi_view.append(torch.stack(channels, dim=1))
    evidence = model.infer(multi_view)
    alpha = {view: evidence[view] + 1 for view in range(3)}
    alpha_fused, uncertainty = model.DS_Combin(alpha)
    return evidence, alpha, alpha_fused, uncertainty


def dense_loss(model: MobileVitClassifier, values: torch.Tensor, labels: torch.Tensor, epoch: int, device: torch.device):
    evidence, alpha, alpha_fused, uncertainty = dense_infer(model, values)
    loss = sum(ce_loss(labels, alpha[view], model.classes, epoch, model.lambda_epochs, device) for view in range(3))
    loss = loss + ce_loss(labels, alpha_fused, model.classes, epoch, model.lambda_epochs, device)
    return evidence, alpha_fused - 1, uncertainty, torch.mean(loss)


class DenseRoNeTCAdapter(nn.Module):
    """Tensor-input wrapper whose outputs can be gathered by DataParallel.

    The recovered author model accepts a Python dictionary and internally moves
    each packet tensor to one fixed device.  This adapter preserves the exact
    embedding/view/DS/loss computation while exposing dense tensors so a frozen
    global batch can be split across GPUs for the much larger USTC protocol.
    """

    def __init__(self, core: MobileVitClassifier):
        super().__init__()
        self.core = core

    def forward(self, values: torch.Tensor, labels: torch.Tensor | None = None, epoch: int = 1):
        evidence, alpha, alpha_fused, uncertainty = dense_infer(self.core, values)
        fused = alpha_fused - 1
        if labels is None:
            return fused, uncertainty
        loss_terms = sum(
            ce_loss(labels, alpha[view], self.core.classes, epoch, self.core.lambda_epochs, values.device)
            for view in range(3)
        )
        loss_terms = loss_terms + ce_loss(
            labels, alpha_fused, self.core.classes, epoch, self.core.lambda_epochs, values.device
        )
        return fused, uncertainty, loss_terms


def parity_check(model: MobileVitClassifier, device: torch.device, packet_num: int, width: int) -> dict:
    model.eval()
    generator = torch.Generator().manual_seed(7)
    values = torch.randint(0, 257, (2, 3, packet_num, width), generator=generator, dtype=torch.long).to(device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    dictionary = {}
    flat = values.reshape(2, 3 * packet_num, width)
    for index in range(3 * packet_num):
        dictionary[index] = flat[:, index]
    with torch.no_grad():
        original = model(dictionary, labels, 1, device)
        dense = dense_loss(model, values, labels, 1, device)
    differences = [float(torch.max(torch.abs(original[i] - dense[i])).item()) if torch.is_tensor(original[i]) else 0.0 for i in range(1, 4)]
    if max(differences) > 1e-6:
        raise RuntimeError(f"dense adapter parity failed: {differences}")
    return {"status": "PASS", "max_abs_differences_evidence_uncertainty_loss": differences}


def protocol_rows(dataset: str, protocol_id: str):
    rows = [row for row in read_csv(ROOT / "protocol_manifest.csv") if row["dataset"] == dataset and row["protocol_id"] == protocol_id]
    if not rows:
        raise RuntimeError("protocol rows not found")
    return rows


def loaders(dataset: str, protocol_id: str, batch_size: int, eval_batch_size: int, workers: int, seed: int):
    rows = protocol_rows(dataset, protocol_id)
    flow_ids = np.load(cache_dir(dataset) / "flow_uids.npy", allow_pickle=False)
    index = {str(uid): offset for offset, uid in enumerate(flow_ids)}
    output = {}
    for role in ("known_train", "known_validation", "known_test", "unknown_test"):
        selected = [row for row in rows if row["role"] == role]
        indices = np.asarray([index[row["flow_uid"]] for row in selected], dtype=np.int64)
        labels = np.asarray([int(row["local_label"]) for row in selected], dtype=np.int64)
        dataset_obj = FrozenByteDataset(cache_dir(dataset) / "views.npy", indices, labels, [row["flow_uid"] for row in selected])
        kwargs = {"batch_size": batch_size if role == "known_train" else eval_batch_size, "shuffle": role == "known_train", "num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}
        if role == "known_train":
            kwargs["generator"] = torch.Generator().manual_seed(seed)
        output[role] = DataLoader(dataset_obj, **kwargs)
    return output, rows


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=np.arange(n_classes), zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)), "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)), "precision": precision.tolist(),
        "recall": recall.tolist(), "f1": f1.tolist(), "support": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(n_classes)).tolist(),
    }


@torch.no_grad()
def predict(model, loader, device, role: str):
    model.eval()
    labels, predictions, scores, flow_ids = [], [], [], []
    for values, target, uid in loader:
        values = values.to(device, non_blocking=True)
        fused, uncertainty = model(values)
        labels.append(target.numpy())
        predictions.append(fused.argmax(1).cpu().numpy())
        scores.append(uncertainty[:, 0].cpu().numpy())
        flow_ids.extend(uid)
    return {
        "role": role, "labels": np.concatenate(labels), "predictions": np.concatenate(predictions),
        "scores": np.concatenate(scores), "flow_ids": flow_ids,
    }


def train_epoch(model, loader, optimizer, device, epoch):
    model.train(); total_loss = total = correct = 0
    for values, labels, _uid in loader:
        values = values.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        fused, _uncertainty, loss_terms = model(values, labels, epoch)
        loss = loss_terms.mean()
        loss.backward(); optimizer.step()
        total_loss += float(loss.item()) * len(labels); total += len(labels); correct += int((fused.argmax(1) == labels).sum().item())
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config(); spec = next(item for item in cfg["protocols"] if item["dataset"] == args.dataset and item["protocol_id"] == args.protocol_id)
    output = ROOT / ("smoke_runs" if args.smoke else "runs") / args.dataset / args.protocol_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    temp_alias, temp_directory_fd = install_project_local_short_tmp(output)
    seed = int(spec["seed"]); seed_everything(seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    train_cfg = cfg["training"]
    loaders_by_role, rows = loaders(args.dataset, args.protocol_id, int(train_cfg["batch_size"]), int(train_cfg["eval_batch_size"]), int(train_cfg["workers"]), seed)
    class_names = [name for _label, name in sorted({(int(row["local_label"]), row["class_name"]) for row in rows if row["is_unknown"] == "0"})]
    n_classes = len(class_names)
    core_model = MobileVitClassifier(n_classes, 3, int(train_cfg["lambda_epochs"]), packet_num=int(cfg["input"]["packet_num"])).to(device)
    parity = parity_check(core_model, device, int(cfg["input"]["packet_num"]), int(cfg["input"]["byte_num_per_view"]))
    write_json(output / "adapter_parity.json", parity)
    adapter = DenseRoNeTCAdapter(core_model).to(device)
    if args.data_parallel:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices")
        model = nn.DataParallel(adapter, device_ids=list(range(torch.cuda.device_count())))
    else:
        model = adapter
    if args.smoke:
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]), betas=tuple(train_cfg["betas"]), weight_decay=float(train_cfg["weight_decay"]))
        values, labels, _uid = next(iter(loaders_by_role["known_train"]))
        values = values.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        fused, uncertainty, loss_terms = model(values, labels, 1)
        loss = loss_terms.mean()
        loss.backward(); optimizer.step()
        smoke = {"status": "PASS", "mode": "smoke", "dataset": args.dataset, "protocol_id": args.protocol_id, "batch_size": len(labels), "loss": float(loss.item()), "finite_fused": bool(torch.isfinite(fused).all()), "finite_uncertainty": bool(torch.isfinite(uncertainty).all()), "peak_gpu_memory_bytes": {str(index): int(torch.cuda.max_memory_allocated(index)) for index in range(torch.cuda.device_count())}, "visible_gpu_count": torch.cuda.device_count(), "data_parallel": bool(args.data_parallel), "project_local_tmp_alias": temp_alias, "parity": parity}
        write_json(output / "smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2)); return
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]), betas=tuple(train_cfg["betas"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(train_cfg["scheduler_milestones"]), gamma=float(train_cfg["scheduler_gamma"]))
    started = time.time(); history = []; best_accuracy = -1.0; best_epoch = 0
    checkpoint = output / "checkpoint_best.pt"
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy = train_epoch(model, loaders_by_role["known_train"], optimizer, device, epoch)
        val = predict(model, loaders_by_role["known_validation"], device, "known_validation")
        val_metrics = classification_metrics(val["labels"], val["predictions"], n_classes)
        improved = val_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = val_metrics["accuracy"], epoch
            torch.save({"model_state_dict": core_model.state_dict(), "epoch": epoch, "class_names": class_names, "config": cfg, "dataset": args.dataset, "protocol_id": args.protocol_id, "data_parallel_training": bool(args.data_parallel)}, checkpoint)
        history.append({"epoch": epoch, "learning_rate": lr, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_accuracy": val_metrics["accuracy"], "validation_macro_f1": val_metrics["macro_f1"], "validation_weighted_f1": val_metrics["weighted_f1"], "is_best": int(improved), "elapsed_seconds": time.time() - started})
        write_csv(output / "history.csv", history)
        print(json.dumps({"dataset": args.dataset, "protocol": args.protocol_id, **history[-1]}), flush=True)
        scheduler.step()
    saved = torch.load(checkpoint, map_location=device, weights_only=False); core_model.load_state_dict(saved["model_state_dict"])
    val = predict(model, loaders_by_role["known_validation"], device, "known_validation")
    known = predict(model, loaders_by_role["known_test"], device, "known_test")
    unknown = predict(model, loaders_by_role["unknown_test"], device, "unknown_test")
    threshold = quantile_higher(val["scores"], float(cfg["threshold"]["quantile"]))
    known_closed = classification_metrics(known["labels"], known["predictions"], n_classes)
    binary_y = np.concatenate((np.zeros(len(known["scores"]), dtype=np.int64), np.ones(len(unknown["scores"]), dtype=np.int64)))
    binary_scores = np.concatenate((known["scores"], unknown["scores"]))
    binary_pred = (binary_scores > threshold).astype(np.int64)
    result = {
        "status": "PASS", "dataset": args.dataset, "protocol_id": args.protocol_id, "seed": seed,
        "known_classes": class_names, "unknown_classes": sorted({row["class_name"] for row in rows if row["is_unknown"] == "1"}),
        "samples": {role: len(loader.dataset) for role, loader in loaders_by_role.items()},
        "best_epoch": best_epoch, "epochs_completed": int(train_cfg["epochs"]), "best_validation_accuracy": best_accuracy,
        "validation": classification_metrics(val["labels"], val["predictions"], n_classes), "known_test_closed": known_closed,
        "open_set": {
            "threshold_known_validation_p95": threshold, "auroc": float(roc_auc_score(binary_y, binary_scores)),
            "auprc": float(average_precision_score(binary_y, binary_scores)),
            "ufar": float(np.mean(unknown["scores"] <= threshold)), "known_frr": float(np.mean(known["scores"] > threshold)),
            "binary_f1": float(f1_score(binary_y, binary_pred, zero_division=0)),
        },
        "runtime_seconds": time.time() - started, "peak_gpu_memory_bytes": {str(index): int(torch.cuda.max_memory_allocated(index)) for index in range(torch.cuda.device_count())},
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"), "visible_gpu_count": torch.cuda.device_count(), "data_parallel": bool(args.data_parallel), "unknown_train_samples": 0, "unknown_validation_samples": 0,
    }
    prediction_rows = []
    for payload in (val, known, unknown):
        for uid, label, prediction, score in zip(payload["flow_ids"], payload["labels"], payload["predictions"], payload["scores"]):
            prediction_rows.append({"dataset": args.dataset, "protocol_id": args.protocol_id, "role": payload["role"], "flow_uid": uid, "true_label": int(label), "predicted_known_label": int(prediction), "unknown_score": float(score), "threshold": threshold, "predicted_unknown": int(score > threshold)})
    write_csv(output / "predictions.csv", prediction_rows)
    write_json(output / "result.json", result)
    write_json(output / "run_config.json", {"config": cfg, "invocation": vars(args), "strict_unknown_free": True, "test_used_for_selection": False, "project_local_tmp_alias": temp_alias, "project_local_tmp_directory_fd": temp_directory_fd})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("error", category=RuntimeWarning)
    main()
