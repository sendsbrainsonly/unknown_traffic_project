#!/usr/bin/env python3
"""YaTC official-pretrained closed-set training on frozen Stage20 Known Train/Val."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
YATC = PROJECT.parent / "YaTC" / "code"
MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
CACHE = OUT / "yatc_stage20_mfr_cache"
WEIGHT = YATC / "output_dir" / "pretrained-model.pth"
FROZEN = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
WEIGHT_HASH = "66314aa57d4364bad0228835a181158f819d10c857936fbbdbae6dc483211fc6"
sys.path.insert(0, str(YATC))
from yatc_compat import enable_torch_six_compat, load_trusted_checkpoint  # noqa: E402
enable_torch_six_compat()
from timm.loss import LabelSmoothingCrossEntropy  # noqa: E402
from timm.models.layers import trunc_normal_  # noqa: E402
from models_YaTC import TraFormer_YaTC  # noqa: E402
from engine import train_one_epoch  # noqa: E402
from util import lr_decay as lrd  # noqa: E402
from util.misc import NativeScalerWithGradNormCount  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_rows(dataset: str, role: str) -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset and row["closed_role"] == role]
    return sorted(rows, key=lambda row: row["flow_id"])


class MFRDataset(Dataset):
    def __init__(self, dataset: str, role: str, services: list[str]):
        self.rows = frozen_rows(dataset, role)
        root = CACHE / dataset
        phase = "test" if role == "known_test" else "trainval"
        audit = read_json(root / f"cache_audit_{phase}.json")["roles"][role]
        image, ids = root / f"{role}_mfr.npy", root / f"{role}_flow_ids.npy"
        if sha(image) != audit["mfr_sha256"] or sha(ids) != audit["flow_ids_sha256"]:
            raise RuntimeError(f"YaTC MFR hash mismatch: {dataset}/{role}")
        self.images = np.load(image, mmap_mode="r", allow_pickle=False)
        known_ids = [str(x) for x in np.load(ids, allow_pickle=False)]
        if known_ids != [row["flow_id"] for row in self.rows] or self.images.shape != (len(known_ids), 40, 40):
            raise RuntimeError(f"YaTC MFR membership mismatch: {dataset}/{role}")
        if {row["service_label"] for row in self.rows} != set(services):
            raise RuntimeError(f"YaTC class set mismatch: {dataset}/{role}")
        self.labels = [services.index(row["service_label"]) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        # PIL grayscale ToTensor + Normalize([0.5],[0.5]) used by YaTC.
        x = torch.from_numpy(np.array(self.images[index], copy=True)).float().unsqueeze(0)
        return x.div_(255).sub_(0.5).div_(0.5), self.labels[index]


def metric_values(true: list[int], pred: list[int], services: list[str]) -> tuple[dict, list[dict]]:
    matrix = np.zeros((len(services), len(services)), dtype=np.int64)
    for a, b in zip(true, pred):
        matrix[a, b] += 1
    classes = []
    for i, name in enumerate(services):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        classes.append({
            "service": name, "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
            "support": support,
        })
    n = len(true)
    return {
        "accuracy": int(np.trace(matrix)) / n,
        "macro_f1": sum(x["f1"] for x in classes) / len(classes),
        "weighted_f1": sum(x["f1"] * x["support"] for x in classes) / n,
        "confusion_matrix": matrix.tolist(),
    }, classes


@torch.no_grad()
def evaluate(model, loader, device, services: list[str]):
    model.eval()
    true, pred = [], []
    for x, y in loader:
        with torch.cuda.amp.autocast():
            logits = model(x.to(device, non_blocking=True))
        true.extend(int(v) for v in y.numpy())
        pred.extend(int(v) for v in logits.argmax(1).cpu().numpy())
    metrics, classes = metric_values(true, pred, services)
    return metrics, classes, pred


def initialized_model(services: list[str]):
    model = TraFormer_YaTC(num_classes=len(services), drop_path_rate=0.1)
    weights = dict(load_trusted_checkpoint(WEIGHT, map_location="cpu")["model"])
    existing = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in weights and weights[key].shape != existing[key].shape:
            del weights[key]
    interpolate_pos_embed(model, weights)
    info = model.load_state_dict(weights, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)
    return model, {"missing_keys": list(info.missing_keys), "unexpected_keys": list(info.unexpected_keys)}


def main(dataset: str, seed: int, smoke: bool, tag: str) -> None:
    if sha(MANIFEST) != FROZEN or sha(WEIGHT) != WEIGHT_HASH:
        raise RuntimeError("frozen Stage20 or YaTC weight hash changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    scope = read_json(PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "config.json")
    if dataset not in scope["datasets"] or seed not in scope["seeds"]:
        raise RuntimeError("outside paired Stage22 scope")
    services = scope["datasets"][dataset]["services"]
    cache_audit = read_json(CACHE / dataset / "cache_audit_trainval.json")
    if cache_audit["status"] != "PASS" or cache_audit["stage20_manifest_sha256"] != FROZEN:
        raise RuntimeError("YaTC Known Train/Val cache not verified")
    if tag and not all(c.isalnum() or c in "_-" for c in tag):
        raise ValueError("invalid run tag")
    label = "smoke" if smoke else "formal"
    run = OUT / "runs" / "yatc_stage20" / dataset / f"seed{seed}_{label}{'_' + tag if tag else ''}"
    if run.exists():
        raise FileExistsError(f"preserve existing run: {run}")
    run.mkdir(parents=True)
    started = time.time()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed); torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
        train_data, val_data = MFRDataset(dataset, "known_train", services), MFRDataset(dataset, "known_validation", services)
        expected = scope["datasets"][dataset]
        if len(train_data) != expected["expected_train"] or len(val_data) != expected["expected_validation"]:
            raise RuntimeError("Stage20 role counts changed")
        sampler = DistributedSampler(train_data, num_replicas=1, rank=0, shuffle=True, seed=0)
        train_loader = DataLoader(train_data, sampler=sampler, batch_size=64, drop_last=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        write_json(run / "input_preflight.json", {
            "status": "PASS", "dataset": dataset, "seed": seed,
            "known_train": len(train_data), "known_validation": len(val_data),
            "known_test_features_loaded": 0, "unknown_features_loaded": 0,
            "stage20_manifest_sha256": FROZEN,
            "mfr_train_sha256": cache_audit["roles"]["known_train"]["mfr_sha256"],
            "mfr_validation_sha256": cache_audit["roles"]["known_validation"]["mfr_sha256"],
        })
        model, load_info = initialized_model(services)
        model.to(device)
        lr = 0.002 * 64 / 256
        groups = lrd.param_groups_lrd(model, 0.05, no_weight_decay_list=model.no_weight_decay(), layer_decay=0.75)
        optimizer = torch.optim.AdamW(groups, lr=lr)
        scaler = NativeScalerWithGradNormCount()
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        args = SimpleNamespace(accum_iter=1, warmup_epochs=20, min_lr=1e-6, lr=lr,
                               epochs=200, clip_grad=None, output_dir="")
        write_json(run / "config.json", {
            "dataset": dataset, "seed": seed, "method": "YaTC-official-pretrained-Stage20",
            "services": services, "official_pretrained_sha256": WEIGHT_HASH,
            "source_model_sha256": sha(YATC / "models_YaTC.py"),
            "source_engine_sha256": sha(YATC / "engine.py"),
            "stage20_manifest_sha256": FROZEN,
            "training": {"epochs": 200, "batch_size": 64, "drop_last": True,
                         "optimizer": "AdamW", "base_lr": 0.002, "effective_lr": lr,
                         "weight_decay": 0.05, "layer_decay": 0.75, "warmup_epochs": 20,
                         "min_lr": 1e-6, "label_smoothing": 0.1, "drop_path_rate": 0.1,
                         "sampler": "DistributedSampler world_size=1 seed=0", "num_workers": 0,
                         "author_epoch_loop": True},
            "checkpoint_selection": "Known Validation weighted F1 (author macro_f1 field)",
            "load_message": load_info,
            "unknown_training_samples": 0, "unknown_validation_samples": 0,
            "test_selection_samples": 0,
        })
        torch.cuda.reset_peak_memory_stats(device)
        if smoke:
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite YaTC smoke loss")
            loss.backward(); optimizer.step()
            write_json(run / "smoke_result.json", {
                "status": "PASS", "batch": len(y), "loss": float(loss.item()),
                "finite_logits": bool(torch.isfinite(logits).all()),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "known_test_features_loaded": 0,
            })
            (run / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
            print(json.dumps({"status": "PASS_SMOKE", "dataset": dataset, "seed": seed}), flush=True)
            return
        best_f1, best_epoch = -1.0, -1
        checkpoint_path = run / "model_best.pt"
        with (run / "training_history.jsonl").open("x", encoding="utf-8") as handle:
            for epoch in range(200):
                sampler.set_epoch(epoch)
                train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device,
                                              epoch, scaler, None, None, log_writer=None, args=args)
                validation, _classes, _predicted = evaluate(model, val_loader, device, services)
                if validation["weighted_f1"] > best_f1:
                    best_f1, best_epoch = validation["weighted_f1"], epoch + 1
                    torch.save({
                        "model_state_dict": model.state_dict(), "epoch": best_epoch,
                        "known_validation_weighted_f1": best_f1,
                        "services": services, "seed": seed,
                        "official_pretrained_sha256": WEIGHT_HASH,
                    }, checkpoint_path)
                event = {
                    "epoch": epoch + 1, "train": train_stats, "validation": validation,
                    "best_epoch": best_epoch, "best_validation_weighted_f1": best_f1,
                    "elapsed_seconds": time.time() - started,
                }
                handle.write(json.dumps(event, default=float) + "\n")
                handle.flush()
                print(json.dumps({"dataset": dataset, "seed": seed, "epoch": epoch + 1,
                                  "val_weighted_f1": validation["weighted_f1"],
                                  "best": best_f1}), flush=True)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        validation, per_class, predicted = evaluate(model, val_loader, device, services)
        if abs(validation["weighted_f1"] - best_f1) > 1e-12:
            raise RuntimeError("best checkpoint Known Validation replay failed")
        prediction_path = run / "validation_predictions.csv"
        with prediction_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "dataset", "seed", "role", "method", "flow_id", "true_service",
                "predicted_service", "correct",
            ])
            writer.writeheader()
            for row, value in zip(val_data.rows, predicted):
                writer.writerow({
                    "dataset": dataset, "seed": seed, "role": "known_validation",
                    "method": "YaTC-official-pretrained-Stage20",
                    "flow_id": row["flow_id"], "true_service": row["service_label"],
                    "predicted_service": services[value],
                    "correct": int(row["service_label"] == services[value]),
                })
        write_json(run / "metrics.json", {
            "dataset": dataset, "seed": seed, "method": "YaTC-official-pretrained-Stage20",
            "best_epoch": best_epoch, "completed_epochs": 200,
            "validation": validation, "validation_per_class": per_class,
            "checkpoint_sha256": sha(checkpoint_path),
            "validation_predictions_sha256": sha(prediction_path),
            "official_pretrained_sha256": WEIGHT_HASH,
            "stage20_manifest_sha256": FROZEN,
            "runtime_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "known_test_features_loaded": 0,
            "unknown_training_samples": 0, "unknown_validation_samples": 0,
            "test_selection_samples": 0,
        })
        if sha(MANIFEST) != FROZEN or sha(WEIGHT) != WEIGHT_HASH:
            raise RuntimeError("protected input hash changed during YaTC training")
        (run / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
        print(json.dumps({"status": "SUCCESS", "dataset": dataset, "seed": seed,
                          "validation": {key: validation[key] for key in
                                         ("accuracy", "macro_f1", "weighted_f1")}}), flush=True)
    except BaseException as exc:
        write_json(run / "FAILURE.json", {
            "status": "FAILED", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_partial_artifacts": True,
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-tag", default="")
    cli = parser.parse_args()
    main(cli.dataset, cli.seed, cli.smoke, cli.run_tag)
