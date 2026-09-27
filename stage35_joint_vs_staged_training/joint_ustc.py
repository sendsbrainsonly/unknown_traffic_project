#!/usr/bin/env python3
"""Matched USTC three-view joint-training ablation; Test is a separate action."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STAGE34 = PROJECT / "stage34_ustc_cic_closed_set"
STAGE31 = PROJECT / "stage31_four_dataset_three_view_equal"
sys.path.insert(0, str(STAGE34))
sys.path.insert(0, str(STAGE31))
import ustc_runner  # noqa: E402
import train_tf_fig_branch as tf_branch  # noqa: E402
import train_yatc_branch as yatc_branch  # noqa: E402
import train_equal_fusion as fusion  # noqa: E402

SEED = 2022
EPOCHS = 20
EFFECTIVE_BATCH = 64
VIEWS = ("trafficformer", "graph", "yatc")
DIMS = (768, 128, 192)
TRAINVAL = STAGE34 / "input_caches/ustc/A-2"
RUN = ROOT / "runs/ustc/joint_e2e20"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def selected_rows(role: str, classes: list[str]) -> list[tuple[str, str]]:
    raw = ustc_runner.selected()
    name = {"known_train": "train", "known_validation": "val", "known_test": "test"}[role]
    rows = sorted((r["flow_id"], r["class_name"]) for r in raw
                  if r["original_split"] == name and r["class_name"] in classes)
    if not rows or len({uid for uid, _ in rows}) != len(rows):
        raise RuntimeError(f"empty or duplicated {role} flow IDs")
    if {label for _, label in rows} != set(classes):
        raise RuntimeError(f"{role} does not contain all Known classes")
    return rows


def source_guard() -> dict:
    expected = json.loads((STAGE34 / "ustc_sample_audit.json").read_text())
    manifest = STAGE34 / "ustc_a2_10pct_manifest.csv"
    if sha(manifest) != expected["sample_manifest_sha256"]:
        raise RuntimeError("Stage34 USTC sample manifest changed")
    if sha(ustc_runner.SOURCE) != expected["source_sha256"]:
        raise RuntimeError("USTC source alignment changed")
    if sha(ustc_runner.CONFIG) != expected["a2_config_sha256"]:
        raise RuntimeError("USTC A-2 Known class configuration changed")
    tf_config = json.loads(tf_branch.source.CONFIG.read_text())
    tf_hash = sha(tf_branch.source.PRETRAINED_MODEL)
    ya_hash = sha(yatc_branch.source.WEIGHT)
    if tf_hash != tf_config["pretrained_model_sha256"]:
        raise RuntimeError("official TrafficFormer weight hash changed")
    if ya_hash != yatc_branch.source.WEIGHT_HASH:
        raise RuntimeError("official YaTC weight hash changed")
    return {"sample_manifest_sha256": sha(manifest),
            "trafficformer_pretrained_sha256": tf_hash,
            "yatc_pretrained_sha256": ya_hash}


def code_guard() -> dict:
    lock = json.loads((ROOT / "code_lock.json").read_text())
    for relative, digest in lock["sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"frozen Stage35 execution code changed: {relative}")
    return lock


def cache_roots(role: str) -> tuple[Path, Path]:
    if role == "known_test":
        return TRAINVAL / "tf_fig_test", TRAINVAL / "yatc_mfr_test"
    return TRAINVAL / "tf_fig", TRAINVAL / "yatc_mfr"


def aligned_index(role: str, rows: list[tuple[str, str]]) -> tuple[np.ndarray, Path, Path]:
    tf_root, mfr_root = cache_roots(role)
    for root in (tf_root, mfr_root):
        audit = json.loads((root / "cache_audit.json").read_text())
        if audit["status"] != "PASS":
            raise RuntimeError(f"input cache audit not PASS: {root}")
    tf_ids = np.load(tf_root / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if len(tf_ids) != len(set(tf_ids)):
        raise RuntimeError("duplicated TrafficFormer/FIG cache ID")
    pos = {uid: i for i, uid in enumerate(tf_ids)}
    wanted = {uid for uid, _ in rows}
    if not wanted.issubset(pos):
        raise RuntimeError(f"TrafficFormer/FIG {role} ID set mismatch")
    if role == "known_test" and set(pos) != wanted:
        raise RuntimeError("TrafficFormer/FIG Test cache contains unexpected IDs")
    mfr_ids = np.load(mfr_root / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if mfr_ids != [uid for uid, _ in rows]:
        raise RuntimeError(f"YaTC {role} sorted flow ID mismatch")
    return np.asarray([pos[uid] for uid, _ in rows], dtype=np.int64), tf_root, mfr_root


def preflight() -> dict:
    if (ROOT / "preflight.json").exists():
        raise FileExistsError(ROOT / "preflight.json")
    guard = source_guard()
    selected, classes, split_hash = ustc_runner.protocol_rows("ustc", "A-2")
    counts = {}
    cache_hashes = {}
    for role in ("known_train", "known_validation"):
        rows = selected_rows(role, classes)
        pos, tf_root, mfr_root = aligned_index(role, rows)
        counts[role] = len(rows)
        if len(pos) != len(rows):
            raise RuntimeError(f"{role} index length mismatch")
        for filename in ("token_ids.npy", "segments.npy", "fig_x.npy", "fig_adj.npy", "fig_mask.npy"):
            array = np.load(tf_root / filename, mmap_mode="r", allow_pickle=False)
            if len(array) != len(selected):
                raise RuntimeError(f"{filename} first dimension mismatch")
            cache_hashes[f"tf_fig/{filename}"] = sha(tf_root / filename)
        mfr = np.load(mfr_root / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
        if mfr.shape != (len(rows), 40, 40):
            raise RuntimeError(f"{role} MFR shape mismatch: {mfr.shape}")
        cache_hashes[f"yatc_mfr/{role}_mfr.npy"] = sha(mfr_root / f"{role}_mfr.npy")
    if counts != {"known_train": 34665, "known_validation": 4333}:
        raise RuntimeError(f"unexpected Stage34 USTC split counts: {counts}")
    result = {"status": "PASS", "dataset": "USTC-TFC2016", "protocol": "A-2_17class_10pct",
              "classes": classes, "counts": counts, "split_sha256": split_hash,
              **guard, "cache_sha256": cache_hashes, "unknown_feature_values_loaded": 0,
              "test_feature_values_loaded": 0, "selection_seed": SEED}
    write_json(ROOT / "preflight.json", result)
    print(json.dumps({"status": "PASS", "counts": counts, "classes": len(classes)}), flush=True)
    return result


def graph_scaler(classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = selected_rows("known_train", classes)
    pos, root, _ = aligned_index("known_train", rows)
    x = np.asarray(np.load(root / "fig_x.npy", mmap_mode="r", allow_pickle=False)[pos], dtype=np.float32)
    mask = np.asarray(np.load(root / "fig_mask.npy", mmap_mode="r", allow_pickle=False)[pos], dtype=bool)
    nodes = x[mask]
    center = nodes.mean(0)
    std = nodes.std(0)
    std[std < 1e-8] = 1.0
    return center.astype(np.float32), std.astype(np.float32)


class ThreeViewFlows(Dataset):
    def __init__(self, role: str, classes: list[str], graph_mean: np.ndarray,
                 graph_std: np.ndarray):
        self.rows = selected_rows(role, classes)
        self.ids = [uid for uid, _ in self.rows]
        self.labels = np.asarray([classes.index(label) for _, label in self.rows], dtype=np.int64)
        pos, root, mfr_root = aligned_index(role, self.rows)
        self.tokens = np.asarray(np.load(root / "token_ids.npy", mmap_mode="r", allow_pickle=False)[pos]).copy()
        self.segments = np.asarray(np.load(root / "segments.npy", mmap_mode="r", allow_pickle=False)[pos]).copy()
        raw_x = np.asarray(np.load(root / "fig_x.npy", mmap_mode="r", allow_pickle=False)[pos], dtype=np.float32)
        self.mask = np.asarray(np.load(root / "fig_mask.npy", mmap_mode="r", allow_pickle=False)[pos], dtype=bool)
        self.graph_x = ((raw_x - graph_mean) / graph_std).astype(np.float32)
        self.graph_x[~self.mask] = 0.0
        raw_adj = np.asarray(np.load(root / "fig_adj.npy", mmap_mode="r", allow_pickle=False)[pos])
        self.graph_adj = tf_branch.source.normalize_adjacency(raw_adj, self.mask)
        self.mfr = np.asarray(np.load(mfr_root / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)).copy()
        if self.mfr.shape != (len(self.rows), 40, 40):
            raise RuntimeError(f"{role} MFR shape mismatch")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.mfr[index].copy()).float().unsqueeze(0)
        image = image.div_(255.0).sub_(0.5).div_(0.5)
        return (torch.from_numpy(self.tokens[index].astype(np.int64)),
                torch.from_numpy(self.segments[index].astype(np.int64)),
                torch.from_numpy(self.graph_x[index]),
                torch.from_numpy(self.graph_adj[index]),
                torch.from_numpy(self.mask[index]), image, int(self.labels[index]))


def load_batches(dataset: ThreeViewFlows, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=True, generator=torch.Generator().manual_seed(SEED) if shuffle else None)


def to_device(batch, device):
    src, seg, x, adj, mask, mfr, label = batch
    return (src.to(device, non_blocking=True), seg.to(device, non_blocking=True),
            x.to(device, non_blocking=True), adj.to(device, non_blocking=True),
            mask.to(device, non_blocking=True), mfr.to(device, non_blocking=True),
            label.to(device, non_blocking=True))


class JointThreeView(nn.Module):
    def __init__(self, classes: list[str], total_steps: int):
        super().__init__()
        seed_all()
        self.tf_args = tf_branch.source.make_tf_args(len(classes), total_steps)
        self.tf_args.batch_size = EFFECTIVE_BATCH
        tf_module = tf_branch.source.load_tf_module()
        self.tf = tf_module.Classifier(self.tf_args)
        load = self.tf.load_state_dict(torch.load(tf_branch.source.PRETRAINED_MODEL,
                                                 map_location="cpu", weights_only=True), strict=False)
        self.tf_load = {"missing_keys": list(load.missing_keys),
                        "unexpected_keys": list(load.unexpected_keys)}
        seed_all()
        self.graph = tf_branch.TAGCN(in_dim=7, hidden=128, labels_num=len(classes),
                                    k_hops=2, dropout=0.5)
        seed_all()
        self.yatc, self.yatc_load = yatc_branch.source.initialized_model(classes)
        seed_all()
        self.adapters = fusion.Adapters(len(classes))
        self.head = fusion.EqualFusion(len(classes), np.zeros(3), np.ones(3))
        for name, dim in zip(VIEWS, DIMS, strict=True):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

    def set_scalers(self, means: list[np.ndarray], stds: list[np.ndarray]) -> None:
        for name, mean, std in zip(VIEWS, means, stds, strict=True):
            getattr(self, f"{name}_mean").copy_(torch.as_tensor(mean, device=getattr(self, f"{name}_mean").device))
            getattr(self, f"{name}_std").copy_(torch.as_tensor(std, device=getattr(self, f"{name}_std").device))

    def raw_features(self, batch):
        src, seg, x, adj, mask, mfr, _ = batch
        _, z_tf = tf_branch.source.tf_forward(self.tf, src, seg)
        _, z_graph = self.graph(x, adj, mask)
        with torch.cuda.amp.autocast():
            z_yatc = self.yatc.forward_features(mfr)
        return z_tf.float(), z_graph.float(), z_yatc.float()

    def forward(self, batch):
        raw = self.raw_features(batch)
        means = [getattr(self, f"{name}_mean") for name in VIEWS]
        stds = [getattr(self, f"{name}_std") for name in VIEWS]
        mu = [view((z - mean) / std)[0] for view, z, mean, std in
              zip(self.adapters.views, raw, means, stds, strict=True)]
        stacked = torch.stack(mu, dim=1)
        logits, _ = self.head(stacked, torch.zeros((len(stacked), 3), device=stacked.device))
        return logits


def metrics(truth: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(classes)), zero_division=0)
    return {"accuracy": float(np.mean(truth == pred)), "macro_f1": float(f.mean()),
            "weighted_f1": float(np.average(f, weights=support)),
            "per_class": [{"class": name, "precision": float(p[i]), "recall": float(r[i]),
                           "f1": float(f[i]), "support": int(support[i])}
                          for i, name in enumerate(classes)]}


@torch.no_grad()
def predict(model: JointThreeView, loader: DataLoader, device: torch.device):
    model.eval()
    logits, truth = [], []
    for batch in loader:
        moved = to_device(batch, device)
        logits.append(model(moved).float().cpu().numpy())
        truth.append(moved[-1].cpu().numpy())
    return np.concatenate(logits), np.concatenate(truth)


@torch.no_grad()
def fit_initial_feature_scalers(model: JointThreeView, loader: DataLoader,
                                device: torch.device):
    model.eval()
    sums = [np.zeros(d, dtype=np.float64) for d in DIMS]
    squares = [np.zeros(d, dtype=np.float64) for d in DIMS]
    count = 0
    for batch in loader:
        moved = to_device(batch, device)
        for j, value in enumerate(model.raw_features(moved)):
            x = value.float().cpu().numpy().astype(np.float64)
            sums[j] += x.sum(0)
            squares[j] += (x*x).sum(0)
        count += len(moved[-1])
    if count != len(loader.dataset):
        raise RuntimeError("train-only initial feature scaler count mismatch")
    means = [v / count for v in sums]
    stds = [np.sqrt(np.maximum(squares[j] / count - means[j]**2, 0.0)) for j in range(3)]
    for std in stds:
        std[std < 1e-8] = 1.0
    means = [x.astype(np.float32) for x in means]
    stds = [x.astype(np.float32) for x in stds]
    model.set_scalers(means, stds)
    return means, stds, count


def smoke(microbatch: int) -> None:
    if microbatch not in (16, 32):
        raise ValueError("smoke microbatch must be 16 or 32")
    run = ROOT / f"smoke_batch{microbatch}"
    run.mkdir(exist_ok=False)
    try:
        info = json.loads((ROOT / "preflight.json").read_text())
        if info["status"] != "PASS" or source_guard()["sample_manifest_sha256"] != info["sample_manifest_sha256"]:
            raise RuntimeError("preflight/source guard failed")
        classes = info["classes"]
        mean, std = graph_scaler(classes)
        train = ThreeViewFlows("known_train", classes, mean, std)
        device = torch.device("cuda:0")
        model = JointThreeView(classes, math.ceil(len(train)/EFFECTIVE_BATCH)*EPOCHS).to(device)
        batch = to_device(next(iter(load_batches(train, microbatch, False))), device)
        torch.cuda.reset_peak_memory_stats(device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch[-1])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite smoke fusion loss")
        loss.backward()
        result = {"status": "PASS", "microbatch": microbatch, "effective_batch": EFFECTIVE_BATCH,
                  "loss": float(loss.detach()), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                  "sample_count": len(batch[-1]), "source_sample_sha256": info["sample_manifest_sha256"]}
        write_json(run / "result.json", result)
        (run / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(run / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


def optimizer_smoke(microbatch: int) -> None:
    if microbatch not in (16, 32):
        raise ValueError("optimizer smoke microbatch must be 16 or 32")
    output = ROOT / f"optimizer_smoke_batch{microbatch}"
    output.mkdir(exist_ok=False)
    try:
        info = json.loads((ROOT / "preflight.json").read_text())
        if info["status"] != "PASS" or source_guard()["sample_manifest_sha256"] != info["sample_manifest_sha256"]:
            raise RuntimeError("preflight/source guard failed")
        seed_all()
        classes = info["classes"]
        mean, std = graph_scaler(classes)
        train_ds = ThreeViewFlows("known_train", classes, mean, std)
        device = torch.device("cuda:0")
        model = JointThreeView(classes, math.ceil(len(train_ds)/EFFECTIVE_BATCH)*EPOCHS).to(device)
        batch = to_device(next(iter(load_batches(train_ds, microbatch, False))), device)
        tf_module = tf_branch.source.load_tf_module()
        tf_opt, tf_scheduler = tf_module.build_optimizer(model.tf_args, model.tf)
        graph_opt = torch.optim.Adam(model.graph.parameters(), lr=1e-3)
        yatc_groups = yatc_branch.source.lrd.param_groups_lrd(
            model.yatc, 0.05, no_weight_decay_list=model.yatc.no_weight_decay(), layer_decay=0.75)
        yatc_opt = torch.optim.AdamW(yatc_groups, lr=5e-4)
        fusion_opt = torch.optim.Adam(list(model.adapters.parameters()) + list(model.head.parameters()), lr=1e-3)
        optimizers = (tf_opt, graph_opt, yatc_opt, fusion_opt)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        set_yatc_lr(yatc_opt, 1, math.ceil(len(train_ds)/EFFECTIVE_BATCH)*EPOCHS)
        scaler = torch.cuda.amp.GradScaler()
        loss = F.cross_entropy(model(batch), batch[-1])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite optimizer smoke loss")
        scaler.scale(loss).backward()
        gradient_groups = {}
        for name, opt in zip(("trafficformer", "graph", "yatc", "fusion"), optimizers, strict=True):
            norms = [p.grad.detach().float().norm() for group in opt.param_groups for p in group["params"]
                     if p.grad is not None]
            if not norms:
                raise RuntimeError(f"{name} has no gradient in final fused classification path")
            gradient_groups[name] = float(torch.stack(norms).norm().cpu())
            if not math.isfinite(gradient_groups[name]) or gradient_groups[name] <= 0:
                raise RuntimeError(f"invalid {name} gradient norm: {gradient_groups[name]}")
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        tf_scheduler.step()
        result = {"status": "PASS", "loss": float(loss.detach()), "microbatch": microbatch,
                  "gradient_norms": gradient_groups, "optimizer_groups_stepped": 4,
                  "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                  "test_feature_values_loaded": 0, "unknown_samples_used": 0}
        write_json(output / "result.json", result)
        (output / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(output / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


def set_yatc_lr(optimizer, step: int, total_steps: int) -> None:
    fraction = step / max(total_steps, 1)
    if fraction < 0.1:
        base = 5e-4 * fraction / 0.1
    else:
        base = 1e-6 + 0.5 * (5e-4 - 1e-6) * (1 + math.cos(math.pi * (fraction-0.1)/0.9))
    for group in optimizer.param_groups:
        group["lr"] = base * group.get("lr_scale", 1.0)


def train(microbatch: int) -> None:
    if microbatch not in (16, 32) or EFFECTIVE_BATCH % microbatch:
        raise ValueError("invalid microbatch")
    smoke_result = ROOT / f"smoke_batch{microbatch}/result.json"
    if not smoke_result.is_file() or json.loads(smoke_result.read_text())["status"] != "PASS":
        raise RuntimeError("matching memory smoke missing")
    optimizer_result = ROOT / f"optimizer_smoke_batch{microbatch}/result.json"
    if not optimizer_result.is_file() or json.loads(optimizer_result.read_text())["status"] != "PASS":
        raise RuntimeError("matching all-branch optimizer smoke missing")
    RUN.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        code_guard()
        info = json.loads((ROOT / "preflight.json").read_text())
        guard = source_guard()
        if guard["sample_manifest_sha256"] != info["sample_manifest_sha256"]:
            raise RuntimeError("sample source changed since preflight")
        classes = info["classes"]
        seed_all()
        device = torch.device("cuda:0")
        graph_mean, graph_std = graph_scaler(classes)
        np.savez(RUN / "known_train_graph_scaler.npz", mean=graph_mean, std=graph_std)
        train_ds = ThreeViewFlows("known_train", classes, graph_mean, graph_std)
        val_ds = ThreeViewFlows("known_validation", classes, graph_mean, graph_std)
        if (len(train_ds), len(val_ds)) != (34665, 4333):
            raise RuntimeError("frozen Train/Validation count drift")
        model = JointThreeView(classes, math.ceil(len(train_ds)/EFFECTIVE_BATCH)*EPOCHS).to(device)
        train_ordered = load_batches(train_ds, microbatch, False)
        means, stds, scaler_count = fit_initial_feature_scalers(model, train_ordered, device)
        np.savez(RUN / "known_train_initial_feature_scalers.npz", **{
            f"{name}_{kind}": value for name, mean, std in zip(VIEWS, means, stds, strict=True)
            for kind, value in (("mean", mean), ("std", std))})
        train_loader = load_batches(train_ds, microbatch, True)
        val_loader = load_batches(val_ds, microbatch, False)
        tf_module = tf_branch.source.load_tf_module()
        tf_opt, tf_scheduler = tf_module.build_optimizer(model.tf_args, model.tf)
        graph_opt = torch.optim.Adam(model.graph.parameters(), lr=1e-3)
        yatc_groups = yatc_branch.source.lrd.param_groups_lrd(
            model.yatc, 0.05, no_weight_decay_list=model.yatc.no_weight_decay(), layer_decay=0.75)
        yatc_opt = torch.optim.AdamW(yatc_groups, lr=5e-4)
        fusion_opt = torch.optim.Adam(list(model.adapters.parameters()) + list(model.head.parameters()), lr=1e-3)
        optimizers = (tf_opt, graph_opt, yatc_opt, fusion_opt)
        grad_scaler = torch.cuda.amp.GradScaler()
        accumulation = EFFECTIVE_BATCH // microbatch
        steps_per_epoch = math.ceil(len(train_ds)/EFFECTIVE_BATCH)
        total_steps = steps_per_epoch * EPOCHS
        config = {"dataset": "USTC-TFC2016", "protocol": "A-2_17class_10pct", "seed": SEED,
                  "classes": classes, "sample_manifest_sha256": info["sample_manifest_sha256"],
                  "cache_sha256": info["cache_sha256"], "official_weight_sha256": guard,
                  "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH, "microbatch": microbatch,
                  "accumulation": accumulation, "steps_per_epoch": steps_per_epoch,
                  "loss": "fused cross-entropy only", "fusion": "three 64D adapters, fixed 1/3, concat, MLP",
                  "feature_normalization": "fixed initial-encoder Known-Train mean/std",
                  "tf_lr": 6e-5, "graph_lr": 1e-3, "yatc_lr": 5e-4, "fusion_lr": 1e-3,
                  "yatc_weight_decay": 0.05, "yatc_layer_decay": 0.75, "yatc_warmup_epochs": 2,
                  "checkpoint_selection": "fused Known Validation Macro-F1",
                  "unknown_samples_used": 0, "test_feature_values_loaded": 0}
        write_json(RUN / "config.json", config)
        best_f1, best_epoch, best_metrics = -1.0, -1, None
        step = 0
        torch.cuda.reset_peak_memory_stats(device)
        with (RUN / "training_history.jsonl").open("x") as history:
            for epoch in range(1, EPOCHS+1):
                model.train()
                total_loss, total_seen = 0.0, 0
                for index, batch in enumerate(train_loader):
                    moved = to_device(batch, device)
                    if index % accumulation == 0:
                        for opt in optimizers:
                            opt.zero_grad(set_to_none=True)
                        set_yatc_lr(yatc_opt, step, total_steps)
                    logits = model(moved)
                    loss = F.cross_entropy(logits, moved[-1])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nonfinite joint loss: epoch={epoch}, batch={index}")
                    group_start = (index // accumulation) * EFFECTIVE_BATCH
                    group_size = min(EFFECTIVE_BATCH, len(train_ds)-group_start)
                    grad_scaler.scale(loss * (len(moved[-1]) / group_size)).backward()
                    total_loss += float(loss.detach()) * len(moved[-1])
                    total_seen += len(moved[-1])
                    if (index+1) % accumulation == 0 or index+1 == len(train_loader):
                        before_scale = grad_scaler.get_scale()
                        for opt in optimizers:
                            grad_scaler.step(opt)
                        grad_scaler.update()
                        if grad_scaler.get_scale() >= before_scale:
                            tf_scheduler.step()
                        step += 1
                logits_val, truth_val = predict(model, val_loader, device)
                val_metrics = metrics(truth_val, logits_val.argmax(1), classes)
                event = {"epoch": epoch, "train_loss": total_loss/total_seen,
                         "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                         "val_weighted_f1": val_metrics["weighted_f1"], "optimizer_steps": step,
                         "elapsed_seconds": time.time()-started,
                         "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
                history.write(json.dumps(event) + "\n")
                history.flush()
                write_json(RUN / "progress.json", event)
                print(json.dumps(event), flush=True)
                if val_metrics["macro_f1"] > best_f1:
                    best_f1, best_epoch, best_metrics = val_metrics["macro_f1"], epoch, val_metrics
                    torch.save({"state_dict": model.state_dict(), "best_epoch": epoch,
                                "validation": val_metrics, "config": config}, RUN / "model_best.pt")
        if step != total_steps or best_epoch < 1:
            raise RuntimeError(f"optimizer step or checkpoint count mismatch: {step}/{total_steps}")
        result = {"status": "PASS", "completed_epochs": EPOCHS, "best_epoch": best_epoch,
                  "best_validation": best_metrics, "optimizer_steps": step, "elapsed_seconds": time.time()-started,
                  "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                  "checkpoint_sha256": sha(RUN / "model_best.pt"),
                  "initial_feature_scaler_sha256": sha(RUN / "known_train_initial_feature_scalers.npz"),
                  "graph_scaler_sha256": sha(RUN / "known_train_graph_scaler.npz"),
                  "test_feature_values_loaded": 0, "unknown_samples_used": 0}
        write_json(RUN / "train_summary.json", result)
        (RUN / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(RUN / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                          "elapsed_seconds": time.time()-started})
        raise


def evaluate() -> None:
    output = RUN / "known_test_evaluation"
    output.mkdir(exist_ok=False)
    try:
        code_guard()
        complete = json.loads((STAGE34 / "completion_verification.json").read_text())
        baseline_path = STAGE34 / "runs/ustc/A-2/known_test_evaluation/results.json"
        baseline = json.loads(baseline_path.read_text())
        if complete["status"] != "PASS" or baseline["status"] != "PASS":
            raise RuntimeError("Stage34 staged baseline not complete and verified")
        train_report = json.loads((RUN / "train_summary.json").read_text())
        if train_report["status"] != "PASS" or not (RUN / "SUCCESS").is_file():
            raise RuntimeError("joint checkpoint not frozen")
        if sha(RUN / "model_best.pt") != train_report["checkpoint_sha256"]:
            raise RuntimeError("joint checkpoint hash changed")
        info = json.loads((ROOT / "preflight.json").read_text())
        if source_guard()["sample_manifest_sha256"] != info["sample_manifest_sha256"]:
            raise RuntimeError("source changed after training")
        classes = info["classes"]
        graph_scale = np.load(RUN / "known_train_graph_scaler.npz", allow_pickle=False)
        feature_scale = np.load(RUN / "known_train_initial_feature_scalers.npz", allow_pickle=False)
        if sha(RUN / "known_train_graph_scaler.npz") != train_report["graph_scaler_sha256"] or \
                sha(RUN / "known_train_initial_feature_scalers.npz") != train_report["initial_feature_scaler_sha256"]:
            raise RuntimeError("Known Train scalers changed")
        test_ds = ThreeViewFlows("known_test", classes, graph_scale["mean"], graph_scale["std"])
        if len(test_ds) != 4333:
            raise RuntimeError("frozen Test sample count changed")
        device = torch.device("cuda:0")
        model = JointThreeView(classes, math.ceil(34665/EFFECTIVE_BATCH)*EPOCHS).to(device)
        checkpoint = torch.load(RUN / "model_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.set_scalers([feature_scale[f"{name}_mean"] for name in VIEWS],
                          [feature_scale[f"{name}_std"] for name in VIEWS])
        logits, truth = predict(model, load_batches(test_ds, 32, False), device)
        pred = logits.argmax(1)
        joint_metrics = metrics(truth, pred, classes)
        if baseline["known_test_samples"] != len(test_ds) or baseline["known_classes"] != classes:
            raise RuntimeError("Stage34 Test class/sample mismatch")
        baseline_csv = STAGE34 / "runs/ustc/A-2/known_test_evaluation/sample_predictions.csv"
        with baseline_csv.open(newline="", encoding="utf-8") as f:
            baseline_rows = list(csv.DictReader(f))
        baseline_by_id = {r["flow_id"]: r for r in baseline_rows}
        if len(baseline_by_id) != len(test_ds) or set(baseline_by_id) != set(test_ds.ids):
            raise RuntimeError("Stage34 and Stage35 Test flow IDs differ")
        staged_pred = np.asarray([classes.index(baseline_by_id[uid]["predicted_class"])
                                  for uid in test_ds.ids], dtype=np.int64)
        for uid, y in zip(test_ds.ids, truth, strict=True):
            if baseline_by_id[uid]["true_class"] != classes[int(y)]:
                raise RuntimeError(f"Test truth mismatch for {uid}")
        staged_metrics = metrics(truth, staged_pred, classes)
        with (output / "sample_predictions.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("flow_id", "true_class", "joint_prediction", "staged_prediction",
                             "joint_correct", "staged_correct"))
            for uid, y, j, s in zip(test_ds.ids, truth, pred, staged_pred, strict=True):
                writer.writerow((uid, classes[int(y)], classes[int(j)], classes[int(s)],
                                 int(j == y), int(s == y)))
        np.save(output / "joint_logits.npy", logits.astype(np.float32), allow_pickle=False)
        np.save(output / "joint_confusion.npy", confusion_matrix(truth, pred,
                labels=np.arange(len(classes))).astype(np.int64), allow_pickle=False)
        np.save(output / "staged_confusion.npy", confusion_matrix(truth, staged_pred,
                labels=np.arange(len(classes))).astype(np.int64), allow_pickle=False)
        result = {"status": "PASS", "dataset": "USTC-TFC2016", "protocol": "A-2_17class_10pct",
                  "sample_manifest_sha256": info["sample_manifest_sha256"], "known_test_samples": len(test_ds),
                  "classes": classes, "joint": joint_metrics, "staged": staged_metrics,
                  "delta_joint_minus_staged": {key: joint_metrics[key]-staged_metrics[key]
                                                for key in ("accuracy", "macro_f1", "weighted_f1")},
                  "joint_only_correct": int(np.sum((pred == truth) & (staged_pred != truth))),
                  "staged_only_correct": int(np.sum((pred != truth) & (staged_pred == truth))),
                  "both_correct": int(np.sum((pred == truth) & (staged_pred == truth))),
                  "both_wrong": int(np.sum((pred != truth) & (staged_pred != truth))),
                  "joint_checkpoint_sha256": train_report["checkpoint_sha256"],
                  "staged_result_sha256": sha(baseline_path),
                  "unknown_samples_loaded": 0, "test_parameter_selection": 0}
        write_json(output / "comparison.json", result)
        (output / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "delta": result["delta_joint_minus_staged"]}), flush=True)
    except BaseException as exc:
        write_json(output / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "smoke", "optimizer-smoke", "train", "evaluate"))
    parser.add_argument("--microbatch", type=int, default=32)
    args = parser.parse_args()
    if args.action in ("smoke", "optimizer-smoke", "train", "evaluate") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.action == "preflight":
        preflight()
    elif args.action == "smoke":
        smoke(args.microbatch)
    elif args.action == "optimizer-smoke":
        optimizer_smoke(args.microbatch)
    elif args.action == "train":
        train(args.microbatch)
    else:
        evaluate()


if __name__ == "__main__":
    main()
