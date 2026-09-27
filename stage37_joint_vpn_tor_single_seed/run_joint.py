#!/usr/bin/env python3
"""Stage20 matched-flow, three-view end-to-end closed-set pilot (seed 2022)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "stage35_joint_vs_staged_training"))
import joint_ustc as joint  # noqa: E402; reuses the verified three-view model/optimizers

MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
INPUTS = PROJECT / "stage21_coarse_service_ours_e3_benchmark/feature_cache"
MFR = PROJECT / "stage23_closed_set_method_table/yatc_stage20_mfr_cache"
CLASSES = {
    "iscx_vpn": ["Communication", "File-Transfer", "P2P", "Streaming"],
    "iscx_tor": ["Browsing", "Communication", "File-Transfer", "P2P", "Streaming"],
}
EXPECTED = {
    "iscx_vpn": (8764, 1098, 1093),
    "iscx_tor": (8946, 1118, 1117),
}
ROLES = ("known_train", "known_validation", "known_test")
FROZEN_MANIFEST_SHA = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def label_of(service: str) -> str:
    return "Communication" if service in ("Chat", "Email", "VoIP") else service


def rows_for(dataset: str, role: str) -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = sorted((r for r in csv.DictReader(handle)
                       if r["dataset"] == dataset and r["closed_role"] == role),
                      key=lambda r: r["flow_id"])
    if not rows or len({r["flow_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"empty or duplicate Stage20 {dataset}/{role}")
    if {label_of(r["service_label"]) for r in rows} != set(CLASSES[dataset]):
        raise RuntimeError(f"missing coarse class in {dataset}/{role}")
    return rows


def guards(dataset: str) -> dict:
    if joint.sha(MANIFEST) != FROZEN_MANIFEST_SHA:
        raise RuntimeError("Stage20 manifest hash changed")
    cache = INPUTS / dataset / "e3_t8_inputs.npz"
    audit_path = MFR / dataset / "cache_audit_trainval.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "PASS" or audit["stage20_manifest_sha256"] != FROZEN_MANIFEST_SHA:
        raise RuntimeError("Stage20 YaTC input audit failed")
    tf_config = json.loads(joint.tf_branch.source.CONFIG.read_text())
    tf_hash = joint.sha(joint.tf_branch.source.PRETRAINED_MODEL)
    yatc_hash = joint.sha(joint.yatc_branch.source.WEIGHT)
    if tf_hash != tf_config["pretrained_model_sha256"] or yatc_hash != joint.yatc_branch.source.WEIGHT_HASH:
        raise RuntimeError("official pretrained weight hash changed")
    return {"stage20_manifest_sha256": FROZEN_MANIFEST_SHA,
            "e3_t8_input_sha256": joint.sha(cache),
            "yatc_trainval_audit_sha256": joint.sha(audit_path),
            "trafficformer_pretrained_sha256": tf_hash,
            "yatc_pretrained_sha256": yatc_hash,
            "joint_model_source_sha256": joint.sha(Path(joint.__file__))}


def graph_scale(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    rows = rows_for(dataset, "known_train")
    with np.load(INPUTS / dataset / "e3_t8_inputs.npz", allow_pickle=False) as cache:
        ids = {str(uid): i for i, uid in enumerate(cache["flow_ids"])}
        positions = [ids[r["flow_id"]] for r in rows]
        graph = cache["fig_x"][positions]
        mask = cache["fig_mask"][positions].astype(bool)
    nodes = graph[mask]
    mean, std = nodes.mean(0), nodes.std(0)
    std[std < 1e-8] = 1
    return mean.astype(np.float32), std.astype(np.float32)


class Flows(Dataset):
    def __init__(self, dataset: str, role: str, mean: np.ndarray, std: np.ndarray):
        self.rows = rows_for(dataset, role)
        self.ids = [r["flow_id"] for r in self.rows]
        self.labels = np.asarray([CLASSES[dataset].index(label_of(r["service_label"]))
                                  for r in self.rows], dtype=np.int64)
        with np.load(INPUTS / dataset / "e3_t8_inputs.npz", allow_pickle=False) as cache:
            cached_ids = [str(uid) for uid in cache["flow_ids"]]
            if len(set(cached_ids)) != len(cached_ids):
                raise RuntimeError("duplicate E3 raw input cache IDs")
            index = {uid: i for i, uid in enumerate(cached_ids)}
            if not set(self.ids).issubset(index):
                raise RuntimeError("Stage20/E3 raw input ID mismatch")
            positions = [index[uid] for uid in self.ids]
            self.tokens = cache["token_ids"][positions].astype(np.int64)
            self.segments = cache["segments"][positions].astype(np.int64)
            raw_graph = cache["fig_x"][positions].astype(np.float32)
            self.mask = cache["fig_mask"][positions].astype(bool)
            raw_adj = cache["fig_adj"][positions]
        self.graph_x = ((raw_graph - mean) / std).astype(np.float32)
        self.graph_x[~self.mask] = 0
        self.graph_adj = joint.tf_branch.source.normalize_adjacency(raw_adj, self.mask)
        root = MFR / dataset
        known = np.load(root / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if known != self.ids:
            raise RuntimeError(f"Stage20/YaTC {dataset}/{role} ID alignment mismatch")
        self.mfr = np.load(root / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
        if self.mfr.shape != (len(self.ids), 40, 40):
            raise RuntimeError("YaTC image shape mismatch")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        image = torch.from_numpy(np.array(self.mfr[i], copy=True)).float().unsqueeze(0)
        image = image.div_(255).sub_(0.5).div_(0.5)
        return (torch.from_numpy(self.tokens[i]), torch.from_numpy(self.segments[i]),
                torch.from_numpy(self.graph_x[i]), torch.from_numpy(self.graph_adj[i]),
                torch.from_numpy(self.mask[i]), image, int(self.labels[i]))


def train(dataset: str) -> None:
    root = HERE / "runs" / dataset
    run = root / "seed2022_joint_e2e20"
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        joint.seed_all()
        source_hashes = guards(dataset)
        mean, std = graph_scale(dataset)
        np.savez(run / "known_train_graph_scaler.npz", mean=mean, std=std)
        train_ds = Flows(dataset, "known_train", mean, std)
        val_ds = Flows(dataset, "known_validation", mean, std)
        if (len(train_ds), len(val_ds)) != EXPECTED[dataset][:2]:
            raise RuntimeError("frozen train/val count drift")
        if set(train_ds.ids) & set(val_ds.ids):
            raise RuntimeError("train/val overlap")
        device = torch.device("cuda:0")
        steps_epoch = math.ceil(len(train_ds) / 64)
        total_steps = steps_epoch * 20
        model = joint.JointThreeView(CLASSES[dataset], total_steps).to(device)
        means, stds, scaler_n = joint.fit_initial_feature_scalers(
            model, joint.load_batches(train_ds, 32, False), device)
        np.savez(run / "known_train_initial_feature_scalers.npz", **{
            f"{name}_{kind}": value
            for name, m, s in zip(joint.VIEWS, means, stds, strict=True)
            for kind, value in (("mean", m), ("std", s))})
        tf_module = joint.tf_branch.source.load_tf_module()
        tf_opt, tf_scheduler = tf_module.build_optimizer(model.tf_args, model.tf)
        graph_opt = torch.optim.Adam(model.graph.parameters(), lr=1e-3)
        yatc_groups = joint.yatc_branch.source.lrd.param_groups_lrd(
            model.yatc, 0.05, no_weight_decay_list=model.yatc.no_weight_decay(), layer_decay=0.75)
        yatc_opt = torch.optim.AdamW(yatc_groups, lr=5e-4)
        fusion_opt = torch.optim.Adam(list(model.adapters.parameters()) + list(model.head.parameters()), lr=1e-3)
        optimizers = (tf_opt, graph_opt, yatc_opt, fusion_opt)
        scaler = torch.cuda.amp.GradScaler()
        config = {"dataset": dataset, "seed": 2022, "epochs": 20, "microbatch": 32,
                  "effective_batch": 64, "classes": CLASSES[dataset],
                  "known_train": len(train_ds), "known_validation": len(val_ds),
                  "known_test_loaded_during_train": 0, "unknown_used": 0,
                  "initial_feature_scaler_samples": scaler_n,
                  "checkpoint_selection": "Known Validation Macro-F1",
                  "model": "Stage35 three-view end-to-end equal-feature fusion",
                  "source_hashes": source_hashes}
        save_json(run / "config.json", config)
        best, best_epoch, best_val = -1.0, 0, None
        step = 0
        train_loader = joint.load_batches(train_ds, 32, True)
        val_loader = joint.load_batches(val_ds, 32, False)
        torch.cuda.reset_peak_memory_stats(device)
        with (run / "training_history.jsonl").open("x") as history:
            for epoch in range(1, 21):
                model.train()
                loss_sum = seen = 0
                for i, batch in enumerate(train_loader):
                    moved = joint.to_device(batch, device)
                    if i % 2 == 0:
                        for opt in optimizers:
                            opt.zero_grad(set_to_none=True)
                        joint.set_yatc_lr(yatc_opt, step, total_steps)
                    loss = F.cross_entropy(model(moved), moved[-1])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nonfinite loss at epoch {epoch}, batch {i}")
                    group_size = min(64, len(train_ds) - (i // 2) * 64)
                    scaler.scale(loss * (len(moved[-1]) / group_size)).backward()
                    loss_sum += float(loss.detach()) * len(moved[-1])
                    seen += len(moved[-1])
                    if (i + 1) % 2 == 0 or i + 1 == len(train_loader):
                        previous = scaler.get_scale()
                        for opt in optimizers:
                            scaler.step(opt)
                        scaler.update()
                        if scaler.get_scale() >= previous:
                            tf_scheduler.step()
                        step += 1
                logits, truth = joint.predict(model, val_loader, device)
                val = joint.metrics(truth, logits.argmax(1), CLASSES[dataset])
                event = {"dataset": dataset, "epoch": epoch, "train_loss": loss_sum / seen,
                         "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                         "val_weighted_f1": val["weighted_f1"], "optimizer_steps": step,
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)}
                history.write(json.dumps(event) + "\n")
                history.flush()
                save_json(run / "progress.json", event)
                print(json.dumps(event), flush=True)
                if val["macro_f1"] > best:
                    best, best_epoch, best_val = val["macro_f1"], epoch, val
                    torch.save({"state_dict": model.state_dict(), "config": config,
                                "best_epoch": best_epoch, "validation": val}, run / "model_best.pt")
        if step != total_steps or best_epoch < 1:
            raise RuntimeError("optimizer steps or checkpoint mismatch")
        result = {"status": "PASS", "best_epoch": best_epoch, "best_validation": best_val,
                  "completed_epochs": 20, "optimizer_steps": step,
                  "checkpoint_sha256": joint.sha(run / "model_best.pt"),
                  "source_hashes": source_hashes, "elapsed_seconds": time.time() - started,
                  "known_test_loaded_during_train": 0, "unknown_used": 0}
        save_json(run / "train_summary.json", result)
        (run / "TRAIN_SUCCESS").write_text("TRAIN_SUCCESS\n")
        print(json.dumps({"dataset": dataset, "status": "TRAIN_PASS", "best_epoch": best_epoch,
                          "val_macro_f1": best}), flush=True)
    except BaseException as exc:
        save_json(run / "TRAIN_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


def evaluate(dataset: str) -> None:
    run = HERE / "runs" / dataset / "seed2022_joint_e2e20"
    output = run / "known_test_evaluation"
    output.mkdir(exist_ok=False)
    try:
        report = json.loads((run / "train_summary.json").read_text())
        if report["status"] != "PASS" or not (run / "TRAIN_SUCCESS").is_file():
            raise RuntimeError("training incomplete")
        if report["source_hashes"] != guards(dataset):
            raise RuntimeError("frozen input/weight/source hashes changed")
        if joint.sha(run / "model_best.pt") != report["checkpoint_sha256"]:
            raise RuntimeError("checkpoint hash changed")
        scales = np.load(run / "known_train_graph_scaler.npz", allow_pickle=False)
        feature_scale = np.load(run / "known_train_initial_feature_scalers.npz", allow_pickle=False)
        test_ds = Flows(dataset, "known_test", scales["mean"], scales["std"])
        if len(test_ds) != EXPECTED[dataset][2]:
            raise RuntimeError("frozen Test count drift")
        train_ids = set(r["flow_id"] for r in rows_for(dataset, "known_train"))
        val_ids = set(r["flow_id"] for r in rows_for(dataset, "known_validation"))
        if set(test_ds.ids) & (train_ids | val_ids):
            raise RuntimeError("Test flow overlap")
        device = torch.device("cuda:0")
        model = joint.JointThreeView(CLASSES[dataset], math.ceil(EXPECTED[dataset][0] / 64) * 20).to(device)
        checkpoint = torch.load(run / "model_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.set_scalers([feature_scale[f"{name}_mean"] for name in joint.VIEWS],
                          [feature_scale[f"{name}_std"] for name in joint.VIEWS])
        logits, truth = joint.predict(model, joint.load_batches(test_ds, 32, False), device)
        pred = logits.argmax(1)
        result = {"status": "PASS", "dataset": dataset, "seed": 2022,
                  "known_test_samples": len(test_ds), "classes": CLASSES[dataset],
                  "known_test": joint.metrics(truth, pred, CLASSES[dataset]),
                  "best_epoch": report["best_epoch"],
                  "checkpoint_sha256": report["checkpoint_sha256"],
                  "source_hashes": report["source_hashes"], "test_used_for_selection": 0}
        save_json(output / "results.json", result)
        np.save(output / "logits.npy", logits.astype(np.float32), allow_pickle=False)
        np.save(output / "confusion_matrix.npy", confusion_matrix(truth, pred,
                labels=np.arange(len(CLASSES[dataset]))), allow_pickle=False)
        with (output / "sample_predictions.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("flow_id", "true_class", "predicted_class", "correct"))
            for uid, y, p in zip(test_ds.ids, truth, pred, strict=True):
                writer.writerow((uid, CLASSES[dataset][int(y)], CLASSES[dataset][int(p)], int(y == p)))
        (output / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"dataset": dataset, "status": "TEST_PASS",
                          "macro_f1": result["known_test"]["macro_f1"]}), flush=True)
    except BaseException as exc:
        save_json(output / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--dataset", choices=tuple(CLASSES), required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        info = guards(args.dataset)
        roles = {role: len(rows_for(args.dataset, role)) for role in ROLES}
        if tuple(roles.values()) != EXPECTED[args.dataset]:
            raise RuntimeError("frozen Stage20 role count drift")
        mean, std = graph_scale(args.dataset)
        train_ds = Flows(args.dataset, "known_train", mean, std)
        val_ds = Flows(args.dataset, "known_validation", mean, std)
        if len(train_ds) != roles["known_train"] or len(val_ds) != roles["known_validation"]:
            raise RuntimeError("input cache role count drift")
        if set(train_ds.ids) & set(val_ds.ids):
            raise RuntimeError("Train/Validation flow overlap")
        print(json.dumps({"status": "PASS", "dataset": args.dataset,
                          "roles": roles, "hashes": info}), flush=True)
    elif args.action == "train":
        train(args.dataset)
    else:
        evaluate(args.dataset)
