#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage22_common import CONFIG, CLOSED_MANIFEST, OUT, PRETRAINED_MODEL, PROJECT, RUN_ROOT, cache_path, dataset_rows, read_json, sha256_file, write_csv, write_json

STAGE17_SCRIPTS = PROJECT / "stage17_encoder_recovery_and_open_set_pilot" / "scripts"
sys.path.insert(0, str(STAGE17_SCRIPTS))
from train_pilot_run import (
    classification_metrics,
    infer_graph,
    infer_tf,
    make_tf_args,
    normalize_adjacency,
    predict_linear,
    seed_everything,
    tf_forward,
    train_fusion,
    train_graph,
)


def load_tf_module():
    path = PROJECT / "tf_runtime" / "code" / "fine-tuning" / "run_classifier.py"
    spec = importlib.util.spec_from_file_location("stage22_tf_run_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_tf_pretrained(train, val, labels_num: int, seed: int, device: torch.device, output: Path, config: dict):
    seed_everything(seed)
    spec = config["training"]["trafficformer"]
    epochs = int(spec["epochs"])
    batch_size = int(spec["batch_size"])
    steps = math.ceil(len(train[0]) / batch_size) * epochs
    tf_args = make_tf_args(labels_num, steps)
    tf_args.batch_size = batch_size
    module = load_tf_module()
    model = module.Classifier(tf_args)
    state = torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    load_report = {
        "pretrained_path": str(PRETRAINED_MODEL),
        "pretrained_sha256": sha256_file(PRETRAINED_MODEL),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    model = model.to(device)
    optimizer, scheduler = module.build_optimizer(tf_args, model)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train[0]).long(), torch.from_numpy(train[1]).long(), torch.from_numpy(train[2]).long()),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = []
    best = None
    best_f1 = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for src, seg, y in loader:
            src, seg, y = src.to(device), seg.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = tf_forward(model, src, seg)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        _, train_pred = infer_tf(model, train[0], train[1], device)
        _, val_pred = infer_tf(model, val[0], val[1], device)
        train_metrics = classification_metrics(train[2], train_pred)
        val_metrics = classification_metrics(val[2], val_pred)
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        print(json.dumps({"encoder": "E1", **row}), flush=True)
    if best is None:
        raise RuntimeError("TrafficFormer training produced no checkpoint")
    model.load_state_dict(best)
    torch.save({
        "state_dict": best,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "initialization": "official_120k_pretrained",
        "pretrained_load_report": load_report,
        "config": spec,
    }, output)
    return model, history, best_epoch, load_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read_json(CONFIG)
    run = RUN_ROOT / args.dataset / f"seed{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    start = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    if sha256_file(PRETRAINED_MODEL) != config["pretrained_model_sha256"]:
        raise RuntimeError("official pretrained model SHA256 mismatch")
    cache = np.load(cache_path(args.dataset), allow_pickle=False)
    rows = dataset_rows(args.dataset)
    services = list(config["datasets"][args.dataset]["services"])
    service_to_label = {service: index for index, service in enumerate(services)}
    pos = {str(flow_id): index for index, flow_id in enumerate(cache["flow_ids"])}
    by_role = {role: [] for role in ("known_train", "known_validation", "known_test")}
    for row in rows:
        by_role[row["closed_role"]].append(row)
    for role in by_role:
        by_role[role].sort(key=lambda row: row["flow_id"])
    idx = {role: np.asarray([pos[row["flow_id"]] for row in values]) for role, values in by_role.items()}
    labels = {role: np.asarray([service_to_label[row["service_label"]] for row in values], dtype=np.int64) for role, values in by_role.items()}
    if set(labels["known_train"]) != set(range(len(services))):
        raise RuntimeError("training split does not cover every Service")

    tokens = cache["token_ids"]
    segments = cache["segments"]
    graph_x = cache["fig_x"].astype(np.float32)
    graph_mask = cache["fig_mask"]
    train_nodes = graph_x[idx["known_train"]][graph_mask[idx["known_train"]]]
    node_mean = train_nodes.mean(0)
    node_std = train_nodes.std(0)
    node_std[node_std < 1e-8] = 1
    graph_x = (graph_x - node_mean) / node_std
    graph_adj = normalize_adjacency(cache["fig_adj"], graph_mask)

    tf_train = (tokens[idx["known_train"]], segments[idx["known_train"]], labels["known_train"])
    tf_val = (tokens[idx["known_validation"]], segments[idx["known_validation"]], labels["known_validation"])
    graph_train = (graph_x[idx["known_train"]], graph_adj[idx["known_train"]], graph_mask[idx["known_train"]], labels["known_train"])
    graph_val = (graph_x[idx["known_validation"]], graph_adj[idx["known_validation"]], graph_mask[idx["known_validation"]], labels["known_validation"])

    tf_model, tf_history, tf_best, load_report = train_tf_pretrained(
        tf_train, tf_val, len(services), args.seed, device, run / "E1_model_best.pt", config
    )
    e1 = {role: infer_tf(tf_model, tokens[idx[role]], segments[idx[role]], device) for role in idx}
    del tf_model
    torch.cuda.empty_cache()

    graph_model, graph_history, graph_best = train_graph(graph_train, graph_val, len(services), args.seed, device, run / "E2_model_best.pt")
    e2 = {role: infer_graph(graph_model, graph_x[idx[role]], graph_adj[idx[role]], graph_mask[idx[role]], device) for role in idx}
    del graph_model
    torch.cuda.empty_cache()

    tf_mean = e1["known_train"][0].mean(0)
    tf_std = e1["known_train"][0].std(0)
    tf_std[tf_std < 1e-8] = 1
    graph_mean = e2["known_train"][0].mean(0)
    graph_std = e2["known_train"][0].std(0)
    graph_std[graph_std < 1e-8] = 1
    e3_z = {
        role: np.concatenate([
            (e1[role][0] - tf_mean) / tf_std,
            (e2[role][0] - graph_mean) / graph_std,
        ], axis=1).astype(np.float32)
        for role in idx
    }
    fusion, fusion_history, fusion_best = train_fusion(
        e3_z["known_train"], labels["known_train"],
        e3_z["known_validation"], labels["known_validation"],
        args.seed, device, run / "E3_model_best.pt",
    )
    e3_pred = {role: predict_linear(fusion, e3_z[role], device) for role in idx}

    results = []
    prediction_rows = []
    branch_data = {
        "E1": {role: e1[role][1] for role in idx},
        "E2": {role: e2[role][1] for role in idx},
        "E3": e3_pred,
    }
    best_epochs = {"E1": tf_best, "E2": graph_best, "E3": fusion_best}
    for encoder, predictions in branch_data.items():
        validation_metrics = classification_metrics(labels["known_validation"], predictions["known_validation"])
        test_metrics = classification_metrics(labels["known_test"], predictions["known_test"])
        results.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "method": "PRETRAINED-TF-E3-T8" if encoder == "E3" else ("PRETRAINED-TRAFFICFORMER" if encoder == "E1" else "TAGCN-T8"),
            "encoder": encoder,
            "best_epoch": best_epochs[encoder],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_weighted_f1": validation_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        })
        for role in ("known_validation", "known_test"):
            for row, truth, prediction in zip(by_role[role], labels[role], predictions[role]):
                prediction_rows.append({
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "role": role,
                    "encoder": encoder,
                    "flow_id": row["flow_id"],
                    "true_service": services[int(truth)],
                    "predicted_service": services[int(prediction)],
                    "correct": int(truth == prediction),
                })
    write_csv(run / "results.csv", results)
    write_csv(run / "predictions.csv", prediction_rows)
    write_json(run / "training_history.json", {"E1": tf_history, "E2": graph_history, "E3": fusion_history})
    write_json(run / "pretrained_load_report.json", load_report)
    np.savez_compressed(
        run / "representations.npz",
        **{f"{role}_e1_z": e1[role][0] for role in idx},
        **{f"{role}_e2_z": e2[role][0] for role in idx},
        **{f"{role}_e3_z": e3_z[role] for role in idx},
        **{f"{role}_e1_pred": e1[role][1] for role in idx},
        **{f"{role}_e2_pred": e2[role][1] for role in idx},
        **{f"{role}_e3_pred": e3_pred[role] for role in idx},
        **{f"{role}_flow_ids": np.asarray([row["flow_id"] for row in by_role[role]]) for role in idx},
    )
    artifacts = {path.name: sha256_file(path) for path in run.iterdir() if path.is_file()}
    write_json(run / "run_manifest.json", {
        "status": "SUCCESS",
        "dataset": args.dataset,
        "seed": args.seed,
        "method": "PRETRAINED-TF-E3-T8",
        "source_manifest_sha256": sha256_file(CLOSED_MANIFEST),
        "source_cache_sha256": sha256_file(cache_path(args.dataset)),
        "pretrained_model_sha256": sha256_file(PRETRAINED_MODEL),
        "checkpoint_selection": "Known Validation Macro-F1",
        "test_selection_samples": 0,
        "samples": {role: len(values) for role, values in by_role.items()},
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "device": str(device),
        "artifacts": artifacts,
    })
    (run / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "run": str(run), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
