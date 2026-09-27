#!/usr/bin/env python3
"""Known-only Stage24 branch pilot using frozen Stage22 E1 representations."""
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

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "stage17_encoder_recovery_and_open_set_pilot" / "scripts"))
from train_pilot_run import normalize_adjacency, predict_linear, train_fusion  # noqa: E402
from structural_models import make_features, make_model  # noqa: E402

STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
STAGE21 = PROJECT / "stage21_coarse_service_ours_e3_benchmark" / "feature_cache"
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
EXPECTED_MANIFEST_SHA = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
METHODS = ("S1", "G1", "G2", "G2-shuffle")
ROLES = ("known_train", "known_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def scores(truth: np.ndarray, pred: np.ndarray, labels: list[str]) -> tuple[dict, list[dict], np.ndarray]:
    classes = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=classes, zero_division=0)
    rows = [{"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(labels)]
    result = {"accuracy": float(accuracy_score(truth, pred)), "macro_f1": float(f1.mean()),
              "weighted_f1": float(np.average(f1, weights=support))}
    matrix = confusion_matrix(truth, pred, labels=classes)
    return result, rows, matrix


def load_known_data(dataset: str, seed: int, method: str, labels: list[str]):
    with STAGE20.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    rows_by_role = {role: sorted((r for r in rows if r["closed_role"] == role),
                                 key=lambda r: r["flow_id"]) for role in ROLES}
    if sum(len(rows_by_role[role]) for role in ROLES) != (9862 if dataset == "iscx_vpn" else 10064):
        raise RuntimeError("Known Train/Validation size mismatch")
    cache_path = STAGE21 / dataset / "e3_t8_inputs.npz"
    cache = np.load(cache_path, allow_pickle=False)
    pos = {str(flow_id): i for i, flow_id in enumerate(cache["flow_ids"])}
    indices = {role: np.asarray([pos[row["flow_id"]] for row in rows_by_role[role]])
               for role in ROLES}
    runs = STAGE22 / "runs" / dataset / f"seed{seed}"
    old = np.load(runs / "representations.npz", allow_pickle=False)
    label_idx = {name: i for i, name in enumerate(labels)}
    data = {}
    for role in ROLES:
        flow_ids = np.asarray([row["flow_id"] for row in rows_by_role[role]])
        if not np.array_equal(flow_ids, old[f"{role}_flow_ids"]):
            raise RuntimeError(f"Stage22 E1 alignment mismatch: {dataset}/{seed}/{role}")
        truth = np.asarray([label_idx[row["service_label"]] for row in rows_by_role[role]],
                           dtype=np.int64)
        raw_x = np.asarray(cache["fig_x"][indices[role]], dtype=np.float32)
        raw_mask = np.asarray(cache["fig_mask"][indices[role]], dtype=bool)
        graph_adj = np.asarray(cache["fig_adj"][indices[role]], dtype=np.uint8)
        feats = make_features(raw_x, raw_mask, flow_ids, seed,
                              shuffle_bursts=method == "G2-shuffle")
        data[role] = {"flow_ids": flow_ids, "truth": truth, "features": feats,
                      "adj": normalize_adjacency(graph_adj, raw_mask),
                      "e1_z": np.asarray(old[f"{role}_e1_z"], dtype=np.float32),
                      "e1_pred": np.asarray(old[f"{role}_e1_pred"], dtype=np.int64),
                      "e3_pred": np.asarray(old[f"{role}_e3_pred"], dtype=np.int64),
                      "packet_count": np.asarray(cache["packet_counts"][indices[role]], dtype=np.int64)}
        if len(data[role]["e1_z"]) != len(truth) or set(np.unique(truth)) != set(range(len(labels))):
            raise RuntimeError(f"known class coverage mismatch: {dataset}/{role}")
    return data, {"stage20_manifest_sha256": sha256(STAGE20),
                  "stage21_cache_sha256": sha256(cache_path),
                  "stage22_e1_checkpoint_sha256": sha256(runs / "E1_model_best.pt"),
                  "stage22_representations_sha256": sha256(runs / "representations.npz")}


def tensor_data(data: dict) -> dict[str, TensorDataset]:
    train_nodes = data["known_train"]["features"].nodes
    train_mask = data["known_train"]["features"].mask
    node_mean = train_nodes[train_mask].mean(axis=0)
    node_std = train_nodes[train_mask].std(axis=0)
    node_std[node_std < 1e-8] = 1.0
    train_stats = data["known_train"]["features"].stats
    stat_mean = train_stats.mean(axis=0)
    stat_std = train_stats.std(axis=0)
    stat_std[stat_std < 1e-8] = 1.0
    result = {}
    for role in ROLES:
        d = data[role]
        f = d["features"]
        nodes = ((f.nodes - node_mean) / node_std).astype(np.float32)
        nodes[~f.mask] = 0.0
        stats = ((f.stats - stat_mean) / stat_std).astype(np.float32)
        if not np.isfinite(nodes).all() or not np.isfinite(stats).all():
            raise RuntimeError(f"non-finite normalized features: {role}")
        result[role] = TensorDataset(
            torch.from_numpy(nodes), torch.from_numpy(d["adj"]).float(),
            torch.from_numpy(f.mask), torch.from_numpy(stats),
            torch.from_numpy(f.burst_ids), torch.from_numpy(f.burst_count),
            torch.from_numpy(f.burst_order), torch.from_numpy(d["truth"]))
    return result


@torch.no_grad()
def infer(model: nn.Module, ds: TensorDataset, device: torch.device,
          batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, preds = [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        features = tuple(item.to(device, non_blocking=True) for item in batch[:-1])
        logits, z = model(features)
        zs.append(z.detach().cpu().numpy())
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(zs).astype(np.float32), np.concatenate(preds).astype(np.int64)


def run(dataset: str, seed: int, method: str, output: Path, device_name: str) -> dict:
    if method not in METHODS or dataset not in ("iscx_vpn", "iscx_tor"):
        raise ValueError("unsupported preregistered dataset/method")
    if seed not in (2022, 2023):
        raise ValueError("pilot seed must be 2022 or 2023")
    if sha256(STAGE20) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Stage20 frozen manifest changed")
    if not (OUT / "preflight.json").is_file() or json.loads((OUT / "preflight.json").read_text())["status"] != "PASS":
        raise RuntimeError("Stage24 preflight has not passed")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage24 formal pilot requires CUDA")
    if (output / "result.json").exists():
        raise RuntimeError(f"refusing to overwrite completed run: {output}")
    seed_all(seed)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    labels = json.loads((STAGE22 / "config.json").read_text())["datasets"][dataset]["services"]
    start = time.monotonic()
    data, input_hashes = load_known_data(dataset, seed, method, labels)
    tensors = tensor_data(data)
    model = make_model(method, len(labels)).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tensors["known_train"], batch_size=64, shuffle=True,
                        generator=generator, num_workers=0)
    history = []
    best_state = None
    best_epoch = -1
    best_f1 = -1.0
    val_truth = data["known_validation"]["truth"]
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for batch in loader:
            features = tuple(item.to(device, non_blocking=True) for item in batch[:-1])
            truth = batch[-1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _z = model(features)
            loss = loss_fn(logits, truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite branch loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(truth)
        _val_z, val_pred = infer(model, tensors["known_validation"], device)
        val_metrics, _rows, _matrix = scores(val_truth, val_pred, labels)
        row = {"epoch": epoch, "train_loss": total_loss / len(tensors["known_train"]),
               "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
               "val_weighted_f1": val_metrics["weighted_f1"]}
        history.append(row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == 50:
            print(json.dumps({"event": "branch_epoch", "dataset": dataset, "seed": seed,
                              "method": method, **row}), flush=True)
    if best_state is None:
        raise RuntimeError("branch did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "method": method, "seed": seed,
                "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
                "parameter_count": parameter_count, "input_hashes": input_hashes,
                "train_only_normalization": True}, output / "branch_best.pt")
    branch = {role: infer(model, tensors[role], device) for role in ROLES}
    del model
    torch.cuda.empty_cache()

    a = data["known_train"]["e1_z"]
    b = branch["known_train"][0]
    a_mean, a_std = a.mean(axis=0), a.std(axis=0)
    b_mean, b_std = b.mean(axis=0), b.std(axis=0)
    a_std[a_std < 1e-8] = 1.0
    b_std[b_std < 1e-8] = 1.0
    fused = {role: np.concatenate(((data[role]["e1_z"] - a_mean) / a_std,
                                   (branch[role][0] - b_mean) / b_std), axis=1).astype(np.float32)
             for role in ROLES}
    fusion, fusion_history, fusion_epoch = train_fusion(
        fused["known_train"], data["known_train"]["truth"],
        fused["known_validation"], val_truth, seed, device, output / "fusion_best.pt")
    fusion_pred = predict_linear(fusion, fused["known_validation"], device)
    method_pred = {"B0": data["known_validation"]["e1_pred"],
                   "B1": data["known_validation"]["e3_pred"],
                   "branch": branch["known_validation"][1], method: fusion_pred}
    metric_rows = []
    class_rows = []
    confusion_rows = []
    for name, pred in method_pred.items():
        metrics, by_class, matrix = scores(val_truth, pred, labels)
        metric_rows.append({"dataset": dataset, "seed": seed, "method": name,
                            "role": "known_validation", **metrics})
        class_rows.extend({"dataset": dataset, "seed": seed, "method": name, **row}
                          for row in by_class)
        confusion_rows.extend({"dataset": dataset, "seed": seed, "method": name,
                               "true_class": labels[i], "predicted_class": labels[j], "count": int(matrix[i, j])}
                              for i in range(len(labels)) for j in range(len(labels)))
    write_csv(output / "validation_metrics.csv", metric_rows,
              ["dataset", "seed", "method", "role", "accuracy", "macro_f1", "weighted_f1"])
    write_csv(output / "validation_per_class.csv", class_rows,
              ["dataset", "seed", "method", "class", "precision", "recall", "f1", "support"])
    write_csv(output / "validation_confusion.csv", confusion_rows,
              ["dataset", "seed", "method", "true_class", "predicted_class", "count"])
    prediction_rows = []
    for i, flow_id in enumerate(data["known_validation"]["flow_ids"]):
        prediction_rows.append({"dataset": dataset, "seed": seed, "flow_id": str(flow_id),
                                "role": "known_validation", "true_class": labels[int(val_truth[i])],
                                "packet_count": int(data["known_validation"]["packet_count"][i]),
                                **{f"pred_{name}": labels[int(pred[i])] for name, pred in method_pred.items()}})
    write_csv(output / "validation_predictions.csv", prediction_rows,
              ["dataset", "seed", "flow_id", "role", "true_class", "packet_count",
               "pred_B0", "pred_B1", "pred_branch", f"pred_{method}"])
    np.savez_compressed(output / "known_representations.npz",
                        known_train_branch_z=branch["known_train"][0],
                        known_validation_branch_z=branch["known_validation"][0],
                        known_train_fused_z=fused["known_train"],
                        known_validation_fused_z=fused["known_validation"],
                        known_train_flow_ids=data["known_train"]["flow_ids"],
                        known_validation_flow_ids=data["known_validation"]["flow_ids"])
    dump_json(output / "training_history.json", {"branch": history, "fusion": fusion_history})
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - start
    result = {"status": "SUCCESS", "dataset": dataset, "seed": seed, "method": method,
              "claim_scope": "Known Train/Validation closed-set development only",
              "train_samples": len(data["known_train"]["truth"]),
              "val_samples": len(val_truth), "num_classes": len(labels),
              "branch_best_epoch": best_epoch, "fusion_best_epoch": fusion_epoch,
              "branch_best_val_macro_f1": best_f1,
              "validation": {r["method"]: {k: r[k] for k in ("accuracy", "macro_f1", "weighted_f1")}
                             for r in metric_rows},
              "parameter_count": parameter_count, "wall_seconds": elapsed,
              "peak_gpu_memory_allocated_bytes": peak,
              "input_hashes": input_hashes,
              "checkpoint_hashes": {"branch": sha256(output / "branch_best.pt"),
                                    "fusion": sha256(output / "fusion_best.pt")},
              "known_test_feature_usage": 0, "unknown_test_feature_usage": 0,
              "threshold_calibration_usage": 0}
    dump_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (args.output / "manifest.json").is_file():
        raise RuntimeError("run bundle must be initialized before training")
    try:
        result = run(args.dataset, args.seed, args.method, args.output, args.device)
    except Exception:
        (args.output / "FAILURE.txt").write_text(traceback.format_exc(), encoding="utf-8")
        meta = json.loads((args.output / "manifest.json").read_text())
        meta["status"] = "failed"
        meta["limitations"] = ["Preserved traceback in FAILURE.txt; no accepted result for this run."]
        dump_json(args.output / "manifest.json", meta)
        raise
    meta = json.loads((args.output / "manifest.json").read_text())
    meta.update({"status": "success", "configuration": {"dataset": args.dataset, "seed": args.seed,
                  "method": args.method, "branch_epochs": 50, "branch_batch": 64,
                  "branch_optimizer": "Adam lr=1e-3", "fusion_epochs": 30,
                  "fusion_batch": 256, "checkpoint_rule": "Known Validation Macro-F1"},
                 "core_results": result["validation"], "inputs": result["input_hashes"],
                 "limitations": ["Stage20 Test already exposed; this run only used Known Train/Validation.",
                                 "Official TrafficFormer pretraining Unknown-class exposure is unverified."],
                 "next_step": "Aggregate preregistered Known Validation pilot; do not select by Test."})
    dump_json(args.output / "manifest.json", meta)
    (args.output / "RESULTS.md").write_text(
        f"# Stage 24 pilot — {args.dataset} / {args.seed} / {args.method}\n\n"
        "- Status: `success / KNOWN_ONLY_PILOT`\n"
        "- Experiment type: ablation; claim scope: diagnostic.\n\n"
        "## Data and split\n\n"
        f"- Stage20 frozen Service task; train={result['train_samples']}, validation={result['val_samples']}.\n"
        "- Known Test and Unknown Test feature usage: 0/0.\n\n"
        "## Configuration and execution\n\n"
        "- First 8 observed packets; branch 50 epochs, batch 64, Adam LR 1e-3.\n"
        "- Fusion 30 epochs, batch 256, Adam LR 1e-3; both checkpoint rules use Known Validation Macro-F1.\n"
        f"- Branch best epoch {result['branch_best_epoch']}; fusion best epoch {result['fusion_best_epoch']}.\n"
        f"- Branch parameters {result['parameter_count']}; elapsed {result['wall_seconds']:.1f}s; peak GPU allocation {result['peak_gpu_memory_allocated_bytes']} bytes.\n\n"
        "## Core results\n\n"
        f"- B0/E1 Known Validation Macro-F1: {result['validation']['B0']['macro_f1']:.6f}.\n"
        f"- B1/E3 Known Validation Macro-F1: {result['validation']['B1']['macro_f1']:.6f}.\n"
        f"- {args.method} Known Validation Macro-F1: {result['validation'][args.method]['macro_f1']:.6f}.\n\n"
        "## Preserved evidence\n\n"
        "- result.json, training_history.json, validation metrics/per-class/confusion/predictions CSV, Known-only representations, branch/fusion checkpoints and SHA256 in result.json.\n\n"
        "## Limitations\n\n"
        "- Pilot only; Stage20 Test already exposed; no Test-based selection, open-set result or independent-generalization claim.\n"
        "- Official TrafficFormer pretraining exposure to future Unknown classes remains unverified.\n\n"
        "## Conclusion and next step\n\n"
        "- Aggregate all preregistered methods/seeds before any candidate decision.\n",
        encoding="utf-8")
    (args.output / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    print(json.dumps({"event": "run_success", "dataset": args.dataset, "seed": args.seed,
                      "method": args.method, "val_macro_f1": result["validation"][args.method]["macro_f1"]}),
          flush=True)


if __name__ == "__main__":
    main()
