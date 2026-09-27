#!/usr/bin/env python3
"""One frozen E3/YaTC 0.5 probability ensemble on matched Stage20 flows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
STAGE23 = PROJECT / "stage23_closed_set_method_table"
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
STAGE21 = PROJECT / "stage21_coarse_service_ours_e3_benchmark" / "feature_cache"
sys.path.insert(0, str(STAGE23 / "scripts"))
from train_yatc_closed import MFRDataset, TraFormer_YaTC  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def coarse(service: str) -> str:
    return "Communication" if service in {"Chat", "Email", "VoIP"} else service


def historic(path: Path, role: str, encoder: str | None = None) -> dict[str, dict]:
    rows = read_csv(path)
    rows = [r for r in rows if r["role"] == role and (encoder is None or r.get("encoder") == encoder)]
    result = {r["flow_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate historical prediction: {path}")
    return result


def metrics(truth: list[str], pred: list[str], labels: list[str]) -> tuple[dict, list[dict]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=labels, zero_division=0)
    return ({"accuracy": float(accuracy_score(truth, pred)),
             "macro_f1": float(np.mean(f1)),
             "weighted_f1": float(np.average(f1, weights=support))},
            [{"class": name, "precision": float(p), "recall": float(r),
              "f1": float(f), "support": int(n)}
             for name, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)])


@torch.no_grad()
def yatc_logits(model: torch.nn.Module, dataset: MFRDataset, device: torch.device) -> np.ndarray:
    outputs = []
    model.eval()
    for x, _label in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
        with torch.cuda.amp.autocast():
            out = model(x.to(device, non_blocking=True))
        outputs.append(out.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def e3_logits(state: dict, embedding: np.ndarray, device: torch.device) -> np.ndarray:
    weight = state["weight"].to(device)
    bias = state["bias"].to(device)
    pieces = []
    for start in range(0, len(embedding), 512):
        x = torch.from_numpy(embedding[start:start + 512]).float().to(device)
        pieces.append(torch.nn.functional.linear(x, weight, bias).float().cpu().numpy())
    return np.concatenate(pieces, axis=0)


def probabilities(logits: np.ndarray) -> np.ndarray:
    v = logits.astype(np.float64)
    v -= v.max(axis=1, keepdims=True)
    exp = np.exp(v)
    return exp / exp.sum(axis=1, keepdims=True)


def source_paths(dataset: str, seed: int) -> dict[str, Path]:
    e3run = STAGE22 / "runs" / dataset / f"seed{seed}"
    yatc = STAGE23 / "runs" / "yatc_stage20" / dataset / f"seed{seed}_formal"
    cache = STAGE23 / "yatc_stage20_mfr_cache" / dataset
    paths = {"stage20_manifest": STAGE20,
             "stage21_cache": STAGE21 / dataset / "e3_t8_inputs.npz",
             "stage22_config": STAGE22 / "config.json",
             "e3_checkpoint": e3run / "E3_model_best.pt",
             "e3_representations": e3run / "representations.npz",
             "e3_predictions": e3run / "predictions.csv",
             "yatc_checkpoint": yatc / "model_best.pt",
             "yatc_config": yatc / "config.json",
             "yatc_metrics": yatc / "metrics.json",
             "yatc_val_predictions": yatc / "validation_predictions.csv",
             "yatc_test_predictions": yatc / "test_predictions_cuda.csv"}
    for role in ("known_validation", "known_test"):
        paths[f"yatc_{role}_mfr"] = cache / f"{role}_mfr.npy"
        paths[f"yatc_{role}_ids"] = cache / f"{role}_flow_ids.npy"
    return paths


def run(dataset: str, seed: int) -> None:
    start = time.monotonic()
    run_dir = OUT / "runs" / dataset / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source = source_paths(dataset, seed)
    before = {name: sha(path) for name, path in source.items()}
    write_json(run_dir / "source_hashes_before.json", before)
    if before["stage20_manifest"] != "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb":
        raise RuntimeError("Stage20 frozen manifest hash mismatch")
    config = json.loads(source["stage22_config"].read_text())
    services = config["datasets"][dataset]["services"]
    yatc_config = json.loads(source["yatc_config"].read_text())
    yatc_metrics = json.loads(source["yatc_metrics"].read_text())
    if yatc_config["services"] != services or yatc_metrics["checkpoint_sha256"] != before["yatc_checkpoint"]:
        raise RuntimeError("YaTC class order or checkpoint identity mismatch")
    coarse_labels = sorted({coarse(x) for x in services})
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("frozen YaTC inference requires a selected GPU")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(source["yatc_checkpoint"], map_location=device, weights_only=False)
    if checkpoint["services"] != services or checkpoint["epoch"] != yatc_metrics["best_epoch"]:
        raise RuntimeError("YaTC saved selection mismatch")
    model = TraFormer_YaTC(num_classes=len(services), drop_path_rate=0.1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    e3_saved = torch.load(source["e3_checkpoint"], map_location="cpu", weights_only=True)
    if e3_saved["seed"] != seed or e3_saved["input_dim"] != 896:
        raise RuntimeError("E3 checkpoint identity mismatch")
    e3_state = e3_saved["state_dict"]
    packet_cache = np.load(source["stage21_cache"], allow_pickle=False)
    packet_counts = dict(zip(packet_cache["flow_ids"].tolist(), packet_cache["packet_counts"].astype(int).tolist(), strict=True))
    packet_cache.close()

    all_metrics, class_metrics, sample_rows = [], [], []
    logit_pack = {}
    with np.load(source["e3_representations"], allow_pickle=False) as representation:
        for role in ("known_validation", "known_test"):
            yatc_data = MFRDataset(dataset, role, services)
            flow_ids = [row["flow_id"] for row in yatc_data.rows]
            e3_ids = representation[f"{role}_flow_ids"].tolist()
            if flow_ids != e3_ids:
                raise RuntimeError(f"E3/YaTC flow order mismatch: {dataset}/{seed}/{role}")
            true = [row["service_label"] for row in yatc_data.rows]
            y_logits = yatc_logits(model, yatc_data, device)
            e_logits = e3_logits(e3_state, representation[f"{role}_e3_z"], device)
            if y_logits.shape != e_logits.shape or y_logits.shape != (len(flow_ids), len(services)):
                raise RuntimeError("logit shape mismatch")
            e_historical = historic(source["e3_predictions"], role, "E3")
            y_historical = historic(source["yatc_val_predictions"] if role == "known_validation"
                                    else source["yatc_test_predictions"], role)
            if set(flow_ids) != set(e_historical) or set(flow_ids) != set(y_historical):
                raise RuntimeError("historical sample coverage mismatch")
            for i, flow_id in enumerate(flow_ids):
                if true[i] != e_historical[flow_id]["true_service"] or true[i] != y_historical[flow_id]["true_service"]:
                    raise RuntimeError(f"historical true label mismatch: {flow_id}")
                if services[int(e_logits[i].argmax())] != e_historical[flow_id]["predicted_service"]:
                    raise RuntimeError(f"E3 argmax parity failed: {flow_id}")
                if services[int(y_logits[i].argmax())] != y_historical[flow_id]["predicted_service"]:
                    raise RuntimeError(f"YaTC argmax parity failed: {flow_id}")
            ep, yp = probabilities(e_logits), probabilities(y_logits)
            fused = 0.5 * ep + 0.5 * yp
            coarse_index = {name: [i for i, service in enumerate(services) if coarse(service) == name]
                            for name in coarse_labels}
            pred = {}
            for method, probs in (("E3", ep), ("YaTC", yp), ("Fusion", fused)):
                pred[(method, "fine")] = [services[i] for i in probs.argmax(1)]
                cp = np.stack([probs[:, coarse_index[label]].sum(1) for label in coarse_labels], axis=1)
                pred[(method, "coarse")] = [coarse_labels[i] for i in cp.argmax(1)]
            for i, flow_id in enumerate(flow_ids):
                item = {"dataset": dataset, "seed": seed, "role": role, "flow_id": flow_id,
                        "packet_count_capped8": packet_counts[flow_id], "true_fine": true[i],
                        "true_coarse": coarse(true[i]), "retained_ge2": int(packet_counts[flow_id] >= 2)}
                for method in ("E3", "YaTC", "Fusion"):
                    item[f"pred_{method}_fine"] = pred[(method, "fine")][i]
                    item[f"pred_{method}_coarse"] = pred[(method, "coarse")][i]
                sample_rows.append(item)
            for scope in ("all", "ge2"):
                eligible = [i for i, flow_id in enumerate(flow_ids) if scope == "all" or packet_counts[flow_id] >= 2]
                for level, labels in (("fine", services), ("coarse", coarse_labels)):
                    y_true = [true[i] if level == "fine" else coarse(true[i]) for i in eligible]
                    for method in ("E3", "YaTC", "Fusion"):
                        y_pred = [pred[(method, level)][i] for i in eligible]
                        result, per_class = metrics(y_true, y_pred, labels)
                        all_metrics.append({"dataset": dataset, "seed": seed, "role": role,
                                            "scope": scope, "level": level, "method": method,
                                            "samples": len(eligible), **result})
                        class_metrics.extend({"dataset": dataset, "seed": seed, "role": role,
                                              "scope": scope, "level": level, "method": method, **row}
                                             for row in per_class)
            logit_pack[f"{role}_flow_ids"] = np.asarray(flow_ids)
            logit_pack[f"{role}_e3_logits"] = e_logits.astype(np.float32)
            logit_pack[f"{role}_yatc_logits"] = y_logits.astype(np.float32)
    write_csv(run_dir / "run_metrics.csv", all_metrics)
    write_csv(run_dir / "per_class.csv", class_metrics)
    write_csv(run_dir / "sample_predictions.csv", sample_rows)
    np.savez_compressed(run_dir / "frozen_logits.npz", **logit_pack)
    after = {name: sha(path) for name, path in source.items()}
    write_json(run_dir / "source_hashes_after.json", after)
    if after != before:
        raise RuntimeError("source hash changed during frozen inference")
    result = {"status": "PASS", "dataset": dataset, "seed": seed, "source_assets_checked": len(before),
              "validation_samples": sum(row["role"] == "known_validation" for row in sample_rows),
              "test_samples": sum(row["role"] == "known_test" for row in sample_rows),
              "e3_yatc_argmax_parity": "PASS", "unknown_usage": 0, "weight_updates": 0,
              "fusion_rule": "0.5*softmax(E3_logits)+0.5*softmax(YaTC_logits)",
              "runtime_seconds": time.monotonic() - start, "device": str(device)}
    write_json(run_dir / "verification.json", result)
    (run_dir / "SUCCESS").write_text("verified\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    args = parser.parse_args()
    run(args.dataset, args.seed)
