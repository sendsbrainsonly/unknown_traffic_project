#!/usr/bin/env python3
"""Frozen YaTC 192-D features plus Stage22 E3 branches on one Stage20 protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
S22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
S23 = PROJECT / "stage23_closed_set_method_table"
S20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
sys.path.insert(0, str(S23 / "scripts"))
from train_yatc_closed import MFRDataset, TraFormer_YaTC  # noqa: E402

ROLES = ("known_train", "known_validation", "known_test")
VARIANTS = ("F1_LinearConcat", "F2_EqualProjected", "F3_FeatureGate")
FROZEN_S20_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coarse(name: str) -> str:
    return "Communication" if name in {"Chat", "Email", "VoIP"} else name


def source_paths(dataset: str, seed: int) -> dict[str, Path]:
    e = S22 / "runs" / dataset / f"seed{seed}"
    y = S23 / "runs" / "yatc_stage20" / dataset / f"seed{seed}_formal"
    c = S23 / "yatc_stage20_mfr_cache" / dataset
    paths = {
        "stage20_manifest": S20,
        "stage22_config": S22 / "config.json",
        "e3_representations": e / "representations.npz",
        "e3_checkpoint": e / "E3_model_best.pt",
        "e3_predictions": e / "predictions.csv",
        "yatc_checkpoint": y / "model_best.pt",
        "yatc_config": y / "config.json",
        "yatc_metrics": y / "metrics.json",
        "yatc_validation_predictions": y / "validation_predictions.csv",
        "yatc_test_predictions": y / "test_predictions_cuda.csv",
    }
    for role in ROLES:
        paths[f"{role}_mfr"] = c / f"{role}_mfr.npy"
        paths[f"{role}_flow_ids"] = c / f"{role}_flow_ids.npy"
    return paths


class FeatureHead(nn.Module):
    def __init__(self, variant: str, classes: int):
        super().__init__()
        self.variant = variant
        if variant == VARIANTS[0]:
            self.classifier = nn.Linear(1088, classes)
        else:
            self.projections = nn.ModuleList([nn.Linear(d, 128) for d in (768, 128, 192)])
            self.classifier = nn.Linear(128, classes)
            if variant == VARIANTS[2]:
                self.gate = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.variant == VARIANTS[0]:
            return self.classifier(x), None
        parts = (x[:, :768], x[:, 768:896], x[:, 896:])
        projected = [torch.relu(proj(part)) for proj, part in zip(self.projections, parts)]
        stack = torch.stack(projected, dim=1)
        if self.variant == VARIANTS[2]:
            weights = torch.softmax(self.gate(torch.cat(projected, dim=1)), dim=1)
        else:
            weights = torch.full((len(x), 3), 1 / 3, device=x.device, dtype=x.dtype)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused), weights


def seeded(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_head(model: FeatureHead, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    logits, weights = [], []
    for start in range(0, len(x), 512):
        y, w = model(torch.from_numpy(x[start:start + 512]).to(device))
        logits.append(y.float().cpu().numpy())
        if w is not None:
            weights.append(w.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(weights) if weights else None


def metric_rows(
    dataset: str, seed: int, role: str, method: str, truth: list[str],
    predicted: list[str], services: list[str],
) -> tuple[list[dict], list[dict]]:
    results, classes = [], []
    for level in ("fine", "coarse"):
        names = services if level == "fine" else sorted({coarse(s) for s in services})
        actual = truth if level == "fine" else [coarse(s) for s in truth]
        chosen = predicted if level == "fine" else [coarse(s) for s in predicted]
        p, r, f, n = precision_recall_fscore_support(actual, chosen, labels=names, zero_division=0)
        results.append({
            "dataset": dataset, "seed": seed, "role": role, "level": level, "method": method,
            "samples": len(actual), "accuracy": float(accuracy_score(actual, chosen)),
            "macro_f1": float(np.mean(f)), "weighted_f1": float(np.average(f, weights=n)),
        })
        classes.extend({
            "dataset": dataset, "seed": seed, "role": role, "level": level, "method": method,
            "class": name, "precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(ni),
        } for name, pi, ri, fi, ni in zip(names, p, r, f, n, strict=True))
    return results, classes


def fit_head(
    variant: str, train_x: np.ndarray, train_y: np.ndarray,
    val_x: np.ndarray, val_y: np.ndarray, seed: int,
    device: torch.device, checkpoint: Path,
) -> tuple[FeatureHead, list[dict], int]:
    seeded(seed)
    model = FeatureHead(variant, int(train_y.max()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=256, shuffle=True, generator=generator, num_workers=0,
    )
    history, best_state, best_epoch, best_f1 = [], None, -1, -1.0
    for epoch in range(1, 31):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{variant}: nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
        val_logits, _ = predict_head(model, val_x, device)
        val_pred = val_logits.argmax(1)
        f1 = precision_recall_fscore_support(
            val_y, val_pred, labels=np.arange(model.classifier.out_features), zero_division=0,
        )[2].mean()
        row = {"variant": variant, "epoch": epoch, "train_loss": loss_sum / len(train_y),
               "val_accuracy": float(np.mean(val_pred == val_y)), "val_macro_f1": float(f1)}
        history.append(row)
        if f1 > best_f1:
            best_f1, best_epoch = float(f1), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"{variant}: no checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "variant": variant, "seed": seed,
                "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
                "training": {"epochs_completed": 30, "batch_size": 256, "optimizer": "Adam",
                             "learning_rate": 1e-3, "weight_decay": 0, "selection": "Known Val fine Macro-F1"}},
               checkpoint)
    return model, history, best_epoch


@torch.no_grad()
def extract_yatc(model: nn.Module, dataset: MFRDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features, logits = [], []
    for x, _ in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
        with torch.cuda.amp.autocast():
            f = model.forward_features(x.to(device, non_blocking=True))
            y = model.head(f)
            if not features:
                direct = model(x.to(device, non_blocking=True))
                if not torch.allclose(y.float(), direct.float(), rtol=1e-4, atol=1e-4):
                    raise RuntimeError("YaTC penultimate head does not replay frozen forward")
        if f.ndim != 2 or f.shape[1] != 192 or not torch.isfinite(f).all():
            raise RuntimeError("YaTC feature shape/nonfinite")
        features.append(f.float().cpu().numpy())
        logits.append(y.float().cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def historical_prediction(path: Path, role: str, encoder: str | None = None) -> dict[str, str]:
    rows = [r for r in read_csv(path) if r["role"] == role and (encoder is None or r.get("encoder") == encoder)]
    result = {r["flow_id"]: r["predicted_service"] for r in rows}
    if len(rows) != len(result):
        raise RuntimeError(f"duplicate historical predictions: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    args = parser.parse_args()
    run = OUT / "runs" / args.dataset / f"seed{args.seed}"
    if not (run / "manifest.json").is_file() or (run / "SUCCESS").exists() or (run / "features.npz").exists():
        raise RuntimeError(f"run bundle missing or already used: {run}")
    started = time.monotonic()
    paths = source_paths(args.dataset, args.seed)
    before = {name: sha(path) for name, path in paths.items()}
    write_json(run / "source_hashes_before.json", before)
    if before["stage20_manifest"] != FROZEN_S20_HASH:
        raise RuntimeError("Stage20 frozen manifest mismatch")
    config = json.loads(paths["stage22_config"].read_text())
    services = list(config["datasets"][args.dataset]["services"])
    yatc_config = json.loads(paths["yatc_config"].read_text())
    yatc_metrics = json.loads(paths["yatc_metrics"].read_text())
    if yatc_config["services"] != services or yatc_metrics["checkpoint_sha256"] != before["yatc_checkpoint"]:
        raise RuntimeError("YaTC class order/checkpoint hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU selector did not assign a CUDA device")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    checkpoint = torch.load(paths["yatc_checkpoint"], map_location=device, weights_only=False)
    if checkpoint["services"] != services or checkpoint["epoch"] != yatc_metrics["best_epoch"]:
        raise RuntimeError("YaTC best selection mismatch")
    model = TraFormer_YaTC(num_classes=len(services), drop_path_rate=0.1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    e3_state = torch.load(paths["e3_checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
    if e3_state["weight"].shape != (len(services), 896):
        raise RuntimeError("E3 frozen head shape mismatch")
    frozen_features: dict[str, np.ndarray] = {}
    frozen_logits: dict[str, np.ndarray] = {}
    roles: dict[str, dict] = {}

    def load_role(role: str) -> None:
        ydata = MFRDataset(args.dataset, role, services)
        ids = np.asarray([r["flow_id"] for r in ydata.rows], dtype=str)
        with np.load(paths["e3_representations"], allow_pickle=False) as rep:
            eids = rep[f"{role}_flow_ids"].astype(str)
            e3 = rep[f"{role}_e3_z"].astype(np.float32)
        if not np.array_equal(ids, eids) or e3.shape != (len(ids), 896):
            raise RuntimeError(f"{role}: E3/YaTC flow alignment or shape mismatch")
        labels = np.asarray(ydata.labels, dtype=np.int64)
        feat, ylogits = extract_yatc(model, ydata, device)
        e3logits = (e3 @ e3_state["weight"].numpy().T + e3_state["bias"].numpy()).astype(np.float32)
        if not np.isfinite(feat).all() or not np.isfinite(e3).all():
            raise RuntimeError(f"{role}: nonfinite features")
        if role != "known_train":
            e_hist = historical_prediction(paths["e3_predictions"], role, "E3")
            y_hist = historical_prediction(
                paths["yatc_validation_predictions"] if role == "known_validation" else paths["yatc_test_predictions"],
                role,
            )
            if len(e_hist) != len(ids) or len(y_hist) != len(ids):
                raise RuntimeError(f"{role}: historical sample count mismatch")
            for i, flow_id in enumerate(ids):
                if services[int(e3logits[i].argmax())] != e_hist[flow_id]:
                    raise RuntimeError(f"{role}: E3 historical parity failed: {flow_id}")
                if services[int(ylogits[i].argmax())] != y_hist[flow_id]:
                    raise RuntimeError(f"{role}: YaTC historical parity failed: {flow_id}")
        roles[role] = {"ids": ids, "labels": labels, "e3": e3}
        frozen_features[role] = feat.astype(np.float32)
        frozen_logits[f"{role}_E3"] = e3logits
        frozen_logits[f"{role}_YaTC"] = ylogits.astype(np.float32)

    # No Known Test values are loaded before all heads have been selected.
    for role in ("known_train", "known_validation"):
        load_role(role)
    y_mean = frozen_features["known_train"].mean(0)
    y_std = frozen_features["known_train"].std(0)
    y_std[y_std < 1e-8] = 1.0
    x = {role: np.concatenate([roles[role]["e3"], (frozen_features[role] - y_mean) / y_std], axis=1).astype(np.float32)
         for role in ("known_train", "known_validation")}
    models: dict[str, FeatureHead] = {}
    best_epochs: dict[str, int] = {}
    histories: list[dict] = []
    for variant in VARIANTS:
        fitted, history, best_epoch = fit_head(
            variant, x["known_train"], roles["known_train"]["labels"],
            x["known_validation"], roles["known_validation"]["labels"],
            args.seed, device, run / f"{variant}_best.pt",
        )
        models[variant], best_epochs[variant] = fitted, best_epoch
        histories.extend(history)
        print(json.dumps({"dataset": args.dataset, "seed": args.seed, "variant": variant,
                          "best_epoch": best_epoch, "best_val_macro_f1": max(h["val_macro_f1"] for h in history)}), flush=True)
    # Post-selection only.
    load_role("known_test")
    x["known_test"] = np.concatenate(
        [roles["known_test"]["e3"], (frozen_features["known_test"] - y_mean) / y_std], axis=1,
    ).astype(np.float32)
    np.savez_compressed(run / "features.npz", y_mean=y_mean, y_std=y_std,
                        **{f"{r}_flow_ids": roles[r]["ids"] for r in ROLES},
                        **{f"{r}_yatc": frozen_features[r] for r in ROLES})
    metrics, per_class, predictions, gates = [], [], [], []
    all_logits: dict[str, np.ndarray] = dict(frozen_logits)
    for role in ROLES:
        for variant, fitted in models.items():
            values, weights = predict_head(fitted, x[role], device)
            all_logits[f"{role}_{variant}"] = values
            if weights is not None and variant == VARIANTS[2]:
                for flow_id, w in zip(roles[role]["ids"], weights, strict=True):
                    gates.append({"dataset": args.dataset, "seed": args.seed, "role": role, "flow_id": flow_id,
                                  "weight_trafficformer": float(w[0]), "weight_fig": float(w[1]),
                                  "weight_yatc": float(w[2])})
        truth = [services[i] for i in roles[role]["labels"]]
        for method in ("E3", "YaTC", *VARIANTS):
            pred = [services[i] for i in all_logits[f"{role}_{method}"].argmax(1)]
            m, c = metric_rows(args.dataset, args.seed, role, method, truth, pred, services)
            metrics.extend(m)
            per_class.extend(c)
            predictions.extend({"dataset": args.dataset, "seed": args.seed, "role": role, "flow_id": fid,
                                "true_service": actual, "method": method, "predicted_service": chosen,
                                "correct": int(actual == chosen), "true_coarse": coarse(actual),
                                "predicted_coarse": coarse(chosen), "coarse_correct": int(coarse(actual) == coarse(chosen))}
                               for fid, actual, chosen in zip(roles[role]["ids"], truth, pred, strict=True))
    np.savez_compressed(run / "logits.npz", **all_logits)
    write_csv(run / "training_history.csv", histories)
    write_csv(run / "run_metrics.csv", metrics)
    write_csv(run / "per_class.csv", per_class)
    write_csv(run / "sample_predictions.csv", predictions)
    write_csv(run / "gate_weights.csv", gates)
    after = {name: sha(path) for name, path in paths.items()}
    write_json(run / "source_hashes_after.json", after)
    if before != after:
        raise RuntimeError("frozen source changed during run")
    verification = {
        "status": "PASS", "dataset": args.dataset, "seed": args.seed,
        "samples": {r: len(roles[r]["ids"]) for r in ROLES},
        "e3_yatc_val_test_historical_parity": "PASS",
        "source_hashes_unchanged": True, "encoder_weight_updates": 0, "unknown_usage": 0,
        "test_loaded_after_head_selection": True, "best_epochs": best_epochs,
        "checkpoints": {v: sha(run / f"{v}_best.pt") for v in VARIANTS},
        "runtime_seconds": time.monotonic() - started,
        "physical_gpu_local_id": 0,
    }
    write_json(run / "verification.json", verification)
    (run / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status": "SUCCESS", **verification}), flush=True)


if __name__ == "__main__":
    main()
