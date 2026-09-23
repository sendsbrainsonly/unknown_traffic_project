#!/usr/bin/env python3
"""Train all preregistered E4 variants for one frozen Stage17 protocol/seed."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from e4_model import E4Encoder
from stage18_common import (
    ACCEPTANCE_TARGETS,
    CACHE,
    CACHE_SHA256,
    OUT,
    SCORES,
    SEEDS,
    SERVICES,
    VARIANT_ORDER,
    apply_robust_scaler,
    fit_robust_scaler,
    load_frozen_protocol,
    percentile,
    protocol_id,
    seed_everything,
    select_variant_channels,
    sha256_file,
    variant_spec,
    write_csv,
    write_json,
)


def classification_metrics(y: np.ndarray, pred: np.ndarray, labels_num: int = 5) -> dict:
    labels = list(range(labels_num))
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": [
            {"local_label": i, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
            for i in labels
        ],
    }


@torch.no_grad()
def infer(model: E4Encoder, x: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int = 512):
    model.eval()
    logits_all, embedding_all = [], []
    for start in range(0, len(x), batch_size):
        tx = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        tm = torch.as_tensor(mask[start : start + batch_size], dtype=torch.bool, device=device)
        logits, embedding = model(tx, tm)
        logits_all.append(logits.cpu().numpy())
        embedding_all.append(embedding.cpu().numpy())
    logits = np.concatenate(logits_all).astype(np.float32)
    embedding = np.concatenate(embedding_all).astype(np.float32)
    return logits, embedding, logits.argmax(1).astype(np.int64)


def train_model(
    variant: str,
    x_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
    checkpoint: Path,
    scaler: dict,
    label_names: dict[int, str],
):
    seed_everything(seed)
    spec = variant_spec(variant)
    model = E4Encoder(
        input_dim=x_train.shape[-1],
        labels_num=5,
        embedding_dim=128,
        multi_scale=bool(spec["multi_scale"]),
        recurrent=bool(spec["recurrent"]),
        dropout=0.2,
    ).to(device)
    counts = np.bincount(y_train, minlength=5).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    class_weight /= class_weight.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(mask_train).bool(),
            torch.from_numpy(y_train).long(),
        ),
        batch_size=128,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = []
    best_state = None
    best_epoch = 0
    best_key = (-1.0, -1.0)
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        seen = 0
        train_true, train_pred = [], []
        for bx, bm, by in loader:
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bx, bm)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(by)
            seen += len(by)
            train_true.append(by.detach().cpu().numpy())
            train_pred.append(logits.argmax(1).detach().cpu().numpy())
        val_logits, _, val_pred = infer(model, x_val, mask_val, device)
        del val_logits
        train_true_np = np.concatenate(train_true)
        train_pred_np = np.concatenate(train_pred)
        train_metrics = classification_metrics(train_true_np, train_pred_np)
        val_metrics = classification_metrics(y_val, val_pred)
        history.append({
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(seen, 1),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        })
        key = (val_metrics["macro_f1"], val_metrics["accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        scheduler.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == 100:
            print(json.dumps({"variant": variant, "epoch": epoch, "val_macro_f1": val_metrics["macro_f1"], "lr": optimizer.param_groups[0]["lr"]}), flush=True)
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "variant": variant,
        "variant_spec": spec,
        "input_dim": int(x_train.shape[-1]),
        "labels_num": 5,
        "embedding_dim": 128,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_key[0],
        "best_val_accuracy": best_key[1],
        "feature_scaler": scaler,
        "label_names": {str(k): v for k, v in label_names.items()},
        "training_config": {
            "epochs": 100,
            "batch_size": 128,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler_milestones": [50, 80],
            "scheduler_gamma": 0.1,
            "loss": "inverse_sqrt_frequency_weighted_cross_entropy",
            "class_weight": class_weight.tolist(),
            "gradient_clip_norm": 5.0,
            "selection": "Known Validation Macro-F1 then Accuracy",
        },
        "strict_unknown_free": True,
        "unknown_training_samples": 0,
        "unknown_validation_samples": 0,
        "known_test_selection_samples": 0,
    }, checkpoint)
    return model, history, best_epoch, best_key


def fit_scores(train_embedding: np.ndarray, train_y: np.ndarray):
    mean = train_embedding.mean(0)
    std = train_embedding.std(0)
    std[std < 1e-8] = 1.0
    z = (train_embedding - mean) / std
    centroids = np.stack([z[train_y == label].mean(0) for label in range(5)]).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32), centroids


def calculate_scores(logits: np.ndarray, embedding: np.ndarray, embedding_mean: np.ndarray, embedding_std: np.ndarray, centroids: np.ndarray):
    shifted = logits - logits.max(1, keepdims=True)
    exp = np.exp(shifted)
    probability = exp / exp.sum(1, keepdims=True)
    msp = 1.0 - probability.max(1)
    energy = -np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) - logits.max(1)
    z = (embedding - embedding_mean) / embedding_std
    distance = np.square(z[:, None, :] - centroids[None, :, :]).sum(2).min(1)
    return {
        "feature_distance": distance.astype(np.float64),
        "msp": msp.astype(np.float64),
        "energy": energy.astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unknown-service", choices=SERVICES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--variants", default=",".join(VARIANT_ORDER))
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if any(item not in VARIANT_ORDER for item in variants):
        raise ValueError(f"invalid variants: {variants}")
    device = torch.device("cuda:0")
    run_dir = OUT / "runs" / protocol_id(args.unknown_service) / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    frozen = load_frozen_protocol(args.unknown_service)
    indices, labels, rows = frozen["indices"], frozen["labels"], frozen["rows"]
    scaler = fit_robust_scaler(frozen["raw_x"][indices["known_train"]], frozen["mask"][indices["known_train"]])
    scaled_all = apply_robust_scaler(frozen["raw_x"], frozen["mask"], scaler)
    label_names = {}
    for row in rows["known_train"]:
        label_names[int(row["local_label"])] = row["service_label"]
    if sorted(label_names) != list(range(5)):
        raise RuntimeError(f"invalid local label map: {label_names}")

    closed_rows, open_rows, per_class_rows, prediction_rows, run_artifacts = [], [], [], [], []
    for variant in variants:
        spec = variant_spec(variant)
        selected_all = select_variant_channels(scaled_all, str(spec["features"]))
        x_train = selected_all[indices["known_train"]]
        x_val = selected_all[indices["known_validation"]]
        mask_train = frozen["mask"][indices["known_train"]]
        mask_val = frozen["mask"][indices["known_validation"]]
        checkpoint = run_dir / f"{variant}_best.pt"
        model, history, best_epoch, best_key = train_model(
            variant, x_train, mask_train, labels["known_train"], x_val, mask_val, labels["known_validation"],
            args.seed, device, checkpoint, scaler, label_names,
        )
        history_path = run_dir / f"{variant}_history.csv"
        write_csv(history_path, history)
        outputs = {}
        for role in ("known_train", "known_validation", "known_test", "unknown_test"):
            role_idx = indices[role]
            outputs[role] = infer(model, selected_all[role_idx], frozen["mask"][role_idx], device)
        train_logits, train_embedding, train_pred = outputs["known_train"]
        del train_logits, train_pred
        embedding_mean, embedding_std, centroids = fit_scores(train_embedding, labels["known_train"])
        score_by_role = {
            role: calculate_scores(outputs[role][0], outputs[role][1], embedding_mean, embedding_std, centroids)
            for role in ("known_validation", "known_test", "unknown_test")
        }
        closed = classification_metrics(labels["known_test"], outputs["known_test"][2])
        closed_rows.append({
            "protocol_id": protocol_id(args.unknown_service),
            "unknown_service": args.unknown_service,
            "seed": args.seed,
            "variant": variant,
            "feature_mode": spec["features"],
            "multi_scale": int(bool(spec["multi_scale"])),
            "recurrent": int(bool(spec["recurrent"])),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_key[0],
            "best_val_accuracy": best_key[1],
            "known_test_accuracy": closed["accuracy"],
            "known_test_macro_f1": closed["macro_f1"],
            "known_test_weighted_f1": closed["weighted_f1"],
            "checkpoint_path": str(checkpoint.relative_to(OUT)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "runtime_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "status": "SUCCESS",
        })
        for metric in closed["per_class"]:
            per_class_rows.append({
                "protocol_id": protocol_id(args.unknown_service),
                "unknown_service": args.unknown_service,
                "seed": args.seed,
                "variant": variant,
                "local_label": metric["local_label"],
                "service": label_names[metric["local_label"]],
                **{key: metric[key] for key in ("precision", "recall", "f1", "support")},
            })
        thresholds = {score: {str(target): percentile(score_by_role["known_validation"][score], target) for target in ACCEPTANCE_TARGETS} for score in SCORES}
        binary = np.r_[np.zeros(len(indices["known_test"]), dtype=np.int64), np.ones(len(indices["unknown_test"]), dtype=np.int64)]
        for score_name in SCORES:
            known_score = score_by_role["known_test"][score_name]
            unknown_score = score_by_role["unknown_test"][score_name]
            combined_score = np.r_[known_score, unknown_score]
            auroc = float(roc_auc_score(binary, combined_score))
            auprc = float(average_precision_score(binary, combined_score))
            for target in ACCEPTANCE_TARGETS:
                threshold = thresholds[score_name][str(target)]
                open_rows.append({
                    "protocol_id": protocol_id(args.unknown_service),
                    "unknown_service": args.unknown_service,
                    "seed": args.seed,
                    "variant": variant,
                    "score": score_name,
                    "known_acceptance_target": target,
                    "threshold": threshold,
                    "val_known_acceptance": float(np.mean(score_by_role["known_validation"][score_name] < threshold)),
                    "known_test_acceptance": float(np.mean(known_score < threshold)),
                    "known_test_frr": float(np.mean(known_score >= threshold)),
                    "ufar": float(np.mean(unknown_score < threshold)),
                    "auroc": auroc,
                    "auprc": auprc,
                    "known_samples": len(known_score),
                    "unknown_samples": len(unknown_score),
                    "status": "SUCCESS",
                })
        for role in ("known_validation", "known_test", "unknown_test"):
            logits, embedding, pred = outputs[role]
            role_scores = score_by_role[role]
            for i, row in enumerate(rows[role]):
                item = {
                    "protocol_id": protocol_id(args.unknown_service),
                    "unknown_service": args.unknown_service,
                    "seed": args.seed,
                    "variant": variant,
                    "role": role,
                    "flow_id": row["flow_id"],
                    "true_service": row["service_label"],
                    "true_local_label": row["local_label"],
                    "predicted_local_label": int(pred[i]),
                    "max_softmax_probability": float(1.0 - role_scores["msp"][i]),
                    "feature_distance": float(role_scores["feature_distance"][i]),
                    "msp": float(role_scores["msp"][i]),
                    "energy": float(role_scores["energy"][i]),
                }
                for score_name in SCORES:
                    for target in ACCEPTANCE_TARGETS:
                        suffix = str(int(target * 100))
                        threshold = thresholds[score_name][str(target)]
                        item[f"{score_name}_threshold_{suffix}"] = threshold
                        item[f"{score_name}_rejected_{suffix}"] = int(role_scores[score_name][i] >= threshold)
                prediction_rows.append(item)
        embedding_path = run_dir / f"{variant}_embeddings.npz"
        np.savez_compressed(
            embedding_path,
            embedding_mean=embedding_mean,
            embedding_std=embedding_std,
            centroids=centroids,
            **{f"{role}_embedding": outputs[role][1] for role in outputs},
            **{f"{role}_logits": outputs[role][0] for role in outputs},
            **{f"{role}_pred": outputs[role][2] for role in outputs},
            **{f"{role}_flow_ids": np.asarray([row["flow_id"] for row in rows[role]]) for role in outputs},
        )
        run_artifacts.extend([checkpoint, history_path, embedding_path])
        del model
        torch.cuda.empty_cache()

    write_csv(run_dir / "closed_set_results.csv", closed_rows)
    write_csv(run_dir / "open_set_results.csv", open_rows)
    write_csv(run_dir / "per_class_results.csv", per_class_rows)
    write_csv(run_dir / "predictions.csv", prediction_rows)
    run_artifacts.extend([run_dir / "closed_set_results.csv", run_dir / "open_set_results.csv", run_dir / "per_class_results.csv", run_dir / "predictions.csv"])
    hashes = {path.name: sha256_file(path) for path in run_artifacts}
    write_json(run_dir / "run_manifest.json", {
        "status": "SUCCESS",
        "protocol_id": protocol_id(args.unknown_service),
        "unknown_service": args.unknown_service,
        "seed": args.seed,
        "variants": list(variants),
        "strict_unknown_free": True,
        "unknown_training_samples": 0,
        "unknown_validation_samples": 0,
        "unknown_support_samples": 0,
        "unknown_normalization_samples": 0,
        "unknown_threshold_samples": 0,
        "known_test_selection_samples": 0,
        "cache_sha256": CACHE_SHA256,
        "cache_current_sha256": sha256_file(CACHE),
        "artifacts": hashes,
        "runtime_seconds": time.time() - started,
    })
    (run_dir / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "run": str(run_dir), "variants": list(variants), "closed_rows": len(closed_rows), "open_rows": len(open_rows)}), flush=True)


if __name__ == "__main__":
    main()
