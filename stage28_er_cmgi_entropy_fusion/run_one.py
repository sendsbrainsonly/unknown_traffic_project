#!/usr/bin/env python3
"""Stage 28 Known-only, paper-inspired two-view 2x2 diagnostic.

This is an adaptation of ER-CMGI mechanisms, not the paper's original model.
No Test or Unknown key is opened by this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from preflight import OUT, manifest_labels, load_pair, sha256, sources

VARIANTS = ("P0_A0_equal", "P1_A0_entropy", "P2_A1_equal", "P3_A1_entropy")
LATENT = 64
EPOCHS = 30
BATCH = 256


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scores(y: np.ndarray, predicted: np.ndarray, nclasses: int) -> dict[str, float]:
    _, _, f1, support = precision_recall_fscore_support(
        y, predicted, labels=np.arange(nclasses), zero_division=0,
    )
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.average(f1, weights=support))}


def zscore(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train.mean(0).astype(np.float32)
    std = train.std(0).astype(np.float32)
    zeros = std < 1e-8
    std[zeros] = 1.0
    output = ((train - mean) / std).astype(np.float32), ((val - mean) / std).astype(np.float32)
    if not all(np.isfinite(arr).all() for arr in output):
        raise RuntimeError("nonfinite standardized feature")
    return *output, {"mean": mean, "std": std, "zero_variance": int(zeros.sum())}


class ViewEncoder(nn.Module):
    def __init__(self, input_dim: int, nclasses: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        self.mean = nn.Linear(128, LATENT)
        self.logvar = nn.Linear(128, LATENT)
        self.classifier = nn.Linear(LATENT, nclasses)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.mean(hidden), self.logvar(hidden).clamp(-8.0, 4.0)


class AdapterPair(nn.Module):
    def __init__(self, nclasses: int, generative: bool):
        super().__init__()
        self.content = ViewEncoder(960, nclasses)
        self.behavior = ViewEncoder(128, nclasses)
        self.generative = generative
        if generative:
            self.c_to_b = nn.Sequential(nn.Linear(LATENT, 128), nn.ReLU(), nn.Linear(128, LATENT))
            self.b_to_c = nn.Sequential(nn.Linear(LATENT, 128), nn.ReLU(), nn.Linear(128, LATENT))

    def train_loss(self, content: torch.Tensor, behavior: torch.Tensor, label: torch.Tensor,
                   target_content: torch.Tensor | None, target_behavior: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
        mc, lc = self.content(content)
        mb, lb = self.behavior(behavior)
        zc = mc + torch.randn_like(mc) * torch.exp(0.5 * lc)
        zb = mb + torch.randn_like(mb) * torch.exp(0.5 * lb)
        ce = 0.5 * (F.cross_entropy(self.content.classifier(zc), label) +
                    F.cross_entropy(self.behavior.classifier(zb), label))
        kl = 0.5 * (-0.5 * (1 + lc - mc.square() - lc.exp()).sum(1).mean() / LATENT -
                    0.5 * (1 + lb - mb.square() - lb.exp()).sum(1).mean() / LATENT)
        gen = torch.zeros((), device=content.device)
        if self.generative:
            assert target_content is not None and target_behavior is not None
            gen = 0.5 * (F.mse_loss(self.c_to_b(mc), target_behavior) +
                         F.mse_loss(self.b_to_c(mb), target_content))
        return ce + 0.05 * kl + 0.1 * gen, {"ce": float(ce.detach()), "kl": float(kl.detach()),
                                             "cross_mse": float(gen.detach())}


@torch.no_grad()
def encode(model: AdapterPair, content: np.ndarray, behavior: np.ndarray,
           device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    chunks = [[], [], []]
    for start in range(0, len(content), 512):
        c = torch.from_numpy(content[start:start + 512]).to(device)
        b = torch.from_numpy(behavior[start:start + 512]).to(device)
        mc, lc = model.content(c)
        mb, lb = model.behavior(b)
        h = 0.5 * np.log(2 * np.pi * np.e) + 0.5 * torch.stack([lc.mean(1), lb.mean(1)], dim=1)
        for target, value in zip(chunks, (mc, mb, h), strict=True):
            target.append(value.cpu().numpy().astype(np.float32))
    return tuple(np.concatenate(part) for part in chunks)  # type: ignore[return-value]


def fit_adapter(name: str, train: dict, val: dict, pca: dict[str, np.ndarray],
                seed: int, device: torch.device, run: Path, services: list[str]) -> tuple[dict, dict]:
    seed_all(seed)
    model = AdapterPair(len(services), generative=name == "A1").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(*(torch.from_numpy(a) for a in
        (train["content"], train["behavior"], train["labels"],
         pca["train_content"], pca["train_behavior"])))
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history: list[dict] = []
    best_score, best_epoch, best_state = -1.0, -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        aggregate = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "cross_mse": 0.0}
        for content, behavior, label, target_c, target_b in loader:
            content, behavior, label = content.to(device), behavior.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.train_loss(content, behavior, label,
                                            target_c.to(device) if model.generative else None,
                                            target_b.to(device) if model.generative else None)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} nonfinite epoch {epoch}")
            loss.backward()
            optimizer.step()
            for key, value in {"loss": float(loss.detach()), **parts}.items():
                aggregate[key] += value * len(label)
        model.eval()
        with torch.no_grad():
            predictions = [[], []]
            for start in range(0, len(val["labels"]), 512):
                c = torch.from_numpy(val["content"][start:start + 512]).to(device)
                b = torch.from_numpy(val["behavior"][start:start + 512]).to(device)
                for dest, encoder, data in zip(predictions, (model.content, model.behavior), (c, b), strict=True):
                    mu, _ = encoder(data)
                    dest.extend(encoder.classifier(mu).argmax(1).cpu().numpy().tolist())
            f1s = [scores(val["labels"], np.asarray(pred), len(services))["macro_f1"] for pred in predictions]
        mean_f1 = float(np.mean(f1s))
        history.append({"adapter": name, "epoch": epoch,
                        **{f"train_{key}": value / len(train["labels"]) for key, value in aggregate.items()},
                        "val_content_macro_f1": f1s[0], "val_behavior_macro_f1": f1s[1],
                        "val_mean_macro_f1": mean_f1})
        if mean_f1 > best_score:
            best_score, best_epoch = mean_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(json.dumps({"phase": name, "epoch": epoch, "val_mean_macro_f1": mean_f1}), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "adapter": name, "seed": seed,
                "best_epoch": best_epoch, "best_val_mean_macro_f1": best_score,
                "epochs_completed": EPOCHS, "selection": "Known Validation mean unimodal Macro-F1"},
               run / f"{name}_adapter_best.pt")
    encoded = {role: encode(model, item["content"], item["behavior"], device)
               for role, item in (("known_train", train), ("known_validation", val))}
    return {"history": history, "best_epoch": best_epoch, "best_f1": best_score,
            "checkpoint_sha256": sha256(run / f"{name}_adapter_best.pt")}, encoded


class Fusion(nn.Module):
    def __init__(self, nclasses: int, entropy: bool, median: np.ndarray, mad: np.ndarray):
        super().__init__()
        self.entropy = entropy
        self.register_buffer("median", torch.from_numpy(median.astype(np.float32)))
        self.register_buffer("mad", torch.from_numpy(mad.astype(np.float32)))
        self.mix_logits = nn.Parameter(torch.zeros(2)) if entropy else None
        self.fuser = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(128, 128), nn.ReLU())
        self.classifier = nn.Linear(128, nclasses)

    def forward(self, mc: torch.Tensor, mb: torch.Tensor, h: torch.Tensor,
                force_equal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.entropy and not force_equal:
            u = F.softplus((h - self.median) / (self.mad + 1e-6)) + 0.1
            high = u / u.sum(1, keepdim=True)
            inv = u.reciprocal()
            low = inv / inv.sum(1, keepdim=True)
            lam = 0.1 + 0.8 * torch.sigmoid(self.mix_logits)
            raw = lam * high + (1 - lam) * low
            weights = raw / raw.sum(1, keepdim=True)
        else:
            weights = torch.full((len(mc), 2), 0.5, device=mc.device, dtype=mc.dtype)
        fused = self.fuser(torch.cat((mc * weights[:, :1], mb * weights[:, 1:]), dim=1))
        return self.classifier(fused), weights


@torch.no_grad()
def infer(model: Fusion, encoded: tuple[np.ndarray, np.ndarray, np.ndarray],
          device: torch.device, force_equal: bool = False, entropy_override: np.ndarray | None = None,
          batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, weights = [], []
    mc, mb, h = encoded
    if entropy_override is not None:
        h = entropy_override
    for start in range(0, len(mc), batch_size):
        args = [torch.from_numpy(a[start:start + batch_size]).to(device) for a in (mc, mb, h)]
        values, w = model(*args, force_equal=force_equal)
        logits.append(values.cpu().numpy())
        weights.append(w.cpu().numpy())
    return np.concatenate(logits), np.concatenate(weights)


def fit_fusion(variant: str, encoded: dict, train_y: np.ndarray, val_y: np.ndarray,
               seed: int, device: torch.device, run: Path, nclasses: int) -> tuple[dict, np.ndarray, np.ndarray, Fusion]:
    seed_all(seed)
    train = encoded["known_train"]
    val = encoded["known_validation"]
    median = np.median(train[2], axis=0)
    mad = np.median(np.abs(train[2] - median), axis=0)
    if not np.isfinite(median).all() or not np.isfinite(mad).all():
        raise RuntimeError("invalid Known Train entropy calibration")
    model = Fusion(nclasses, variant.endswith("entropy"), median, mad).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in (*train, train_y))),
                        batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    history = []
    best_f1, best_epoch, best_state = -1.0, -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, grad_sum, steps = 0.0, 0.0, 0
        for mc, mb, h, label in loader:
            mc, mb, h, label = (a.to(device) for a in (mc, mb, h, label))
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(mc, mb, h)
            loss = F.cross_entropy(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{variant} nonfinite epoch {epoch}")
            loss.backward()
            grad_sum += float(torch.linalg.vector_norm(torch.stack([
                p.grad.norm() for p in model.parameters() if p.grad is not None])).detach())
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label)
            steps += 1
        val_logits, val_weights = infer(model, val, device)
        metric = scores(val_y, val_logits.argmax(1), nclasses)
        lam = (0.1 + 0.8 * torch.sigmoid(model.mix_logits)).detach().cpu().numpy().tolist() if model.entropy else [0.5, 0.5]
        history.append({"variant": variant, "epoch": epoch, "train_loss": loss_sum / len(train_y),
                        "val_accuracy": metric["accuracy"], "val_macro_f1": metric["macro_f1"],
                        "val_weighted_f1": metric["weighted_f1"], "lambda_content": lam[0],
                        "lambda_behavior": lam[1], "weight_content_mean": float(val_weights[:, 0].mean()),
                        "weight_behavior_mean": float(val_weights[:, 1].mean()),
                        "weight_collapse_ratio": float((val_weights.max(1) > 0.95).mean()),
                        "gradient_norm_mean": grad_sum / max(steps, 1)})
        if metric["macro_f1"] > best_f1:
            best_f1, best_epoch = metric["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(json.dumps({"phase": variant, "epoch": epoch, "val_macro_f1": metric["macro_f1"]}), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "variant": variant, "seed": seed,
                "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                "epochs_completed": EPOCHS, "selection": "Known Validation fine Macro-F1",
                "known_train_entropy_median": median.tolist(), "known_train_entropy_mad": mad.tolist()},
               run / f"{variant}_best.pt")
    logits, weights = infer(model, val, device)
    return {"history": history, "best_epoch": best_epoch,
            "checkpoint_sha256": sha256(run / f"{variant}_best.pt"),
            "entropy_median": median.tolist(), "entropy_mad": mad.tolist()}, logits, weights, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--encoder-seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--training-seed", type=int, choices=range(2022, 2027), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("no selected CUDA GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run / "manifest.json").is_file() or (run / "training_history.csv").exists():
        raise RuntimeError(f"run bundle missing or already used: {run}")
    started = time.monotonic()
    source_before = {name: sha256(path) for name, path in sources(args.dataset, args.encoder_seed).items()}
    save_json(run / "source_hashes_before.json", source_before)
    expected = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    if source_before != expected[f"{args.dataset}/{args.encoder_seed}"]:
        raise RuntimeError("frozen source hash differs from Stage28 preflight")
    pair = load_pair(args.dataset, args.encoder_seed, manifest_labels()[args.dataset])
    train_raw, val_raw = (pair["roles"][role] for role in ("known_train", "known_validation"))
    content_train, content_val, content_norm = zscore(train_raw["semantic"], val_raw["semantic"])
    behavior_train, behavior_val, behavior_norm = zscore(train_raw["behavioral"], val_raw["behavioral"])
    np.savez_compressed(run / "known_train_normalization.npz", content_mean=content_norm["mean"],
                        content_std=content_norm["std"], behavior_mean=behavior_norm["mean"],
                        behavior_std=behavior_norm["std"])
    pca_targets = {}
    for name, tr, va in (("content", content_train, content_val),
                         ("behavior", behavior_train, behavior_val)):
        pca = PCA(n_components=LATENT, svd_solver="randomized", random_state=args.training_seed)
        pca_targets[f"train_{name}"] = pca.fit_transform(tr).astype(np.float32)
        pca_targets[f"val_{name}"] = pca.transform(va).astype(np.float32)
        np.savez_compressed(run / f"pca_{name}_train_only.npz", mean=pca.mean_, components=pca.components_,
                            explained_variance=pca.explained_variance_, train_samples=len(tr))
    train = {"content": content_train, "behavior": behavior_train, "labels": train_raw["labels"]}
    val = {"content": content_val, "behavior": behavior_val, "labels": val_raw["labels"]}
    encoded, adapter_data, history = {}, {}, []
    for adapter in ("A0", "A1"):
        adapter_data[adapter], encoded[adapter] = fit_adapter(
            adapter, train, val, pca_targets, args.training_seed, device, run, pair["services"])
        history.extend(adapter_data[adapter]["history"])
    metrics, per_class, predictions, weights_rows = [], [], [], []
    logits_to_save: dict[str, np.ndarray] = {}
    fusion_data = {}
    for variant in VARIANTS:
        adapter = "A0" if variant.startswith(("P0", "P1")) else "A1"
        result, logits, weights, model = fit_fusion(
            variant, encoded[adapter], train["labels"], val["labels"],
            args.training_seed, device, run, len(pair["services"]))
        fusion_data[variant] = {k: v for k, v in result.items() if k != "history"}
        history.extend(result["history"])
        logits_to_save[variant] = logits
        predicted = logits.argmax(1)
        metric = scores(val["labels"], predicted, len(pair["services"]))
        metrics.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                        "training_seed": args.training_seed, "variant": variant,
                        "role": "known_validation", "samples": len(predicted),
                        "best_epoch": result["best_epoch"], **metric})
        precision, recall, f1, support = precision_recall_fscore_support(
            val["labels"], predicted, labels=np.arange(len(pair["services"])), zero_division=0)
        for i, service in enumerate(pair["services"]):
            per_class.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                              "training_seed": args.training_seed, "variant": variant,
                              "class": service, "precision": precision[i], "recall": recall[i],
                              "f1": f1[i], "support": support[i]})
        for flow_id, true, pred, w, h in zip(val_raw["flow_ids"], val["labels"], predicted,
                                             weights, encoded[adapter]["known_validation"][2], strict=True):
            predictions.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                                "training_seed": args.training_seed, "variant": variant,
                                "flow_id": flow_id, "true_service": pair["services"][true],
                                "predicted_service": pair["services"][pred], "correct": int(true == pred)})
            weights_rows.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                                 "training_seed": args.training_seed, "variant": variant,
                                 "flow_id": flow_id, "true_service": pair["services"][true],
                                 "correct": int(true == pred), "H_content": float(h[0]),
                                 "H_behavior": float(h[1]), "w_content": float(w[0]),
                                 "w_behavior": float(w[1])})
        if variant == "P3_A1_entropy":
            val_encoded = encoded[adapter]["known_validation"]
            equal_logits, _ = infer(model, val_encoded, device, force_equal=True)
            shuffle = np.random.default_rng(args.training_seed + 991).permutation(len(val["labels"]))
            shuffle_logits, _ = infer(model, val_encoded, device,
                                      entropy_override=val_encoded[2][shuffle])
            batch_logits, _ = infer(model, val_encoded, device, batch_size=17)
            if not np.allclose(batch_logits, logits, atol=1e-5, rtol=1e-5):
                raise RuntimeError("batch-invariance failed")
            counterfactual = {"main": metric,
                              "forced_equal": scores(val["labels"], equal_logits.argmax(1), len(pair["services"])),
                              "shuffled_entropy": scores(val["labels"], shuffle_logits.argmax(1), len(pair["services"])),
                              "batch_invariance_max_abs": float(np.max(np.abs(batch_logits - logits))),
                              "shuffled_entropy_seed": args.training_seed + 991}
            logits_to_save["P3_forced_equal"] = equal_logits
            logits_to_save["P3_shuffled_entropy"] = shuffle_logits
        print(json.dumps({"variant": variant, "best_epoch": result["best_epoch"],
                          "val_macro_f1": metric["macro_f1"]}), flush=True)
    save_csv(run / "training_history.csv", history)
    save_csv(run / "run_metrics.csv", metrics)
    save_csv(run / "per_class.csv", per_class)
    save_csv(run / "known_validation_predictions.csv", predictions)
    save_csv(run / "known_validation_entropy_weights.csv", weights_rows)
    np.savez_compressed(run / "known_validation_logits.npz", flow_ids=val_raw["flow_ids"],
                        labels=val["labels"], **logits_to_save)
    save_json(run / "counterfactual.json", counterfactual)
    source_after = {name: sha256(path) for name, path in sources(args.dataset, args.encoder_seed).items()}
    save_json(run / "source_hashes_after.json", source_after)
    if source_before != source_after:
        raise RuntimeError("frozen source hash changed")
    verification = {"status": "PASS", "dataset": args.dataset, "encoder_seed": args.encoder_seed,
                    "training_seed": args.training_seed, "train_samples": len(train["labels"]),
                    "validation_samples": len(val["labels"]), "test_samples_loaded": 0,
                    "unknown_samples_loaded": 0, "pca_fit_role": "known_train",
                    "standardization_fit_role": "known_train", "entropy_calibration_role": "known_train",
                    "checkpoint_selection_role": "known_validation", "frozen_hashes_unchanged": True,
                    "adapter": {a: {k: v for k, v in row.items() if k != "history"}
                                for a, row in adapter_data.items()},
                    "fusion": fusion_data, "counterfactual": counterfactual,
                    "runtime_seconds": time.monotonic() - started}
    save_json(run / "verification.json", verification)
    (run / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status": "SUCCESS", "dataset": args.dataset,
                      "encoder_seed": args.encoder_seed, "training_seed": args.training_seed,
                      "metrics": metrics, "runtime_seconds": verification["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
