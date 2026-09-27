#!/usr/bin/env python3
"""Known-only E3-native two-branch entropy fusion, no YaTC."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from preflight import OUT, labels, load_pair, paths, sha

EPOCHS = 30
BATCH = 256
LATENT = 64


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path, records):
    if not records:
        raise RuntimeError(f"empty records: {path}")
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(truth, prediction, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(classes), zero_division=0)
    result = {"accuracy": float(np.mean(truth == prediction)),
              "macro_f1": float(f1.mean()),
              "weighted_f1": float(np.average(f1, weights=support))}
    return result, (precision, recall, f1, support)


class View(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        self.mu = nn.Linear(128, LATENT)
        self.logvar = nn.Linear(128, LATENT)
        self.classifier = nn.Linear(LATENT, classes)

    def forward(self, x):
        h = self.backbone(x)
        return self.mu(h), self.logvar(h).clamp(-8, 4)


class Pair(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.trafficformer = View(768, classes)
        self.graph = View(128, classes)

    def training_loss(self, x_t, x_g, y):
        mt, lt = self.trafficformer(x_t)
        mg, lg = self.graph(x_g)
        zt = mt + torch.randn_like(mt) * torch.exp(lt * 0.5)
        zg = mg + torch.randn_like(mg) * torch.exp(lg * 0.5)
        ce = 0.5 * (F.cross_entropy(self.trafficformer.classifier(zt), y) +
                    F.cross_entropy(self.graph.classifier(zg), y))
        kl_t = -0.5 * (1 + lt - mt.square() - lt.exp()).sum(1).mean() / LATENT
        kl_g = -0.5 * (1 + lg - mg.square() - lg.exp()).sum(1).mean() / LATENT
        kl = 0.5 * (kl_t + kl_g)
        return ce + 0.05 * kl, ce.detach(), kl.detach()


@torch.no_grad()
def encode(model, data, device):
    model.eval()
    mu_t, mu_g, entropy, pred_t, pred_g = [], [], [], [], []
    for start in range(0, len(data["labels"]), 512):
        t = torch.from_numpy(data["trafficformer"][start:start + 512]).to(device)
        g = torch.from_numpy(data["graph"][start:start + 512]).to(device)
        mt, lt = model.trafficformer(t)
        mg, lg = model.graph(g)
        h = 0.5 * np.log(2 * np.pi * np.e) + 0.5 * torch.stack((lt.mean(1), lg.mean(1)), dim=1)
        mu_t.append(mt.cpu().numpy()); mu_g.append(mg.cpu().numpy())
        entropy.append(h.cpu().numpy())
        pred_t.append(model.trafficformer.classifier(mt).argmax(1).cpu().numpy())
        pred_g.append(model.graph.classifier(mg).argmax(1).cpu().numpy())
    return (np.concatenate(mu_t).astype(np.float32), np.concatenate(mu_g).astype(np.float32),
            np.concatenate(entropy).astype(np.float32), np.concatenate(pred_t), np.concatenate(pred_g))


def train_pair(train, val, classes, seed, device, run):
    seeded(seed)
    model = Pair(classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(train[key]) for key in
        ("trafficformer", "graph", "labels"))), batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, best_epoch, state = [], -1.0, -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total = ce_total = kl_total = 0.0
        for t, g, y in loader:
            t, g, y = t.to(device), g.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, kl = model.training_loss(t, g, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite adapter loss epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(y)
            ce_total += float(ce) * len(y)
            kl_total += float(kl) * len(y)
        encoded = encode(model, val, device)
        ft = metrics(val["labels"], encoded[3], classes)[0]["macro_f1"]
        fg = metrics(val["labels"], encoded[4], classes)[0]["macro_f1"]
        score = (ft + fg) / 2
        history.append({"phase": "adapter", "epoch": epoch, "train_loss": loss_total / len(train["labels"]),
                        "train_ce": ce_total / len(train["labels"]), "train_kl": kl_total / len(train["labels"]),
                        "val_tf_macro_f1": ft, "val_graph_macro_f1": fg, "val_mean_macro_f1": score})
        if score > best:
            best, best_epoch = score, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"phase": "adapter", "epoch": epoch, "val_mean_macro_f1": score}), flush=True)
    assert state is not None
    model.load_state_dict(state)
    torch.save({"state_dict": state, "best_epoch": best_epoch, "seed": seed,
                "epochs_completed": EPOCHS, "selection": "Known Val mean branch Macro-F1"},
               run / "adapter_best.pt")
    return model, history, best_epoch


class Fusion(nn.Module):
    def __init__(self, classes, dynamic, median, mad):
        super().__init__()
        self.dynamic = dynamic
        self.register_buffer("median", torch.as_tensor(median, dtype=torch.float32))
        self.register_buffer("mad", torch.as_tensor(mad, dtype=torch.float32))
        self.shared_lambda_logit = nn.Parameter(torch.zeros(())) if dynamic else None
        self.fuser = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(128, 128), nn.ReLU())
        self.classifier = nn.Linear(128, classes)

    def forward(self, mt, mg, h, force_equal=False):
        if self.dynamic and not force_equal:
            u = F.softplus((h - self.median) / (self.mad + 1e-6)) + 0.1
            high = u / u.sum(1, keepdim=True)
            inverse = u.reciprocal()
            low = inverse / inverse.sum(1, keepdim=True)
            lam = 0.1 + 0.8 * torch.sigmoid(self.shared_lambda_logit)
            weights = lam * high + (1 - lam) * low
        else:
            weights = torch.full((len(mt), 2), 0.5, dtype=mt.dtype, device=mt.device)
        z = torch.cat((mt * weights[:, :1], mg * weights[:, 1:]), dim=1)
        return self.classifier(self.fuser(z)), weights


@torch.no_grad()
def infer(model, encoded, device, force_equal=False, h_override=None, batch=512):
    model.eval()
    mt, mg, h = encoded[:3]
    if h_override is not None:
        h = h_override
    scores, weights = [], []
    for start in range(0, len(mt), batch):
        tensors = [torch.from_numpy(a[start:start + batch]).to(device) for a in (mt, mg, h)]
        y, w = model(*tensors, force_equal=force_equal)
        scores.append(y.cpu().numpy()); weights.append(w.cpu().numpy())
    return np.concatenate(scores), np.concatenate(weights)


def train_fusion(name, train, val, train_y, val_y, classes, seed, device, run):
    seeded(seed)
    center = np.median(train[2], axis=0)
    mad = np.median(np.abs(train[2] - center), axis=0)
    model = Fusion(classes, dynamic=name == "N1_shared_lambda_entropy", median=center, mad=mad).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in (*train[:3], train_y))),
                        batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, best_epoch, state = [], -1.0, -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for mt, mg, h, y in loader:
            mt, mg, h, y = (a.to(device) for a in (mt, mg, h, y))
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(mt, mg, h)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {name} loss epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(y)
        val_logits, w = infer(model, val, device)
        result = metrics(val_y, val_logits.argmax(1), classes)[0]
        lam = float((0.1 + 0.8 * torch.sigmoid(model.shared_lambda_logit)).detach()) if model.dynamic else 0.5
        history.append({"phase": name, "epoch": epoch, "train_loss": loss_total / len(train_y),
                        **{f"val_{key}": value for key, value in result.items()},
                        "shared_lambda": lam, "w_trafficformer_mean": float(w[:, 0].mean()),
                        "w_trafficformer_std": float(w[:, 0].std()),
                        "weight_collapse_ratio": float((w.max(1) > 0.95).mean())})
        if result["macro_f1"] > best:
            best, best_epoch = result["macro_f1"], epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"phase": name, "epoch": epoch, "val_macro_f1": result["macro_f1"]}), flush=True)
    assert state is not None
    model.load_state_dict(state)
    torch.save({"state_dict": state, "name": name, "best_epoch": best_epoch,
                "seed": seed, "epochs_completed": EPOCHS,
                "entropy_median": center.tolist(), "entropy_mad": mad.tolist(),
                "selection": "Known Val fine Service Macro-F1"}, run / f"{name}_best.pt")
    logits, w = infer(model, val, device)
    return model, history, best_epoch, logits, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--encoder-seed", type=int, choices=(2022, 2023), required=True)
    p.add_argument("--training-seed", type=int, choices=range(2022, 2027), required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("no selected CUDA GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run / "manifest.json").is_file() or (run / "SUCCESS").exists():
        raise RuntimeError("run bundle missing or already completed")
    start = time.monotonic()
    source_before = {name: sha(path) for name, path in paths(args.dataset, args.encoder_seed).items()}
    expected = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    write_json(run / "source_hashes_before.json", source_before)
    if source_before != expected[f"{args.dataset}/{args.encoder_seed}"]:
        raise RuntimeError("source hash differs from preflight")
    pair = load_pair(args.dataset, args.encoder_seed, labels()[args.dataset])
    train, val = (pair["roles"][role] for role in ("known_train", "known_validation"))
    classes = len(pair["services"])
    model, history, adapter_epoch = train_pair(train, val, classes, args.training_seed, device, run)
    train_encoded, val_encoded = encode(model, train, device), encode(model, val, device)
    output = []
    per_class = []
    predictions = []
    weights = []
    original = val["e3_original_pred"]
    original_metric, original_pc = metrics(val["labels"], original, classes)
    output.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                   "training_seed": args.training_seed, "method": "E3_original", "best_epoch": -1,
                   "samples": len(original), **original_metric})
    logits_store = {}
    heads = {}
    for name in ("N0_equal", "N1_shared_lambda_entropy"):
        head, curve, epoch, logits, w = train_fusion(name, train_encoded, val_encoded,
            train["labels"], val["labels"], classes, args.training_seed, device, run)
        history.extend(curve)
        heads[name] = head
        logits_store[name] = logits
        result, _ = metrics(val["labels"], logits.argmax(1), classes)
        output.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                       "training_seed": args.training_seed, "method": name,
                       "best_epoch": epoch, "samples": len(val["labels"]), **result})
        if name == "N1_shared_lambda_entropy":
            equal_logits, _ = infer(head, val_encoded, device, force_equal=True)
            permutation = np.random.default_rng(args.training_seed + 991).permutation(len(val["labels"]))
            shuffled_logits, _ = infer(head, val_encoded, device, h_override=val_encoded[2][permutation])
            small_batch_logits, _ = infer(head, val_encoded, device, batch=17)
            difference = float(np.max(np.abs(small_batch_logits - logits)))
            if difference > 1e-5:
                raise RuntimeError(f"batch invariance failed: {difference}")
            counterfactual = {"main": result,
                              "forced_equal": metrics(val["labels"], equal_logits.argmax(1), classes)[0],
                              "shuffled_entropy": metrics(val["labels"], shuffled_logits.argmax(1), classes)[0],
                              "max_batch_abs_logit_difference": difference,
                              "permutation_seed": args.training_seed + 991}
            logits_store["N1_forced_equal"] = equal_logits
            logits_store["N1_shuffled_entropy"] = shuffled_logits
        print(json.dumps({"method": name, "best_epoch": epoch, "val_macro_f1": result["macro_f1"]}), flush=True)
    for name, predicted, w in (("E3_original", original, np.full((len(original), 2), np.nan)),
                               *[(name, logits_store[name].argmax(1),
                                  infer(heads[name], val_encoded, device)[1])
                                 for name in ("N0_equal", "N1_shared_lambda_entropy")]):
        _, pc = metrics(val["labels"], predicted, classes)
        for i, service in enumerate(pair["services"]):
            per_class.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                              "training_seed": args.training_seed, "method": name, "class": service,
                              "precision": float(pc[0][i]), "recall": float(pc[1][i]),
                              "f1": float(pc[2][i]), "support": int(pc[3][i])})
        for fid, actual, chosen, weight, entropy in zip(val["flow_ids"], val["labels"], predicted,
                                                         w, val_encoded[2], strict=True):
            predictions.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                                "training_seed": args.training_seed, "method": name,
                                "flow_id": str(fid), "true_service": pair["services"][int(actual)],
                                "predicted_service": pair["services"][int(chosen)],
                                "correct": int(actual == chosen)})
            if name != "E3_original":
                weights.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                                "training_seed": args.training_seed, "method": name,
                                "flow_id": str(fid), "class": pair["services"][int(actual)],
                                "correct": int(actual == chosen),
                                "H_trafficformer": float(entropy[0]), "H_graph": float(entropy[1]),
                                "w_trafficformer": float(weight[0]), "w_graph": float(weight[1])})
    write_csv(run / "training_history.csv", history)
    write_csv(run / "run_metrics.csv", output)
    write_csv(run / "per_class.csv", per_class)
    write_csv(run / "known_validation_predictions.csv", predictions)
    write_csv(run / "known_validation_entropy_weights.csv", weights)
    np.savez_compressed(run / "known_validation_logits.npz", flow_ids=val["flow_ids"],
                        labels=val["labels"], E3_original_pred=original, **logits_store)
    write_json(run / "counterfactual.json", counterfactual)
    source_after = {name: sha(path) for name, path in paths(args.dataset, args.encoder_seed).items()}
    write_json(run / "source_hashes_after.json", source_after)
    if source_after != source_before:
        raise RuntimeError("source hash changed")
    verification = {"status": "PASS", "dataset": args.dataset,
                    "encoder_seed": args.encoder_seed, "training_seed": args.training_seed,
                    "train_samples": len(train["labels"]), "validation_samples": len(val["labels"]),
                    "native_e3_branches": 2, "yatc_usage": 0, "known_test_usage": 0, "unknown_usage": 0,
                    "adapter_best_epoch": adapter_epoch,
                    "checkpoint_selection_role": "known_validation",
                    "entropy_calibration_role": "known_train",
                    "source_hashes_unchanged": True,
                    "checkpoints": {name: sha(run / name) for name in
                        ("adapter_best.pt", "N0_equal_best.pt", "N1_shared_lambda_entropy_best.pt")},
                    "runtime_seconds": time.monotonic() - start}
    write_json(run / "verification.json", verification)
    (run / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status": "SUCCESS", "dataset": args.dataset,
                      "encoder_seed": args.encoder_seed, "training_seed": args.training_seed,
                      "metrics": output, "counterfactual": counterfactual}), flush=True)


if __name__ == "__main__":
    main()
