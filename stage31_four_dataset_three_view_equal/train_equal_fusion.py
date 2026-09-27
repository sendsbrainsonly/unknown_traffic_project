#!/usr/bin/env python3
"""Stage30 T0-equivalent Known-only adapters and fixed three-view equal fusion."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from preflight import OUT, sha
from train_yatc_branch import protocol_rows

VIEWS = ("trafficformer", "graph", "yatc")
DIMS = (768, 128, 192)
LATENT = 64
EPOCHS = 30
BATCH = 256


def seeded(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(truth, prediction, nclasses):
    p, r, f, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(nclasses), zero_division=0)
    return {"accuracy": float(np.mean(truth == prediction)), "macro_f1": float(f.mean()),
            "weighted_f1": float(np.average(f, weights=support))}, (p, r, f, support)


class View(nn.Module):
    def __init__(self, dim, nclasses):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        self.mean = nn.Linear(128, LATENT)
        self.logvar = nn.Linear(128, LATENT)
        self.classifier = nn.Linear(LATENT, nclasses)

    def forward(self, x):
        hidden = self.backbone(x)
        return self.mean(hidden), self.logvar(hidden).clamp(-8., 4.)


class Adapters(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.views = nn.ModuleList([View(dim, nclasses) for dim in DIMS])

    def training_loss(self, xs, labels):
        ces, kls = [], []
        for module, x in zip(self.views, xs, strict=True):
            mu, logvar = module(x)
            latent = mu + torch.randn_like(mu) * torch.exp(logvar * .5)
            ces.append(F.cross_entropy(module.classifier(latent), labels))
            kls.append(-.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean() / LATENT)
        ce, kl = torch.stack(ces).mean(), torch.stack(kls).mean()
        return ce + .05 * kl, ce.detach(), kl.detach()


@torch.no_grad()
def encode(model, values, device):
    model.eval()
    mu_chunks, h_chunks, pred_chunks = [], [], []
    for start in range(0, len(values["labels"]), 512):
        xs = [torch.from_numpy(values[name][start:start+512]).to(device) for name in VIEWS]
        mus, entropies, predictions = [], [], []
        for module, x in zip(model.views, xs, strict=True):
            mu, logvar = module(x)
            mus.append(mu)
            entropies.append(.5 * np.log(2*np.pi*np.e) + .5 * logvar.mean(1))
            predictions.append(module.classifier(mu).argmax(1))
        mu_chunks.append(torch.stack(mus, 1).cpu().numpy())
        h_chunks.append(torch.stack(entropies, 1).cpu().numpy())
        pred_chunks.append(torch.stack(predictions, 1).cpu().numpy())
    return np.concatenate(mu_chunks).astype(np.float32), np.concatenate(h_chunks).astype(np.float32), np.concatenate(pred_chunks)


class EqualFusion(nn.Module):
    def __init__(self, nclasses, center, mad):
        super().__init__()
        self.dynamic = False
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("mad", torch.as_tensor(mad, dtype=torch.float32))
        self.mix_logits = None
        self.fuser = nn.Sequential(nn.Linear(3*LATENT, 128), nn.ReLU(), nn.Dropout(.1),
                                   nn.Linear(128, 128), nn.ReLU())
        self.classifier = nn.Linear(128, nclasses)

    def forward(self, mu, entropy):
        weight = torch.full((len(mu), 3), 1/3, dtype=mu.dtype, device=mu.device)
        joined = (mu*weight.unsqueeze(-1)).flatten(1)
        return self.classifier(self.fuser(joined)), weight


@torch.no_grad()
def infer(model, encoded, device):
    model.eval()
    logits = []
    for start in range(0, len(encoded[0]), 512):
        mu = torch.from_numpy(encoded[0][start:start+512]).to(device)
        entropy = torch.from_numpy(encoded[1][start:start+512]).to(device)
        y, _ = model(mu, entropy)
        logits.append(y.cpu().numpy())
    return np.concatenate(logits)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def load_views(dataset, protocol, selected, classes):
    root = OUT / "runs" / dataset / protocol
    by_role = {}
    source_hashes = {}
    for role in ("known_train", "known_validation"):
        rows = sorted((uid, cls) for uid, cls, r in selected if r == role)
        ids = [uid for uid, _ in rows]
        labels = np.asarray([classes.index(cls) for _, cls in rows], dtype=np.int64)
        values = {"flow_ids": np.asarray(ids, dtype="U80"), "labels": labels}
        for view, branch in (("trafficformer", "trafficformer"), ("graph", "graph"), ("yatc", "yatc")):
            sub = root / branch
            if not (sub / "SUCCESS").is_file():
                raise RuntimeError(f"branch incomplete: {sub}")
            flow_ids = np.load(sub / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
            if flow_ids != ids:
                raise RuntimeError(f"branch flow-ID mismatch: {view}/{role}")
            path = sub / f"{role}_features.npy"
            feature = np.load(path, mmap_mode="r", allow_pickle=False)
            dim = DIMS[VIEWS.index(view)]
            if feature.shape != (len(ids), dim) or not np.isfinite(feature).all():
                raise RuntimeError(f"bad branch feature: {view}/{role}/{feature.shape}")
            values[view] = np.asarray(feature, dtype=np.float32)
            source_hashes[f"{view}_{role}"] = sha(path)
        by_role[role] = values
    stats = {}
    for view in VIEWS:
        mean = by_role["known_train"][view].mean(0)
        std = by_role["known_train"][view].std(0)
        std[std < 1e-8] = 1
        stats[view] = (mean, std)
        for role in by_role:
            by_role[role][view] = ((by_role[role][view]-mean)/std).astype(np.float32)
    return by_role, stats, source_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    run = OUT / "runs" / args.dataset / args.protocol / "T0_equal"
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    started = time.time()
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        selected, classes, manifest_sha = protocol_rows(args.dataset, args.protocol)
        roles, scalers, hashes = load_views(args.dataset, args.protocol, selected, classes)
        train, val = roles["known_train"], roles["known_validation"]
        np.savez(run / "known_train_view_scalers.npz", **{
            f"{view}_{kind}": value for view, (mean, std) in scalers.items()
            for kind, value in (("mean", mean), ("std", std))})
        save_json(run / "config.json", {"dataset": args.dataset, "protocol": args.protocol,
            "classes": classes, "seed": 2022, "source_split_sha256": manifest_sha,
            "source_feature_hashes": hashes, "views": list(VIEWS), "dimensions": DIMS,
            "adapter_epochs": 30, "head_epochs": 30, "batch_size": 256,
            "adapter_loss": "mean CE + 0.05 mean KL", "head_weight": [1/3]*3,
            "checkpoint_selection": "Known Validation mean unimodal Macro-F1 for adapters; Known Validation Macro-F1 for head",
            "test_selection_samples": 0, "unknown_samples_used": 0})
        seeded(2022)
        adapters = Adapters(len(classes)).to(device)
        optimizer = torch.optim.Adam(adapters.parameters(), lr=1e-3)
        arrays = [torch.from_numpy(train[name]) for name in VIEWS] + [torch.from_numpy(train["labels"])]
        loader = DataLoader(TensorDataset(*arrays), batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(2022), num_workers=0)
        history, best, best_epoch, best_state = [], -1.0, -1, None
        for epoch in range(1, EPOCHS+1):
            adapters.train(); total = 0.0
            for batch in loader:
                xs = [v.to(device) for v in batch[:3]]
                y = batch[3].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = adapters.training_loss(xs, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite adapter loss epoch {epoch}")
                loss.backward(); optimizer.step(); total += float(loss.detach())*len(y)
            _, _, pred = encode(adapters, val, device)
            f1 = [metrics(val["labels"], pred[:, i], len(classes))[0]["macro_f1"] for i in range(3)]
            score = float(np.mean(f1))
            history.append({"phase": "adapters", "epoch": epoch,
                            "train_loss": total/len(train["labels"]),
                            "val_mean_unimodal_macro_f1": score,
                            **{f"val_{name}_macro_f1": value for name, value in zip(VIEWS, f1)}})
            if score > best:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in adapters.state_dict().items()}
            print(json.dumps({"phase": "adapters", "epoch": epoch,
                              "val_mean_macro_f1": score}), flush=True)
        adapters.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                    "source_feature_hashes": hashes}, run / "adapters_best.pt")
        encoded_train, encoded_val = encode(adapters, train, device), encode(adapters, val, device)
        seeded(2022)
        center = np.median(encoded_train[1], axis=0)
        mad = np.median(np.abs(encoded_train[1]-center), axis=0)
        head = EqualFusion(len(classes), center, mad).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(torch.from_numpy(encoded_train[0]),
                            torch.from_numpy(encoded_train[1]), torch.from_numpy(train["labels"])),
                            batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(2022), num_workers=0)
        best, head_epoch, state = -1.0, -1, None
        for epoch in range(1, EPOCHS+1):
            head.train(); total = 0.0
            for mu, entropy, y in loader:
                mu, entropy, y = mu.to(device), entropy.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = head(mu, entropy)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite T0 loss epoch {epoch}")
                loss.backward(); optimizer.step(); total += float(loss.detach())*len(y)
            logits = infer(head, encoded_val, device)
            score = metrics(val["labels"], logits.argmax(1), len(classes))[0]
            history.append({"phase": "T0_equal", "epoch": epoch,
                            "train_loss": total/len(train["labels"]),
                            **{f"val_{name}": value for name, value in score.items()}})
            if score["macro_f1"] > best:
                best, head_epoch = score["macro_f1"], epoch
                state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(json.dumps({"phase": "T0_equal", "epoch": epoch,
                              "val_macro_f1": score["macro_f1"]}), flush=True)
        head.load_state_dict(state)
        torch.save({"state_dict": state, "best_epoch": head_epoch,
                    "source_feature_hashes": hashes, "method": "T0_equal"}, run / "T0_equal_best.pt")
        logits = infer(head, encoded_val, device)
        prediction = logits.argmax(1)
        score, per_class = metrics(val["labels"], prediction, len(classes))
        with (run / "known_validation_predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("flow_id", "true_class", "predicted_class", "correct"))
            writer.writeheader()
            for uid, truth, chosen in zip(val["flow_ids"], val["labels"], prediction, strict=True):
                writer.writerow({"flow_id": str(uid), "true_class": classes[int(truth)],
                                 "predicted_class": classes[int(chosen)], "correct": int(truth==chosen)})
        save_json(run / "known_validation_metrics.json", {"metrics": score, "classes": [
            {"class": name, "precision": float(per_class[0][i]),
             "recall": float(per_class[1][i]), "f1": float(per_class[2][i]),
             "support": int(per_class[3][i])} for i, name in enumerate(classes)],
             "adapter_best_epoch": best_epoch, "head_best_epoch": head_epoch,
             "checkpoint_hashes": {name: sha(run / name) for name in ("adapters_best.pt", "T0_equal_best.pt")}})
        with (run / "training_history.csv").open("w", newline="") as handle:
            fields = list(dict.fromkeys(key for row in history for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        save_json(run / "verification.json", {"status": "PASS", "known_test_usage": 0,
             "unknown_usage": 0, "encoder_updates": 0, "adapter_best_epoch": best_epoch,
             "head_best_epoch": head_epoch, "known_validation_samples": len(val["labels"]),
             "elapsed_seconds": time.time()-started})
        (run / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "dataset": args.dataset,
                          "protocol": args.protocol, "val_macro_f1": score["macro_f1"]}), flush=True)
    except BaseException as exc:
        save_json(run / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
