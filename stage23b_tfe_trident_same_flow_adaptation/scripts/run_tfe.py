#!/usr/bin/env python3
"""Stage20 all-flow eight-packet adaptation of released TFE-GNN model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import dgl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
SOURCE = PROJECT.parent / "TFE-GNN/code"
sys.path.insert(0, str(SOURCE))
import model as original_model  # noqa: E402
from config import ISCXVPNConfig, ISCXTorConfig  # noqa: E402
from optim import GradualWarmupScheduler  # noqa: E402
from utils import construct_graph  # noqa: E402

MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
FROZEN = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
CONFIG = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison/config.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=20000)
def graph_for(tokens: tuple[int, ...]):
    g = construct_graph(list(tokens), w_size=5, k=1)
    if g.num_nodes() == 0:
        # Undefined empty/constant-byte native graph; explicit adapter.
        g = dgl.graph(([0], [0]), num_nodes=1)
        g.ndata["feat"] = torch.tensor([[int(tokens[0])]], dtype=torch.float32)
    return g


class Flows(Dataset):
    def __init__(self, root: Path, positions: np.ndarray, labels: np.ndarray):
        self.header = np.load(root / "header.npy", mmap_mode="r", allow_pickle=False)
        self.payload = np.load(root / "payload.npy", mmap_mode="r", allow_pickle=False)
        self.positions, self.labels = positions, labels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return int(self.positions[i]), int(self.labels[i])

    def collate(self, batch):
        heads, bodies, labels = [], [], []
        for pos, label in batch:
            labels.append(label)
            for slot in range(8):
                heads.append(graph_for(tuple(map(int, self.header[pos, slot]))))
                bodies.append(graph_for(tuple(map(int, self.payload[pos, slot]))))
        return dgl.batch(heads), dgl.batch(bodies), torch.tensor(labels, dtype=torch.long)


def inputs(dataset: str, phase: str, services: list[str]):
    root = OUT / "inputs" / dataset / phase
    audit = json.loads((root / "input_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["stage20_manifest_sha256"] != FROZEN:
        raise RuntimeError("input audit mismatch")
    for name, expected in audit["file_hashes"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"input hash mismatch: {name}")
    with (root / "membership.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [str(x) for x in np.load(root / "flow_ids.npy", allow_pickle=False)]
    if ids != [r["flow_id"] for r in rows]:
        raise RuntimeError("flow IDs misaligned")
    for name, shape in (("header.npy", (len(rows), 8, 40)), ("payload.npy", (len(rows), 8, 150))):
        if np.load(root / name, mmap_mode="r", allow_pickle=False).shape != shape:
            raise RuntimeError(f"input shape mismatch: {name}")
    labels = np.asarray([services.index(r["service_label"]) for r in rows], dtype=np.int64)
    return root, labels, rows, audit


def metrics(y, pred, services):
    labels = np.arange(len(services))
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(n[i])}
                      for i, name in enumerate(services)},
    }


def loader(root, positions, labels, batch, shuffle, seed):
    ds = Flows(root, positions, labels[positions])
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      collate_fn=ds.collate, generator=torch.Generator().manual_seed(seed))


def epoch(model, batches, device, criterion, optimizer=None, scheduler=None, accumulation=1):
    train = optimizer is not None
    model.train(train)
    yy, pp, prob, total_loss = [], [], [], 0.0
    if train:
        optimizer.zero_grad()
    with torch.set_grad_enabled(train):
        for step, (heads, bodies, labels) in enumerate(batches, 1):
            labels = labels.to(device)
            logits = model(heads.to(device), bodies.to(device), labels)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("TFE nonfinite loss")
            if train:
                (loss / accumulation).backward()
                if step % accumulation == 0 or step == len(batches):
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            yy.extend(labels.cpu().numpy().tolist())
            pp.extend(np.argmax(probabilities, axis=1).tolist())
            prob.append(probabilities)
    return np.asarray(yy), np.asarray(pp), np.concatenate(prob), total_loss / len(yy)


def predictions(path, dataset, seed, role, rows, y, pred, proba, services):
    with path.open("x", newline="", encoding="utf-8") as f:
        cols = ["dataset", "seed", "role", "method", "flow_id", "true_service",
                "predicted_service", "correct"] + [f"prob_{s}" for s in services]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row, true, guess, probs in zip(rows, y, pred, proba):
            result = {"dataset": dataset, "seed": seed, "role": role,
                      "method": "TFE-GNN-8-UDP-short", "flow_id": row["flow_id"],
                      "true_service": services[int(true)], "predicted_service": services[int(guess)],
                      "correct": int(true == guess)}
            result.update({f"prob_{s}": float(probs[i]) for i, s in enumerate(services)})
            w.writerow(result)


def run(dataset: str, seed: int, phase: str) -> None:
    if digest(MANIFEST) != FROZEN:
        raise RuntimeError("frozen Stage20 manifest changed")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if seed not in cfg["seeds"]:
        raise ValueError("unregistered seed")
    services = cfg["datasets"][dataset]["services"]
    native = ISCXVPNConfig() if dataset == "iscx_vpn" else ISCXTorConfig()
    original_model.config.FLOW_PAD_TRUNC_LENGTH = 8
    run_dir = OUT / "runs" / "tfe" / dataset / f"seed{seed}"
    if phase == "train":
        if run_dir.exists():
            raise FileExistsError(f"preserve existing run {run_dir}")
        run_dir.mkdir(parents=True)
        root, labels, rows, audit = inputs(dataset, "trainval", services)
        train = np.flatnonzero([r["role"] == "known_train" for r in rows])
        val = np.flatnonzero([r["role"] == "known_validation" for r in rows])
        if (len(train), len(val)) != (cfg["datasets"][dataset]["expected_train"],
                                     cfg["datasets"][dataset]["expected_validation"]):
            raise RuntimeError("frozen membership count changed")
        (run_dir / "preflight.json").write_text(json.dumps({
            "stage20_sha256": FROZEN, "dataset": dataset, "seed": seed,
            "known_train": len(train), "known_validation": len(val),
            "known_test_features_loaded": 0, "unknown_features_loaded": 0,
            "input_hashes": audit["file_hashes"]}, indent=2) + "\n", encoding="utf-8")
    else:
        if not (run_dir / "SELECTION_COMPLETE").is_file():
            raise RuntimeError("TFE Known Validation selection incomplete")
        selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
        if digest(run_dir / selection["checkpoint_file"]) != selection["checkpoint_sha256"]:
            raise RuntimeError("selected TFE checkpoint changed")
        root, labels, rows, audit = inputs(dataset, "test", services)
        val = np.arange(len(rows))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("formal TFE requires CUDA")
    device = torch.device("cuda:0")
    model = original_model.MixTemporalGNN(
        num_classes=len(services), embedding_size=native.EMBEDDING_SIZE,
        h_feats=native.H_FEATS, dropout=native.DROPOUT,
        downstream_dropout=native.DOWNSTREAM_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=native.LABEL_SMOOTHING)
    if phase == "train":
        batch_size = int(native.BATCH_SIZE)
        train_loader = loader(root, train, labels, batch_size, True, seed)
        val_loader = loader(root, val, labels, batch_size, False, seed)
        steps = len(train_loader) * int(native.MAX_EPOCH)
        optimizer = torch.optim.Adam(model.parameters(), lr=native.LR, weight_decay=native.WEIGHT_DECAY)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps - int(steps * native.WARM_UP), eta_min=native.LR_MIN)
        scheduler = GradualWarmupScheduler(
            optimizer, warmup_iter=int(steps * native.WARM_UP), after_scheduler=cosine)
        scheduler.step()  # released-code scheduler placement, recorded as such
        best, best_epoch, best_file = -1.0, None, None
        for number in range(1, int(native.MAX_EPOCH) + 1):
            tr_y, tr_p, _, tr_loss = epoch(model, train_loader, device, criterion, optimizer,
                                           scheduler, int(native.GRADIENT_ACCUMULATION))
            va_y, va_p, va_prob, va_loss = epoch(model, val_loader, device, criterion)
            tr_m, va_m = metrics(tr_y, tr_p, services), metrics(va_y, va_p, services)
            with (run_dir / "training_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": number, "train_loss": tr_loss, "val_loss": va_loss,
                                    "train": tr_m, "val": va_m,
                                    "lr": optimizer.param_groups[0]["lr"]}) + "\n")
            if va_m["macro_f1"] > best:
                best, best_epoch = va_m["macro_f1"], number
                best_file = f"checkpoint_epoch{number:03d}.pt"
                torch.save({"model": model.state_dict(), "services": services,
                            "epoch": number, "val": va_m}, run_dir / best_file)
                predictions(run_dir / f"known_val_epoch{number:03d}.csv", dataset, seed,
                            "known_validation", [rows[i] for i in val], va_y, va_p,
                            va_prob, services)
            print(json.dumps({"event": "epoch", "dataset": dataset, "seed": seed,
                              "epoch": number, "total": native.MAX_EPOCH,
                              "train_loss": tr_loss, "val_loss": va_loss,
                              "val_macro_f1": va_m["macro_f1"], "best": best}), flush=True)
        selection = {"status": "SELECTION_COMPLETE", "method": "TFE-GNN-8-UDP-short",
                     "dataset": dataset, "seed": seed, "train_flows": len(train),
                     "validation_flows": len(val), "known_test_loaded": 0, "unknown_loaded": 0,
                     "best_epoch": best_epoch, "best_val_macro_f1": best,
                     "configured_epochs": int(native.MAX_EPOCH),
                     "completed_epochs": int(native.MAX_EPOCH),
                     "checkpoint_file": best_file, "checkpoint_sha256": digest(run_dir / best_file),
                     "stage20_manifest_sha256": FROZEN, "input_hashes": audit["file_hashes"]}
        (run_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        (run_dir / "SELECTION_COMPLETE").write_text("PASS\n", encoding="utf-8")
    else:
        selected = torch.load(run_dir / selection["checkpoint_file"], map_location=device, weights_only=False)
        model.load_state_dict(selected["model"])
        test_loader = loader(root, val, labels, int(native.BATCH_SIZE), False, seed)
        y, p, prob, loss = epoch(model, test_loader, device, criterion)
        result = metrics(y, p, services)
        predictions(run_dir / "known_test_predictions.csv", dataset, seed, "known_test",
                    rows, y, p, prob, services)
        report = {"status": "COMPLETE", "dataset": dataset, "seed": seed,
                  "method": "TFE-GNN-8-UDP-short", "known_test_flows": len(rows),
                  "test": result, "test_loss": loss,
                  "checkpoint_sha256": selection["checkpoint_sha256"],
                  "stage20_manifest_sha256": FROZEN, "test_input_hashes": audit["file_hashes"]}
        (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (run_dir / "SUCCESS").write_text("PASS\n", encoding="utf-8")
        print(json.dumps({"event": "test_complete", "dataset": dataset, "seed": seed,
                          "test_macro_f1": result["macro_f1"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.phase)
