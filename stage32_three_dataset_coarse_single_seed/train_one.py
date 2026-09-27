#!/usr/bin/env python3
"""Single-seed Stage32 coarse T0; exact Stage31/Stage30 architecture, frozen encoders."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import (COARSE_CLASSES, DATASETS, ROOT, TRAINING_SEED, VIEWS,
                    freeze_sources, load_known, sha256, write_json)

sys.path.insert(0, str(ROOT.parent / "stage31_four_dataset_three_view_equal"))
from train_equal_fusion import Adapters, EqualFusion, encode, infer, metrics, seeded  # noqa: E402

EPOCHS = 30
BATCH = 256


def save_csv(path, rows):
    if not rows or path.exists():
        raise RuntimeError(f"empty/existing CSV: {path}")
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def normalize(train, val):
    stats = {}
    for name in VIEWS:
        mean = train[name].mean(0)
        std = train[name].std(0)
        std[std < 1e-8] = 1
        stats[name] = (mean, std)
        for values in (train, val):
            values[name] = ((values[name] - mean) / std).astype(np.float32)
            if not np.isfinite(values[name]).all():
                raise RuntimeError(f"nonfinite standardized {name}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage32 training requires a live-selected CUDA GPU")
    preflight = json.loads((ROOT / "preflight.json").read_text())
    if preflight["status"] != "PASS" or preflight["test_feature_values_loaded"] != 0:
        raise RuntimeError("Stage32 Known-only preflight did not pass")
    frozen = json.loads((ROOT / "frozen_source_hashes_before.json").read_text())
    before = freeze_sources()
    if before != frozen:
        raise RuntimeError("frozen source changed before training")
    run = ROOT / "runs" / args.dataset
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        classes = COARSE_CLASSES[args.dataset]
        train = load_known(args.dataset, "known_train")
        val = load_known(args.dataset, "known_validation")
        if set(train["flow_ids"].tolist()) & set(val["flow_ids"].tolist()):
            raise RuntimeError("Train/Val flow overlap")
        stats = normalize(train, val)
        np.savez(run / "known_train_view_scalers.npz", **{
            f"{view}_{kind}": value for view, (mean, std) in stats.items()
            for kind, value in (("mean", mean), ("std", std))})
        write_json(run / "config.json", {
            "dataset": args.dataset, "method": "Stage30/31 T0_equal coarse label adaptation",
            "classes": classes, "training_seed": TRAINING_SEED,
            "adapter_epochs": EPOCHS, "head_epochs": EPOCHS,
            "batch_size": BATCH, "optimizer": "Adam(lr=1e-3)",
            "adapter_loss": "mean CE + 0.05 mean KL",
            "head_loss": "cross entropy", "weights": [1/3] * 3,
            "adapter_selection": "Known Validation mean unimodal Macro-F1",
            "head_selection": "Known Validation coarse Macro-F1",
            "train_samples": len(train["labels"]), "validation_samples": len(val["labels"]),
            "source_hashes_before": before, "test_selection_samples": 0,
            "unknown_samples_used": 0, "encoder_updates": 0,
        })
        seeded(TRAINING_SEED)
        adapters = Adapters(len(classes)).to(device)
        optimizer = torch.optim.Adam(adapters.parameters(), lr=1e-3)
        tensors = [torch.from_numpy(train[name]) for name in VIEWS] + [torch.from_numpy(train["labels"])]
        loader = DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(TRAINING_SEED), num_workers=0)
        history, best, best_epoch, best_state = [], -1.0, -1, None
        for epoch in range(1, EPOCHS + 1):
            adapters.train()
            total = 0.0
            for batch in loader:
                xs = [v.to(device) for v in batch[:3]]
                y = batch[3].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = adapters.training_loss(xs, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite adapter loss epoch {epoch}")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(y)
            _, _, pred = encode(adapters, val, device)
            f1 = [metrics(val["labels"], pred[:, i], len(classes))[0]["macro_f1"] for i in range(3)]
            score = float(np.mean(f1))
            history.append({"phase": "adapters", "epoch": epoch,
                            "train_loss": total / len(train["labels"]),
                            "val_mean_unimodal_macro_f1": score,
                            **{f"val_{name}_macro_f1": v for name, v in zip(VIEWS, f1, strict=True)}})
            if score > best:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in adapters.state_dict().items()}
            print(json.dumps({"dataset": args.dataset, "phase": "adapters", "epoch": epoch,
                              "val_mean_macro_f1": score}), flush=True)
        if best_state is None:
            raise RuntimeError("no adapter checkpoint selected")
        adapters.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                    "epochs_completed": EPOCHS, "source_hashes": before}, run / "adapters_best.pt")
        encoded_train, encoded_val = encode(adapters, train, device), encode(adapters, val, device)
        seeded(TRAINING_SEED)
        center = np.median(encoded_train[1], axis=0)
        mad = np.median(np.abs(encoded_train[1] - center), axis=0)
        head = EqualFusion(len(classes), center, mad).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(torch.from_numpy(encoded_train[0]),
                            torch.from_numpy(encoded_train[1]), torch.from_numpy(train["labels"])),
                            batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(TRAINING_SEED), num_workers=0)
        best, head_epoch, state = -1.0, -1, None
        for epoch in range(1, EPOCHS + 1):
            head.train()
            total = 0.0
            for mu, entropy, y in loader:
                mu, entropy, y = mu.to(device), entropy.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = head(mu, entropy)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite T0 loss epoch {epoch}")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(y)
            logits = infer(head, encoded_val, device)
            score = metrics(val["labels"], logits.argmax(1), len(classes))[0]
            history.append({"phase": "T0_equal", "epoch": epoch,
                            "train_loss": total / len(train["labels"]),
                            **{f"val_{name}": value for name, value in score.items()}})
            if score["macro_f1"] > best:
                best, head_epoch = score["macro_f1"], epoch
                state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(json.dumps({"dataset": args.dataset, "phase": "T0_equal", "epoch": epoch,
                              "val_macro_f1": score["macro_f1"]}), flush=True)
        if state is None:
            raise RuntimeError("no T0 checkpoint selected")
        head.load_state_dict(state)
        torch.save({"state_dict": state, "best_epoch": head_epoch,
                    "epochs_completed": EPOCHS, "method": "T0_equal",
                    "source_hashes": before}, run / "T0_equal_best.pt")
        logits = infer(head, encoded_val, device)
        prediction = logits.argmax(1)
        score, pc = metrics(val["labels"], prediction, len(classes))
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        save_csv(run / "training_history.csv", history)
        save_csv(run / "known_validation_predictions.csv", [
            {"flow_id": str(uid), "true_fine": str(fine),
             "true_coarse": classes[int(truth)], "pred_coarse": classes[int(chosen)],
             "correct": int(truth == chosen), "max_probability": float(prob.max())}
            for uid, fine, truth, chosen, prob in zip(val["flow_ids"], val["fine_labels"],
                val["labels"], prediction, probabilities, strict=True)])
        np.save(run / "known_validation_logits.npy", logits.astype(np.float32), allow_pickle=False)
        write_json(run / "known_validation_metrics.json", {
            "metrics": score, "adapter_best_epoch": best_epoch, "head_best_epoch": head_epoch,
            "classes": [{"class": name, "precision": float(pc[0][i]),
                         "recall": float(pc[1][i]), "f1": float(pc[2][i]),
                         "support": int(pc[3][i])} for i, name in enumerate(classes)],
            "checkpoint_hashes": {name: sha256(run / name) for name in ("adapters_best.pt", "T0_equal_best.pt")},
        })
        after = freeze_sources()
        if after != before:
            raise RuntimeError("frozen input hashes changed during training")
        write_json(run / "verification.json", {"status": "PASS", "known_test_usage": 0,
            "unknown_usage": 0, "encoder_updates": 0, "epochs_adapter_completed": EPOCHS,
            "epochs_head_completed": EPOCHS, "adapter_best_epoch": best_epoch,
            "head_best_epoch": head_epoch, "source_hashes_unchanged": True,
            "elapsed_seconds": time.monotonic() - started})
        (run / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "dataset": args.dataset, "val": score,
                          "adapter_best_epoch": best_epoch, "head_best_epoch": head_epoch}), flush=True)
    except BaseException as exc:
        write_json(run / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
