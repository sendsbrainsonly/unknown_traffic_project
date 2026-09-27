#!/usr/bin/env python3
"""Known-only YaTC branch training on a Stage31 frozen protocol."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from preflight import OUT, PROJECT, SOURCES, sha

sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table" / "scripts"))
import train_yatc_closed as source  # noqa: E402


def protocol_rows(dataset: str, protocol: str):
    path = SOURCES[dataset]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if dataset.startswith("iscx"):
        selected = [(r["flow_id_sha256"], r["canonical_class"], r["role"])
                    for r in rows if r["setting"] == "medium" and
                    r["role"] in ("known_train", "known_validation")]
    elif dataset == "vnat":
        selected = [(r["flow_uid"], r["application"],
                     "known_train" if r["split"] == "train" else "known_validation")
                    for r in rows if r["protocol_id"] == protocol and r["class_role"] == "known"
                    and r["split"] in ("train", "validation")]
    else:
        config = json.loads((PROJECT / "stage3_unknown_utility" / "outputs" / "A-2" / "training_config.json").read_text())
        known = set(config["known_classes"])
        align = PROJECT / "opendetect_ustc_encoder_audit" / "outputs" / "input_alignment_manifest.csv"
        with align.open(newline="", encoding="utf-8") as handle:
            selected = [(r["flow_id"], r["class_name"],
                         "known_train" if r["original_split"] == "train" else "known_validation")
                        for r in csv.DictReader(handle) if r["class_name"] in known and
                        r["original_split"] in ("train", "val") and r["input_valid"] == "True"]
    if len(selected) != len({uid for uid, _, _ in selected}):
        raise RuntimeError("duplicate frozen Known flow ID")
    classes = sorted({cls for _, cls, _ in selected})
    if not selected or set(x[2] for x in selected) != {"known_train", "known_validation"}:
        raise RuntimeError("missing Known Train or Validation")
    return selected, classes, sha(path)


def cache_root(dataset: str, protocol: str) -> Path:
    root = OUT / "input_caches" / dataset
    if dataset == "vnat":
        root = root / protocol / ("yatc_mfr_attempt3" if protocol == "medium_seed2025" else "yatc_mfr")
    elif dataset == "ustc":
        root = root / "A-2" / "yatc_mfr"
    else:
        root = root / "yatc_mfr"
    return root


class KnownMFR(Dataset):
    def __init__(self, root: Path, role: str, selected: list[tuple[str, str, str]],
                 classes: list[str], file_role: str | None = None):
        self.rows = sorted((uid, cls) for uid, cls, item_role in selected if item_role == role)
        prefix = file_role or role
        ids = np.load(root / f"{prefix}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if ids != [uid for uid, _ in self.rows]:
            raise RuntimeError(f"MFR ID alignment failed: {root}/{role}")
        self.images = np.load(root / f"{prefix}_mfr.npy", mmap_mode="r", allow_pickle=False)
        if self.images.shape != (len(ids), 40, 40):
            raise RuntimeError("MFR shape mismatch")
        label = {name: i for i, name in enumerate(classes)}
        self.labels = np.asarray([label[cls] for _, cls in self.rows], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.images[i], copy=True)).float().unsqueeze(0)
        return x.div_(255).sub_(0.5).div_(0.5), int(self.labels[i])


@torch.no_grad()
def infer_features(model, dataset, device, output: Path):
    model.eval()
    feature = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                        shape=(len(dataset), 192))
    offset = 0
    for x, _ in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
        with torch.cuda.amp.autocast():
            part = model.forward_features(x.to(device, non_blocking=True))
        if part.shape != (len(x), 192) or not torch.isfinite(part).all():
            raise RuntimeError("YaTC feature shape or finiteness failed")
        feature[offset:offset+len(x)] = part.float().cpu().numpy()
        offset += len(x)
    feature.flush()
    if offset != len(dataset):
        raise RuntimeError("inference count mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    expected = {"iscx_vpn": "medium_seed2022", "iscx_tor": "medium_seed2022",
                "vnat": ("medium_seed2025", "medium_seed2026"), "ustc": "A-2"}[args.dataset]
    if args.protocol not in (expected if isinstance(expected, tuple) else (expected,)):
        raise ValueError("protocol outside fixed Stage31 pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    suffix = "" if args.attempt == 1 else f"_attempt{args.attempt}"
    run = OUT / "runs" / args.dataset / args.protocol / (("yatc_smoke" if args.smoke else "yatc") + suffix)
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    started = time.time()
    try:
        random.seed(2022); np.random.seed(2022); torch.manual_seed(2022)
        torch.cuda.manual_seed_all(2022); torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        if sha(source.WEIGHT) != source.WEIGHT_HASH:
            raise RuntimeError("official YaTC pretrained weight hash mismatch")
        selected, classes, source_hash = protocol_rows(args.dataset, args.protocol)
        cache = cache_root(args.dataset, args.protocol)
        audit = json.loads((cache / "cache_audit.json").read_text())
        if audit["status"] != "PASS":
            raise RuntimeError("MFR input audit did not pass")
        train = KnownMFR(cache, "known_train", selected, classes,
                         "train" if args.dataset == "vnat" else None)
        val = KnownMFR(cache, "known_validation", selected, classes,
                       "validation" if args.dataset == "vnat" else None)
        if set(train.labels) != set(range(len(classes))) or set(val.labels) != set(range(len(classes))):
            raise RuntimeError("class absent from Known Train/Validation")
        source.write_json(run / "config.json", {
            "dataset": args.dataset, "protocol": args.protocol, "seed": 2022,
            "classes": classes, "known_train": len(train), "known_validation": len(val),
            "frozen_split_sha256": source_hash, "pretrained_sha256": source.WEIGHT_HASH,
            "cache_audit_sha256": sha(cache / "cache_audit.json"),
            "training": "Stage23 YaTC author 200 epochs, batch64, AdamW, effective LR0.0005, weight decay0.05, layer decay0.75, warmup20, label smoothing0.1",
            "checkpoint_selection": "Known Validation weighted F1 (Stage23 author rule)",
            "unknown_training_samples": 0, "unknown_validation_samples": 0,
            "test_selection_samples": 0})
        sampler = DistributedSampler(train, num_replicas=1, rank=0, shuffle=True, seed=0)
        train_loader = DataLoader(train, sampler=sampler, batch_size=64, drop_last=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        device = torch.device("cuda:0")
        model, load_info = source.initialized_model(classes)
        model.to(device)
        lr = 0.002 * 64 / 256
        groups = source.lrd.param_groups_lrd(model, 0.05,
                                              no_weight_decay_list=model.no_weight_decay(), layer_decay=0.75)
        optimizer = torch.optim.AdamW(groups, lr=lr)
        scaler = source.NativeScalerWithGradNormCount()
        criterion = source.LabelSmoothingCrossEntropy(smoothing=0.1)
        recipe = SimpleNamespace(accum_iter=1, warmup_epochs=20, min_lr=1e-6,
                                 lr=lr, epochs=200, clip_grad=None, output_dir="")
        source.write_json(run / "pretrained_load.json", load_info)
        if args.smoke:
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite YaTC smoke loss")
            loss.backward(); optimizer.step()
            source.write_json(run / "smoke_result.json", {"status": "PASS", "loss": float(loss),
                                                      "batch": len(y), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))})
            (run / "SUCCESS").write_text("SUCCESS\n")
            print(json.dumps({"status": "PASS_SMOKE", "dataset": args.dataset}), flush=True)
            return
        best, best_epoch = -1.0, -1
        with (run / "training_history.jsonl").open("x") as history:
            for epoch in range(200):
                sampler.set_epoch(epoch)
                train_stats = source.train_one_epoch(model, criterion, train_loader, optimizer,
                                                     device, epoch, scaler, None, None,
                                                     log_writer=None, args=recipe)
                validation, _, _ = source.evaluate(model, val_loader, device, classes)
                if validation["weighted_f1"] > best:
                    best, best_epoch = validation["weighted_f1"], epoch+1
                    torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch,
                                "classes": classes, "validation_weighted_f1": best},
                               run / "model_best.pt")
                event = {"epoch": epoch+1, "train": train_stats,
                         "validation": validation, "best_epoch": best_epoch,
                         "elapsed_seconds": time.time()-started}
                history.write(json.dumps(event, default=float) + "\n"); history.flush()
                print(json.dumps({"dataset": args.dataset, "protocol": args.protocol,
                                  "branch": "yatc", "epoch": epoch+1, "val_macro_f1": validation["macro_f1"],
                                  "val_weighted_f1": validation["weighted_f1"]}), flush=True)
        state = torch.load(run / "model_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        infer_features(model, train, device, run / "known_train_features.npy")
        infer_features(model, val, device, run / "known_validation_features.npy")
        for role, ds in (("known_train", train), ("known_validation", val)):
            np.save(run / f"{role}_flow_ids.npy", np.asarray([uid for uid, _ in ds.rows], dtype="U80"), allow_pickle=False)
        source.write_json(run / "metrics.json", {"status": "PASS", "best_epoch": best_epoch,
                                                  "completed_epochs": 200, "best_val_weighted_f1": best,
                                                  "checkpoint_sha256": sha(run / "model_best.pt"),
                                                  "train_features_sha256": sha(run / "known_train_features.npy"),
                                                  "val_features_sha256": sha(run / "known_validation_features.npy")})
        (run / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "dataset": args.dataset, "branch": "yatc",
                          "best_epoch": best_epoch, "val_weighted_f1": best}), flush=True)
    except BaseException as exc:
        source.write_json(run / "FAILURE.json", {"status": "FAIL", "error": repr(exc),
                                                    "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
