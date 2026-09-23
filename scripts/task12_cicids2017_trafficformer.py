#!/usr/bin/env python3
"""Train a fresh CIC-IDS-2017 TrafficFormer head and export train/val z_t.

The model class and optimizer builder are imported from the pinned repository
copy of TrafficFormer/UER.  Training retains the actual USTC batch size of 32
on the user-selected physical GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TF_CODE = PROJECT_ROOT / "tf_runtime" / "code"
SEED = 7
SEQ_LENGTH = 320
EFFECTIVE_BATCH = 32
MICRO_BATCH = 32
EVAL_BATCH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime():
    sys.path.insert(0, str(TF_CODE))
    sys.path.insert(0, str(TF_CODE / "fine-tuning"))
    classifier = importlib.import_module("run_classifier")
    from uer.utils import str2tokenizer
    from uer.utils.config import load_hyperparam
    from uer.utils.constants import CLS_TOKEN
    from uer.utils.seed import set_seed
    return classifier, str2tokenizer, load_hyperparam, CLS_TOKEN, set_seed


def model_args(args: argparse.Namespace, labels_num: int) -> SimpleNamespace:
    ns = SimpleNamespace(
        vocab_path=str(args.vocab),
        spm_model_path=None,
        config_path=str(args.bert_config),
        pretrained_model_path=str(args.pretrained_model),
        output_model_path=str(args.output_dir / "best_checkpoint.bin"),
        train_path=str(args.input_dir / "train_dataset.tsv"),
        dev_path=str(args.input_dir / "valid_dataset.tsv"),
        test_path=None,
        embedding="word_pos_seg",
        encoder="transformer",
        mask="fully_visible",
        layernorm_positioning="post",
        feed_forward="dense",
        layernorm="normal",
        relative_position_embedding=False,
        relative_attention_buckets_num=32,
        remove_embedding_layernorm=False,
        remove_attention_scale=False,
        remove_transformer_bias=False,
        bidirectional=False,
        factorized_embedding_parameterization=False,
        parameter_sharing=False,
        learning_rate=6e-5,
        warmup=0.1,
        optimizer="adamw",
        scheduler="linear",
        fp16=False,
        fp16_opt_level="O1",
        batch_size=EFFECTIVE_BATCH,
        seq_length=SEQ_LENGTH,
        dropout=0.5,
        epochs_num=args.epochs,
        report_steps=100,
        seed=SEED,
        pooling="first",
        soft_targets=False,
        soft_alpha=0.5,
        labels_num=labels_num,
        is_moe=False,
        vocab_size=None,
        moebert_expert_dim=3072,
        moebert_expert_num=None,
        moebert_route_method="hash-random",
        moebert_route_hash_list=None,
        moebert_load_balance=0.0,
    )
    return ns


def cache_paths(cache_dir: Path, split: str) -> dict[str, Path]:
    return {
        "tokens": cache_dir / f"tokens_{split}.npy",
        "segments": cache_dir / f"segments_{split}.npy",
        "labels": cache_dir / f"labels_{split}.npy",
        "flow_ids": cache_dir / f"flow_ids_{split}.npy",
        "class_names": cache_dir / f"class_names_{split}.npy",
        "source_days": cache_dir / f"source_days_{split}.npy",
        "manifest": cache_dir / f"cache_{split}.json",
    }


def count_tsv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_flow_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for expected, row in enumerate(rows):
        if int(row["row_index"]) != expected:
            raise ValueError(f"non-contiguous flow map at {path}: {expected}")
    return rows


def build_cache(
    split: str,
    input_dir: Path,
    cache_dir: Path,
    tokenizer,
    cls_token: str,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir, split)
    tsv_path = input_dir / ("train_dataset.tsv" if split == "train" else "valid_dataset.tsv")
    map_path = input_dir / f"flow_map_{split}.csv"
    source_hashes = {"tsv": sha256_file(tsv_path), "flow_map": sha256_file(map_path)}
    if paths["manifest"].exists():
        old = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if old.get("source_hashes") == source_hashes and all(
            paths[name].exists() for name in ("tokens", "segments", "labels", "flow_ids", "class_names", "source_days")
        ):
            return old

    n = count_tsv(tsv_path)
    flow_rows = read_flow_map(map_path)
    if len(flow_rows) != n:
        raise ValueError(f"{split}: TSV rows {n} != flow map rows {len(flow_rows)}")
    tokens = np.lib.format.open_memmap(paths["tokens"], mode="w+", dtype=np.int32, shape=(n, SEQ_LENGTH))
    segments = np.lib.format.open_memmap(paths["segments"], mode="w+", dtype=np.uint8, shape=(n, SEQ_LENGTH))
    labels = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.int64, shape=(n,))
    tokens[:] = 0
    segments[:] = 0
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["label", "text_a"]:
            raise ValueError(f"unexpected TSV header: {reader.fieldnames}")
        for index, row in enumerate(reader):
            ids = tokenizer.convert_tokens_to_ids(
                [cls_token] + tokenizer.tokenize(row["text_a"])
            )[:SEQ_LENGTH]
            tokens[index, :len(ids)] = ids
            segments[index, :len(ids)] = 1
            labels[index] = int(row["label"])
            if int(flow_rows[index]["label_id"]) != int(labels[index]):
                raise ValueError(f"{split} label mismatch at row {index}")
            if (index + 1) % 50_000 == 0:
                print(f"tokenize {split}: {index + 1:,}/{n:,}", flush=True)
    tokens.flush()
    segments.flush()
    labels.flush()
    max_fid = max(len(row["flow_id"]) for row in flow_rows)
    max_class = max(len(row["class_name"]) for row in flow_rows)
    max_day = max(len(row["source_day"]) for row in flow_rows)
    np.save(paths["flow_ids"], np.asarray([row["flow_id"] for row in flow_rows], dtype=f"<U{max_fid}"))
    np.save(paths["class_names"], np.asarray([row["class_name"] for row in flow_rows], dtype=f"<U{max_class}"))
    np.save(paths["source_days"], np.asarray([row["source_day"] for row in flow_rows], dtype=f"<U{max_day}"))
    manifest = {
        "split": split,
        "rows": n,
        "seq_length": SEQ_LENGTH,
        "source_hashes": source_hashes,
        "paths": {key: str(value) for key, value in paths.items() if key != "manifest"},
        "tokenizer": "UER bert tokenizer",
        "construction": "[CLS] + tokenize(text_a), truncate to 320, token/segment tail padding 0",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_arrays(cache_dir: Path, split: str):
    paths = cache_paths(cache_dir, split)
    return (
        np.load(paths["tokens"], mmap_mode="r"),
        np.load(paths["segments"], mmap_mode="r"),
        np.load(paths["labels"], mmap_mode="r"),
    )


def tensor_batch(array: np.ndarray, indices: list[int], dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array[indices]), dtype=dtype, device="cuda:0")


def evaluate(model: nn.Module, arrays, batch_size: int) -> dict[str, float]:
    tokens, segments, labels = arrays
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            stop = min(len(labels), start + batch_size)
            indices = list(range(start, stop))
            src = tensor_batch(tokens, indices, torch.long)
            seg = tensor_batch(segments, indices, torch.long)
            tgt = tensor_batch(labels, indices, torch.long)
            loss, logits = model(src, tgt, seg)
            losses.append(float(loss.item()) * len(indices))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            truths.append(np.asarray(labels[start:stop]))
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    return {
        "loss": sum(losses) / len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> None:
    classifier, str2tokenizer, load_hyperparam, cls_token, set_seed = load_runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "token_cache"
    labels_payload = json.loads((args.input_dir / "label_map.json").read_text(encoding="utf-8"))
    labels_num = len(labels_payload["class_to_id"])
    margs = load_hyperparam(model_args(args, labels_num))
    margs.tokenizer = str2tokenizer["bert"](margs)
    build_cache("train", args.input_dir, cache_dir, margs.tokenizer, cls_token)
    build_cache("val", args.input_dir, cache_dir, margs.tokenizer, cls_token)
    train_arrays = load_arrays(cache_dir, "train")
    val_arrays = load_arrays(cache_dir, "val")

    set_seed(SEED)
    model = classifier.Classifier(margs)
    state = torch.load(args.pretrained_model, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    model.to("cuda:0")
    n_train = len(train_arrays[2])
    steps_per_epoch = math.ceil(n_train / EFFECTIVE_BATCH)
    margs.train_steps = int(n_train * args.epochs / EFFECTIVE_BATCH) + 1
    optimizer, scheduler = classifier.build_optimizer(margs, model)
    order = list(range(n_train))
    random.shuffle(order)  # Same one-time shuffle semantics as run_classifier.py.
    checkpoint = args.output_dir / "best_checkpoint.bin"
    metrics: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for step, batch_start in enumerate(range(0, n_train, EFFECTIVE_BATCH), 1):
            batch_indices = order[batch_start:batch_start + EFFECTIVE_BATCH]
            optimizer.zero_grad()
            effective_size = len(batch_indices)
            for micro_start in range(0, effective_size, MICRO_BATCH):
                indices = batch_indices[micro_start:micro_start + MICRO_BATCH]
                src = tensor_batch(train_arrays[0], indices, torch.long)
                seg = tensor_batch(train_arrays[1], indices, torch.long)
                tgt = tensor_batch(train_arrays[2], indices, torch.long)
                loss, _ = model(src, tgt, seg)
                (loss * (len(indices) / effective_size)).backward()
                loss_sum += float(loss.item()) * len(indices)
                sample_count += len(indices)
            optimizer.step()
            scheduler.step()
            if step % 500 == 0 or step == steps_per_epoch:
                print(
                    f"epoch={epoch} step={step}/{steps_per_epoch} "
                    f"train_loss={loss_sum / sample_count:.6f}",
                    flush=True,
                )
        val = evaluate(model, val_arrays, EVAL_BATCH)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "elapsed_seconds": time.time() - started,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        elif epoch - best_epoch >= args.earlystop:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    metrics_path = args.output_dir / "training_metrics.csv"
    save_csv(metrics_path, metrics, list(metrics[0]))
    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classes": labels_payload,
        "train_samples": int(len(train_arrays[2])),
        "val_samples": int(len(val_arrays[2])),
        "pretrained_checkpoint": str(args.pretrained_model.resolve()),
        "pretrained_sha256": sha256_file(args.pretrained_model),
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs_requested": args.epochs,
        "epochs_completed": len(metrics),
        "earlystop_patience": args.earlystop,
        "learning_rate": 6e-5,
        "optimizer": "adamw",
        "scheduler": "linear",
        "warmup": 0.1,
        "seed": SEED,
        "effective_batch_size": EFFECTIVE_BATCH,
        "micro_batch_size": MICRO_BATCH,
        "gradient_accumulation_steps": math.ceil(EFFECTIVE_BATCH / MICRO_BATCH),
        "eval_batch_size": EVAL_BATCH,
        "seq_length": SEQ_LENGTH,
        "pooling": "first",
        "embedding": "word_pos_seg",
        "encoder": "transformer",
        "mask": "fully_visible",
        "device": "cuda:0",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "load_missing_keys": list(incompatible.missing_keys),
        "load_unexpected_keys": list(incompatible.unexpected_keys),
        "no_test_split": True,
        "validation_usage": "early stopping/checkpoint selection and reporting only",
        "command": " ".join(sys.argv),
    }
    config_path = args.output_dir / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.publish_root:
        args.publish_root.mkdir(parents=True, exist_ok=True)
        (args.publish_root / "training_config.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        (args.publish_root / "training_metrics.csv").write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)


def cache_only(args: argparse.Namespace) -> None:
    _, str2tokenizer, load_hyperparam, cls_token, _ = load_runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_payload = json.loads((args.input_dir / "label_map.json").read_text(encoding="utf-8"))
    margs = load_hyperparam(model_args(args, len(labels_payload["class_to_id"])))
    margs.tokenizer = str2tokenizer["bert"](margs)
    cache_dir = args.output_dir / "token_cache"
    manifests = {
        split: build_cache(split, args.input_dir, cache_dir, margs.tokenizer, cls_token)
        for split in ("train", "val")
    }
    print(json.dumps(manifests, indent=2), flush=True)


class EncoderOnly(nn.Module):
    def __init__(self, args):
        super().__init__()
        from uer.layers import str2embedding
        from uer.encoders import str2encoder
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

    def forward(self, src, seg):
        output = self.encoder(self.embedding(src, seg), seg)
        return output[:, 0, :]


def export_embeddings(args: argparse.Namespace) -> None:
    _, str2tokenizer, load_hyperparam, cls_token, set_seed = load_runtime()
    labels_payload = json.loads((args.input_dir / "label_map.json").read_text(encoding="utf-8"))
    margs = load_hyperparam(model_args(args, len(labels_payload["class_to_id"])))
    margs.tokenizer = str2tokenizer["bert"](margs)
    cache_dir = args.output_dir / "token_cache"
    build_cache("train", args.input_dir, cache_dir, margs.tokenizer, cls_token)
    build_cache("val", args.input_dir, cache_dir, margs.tokenizer, cls_token)
    set_seed(SEED)
    model = EncoderOnly(margs)
    state = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise ValueError(f"checkpoint missing encoder keys: {incompatible.missing_keys[:5]}")
    model.to("cuda:0").eval()
    embed_dir = args.output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    summary = {
        "z_t_definition": "TrafficFormer encoder output[:, 0, :] before classification head",
        "dimension": int(margs.hidden_size),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "splits": {},
        "unexpected_checkpoint_keys": list(incompatible.unexpected_keys),
    }
    for split in ("train", "val"):
        arrays = load_arrays(cache_dir, split)
        n = len(arrays[2])
        out_path = embed_dir / f"embedding_{split}.npy"
        out = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(n, int(margs.hidden_size))
        )
        for start in range(0, n, EVAL_BATCH):
            stop = min(n, start + EVAL_BATCH)
            indices = list(range(start, stop))
            src = tensor_batch(arrays[0], indices, torch.long)
            seg = tensor_batch(arrays[1], indices, torch.long)
            with torch.no_grad():
                out[start:stop] = model(src, seg).cpu().numpy().astype(np.float32)
            if stop % 25_600 == 0 or stop == n:
                print(f"embedding {split}: {stop:,}/{n:,}", flush=True)
        out.flush()
        finite = True
        check = np.load(out_path, mmap_mode="r")
        for start in range(0, n, 50_000):
            finite = finite and bool(np.isfinite(check[start:start + 50_000]).all())
        if not finite:
            raise ValueError(f"non-finite embedding values in {split}")
        for name in ("labels", "flow_ids", "class_names", "source_days"):
            source = cache_paths(cache_dir, split)[name]
            destination = embed_dir / f"{name}_{split}.npy"
            if destination.exists():
                destination.unlink()
            os.link(source, destination)
        classes = np.load(embed_dir / f"class_names_{split}.npy", mmap_mode="r")
        days = np.load(embed_dir / f"source_days_{split}.npy", mmap_mode="r")
        counts = Counter(zip(classes.tolist(), days.tolist()))
        for (class_name, source_day), count in sorted(counts.items()):
            manifest_rows.append({
                "split": split,
                "class_name": class_name,
                "source_day": source_day,
                "samples": count,
                "z_t_dimension": int(margs.hidden_size),
                "embedding_path": str(out_path),
                "flow_ids_path": str(embed_dir / f"flow_ids_{split}.npy"),
                "labels_path": str(embed_dir / f"labels_{split}.npy"),
                "source_days_path": str(embed_dir / f"source_days_{split}.npy"),
                "finite": finite,
            })
        summary["splits"][split] = {
            "samples": n,
            "shape": [n, int(margs.hidden_size)],
            "dtype": "float32",
            "finite": finite,
            "embedding_sha256": sha256_file(out_path),
            "flow_ids_sha256": sha256_file(embed_dir / f"flow_ids_{split}.npy"),
            "labels_sha256": sha256_file(embed_dir / f"labels_{split}.npy"),
            "source_days_sha256": sha256_file(embed_dir / f"source_days_{split}.npy"),
        }
    (embed_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path = args.output_dir / "embedding_manifest.csv"
    save_csv(manifest_path, manifest_rows, list(manifest_rows[0]))
    if args.publish_root:
        (args.publish_root / "embedding_manifest.csv").write_text(
            manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "train", "embed"):
        item = sub.add_parser(name)
        item.add_argument("--input-dir", type=Path, required=True)
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--publish-root", type=Path)
        item.add_argument("--pretrained-model", type=Path, default=TF_CODE / "models/pretrained_model.bin")
        item.add_argument("--vocab", type=Path, default=TF_CODE / "models/encryptd_vocab.txt")
        item.add_argument("--bert-config", type=Path, default=TF_CODE / "models/bert/base_config.json")
        item.add_argument("--epochs", type=int, default=3)
        item.add_argument("--earlystop", type=int, default=3)
        if name == "embed":
            item.add_argument("--checkpoint", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.publish_root:
        args.publish_root = args.publish_root.resolve()
    if args.command == "cache":
        cache_only(args)
    elif args.command == "train":
        train(args)
    else:
        export_embeddings(args)


if __name__ == "__main__":
    main()
