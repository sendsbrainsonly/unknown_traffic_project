#!/usr/bin/env python3
"""Reproduce epoch1 without changing the original learning-rate horizon."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ATTEMPT = Path(__file__).resolve().parent
ROOT = ATTEMPT.parent
sys.path.insert(0, str(ROOT))
import protocol
import worker


def transformed_function(source):
    text = inspect.getsource(source.train_tf_pretrained)
    tree = ast.parse(text)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch":
            found.append(node)
    if len(found) != 1 or ast.unparse(found[0].iter) != "range(1, epochs + 1)":
        raise RuntimeError("unexpected inherited outer epoch loop; no adaptation applied")
    before = ast.unparse(tree)
    found[0].iter = ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                            args=[ast.Constant(1), ast.Constant(2)], keywords=[])
    tree = ast.fix_missing_locations(tree)
    after = ast.unparse(tree)
    expected = before.replace("for epoch in range(1, epochs + 1):", "for epoch in range(1, 2):", 1)
    if after != expected:
        raise RuntimeError("adaptation changed more than the outer loop endpoint")
    namespace = dict(source.__dict__)
    exec(compile(tree, "<Stage49 authorized epoch1 replay>", "exec"), namespace)
    description = {"only_change": "outer loop range(1,epochs+1) -> range(1,2)",
                   "original_function_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "adapted_ast_sha256": hashlib.sha256(after.encode()).hexdigest(),
                   "scheduler_horizon_changed": False}
    return namespace["train_tf_pretrained"], description


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; do not fall back to CPU")
    started = time.monotonic()
    protocol.verify_sources()
    runner = worker.configure()
    runner.patch_modules()
    source = runner.tf_graph.source
    function, adaptation = transformed_function(source)
    config = source.read_json(source.CONFIG)
    if config["training"]["trafficformer"]["epochs"] != 20:
        raise RuntimeError("original20-epoch scheduler horizon changed")
    output = ATTEMPT / "selected_trafficformer"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    rows, classes, manifest_sha = runner.protocol_rows("ustc", "A-2")
    cache = worker.CIC / "input_caches/ustc/A-2/tf_fig"
    ids = np.load(cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if set(ids) != {uid for uid, _, _ in rows} or len(ids) != len(rows):
        raise RuntimeError("replay cache IDs do not match frozen Train/Val")
    index = {uid: i for i, uid in enumerate(ids)}
    tokens = np.load(cache / "token_ids.npy", mmap_mode="r", allow_pickle=False)
    segments = np.load(cache / "segments.npy", mmap_mode="r", allow_pickle=False)
    roles, arrays = {}, {}
    for role in ("known_train", "known_validation"):
        selected = sorted((uid, label) for uid, label, item_role in rows if item_role == role)
        positions = np.array([index[uid] for uid, _ in selected], dtype=np.int64)
        labels = np.array([classes.index(label) for _, label in selected], dtype=np.int64)
        roles[role] = selected
        arrays[role] = np.asarray(tokens[positions]), np.asarray(segments[positions]), labels
    protocol.write_json(output / "config.json", {"branch": "trafficformer", "actual_dataset": "CIC-IDS-2017",
              "classes": classes, "seed": 2022, "training_epochs_executed": 1, "scheduler_horizon_epochs": 20,
              "batch_size": 64, "learning_rate": 6e-5, "requested_checkpoint_epoch": 1,
              "selection": "user-selected original epoch1, validated by original first-epoch log",
              "frozen_split_sha256": manifest_sha, "adaptation": adaptation,
              "unknown_samples_used": 0, "test_parameter_selection": 0})
    model, history, epoch, load = function(arrays["known_train"], arrays["known_validation"],
                                         len(classes), 2022, device, output / "model_best.pt", config)
    if len(history) != 1 or epoch != 1:
        raise RuntimeError("not exactly one replayed epoch")
    reference = json.loads((ATTEMPT / "reference_epoch1.json").read_text())["original_epoch1"]
    parity = {}
    for key in ("train_loss", "train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_f1"):
        difference = abs(history[0][key] - reference[key])
        tolerance = 1e-7 if key == "train_loss" else 1e-12
        parity[key] = {"reference": reference[key], "replayed": history[0][key], "difference": difference, "tolerance": tolerance}
        if difference > tolerance:
            protocol.write_json(ATTEMPT / "replay_parity_failed.json", parity)
            raise RuntimeError(f"epoch1 metric parity failed: {key}")
    for role in roles:
        features, _ = source.infer_tf(model, arrays[role][0], arrays[role][1], device)
        if not np.isfinite(features).all():
            raise RuntimeError("nonfinite replay features")
        np.save(output / f"{role}_features.npy", features.astype(np.float32), allow_pickle=False)
        np.save(output / f"{role}_flow_ids.npy", np.asarray([uid for uid, _ in roles[role]], dtype="U80"), allow_pickle=False)
    protocol.write_json(output / "training_history.json", history)
    protocol.write_json(output / "pretrained_load_report.json", load)
    protocol.write_json(output / "metrics.json", {"status": "PASS", "completed_epochs": 1,
                        "scheduler_horizon_epochs": 20, "best_epoch": 1,
                        "checkpoint_sha256": protocol.sha(output / "model_best.pt"),
                        "elapsed_seconds": time.monotonic() - started,
                        "selection": "explicit user request; original epoch1 log parity verified"})
    protocol.verify_sources()
    protocol.write_json(ATTEMPT / "epoch1_replay_verification.json", {"status": "PASS", "selected_epoch": 1,
                        "parity": parity, "protocol_unchanged": True, "original_sources_unchanged": True,
                        "scheduler_horizon_epochs": 20, "training_epochs_executed": 1,
                        "weights_directly_exported_from_original_process": False,
                        "checkpoint_sha256": protocol.sha(output / "model_best.pt")})
    (output / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status": "PASS", "selected_epoch": 1, "parity": parity}), flush=True)


if __name__ == "__main__":
    main()
