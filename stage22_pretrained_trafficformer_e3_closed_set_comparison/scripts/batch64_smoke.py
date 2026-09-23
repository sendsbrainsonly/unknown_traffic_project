#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys

import numpy as np
import torch
import torch.nn as nn

from stage22_common import OUT, PRETRAINED_MODEL, PROJECT, cache_path, write_json

STAGE17_SCRIPTS = PROJECT / "stage17_encoder_recovery_and_open_set_pilot" / "scripts"
sys.path.insert(0, str(STAGE17_SCRIPTS))
from train_pilot_run import make_tf_args, tf_forward


def load_tf_module():
    path = PROJECT / "tf_runtime" / "code" / "fine-tuning" / "run_classifier.py"
    spec = importlib.util.spec_from_file_location("stage22_tf_run_classifier_batch64", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    cache = np.load(cache_path("iscx_vpn"), allow_pickle=False)
    module = load_tf_module()
    tf_args = make_tf_args(6, math.ceil(8764 / 64) * 20)
    tf_args.batch_size = 64
    model = module.Classifier(tf_args)
    state = torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device).train()
    optimizer, scheduler = module.build_optimizer(tf_args, model)
    src = torch.as_tensor(cache["token_ids"][:64], dtype=torch.long, device=device)
    seg = torch.as_tensor(cache["segments"][:64], dtype=torch.long, device=device)
    labels = torch.arange(64, device=device) % 6
    optimizer.zero_grad(set_to_none=True)
    logits, embedding = tf_forward(model, src, seg)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    result = {
        "status": "PASS" if torch.isfinite(loss).item() else "FAIL",
        "batch_size": 64,
        "loss": float(loss.item()),
        "logits_shape": list(logits.shape),
        "embedding_shape": list(embedding.shape),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(OUT / "batch64_smoke_result.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
