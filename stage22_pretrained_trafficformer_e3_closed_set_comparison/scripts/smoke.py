#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys

import numpy as np
import torch

from stage22_common import CONFIG, OUT, PRETRAINED_MODEL, PROJECT, cache_path, read_json, sha256_file, write_json

STAGE17_SCRIPTS = PROJECT / "stage17_encoder_recovery_and_open_set_pilot" / "scripts"
sys.path.insert(0, str(STAGE17_SCRIPTS))
from train_pilot_run import make_tf_args, tf_forward


def load_tf_module():
    path = PROJECT / "tf_runtime" / "code" / "fine-tuning" / "run_classifier.py"
    spec = importlib.util.spec_from_file_location("stage22_tf_run_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read_json(CONFIG)
    device = torch.device(args.device)
    cache = np.load(cache_path("iscx_vpn"), allow_pickle=False)
    module = load_tf_module()
    tf_args = make_tf_args(6, math.ceil(64 / 64) * 20)
    tf_args.batch_size = 64
    model = module.Classifier(tf_args)
    state = torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    src = torch.as_tensor(cache["token_ids"][:2], dtype=torch.long, device=device)
    seg = torch.as_tensor(cache["segments"][:2], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, z = tf_forward(model, src, seg)
    result = {
        "status": "PASS",
        "pretrained_sha256": sha256_file(PRETRAINED_MODEL),
        "expected_sha256": config["pretrained_model_sha256"],
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "logits_shape": list(logits.shape),
        "embedding_shape": list(z.shape),
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "finite_embedding": bool(torch.isfinite(z).all().item()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    if result["pretrained_sha256"] != result["expected_sha256"] or not result["finite_logits"] or not result["finite_embedding"]:
        result["status"] = "FAIL"
    write_json(OUT / "smoke_result.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
