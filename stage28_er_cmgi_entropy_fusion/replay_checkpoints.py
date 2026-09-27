#!/usr/bin/env python3
"""Independently replay saved adapter/fusion checkpoints on Known Validation."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from preflight import OUT, S27, load_pair, manifest_labels, sha256
from run_one import AdapterPair, Fusion, encode, infer, save_json

import importlib.util

spec = importlib.util.spec_from_file_location("stage27_replay_module", S27 / "run_one.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import Stage27 source")
stage27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage27)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    p.add_argument("--encoder-seed", type=int, required=True)
    p.add_argument("--training-seed", type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    result_path = run / "independent_checkpoint_replay.json"
    if result_path.exists() or not (run / "SUCCESS").is_file():
        raise RuntimeError("replay already exists or source run incomplete")
    pair = load_pair(args.dataset, args.encoder_seed, manifest_labels()[args.dataset])
    val = pair["roles"]["known_validation"]
    nclasses = len(pair["services"])
    with np.load(run / "known_train_normalization.npz", allow_pickle=False) as normalizer:
        content = ((val["semantic"] - normalizer["content_mean"]) / normalizer["content_std"]).astype(np.float32)
        behavior = ((val["behavioral"] - normalizer["behavior_mean"]) / normalizer["behavior_std"]).astype(np.float32)
    saved = np.load(run / "known_validation_logits.npz", allow_pickle=False)
    if not np.array_equal(saved["flow_ids"], val["flow_ids"]) or not np.array_equal(saved["labels"], val["labels"]):
        raise RuntimeError("saved validation IDs/labels mismatch")
    device = torch.device("cpu")
    adapters = {}
    for name in ("A0", "A1"):
        ckpt = torch.load(run / f"{name}_adapter_best.pt", map_location="cpu", weights_only=True)
        adapter = AdapterPair(nclasses, generative=name == "A1")
        adapter.load_state_dict(ckpt["state_dict"], strict=True)
        adapters[name] = encode(adapter, content, behavior, device)
    checks = {}
    for method, name in (("P0_A0_equal", "A0"), ("P1_A0_entropy", "A0"),
                         ("P2_A1_equal", "A1"), ("P3_A1_entropy", "A1")):
        ckpt = torch.load(run / f"{method}_best.pt", map_location="cpu", weights_only=True)
        model = Fusion(nclasses, method.endswith("entropy"),
                       np.asarray(ckpt["known_train_entropy_median"], dtype=np.float32),
                       np.asarray(ckpt["known_train_entropy_mad"], dtype=np.float32))
        model.load_state_dict(ckpt["state_dict"], strict=True)
        got, _ = infer(model, adapters[name], device, batch_size=23)
        expected = saved[method]
        maximum = float(np.max(np.abs(got - expected)))
        if maximum > 2e-4 or not np.array_equal(got.argmax(1), expected.argmax(1)):
            raise RuntimeError(f"checkpoint prediction replay mismatch: {method}, max={maximum}")
        checks[method] = {"max_abs_logit_difference": maximum,
                          "prediction_match": True,
                          "checkpoint_sha256": sha256(run / f"{method}_best.pt")}
    saved.close()
    # Replay the two retrained Stage27 baseline heads with the original Stage27 input order.
    baseline_x = np.concatenate((val["semantic"][:, :768], val["behavioral"],
                                 val["semantic"][:, 768:]), axis=1).astype(np.float32)
    with np.load(run / "baseline_known_validation_logits.npz", allow_pickle=False) as values:
        if not np.array_equal(values["flow_ids"], val["flow_ids"]):
            raise RuntimeError("baseline IDs mismatch")
        for method in stage27.VARIANTS[:2]:
            ckpt = torch.load(run / f"baseline_{method}_best.pt", map_location="cpu", weights_only=True)
            model = stage27.FeatureHead(method, nclasses)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            got, _ = stage27.predict_head(model, baseline_x, device)
            expected = values[method]
            maximum = float(np.max(np.abs(got - expected)))
            if maximum > 2e-4 or not np.array_equal(got.argmax(1), expected.argmax(1)):
                raise RuntimeError(f"baseline checkpoint replay mismatch: {method}, max={maximum}")
            checks[method] = {"max_abs_logit_difference": maximum,
                              "prediction_match": True,
                              "checkpoint_sha256": sha256(run / f"baseline_{method}_best.pt")}
    save_json(result_path, {"status": "PASS", "role": "known_validation", "methods": checks,
                            "test_usage": 0, "unknown_usage": 0})
    print(json.dumps({"status": "PASS", "dataset": args.dataset,
                      "encoder_seed": args.encoder_seed, "training_seed": args.training_seed,
                      "methods": len(checks),
                      "max_abs_logit_difference": max(v["max_abs_logit_difference"] for v in checks.values())}),
          flush=True)


if __name__ == "__main__":
    main()
