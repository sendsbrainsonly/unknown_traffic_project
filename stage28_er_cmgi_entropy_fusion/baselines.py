#!/usr/bin/env python3
"""Matched Known-Val YaTC and Stage27 F1/F2 head baselines for Stage28."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from preflight import OUT, S27, load_pair, manifest_labels, sha256, sources
from run_one import save_csv, save_json, scores

spec = importlib.util.spec_from_file_location("stage27_frozen_run_one", S27 / "run_one.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load Stage27 frozen baseline implementation")
stage27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage27)
STAGE27_VARIANTS = stage27.VARIANTS
fit_head = stage27.fit_head
predict_head = stage27.predict_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--encoder-seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--training-seed", type=int, choices=range(2022, 2027), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("no selected CUDA GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    run = OUT / "runs" / args.dataset / f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run / "SUCCESS").exists() or (run / "baseline_metrics.csv").exists():
        raise RuntimeError("Stage28 primary run not complete or baselines already exist")
    actual_sources = {name: sha256(path) for name, path in sources(args.dataset, args.encoder_seed).items()}
    expected_sources = json.loads((OUT / "frozen_input_hashes_before.json").read_text())
    if actual_sources != expected_sources[f"{args.dataset}/{args.encoder_seed}"]:
        raise RuntimeError("frozen source hash differs from Stage28 preflight")
    pair = load_pair(args.dataset, args.encoder_seed, manifest_labels()[args.dataset])
    tr = pair["roles"]["known_train"]
    va = pair["roles"]["known_validation"]
    # Stage27 exact input order: raw TrafficFormer, raw FIG, Known-Train-z-scored YaTC.
    feature = {}
    for role, item in (("known_train", tr), ("known_validation", va)):
        feature[role] = np.concatenate((item["semantic"][:, :768], item["behavioral"],
                                        item["semantic"][:, 768:]), axis=1).astype(np.float32)
    original = S27 / "runs" / args.dataset / f"seed{args.encoder_seed}" / "logits.npz"
    with np.load(original, allow_pickle=False) as values:
        yatc_logits = values["known_validation_YaTC"].astype(np.float32)
    if yatc_logits.shape != (len(va["labels"]), len(pair["services"])):
        raise RuntimeError("YaTC historical Known Val shape mismatch")
    rows, classes, predictions, history = [], [], [], []
    logits_to_save = {"YaTC": yatc_logits}
    for method in ("YaTC", *STAGE27_VARIANTS[:2]):
        if method == "YaTC":
            logits = yatc_logits
            best_epoch = -1  # Frozen historical YaTC checkpoint; no new selection.
        else:
            model, curve, best_epoch = fit_head(method, feature["known_train"], tr["labels"],
                                                feature["known_validation"], va["labels"],
                                                args.training_seed, device, run / f"baseline_{method}_best.pt")
            history.extend(curve)
            logits, _ = predict_head(model, feature["known_validation"], device)
            logits_to_save[method] = logits
        pred = logits.argmax(1)
        metric = scores(va["labels"], pred, len(pair["services"]))
        rows.append({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                     "training_seed": args.training_seed, "method": method,
                     "role": "known_validation", "samples": len(pred),
                     "best_epoch": best_epoch, **metric})
        p, r, f, n = precision_recall_fscore_support(
            va["labels"], pred, labels=np.arange(len(pair["services"])), zero_division=0)
        classes.extend({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                        "training_seed": args.training_seed, "method": method,
                        "class": name, "precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(n[i])}
                       for i, name in enumerate(pair["services"]))
        predictions.extend({"dataset": args.dataset, "encoder_seed": args.encoder_seed,
                            "training_seed": args.training_seed, "method": method,
                            "flow_id": str(flow_id), "true_service": pair["services"][int(actual)],
                            "predicted_service": pair["services"][int(chosen)],
                            "correct": int(actual == chosen)}
                           for flow_id, actual, chosen in zip(va["flow_ids"], va["labels"], pred, strict=True))
        print(json.dumps({"method": method, "val_macro_f1": metric["macro_f1"]}), flush=True)
    save_csv(run / "baseline_metrics.csv", rows)
    save_csv(run / "baseline_per_class.csv", classes)
    save_csv(run / "baseline_predictions.csv", predictions)
    save_csv(run / "baseline_training_history.csv", history)
    np.savez_compressed(run / "baseline_known_validation_logits.npz", flow_ids=va["flow_ids"],
                        labels=va["labels"], **logits_to_save)
    save_json(run / "baseline_verification.json", {
        "status": "PASS", "role": "known_validation", "test_values_loaded": 0,
        "unknown_values_loaded": 0, "original_stage27_logits_sha256": sha256(original),
        "F1_F2_retrained_head_seed": args.training_seed,
        "F1_F2_selection": "Known Validation fine Macro-F1",
        "checkpoint_sha256": {method: sha256(run / f"baseline_{method}_best.pt")
                              for method in STAGE27_VARIANTS[:2]},
    })


if __name__ == "__main__":
    main()
