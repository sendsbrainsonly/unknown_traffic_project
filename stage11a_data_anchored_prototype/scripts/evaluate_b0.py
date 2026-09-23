#!/usr/bin/env python3
"""Evaluate one frozen B0 checkpoint without changing the Open-Detect project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage11a_common import (
    CONFIG_PATH,
    STAGE_ROOT,
    array_hashes,
    build_loaders,
    evaluate_checkpoint,
    load_checkpoint,
    model_for_method,
    read_json,
    set_reproducible,
    sha256_file,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a1", "a2", "a3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    if args.seed != config["seeds"][args.fold]:
        raise ValueError("fold/seed pairing does not match frozen v6")
    output = STAGE_ROOT / "artifacts" / "b0" / args.scenario / f"fold{args.fold}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").is_file():
        print(json.dumps({"status": "SKIP_VERIFIED_SUCCESS", "output": str(output)}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen Open-Detect evaluation")
    device = torch.device("cuda:0")
    set_reproducible(args.seed)
    scenario_cfg = config["scenarios"][args.scenario]
    train_cfg = config["training"]
    data_dir = Path(config["v6_split_root"]) / f"fold{args.fold}_seed{args.seed}"
    b0_dir = Path(config["v6_frozen_root"]) / args.scenario / f"fold{args.fold}_seed{args.seed}"
    b0_cfg = read_json(b0_dir / "run_config.json")
    loaders = build_loaders(
        data_dir,
        int(scenario_cfg["split"]),
        args.seed,
        int(train_cfg["batch_size"]),
        int(train_cfg["eval_batch_size"]),
        int(train_cfg["workers"]),
    )
    if array_hashes(loaders) != b0_cfg["split_array_sha256"]:
        raise RuntimeError("B0 evaluation arrays do not match frozen run_config hashes")
    model = model_for_method("B0", len(loaders.known_classes), int(train_cfg["latent_dim"]), device)
    checkpoint_path = b0_dir / "model_best.pt"
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    results = evaluate_checkpoint(
        model,
        loaders,
        device,
        "B0",
        args.scenario,
        args.seed,
        args.fold,
        output / "density_models.joblib",
    )
    frozen_metrics = read_json(b0_dir / "test_metrics.json")
    frozen_primary = frozen_metrics["open_world"]["balanced_1to1"]
    recomputed = results["native"]["detection"]
    parity = {
        "auroc_abs_error": abs(recomputed["auroc"] - frozen_primary["auroc"]),
        "accuracy_abs_error": abs(
            recomputed["accuracy"] - frozen_primary["paper_validation_95pct"]["accuracy"]
        ),
        "binary_f1_abs_error": abs(
            recomputed["binary_f1"] - frozen_primary["paper_validation_95pct"]["f1"]
        ),
        "threshold_abs_error": abs(
            recomputed["threshold"] - frozen_primary["paper_validation_95pct"]["threshold"]
        ),
    }
    parity["status"] = "PASS" if max(parity.values()) <= 1e-10 else "FAIL"
    if parity["status"] != "PASS":
        raise RuntimeError(f"B0 Native parity failed: {parity}")
    results["frozen_b0"] = {
        "source_run": str(b0_dir.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "validation_accuracy": float(checkpoint["validation_accuracy"]),
        "native_recompute_parity": parity,
        "retrained": False,
        "source_modified": False,
    }
    write_json(
        output / "config.json",
        {
            "evaluation_label": "DEVELOPMENT_RESULT",
            "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
            "scenario": args.scenario,
            "fold": args.fold,
            "seed": args.seed,
            "method": "B0",
            "source_run": str(b0_dir.resolve()),
            "data_dir": str(data_dir.resolve()),
            "checkpoint_selection": "already frozen Known Validation Accuracy",
            "unknown_use": "evaluation only",
        },
    )
    write_json(output / "results.json", results)
    (output / "checkpoint.sha256").write_text(
        f"{results['frozen_b0']['checkpoint_sha256']}  {checkpoint_path}\n", encoding="utf-8"
    )
    (output / "SUCCESS").write_text("STAGE11A_B0_READ_ONLY_EVALUATION_SUCCESS\n", encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "output": str(output), "parity": parity}, indent=2))


if __name__ == "__main__":
    main()
