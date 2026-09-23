#!/usr/bin/env python3
"""Evaluate R0-R3 on one frozen USTC v6 B0 checkpoint; never train."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch

from stage11b_common import (
    CONFIG_PATH,
    METHODS,
    STAGE_ROOT,
    array_hashes,
    build_loaders,
    evaluate_readouts,
    load_checkpoint,
    metric_abs_errors,
    model_for_method,
    read_json,
    set_reproducible,
    sha256_file,
    stage11a_expected,
    state_dict_sha256,
    write_csv,
    write_json,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    if args.seed != int(config["seeds"][args.fold]):
        raise ValueError("fold/seed pairing does not match the frozen v6 campaign")
    scenario_cfg = config["scenarios"][args.scenario]
    run_name = f"fold{args.fold}_seed{args.seed}"
    if args.smoke:
        output = STAGE_ROOT / "artifacts" / "smoke" / args.scenario / run_name
    else:
        output = STAGE_ROOT / "artifacts" / args.scenario / run_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").is_file():
        return {"status": "SKIP_VERIFIED_SUCCESS", "output": str(output)}
    if (output / "FAILURE.json").exists():
        raise RuntimeError(f"Refusing to overwrite preserved failed attempt: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen Open-Detect inference")

    inference_cfg = config["frozen_inference"]
    data_dir = Path(config["v6_split_root"]) / run_name
    b0_dir = Path(config["v6_frozen_root"]) / args.scenario / run_name
    stage11a_dir = Path(config["stage11a_root"]) / "artifacts" / "b0" / args.scenario / run_name
    required_b0 = ["SUCCESS", "model_best.pt", "run_config.json", "test_metrics.json"]
    required_stage11a = ["SUCCESS", "results.json", "density_models.joblib", "checkpoint.sha256"]
    if not all((b0_dir / name).is_file() for name in required_b0):
        raise FileNotFoundError(f"Frozen B0 run is incomplete: {b0_dir}")
    if not all((stage11a_dir / name).is_file() for name in required_stage11a):
        raise FileNotFoundError(f"Frozen Stage11A B0 diagnostic is incomplete: {stage11a_dir}")

    checkpoint_path = b0_dir / "model_best.pt"
    checkpoint_hash_before = sha256_file(checkpoint_path)
    stage11a_results = read_json(stage11a_dir / "results.json")
    if stage11a_results["method"] != "B0":
        raise AssertionError("Stage11A source is not a frozen B0 evaluation")
    if (
        stage11a_results["scenario"] != args.scenario
        or int(stage11a_results["fold"]) != args.fold
        or int(stage11a_results["seed"]) != args.seed
    ):
        raise AssertionError("Stage11A source identity mismatch")
    if stage11a_results["frozen_b0"]["checkpoint_sha256"] != checkpoint_hash_before:
        raise AssertionError("B0 checkpoint hash differs from Stage11A frozen evaluation")

    density_path = stage11a_dir / "density_models.joblib"
    if sha256_file(density_path) != stage11a_results["density_models_sha256"]:
        raise AssertionError("Stage11A frozen density-model hash mismatch")
    frozen_density = joblib.load(density_path)
    if set(frozen_density["models"]) != {"K1", "K2"}:
        raise AssertionError("Only frozen K1/K2 density models are allowed")

    set_reproducible(args.seed)
    loaders = build_loaders(
        data_dir,
        int(scenario_cfg["split"]),
        args.seed,
        int(inference_cfg["eval_batch_size"]),
        int(inference_cfg["eval_batch_size"]),
        int(inference_cfg["workers"]),
    )
    b0_cfg = read_json(b0_dir / "run_config.json")
    observed_arrays = array_hashes(loaders)
    if observed_arrays != b0_cfg["split_array_sha256"]:
        raise AssertionError("Split arrays do not match the frozen B0 run")
    if loaders.known_classes != list(map(int, b0_cfg["known_classes"])):
        raise AssertionError("Known class set differs from frozen B0")
    if loaders.unknown_classes != list(map(int, b0_cfg["unknown_classes"])):
        raise AssertionError("Unknown class set differs from frozen B0")

    device = torch.device("cuda:0")
    model = model_for_method(
        "B0", len(loaders.known_classes), int(inference_cfg["latent_dim"]), device
    )
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    in_memory_hash_before = state_dict_sha256(model)
    class_names = {int(key): value for key, value in config["class_names"].items()}
    evaluation, support_bundle, sample_arrays, per_class_rows, absorption_rows = evaluate_readouts(
        model,
        loaders,
        device,
        frozen_density,
        int(inference_cfg["eval_seed"]),
        class_names,
    )
    in_memory_hash_after = state_dict_sha256(model)
    checkpoint_hash_after = sha256_file(checkpoint_path)
    if in_memory_hash_after != in_memory_hash_before:
        raise AssertionError("Frozen model state changed during inference")
    if checkpoint_hash_after != checkpoint_hash_before:
        raise AssertionError("Frozen checkpoint file changed during inference")

    parity: dict[str, object] = {}
    all_errors: list[float] = []
    for method in ("R0", "R2", "R3"):
        errors = metric_abs_errors(
            evaluation["detectors"][method], stage11a_expected(stage11a_results, method)
        )
        parity[method] = errors
        all_errors.extend(errors.values())
    historical = read_json(b0_dir / "test_metrics.json")["open_world"]["balanced_1to1"]
    r0_detection = evaluation["detectors"]["R0"]["combined_balanced_1to1"]
    historical_errors = {
        "auroc_abs_error": abs(float(r0_detection["auroc"]) - float(historical["auroc"])),
        "threshold_abs_error": abs(
            float(r0_detection["threshold"])
            - float(historical["paper_validation_95pct"]["threshold"])
        ),
        "accuracy_abs_error": abs(
            float(r0_detection["accuracy"])
            - float(historical["paper_validation_95pct"]["accuracy"])
        ),
        "binary_f1_abs_error": abs(
            float(r0_detection["binary_f1"])
            - float(historical["paper_validation_95pct"]["f1"])
        ),
    }
    all_errors.extend(historical_errors.values())
    parity["historical_v6_R0"] = historical_errors
    parity["max_abs_error"] = max(all_errors)
    parity["tolerance"] = 1e-10
    parity["status"] = "PASS" if max(all_errors) <= 1e-10 else "FAIL"
    if parity["status"] != "PASS":
        raise AssertionError(f"Frozen representation/readout parity failed: {parity}")

    support_bundle["source_density_path"] = str(density_path.resolve())
    support_bundle["source_density_sha256"] = sha256_file(density_path)
    support_bundle["checkpoint_sha256"] = checkpoint_hash_before
    support_path = output / "support_models.joblib"
    joblib.dump(support_bundle, support_path, compress=3)
    thresholds = {
        method: {
            "threshold": evaluation["detectors"][method]["combined_balanced_1to1"]["threshold"],
            "source": "Known Validation anomaly score P95 only",
            "numpy_method": "higher",
        }
        for method in METHODS
    }
    write_json(output / "thresholds.json", thresholds)
    np.savez_compressed(output / "sample_outputs.npz", **sample_arrays)
    identity = {"scenario": args.scenario, "fold": args.fold, "seed": args.seed}
    write_csv(output / "per_class_support_analysis.csv", [{**identity, **row} for row in per_class_rows])
    write_csv(output / "unknown_absorption_analysis.csv", [{**identity, **row} for row in absorption_rows])
    write_json(output / "representation_parity.json", parity)
    result = {
        "evaluation_label": "DEVELOPMENT_RESULT",
        "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
        **identity,
        "mode": "SMOKE" if args.smoke else "FORMAL",
        "frozen_encoder": {
            "source_checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "model_state_sha256_before": in_memory_hash_before,
            "model_state_sha256_after": in_memory_hash_after,
            "optimizer": None,
            "backward": False,
            "loss_update": False,
            "checkpoint_update": False,
            "prototype_update": False,
            "new_encoder_training": False,
        },
        "classes": {
            "known": loaders.known_classes,
            "unknown": loaders.unknown_classes,
        },
        "split_array_sha256": observed_arrays,
        "data_roles": {
            "R1_fit": "Known Train deterministic mu only",
            "R2_R3_fit": "read-only Stage11A models fit on the same Known Train deterministic mu",
            "thresholds": "Known Validation P95 only",
            "known_test": "evaluation only",
            "unknown_test": "evaluation and post-hoc absorption analysis only",
        },
        "representation_parity": parity,
        "evaluation": evaluation,
        "artifacts": {
            "support_models": str(support_path.resolve()),
            "support_models_sha256": sha256_file(support_path),
            "sample_outputs": str((output / "sample_outputs.npz").resolve()),
            "source_stage11a_density": str(density_path.resolve()),
            "source_stage11a_density_sha256": sha256_file(density_path),
        },
        "cipher_spectrum_stage9_sample_level_test_used": False,
        "frozen_source_modified": False,
        "next_stage_started": False,
    }
    write_json(
        output / "config.json",
        {
            **identity,
            "mode": result["mode"],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash_before,
            "data_dir": str(data_dir.resolve()),
            "methods": list(METHODS),
            "optimizer": None,
            "threshold": config["threshold"],
            "density_protocol": config["density_protocol"],
            "unknown_use": "evaluation and post-hoc analysis only",
        },
    )
    write_json(output / "results.json", result)
    (output / "checkpoint.sha256").write_text(
        f"{checkpoint_hash_before}  {checkpoint_path.resolve()}\n", encoding="utf-8"
    )
    (output / "SUCCESS").write_text("STAGE11B_FROZEN_READOUT_SUCCESS\n", encoding="utf-8")
    return {
        "status": "SUCCESS",
        "output": str(output),
        "representation_parity": parity["status"],
        "new_encoder_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a1", "a2", "a3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    run_name = f"fold{args.fold}_seed{args.seed}"
    output = (
        STAGE_ROOT / "artifacts" / "smoke" / args.scenario / run_name
        if args.smoke
        else STAGE_ROOT / "artifacts" / args.scenario / run_name
    )
    try:
        result = run(args)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "FAILURE.json",
            {
                "status": "FAILED",
                "scenario": args.scenario,
                "fold": args.fold,
                "seed": args.seed,
                "mode": "SMOKE" if args.smoke else "FORMAL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "preserve_and_do_not_overwrite": True,
                "new_encoder_training": False,
                "cipher_spectrum_stage9_sample_level_test_used": False,
                "config_path": str(CONFIG_PATH.resolve()),
                "forbidden": config["forbidden"],
            },
        )
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
