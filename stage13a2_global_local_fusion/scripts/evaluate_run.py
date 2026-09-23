#!/usr/bin/env python3
"""Evaluate fixed global-local fusion from one frozen Stage 13A-1 run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path

import numpy as np

from stage13a2_common import (
    CONFIG_PATH,
    METHODS,
    STAGE13A1_ROOT,
    STAGE_ROOT,
    SUMMARY,
    canonical_array_sha256,
    fixed_gl_fusion,
    method_metrics,
    read_json,
    robust_parameters,
    sha256_file,
    write_json,
)


def metric_parity(observed: dict[str, object], expected: dict[str, object]) -> float:
    names = ("threshold", "auroc", "auprc", "ufar", "known_frr", "binary_f1", "known_macro_f1")
    return max(abs(float(observed[name]) - float(expected[name])) for name in names)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    if args.seed != int(config["seeds"][args.fold]):
        raise ValueError("fold/seed pairing does not match the frozen campaign")
    provenance = read_json(SUMMARY / "provenance_verification.json")
    if provenance.get("status") != "PASS" or provenance.get("phase") != "before":
        raise RuntimeError("Before-phase provenance must PASS before evaluation")

    run_name = f"fold{args.fold}_seed{args.seed}"
    output = STAGE_ROOT / "artifacts" / args.scenario / run_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").is_file():
        return {"status": "SKIP_VERIFIED_SUCCESS", "output": str(output)}
    if (output / "FAILURE.json").is_file():
        raise RuntimeError(f"Refusing to overwrite preserved failure: {output}")

    source = STAGE13A1_ROOT / "artifacts" / args.scenario / run_name
    source_key = f"{args.scenario}/{run_name}"
    frozen = provenance["frozen_hashes_before"]["runs"][source_key]
    for name in ("results.json", "input_hashes.json", "score_arrays.npz"):
        if sha256_file(source / name) != frozen["artifacts"][name]:
            raise AssertionError(f"Frozen Stage13A-1 artifact changed: {source / name}")
    checkpoint_path = Path(frozen["checkpoint_path"])
    if sha256_file(checkpoint_path) != frozen["checkpoint_sha256"]:
        raise AssertionError("Frozen B0 checkpoint hash changed")

    source_result = read_json(source / "results.json")
    with np.load(source / "score_arrays.npz", allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    roles = ("validation", "known_test", "unknown_test")
    source_scores = {
        "CENTROID": {role: arrays[f"{role}_centroid"] for role in roles},
        "KNN_10": {role: arrays[f"{role}_knn10"] for role in roles},
    }
    for role in roles:
        if source_scores["CENTROID"][role].shape != source_scores["KNN_10"][role].shape:
            raise AssertionError(f"Frozen score shape mismatch for {role}")

    eps = float(config["normalization"]["eps"])
    parameters = {
        method: robust_parameters(source_scores[method]["validation"], eps)
        for method in ("CENTROID", "KNN_10")
    }
    normalized: dict[str, dict[str, np.ndarray]] = {"CENTROID": {}, "KNN_10": {}}
    fusion: dict[str, np.ndarray] = {}
    for role in roles:
        zc, zk, fused = fixed_gl_fusion(
            source_scores["CENTROID"][role],
            source_scores["KNN_10"][role],
            parameters["CENTROID"],
            parameters["KNN_10"],
        )
        normalized["CENTROID"][role] = zc
        normalized["KNN_10"][role] = zk
        fusion[role] = fused

    eval_seed = int(config["evaluation"]["eval_seed"])
    known_labels = arrays["known_test_labels_reindexed"]
    known_predictions = arrays["known_predictions_reindexed"]
    methods = {
        method: method_metrics(
            source_scores[method]["validation"],
            source_scores[method]["known_test"],
            source_scores[method]["unknown_test"],
            known_labels,
            known_predictions,
            eval_seed,
        )
        for method in ("CENTROID", "KNN_10")
    }
    methods["GL_FUSION"] = method_metrics(
        fusion["validation"],
        fusion["known_test"],
        fusion["unknown_test"],
        known_labels,
        known_predictions,
        eval_seed,
    )
    parity = {
        method: metric_parity(methods[method], source_result["methods"][method])
        for method in ("CENTROID", "KNN_10")
    }
    if max(parity.values()) > 1e-12:
        raise AssertionError(f"Frozen Stage13A-1 baseline metric parity failed: {parity}")

    result = {
        "evaluation_label": config["claim_scope"],
        "external_validation_label": config["external_validation_label"],
        "scenario": args.scenario,
        "fold": args.fold,
        "seed": args.seed,
        "status": "SUCCESS",
        "methods": methods,
        "deltas": {
            method: {
                metric: float(methods[method][metric]) - float(methods["CENTROID"][metric])
                for metric in ("auroc", "auprc", "ufar", "known_frr", "binary_f1", "known_macro_f1")
            }
            for method in ("KNN_10", "GL_FUSION")
        },
        "normalization": {
            "fit_split": "Known Validation only",
            "centroid": parameters["CENTROID"],
            "knn10": parameters["KNN_10"],
            "unknown_used": False,
            "test_used": False,
        },
        "fusion": {
            "formula": "0.5 * Zc + 0.5 * Zk",
            "centroid_weight": 0.5,
            "knn10_weight": 0.5,
            "weight_search": False,
            "learned": False,
        },
        "frozen_inputs": {
            "stage13a1_run": str(source.resolve()),
            "stage13a1_manifest_sha256": frozen["stage13a1_manifest_sha256"],
            "stage13a1_results_sha256": frozen["artifacts"]["results.json"],
            "stage13a1_input_hashes_sha256": frozen["artifacts"]["input_hashes.json"],
            "stage13a1_score_arrays_sha256": frozen["artifacts"]["score_arrays.npz"],
            "checkpoint_path": frozen["checkpoint_path"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "stage11b_run": frozen["stage11b_run_dir"],
            "stage11b_sample_outputs_sha256": frozen["stage11b_sample_outputs_sha256"],
            "split_array_sha256": frozen["split_array_sha256"],
            "known_classes": frozen["known_classes"],
            "unknown_classes": frozen["unknown_classes"],
        },
        "parity": {
            "centroid_metric_max_abs_error": parity["CENTROID"],
            "knn10_metric_max_abs_error": parity["KNN_10"],
            "status": "PASS",
        },
        "array_hashes": {
            **{
                f"source_{role}_{method.lower()}": canonical_array_sha256(source_scores[method][role])
                for method in ("CENTROID", "KNN_10")
                for role in roles
            },
            **{f"gl_{role}": canonical_array_sha256(fusion[role]) for role in roles},
        },
        "operations": {
            "new_encoder_training": False,
            "encoder_inference": False,
            "optimizer": None,
            "backward": False,
            "checkpoint_update": False,
            "normalization_fit_split": "Known Validation only",
            "unknown_calibration": False,
            "test_threshold_tuning": False,
            "weight_search": False,
            "allowed_weights": [0.5, 0.5],
            "learned_fusion": False,
            "logvar": False,
            "margin": False,
            "covariance": False,
            "k2": False,
            "external_test_datasets_read": [],
            "cipherspectrum_test_read": False,
            "frozen_experiment_modified": False,
        },
    }
    write_json(output / "config.json", {
        "scenario": args.scenario,
        "fold": args.fold,
        "seed": args.seed,
        "methods": list(METHODS),
        "normalization": config["normalization"],
        "fusion": config["fusion"],
        "threshold": config["threshold"],
        "evaluation": config["evaluation"],
    })
    write_json(output / "input_hashes.json", result["frozen_inputs"])
    write_json(output / "normalization_stats.json", result["normalization"])
    write_json(output / "thresholds.json", {
        method: {
            "threshold": methods[method]["threshold"],
            "source": "Known Validation score P95 only",
            "numpy_method": "higher",
        }
        for method in METHODS
    })
    np.savez_compressed(
        output / "score_arrays.npz",
        validation_centroid=source_scores["CENTROID"]["validation"],
        known_test_centroid=source_scores["CENTROID"]["known_test"],
        unknown_test_centroid=source_scores["CENTROID"]["unknown_test"],
        validation_knn10=source_scores["KNN_10"]["validation"],
        known_test_knn10=source_scores["KNN_10"]["known_test"],
        unknown_test_knn10=source_scores["KNN_10"]["unknown_test"],
        validation_zc=normalized["CENTROID"]["validation"],
        known_test_zc=normalized["CENTROID"]["known_test"],
        unknown_test_zc=normalized["CENTROID"]["unknown_test"],
        validation_zk=normalized["KNN_10"]["validation"],
        known_test_zk=normalized["KNN_10"]["known_test"],
        unknown_test_zk=normalized["KNN_10"]["unknown_test"],
        validation_gl=fusion["validation"],
        known_test_gl=fusion["known_test"],
        unknown_test_gl=fusion["unknown_test"],
        known_test_labels_reindexed=known_labels,
        unknown_test_labels_original=arrays["unknown_test_labels_original"],
        known_predictions_reindexed=known_predictions,
    )
    write_json(output / "results.json", result)
    (output / "SUCCESS").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {"status": "SUCCESS", "output": str(output), "scenario": args.scenario, "seed": args.seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a1", "a2", "a3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))
    except Exception as error:
        output = STAGE_ROOT / "artifacts" / args.scenario / f"fold{args.fold}_seed{args.seed}"
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILURE.json", {
            "status": "FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()

