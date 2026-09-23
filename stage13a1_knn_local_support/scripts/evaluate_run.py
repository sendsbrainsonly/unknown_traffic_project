#!/usr/bin/env python3
"""Evaluate centroid, kNN-5, and kNN-10 from one frozen Stage11B representation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path

import numpy as np

from stage13_common import (
    CONFIG_PATH,
    METHODS,
    STAGE11B_ROOT,
    STAGE_ROOT,
    SUMMARY,
    canonical_array_sha256,
    centroid_scores,
    empirical_centroids,
    local_knn_scores,
    method_metrics,
    read_json,
    sha256_file,
    write_json,
)


def max_abs_error(observed: dict[str, object], expected: dict[str, object]) -> float:
    expected_detection = expected["combined_balanced_1to1"]
    pairs = (
        (observed["auroc"], expected_detection["auroc"]),
        (observed["auprc"], expected_detection["auprc"]),
        (observed["ufar"], expected_detection["ufar"]),
        (observed["known_frr"], expected_detection["known_frr"]),
        (observed["binary_f1"], expected_detection["binary_f1"]),
        (observed["known_macro_f1"], expected["known_test"]["macro_f1"]),
        (observed["threshold"], expected_detection["threshold"]),
    )
    return max(abs(float(left) - float(right)) for left, right in pairs)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    if args.seed != int(config["seeds"][args.fold]):
        raise ValueError("fold/seed pairing does not match the frozen campaign")
    if args.scenario not in config["scenarios"]:
        raise ValueError(f"Unknown scenario: {args.scenario}")
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

    source = STAGE11B_ROOT / "artifacts" / args.scenario / run_name
    source_result = read_json(source / "results.json")
    source_manifest = read_json(source / "manifest.json")
    source_key = f"{args.scenario}/{run_name}"
    frozen = provenance["frozen_hashes_before"]["runs"][source_key]
    if sha256_file(source / "sample_outputs.npz") != frozen["artifacts"]["sample_outputs.npz"]:
        raise AssertionError("Frozen sample_outputs hash changed")
    checkpoint_path = Path(frozen["checkpoint_path"])
    if sha256_file(checkpoint_path) != frozen["checkpoint_sha256"]:
        raise AssertionError("Frozen B0 checkpoint hash changed")

    with np.load(source / "sample_outputs.npz", allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    train_mu = data["train_mu"]
    train_labels = data["train_labels_reindexed"]
    centroids = empirical_centroids(train_mu, train_labels)

    role_values = {
        "train": train_mu,
        "validation": data["validation_mu"],
        "known_test": data["known_test_mu"],
        "unknown_test": data["unknown_test_mu"],
    }
    centroid_by_role: dict[str, tuple[np.ndarray, np.ndarray]] = {
        role: centroid_scores(values, centroids) for role, values in role_values.items()
    }
    parity_score_errors = []
    parity_prediction_errors = []
    for role in ("validation", "known_test", "unknown_test"):
        scores, predictions = centroid_by_role[role]
        parity_score_errors.append(
            float(np.max(np.abs(scores - data[f"{role}_r1_scores"])))
        )
        parity_prediction_errors.append(
            int(np.sum(predictions != data[f"{role}_r1_predictions_reindexed"]))
        )
    if max(parity_score_errors) > 1e-6 or sum(parity_prediction_errors) != 0:
        raise AssertionError("DES-v0 centroid reconstruction does not match Stage11B R1")

    knn_by_role: dict[str, dict[int, np.ndarray]] = {}
    chunk_size = int(config["knn"]["chunk_size"])
    for role, values in role_values.items():
        knn_by_role[role] = local_knn_scores(
            values,
            centroid_by_role[role][1],
            train_mu,
            train_labels,
            config["knn"]["allowed_k"],
            query_train_indices=np.arange(len(train_mu), dtype=np.int64) if role == "train" else None,
            chunk_size=chunk_size,
        )

    method_scores = {
        "CENTROID": {role: centroid_by_role[role][0] for role in role_values},
        "KNN_5": {role: knn_by_role[role][5] for role in role_values},
        "KNN_10": {role: knn_by_role[role][10] for role in role_values},
    }
    known_predictions = centroid_by_role["known_test"][1]
    methods: dict[str, object] = {}
    for method in METHODS:
        methods[method] = method_metrics(
            method_scores[method]["validation"],
            method_scores[method]["known_test"],
            method_scores[method]["unknown_test"],
            data["known_test_labels_reindexed"],
            known_predictions,
            int(config["evaluation"]["eval_seed"]),
        )
        methods[method]["train_score_mean"] = float(method_scores[method]["train"].mean())
        methods[method]["train_score_median"] = float(np.median(method_scores[method]["train"]))

    baseline_error = max_abs_error(
        methods["CENTROID"], source_result["evaluation"]["detectors"]["R1"]
    )
    if baseline_error > 1e-10:
        raise AssertionError(f"Stage11B R1 metric parity failed: {baseline_error}")

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
            for method in ("KNN_5", "KNN_10")
        },
        "frozen_inputs": {
            "stage11b_run": str(source.resolve()),
            "stage11b_manifest_sha256": sha256_file(source / "manifest.json"),
            "sample_outputs_sha256": sha256_file(source / "sample_outputs.npz"),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "split_array_sha256": frozen["split_array_sha256"],
            "known_classes": frozen["known_classes"],
            "unknown_classes": frozen["unknown_classes"],
        },
        "parity": {
            "centroid_score_max_abs_error": max(parity_score_errors),
            "centroid_prediction_mismatches": sum(parity_prediction_errors),
            "centroid_metric_max_abs_error": baseline_error,
            "status": "PASS",
        },
        "array_hashes": {
            "train_mu": canonical_array_sha256(train_mu),
            "train_labels_reindexed": canonical_array_sha256(train_labels),
            "validation_mu": canonical_array_sha256(data["validation_mu"]),
            "known_test_mu": canonical_array_sha256(data["known_test_mu"]),
            "unknown_test_mu": canonical_array_sha256(data["unknown_test_mu"]),
        },
        "operations": {
            "new_encoder_training": False,
            "optimizer": None,
            "backward": False,
            "checkpoint_update": False,
            "self_neighbor_excluded_for_train": True,
            "allowed_k": config["knn"]["allowed_k"],
            "score_fusion": False,
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
        "knn": config["knn"],
        "threshold": config["threshold"],
        "evaluation": config["evaluation"],
        "source_manifest_status": source_manifest["status"],
    })
    write_json(output / "input_hashes.json", result["frozen_inputs"])
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
        validation_centroid=method_scores["CENTROID"]["validation"],
        known_test_centroid=method_scores["CENTROID"]["known_test"],
        unknown_test_centroid=method_scores["CENTROID"]["unknown_test"],
        validation_knn5=method_scores["KNN_5"]["validation"],
        known_test_knn5=method_scores["KNN_5"]["known_test"],
        unknown_test_knn5=method_scores["KNN_5"]["unknown_test"],
        validation_knn10=method_scores["KNN_10"]["validation"],
        known_test_knn10=method_scores["KNN_10"]["known_test"],
        unknown_test_knn10=method_scores["KNN_10"]["unknown_test"],
        known_test_labels_reindexed=data["known_test_labels_reindexed"],
        unknown_test_labels_original=data["unknown_test_labels_original"],
        known_predictions_reindexed=known_predictions,
    )
    write_json(output / "results.json", result)
    (output / "SUCCESS").write_text(
        dt.datetime.now(dt.timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    return {"status": "SUCCESS", "output": str(output), "scenario": args.scenario, "seed": args.seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a1", "a2", "a3"), required=True)
    parser.add_argument("--fold", type=int, required=True)
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

