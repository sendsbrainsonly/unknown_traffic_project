#!/usr/bin/env python3
"""Open Stage 12 test data once, then evaluate paired M0/M1 readouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from stage12_common import (
    CONFIG_PATH,
    STAGE_ROOT,
    array_digest,
    load_config,
    sha256_file,
    write_csv,
    write_json,
)
from train_pretest import (
    TrafficImages,
    collect_mu_logvar,
    des_scores,
    import_open_detect,
    load_npz,
    native_scores,
)


METHODS = ("M0_OPEN_DETECT_NATIVE", "M1_DES_V0")


def expected_runs(config: dict) -> list[tuple[str, str, int]]:
    runs = []
    for dataset in config["datasets"]:
        protocol = json.loads(
            (STAGE_ROOT / "protocol" / dataset / "unknown_class_protocol.json").read_text(encoding="utf-8")
        )
        for setting in protocol["settings"]:
            for seed in config["training_seeds"]:
                runs.append((dataset, setting, int(seed)))
    return runs


def evaluation_code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted([
        Path(__file__), Path(__file__).with_name("stage12_common.py"),
        Path(__file__).with_name("train_pretest.py"),
        Path(__file__).with_name("aggregate_results.py"), CONFIG_PATH,
    ]):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def verify_all_ready(config: dict) -> list[dict]:
    rows = []
    for dataset, setting, seed in expected_runs(config):
        run = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
        freeze_path = run / "pretest_freeze.json"
        if not (run / "READY_FOR_ONE_SHOT_TEST").is_file() or not freeze_path.is_file():
            raise RuntimeError(f"not ready: {run}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["status"] != "READY_FOR_ONE_SHOT_TEST" or freeze["test_metrics_computed"]:
            raise RuntimeError(f"invalid pretest state: {run}")
        if sha256_file(run / "model_best.pt") != freeze["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after freeze: {run}")
        if sha256_file(run / "des_centroids.npy") != freeze["des_centroids_sha256"]:
            raise RuntimeError(f"DES centroids changed after freeze: {run}")
        rows.append({
            "dataset": dataset, "setting": setting, "seed": seed,
            "run": str(run.resolve()), "pretest_freeze_sha256": sha256_file(freeze_path),
            "checkpoint_sha256": freeze["checkpoint_sha256"],
            "des_centroids_sha256": freeze["des_centroids_sha256"],
        })
    return rows


def record_opening(config: dict, ready: list[dict]) -> dict:
    path = STAGE_ROOT / "outputs" / "summary" / "FINAL_TEST_OPENING.json"
    code_hash = evaluation_code_hash()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["evaluation_code_hash"] != code_hash:
            raise RuntimeError("evaluation code changed after FINAL_TEST_OPENING")
        if existing["ready_runs"] != ready:
            raise RuntimeError("pretest run set changed after FINAL_TEST_OPENING")
        return existing
    opening = {
        "status": "FINAL_TEST_OPENED_ONCE",
        "opened_unix_time": time.time(),
        "expected_encoders": len(ready),
        "ready_runs": ready,
        "evaluation_code_hash": code_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "method_change_after_opening_allowed": False,
        "rerun_after_failure_allowed": False,
        "resume_same_frozen_code_after_interruption_allowed": True,
    }
    write_json(path, opening)
    return opening


def method_metrics(
    method: str,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_predictions: np.ndarray,
    known_true: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.concatenate([np.zeros(len(known_scores), dtype=np.int64), np.ones(len(unknown_scores), dtype=np.int64)])
    scores = np.concatenate([known_scores, unknown_scores])
    y_pred = (scores >= threshold).astype(np.int64)
    count = min(len(known_scores), len(unknown_scores))
    rng = np.random.default_rng(0)
    known_idx = rng.choice(len(known_scores), size=count, replace=False)
    unknown_idx = rng.choice(len(unknown_scores), size=count, replace=False)
    balanced_true = np.concatenate([np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)])
    balanced_scores = np.concatenate([known_scores[known_idx], unknown_scores[unknown_idx]])
    balanced_pred = (balanced_scores >= threshold).astype(np.int64)
    return {
        "known_accuracy": float(accuracy_score(known_true, known_predictions)),
        "known_macro_f1": float(f1_score(known_true, known_predictions, average="macro", zero_division=0)),
        "known_frr": float(np.mean(known_scores >= threshold)),
        "ufar": float(np.mean(unknown_scores < threshold)),
        "unknown_recall": float(np.mean(unknown_scores >= threshold)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "natural_binary_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_binary_f1": float(f1_score(balanced_true, balanced_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "known_test_samples": int(len(known_scores)),
        "unknown_test_samples": int(len(unknown_scores)),
        "balanced_each_side": int(count),
    }


def paired_bootstrap(
    m0_known: np.ndarray,
    m0_unknown: np.ndarray,
    m1_known: np.ndarray,
    m1_unknown: np.ndarray,
    threshold0: float,
    threshold1: float,
    iterations: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        known_idx = rng.integers(0, len(m0_known), size=len(m0_known))
        unknown_idx = rng.integers(0, len(m0_unknown), size=len(m0_unknown))
        y = np.concatenate([np.zeros(len(known_idx), dtype=np.int8), np.ones(len(unknown_idx), dtype=np.int8)])
        values = {}
        for method, known, unknown, threshold in (
            ("M0", m0_known, m0_unknown, threshold0),
            ("M1", m1_known, m1_unknown, threshold1),
        ):
            ks, us = known[known_idx], unknown[unknown_idx]
            combined = np.concatenate([ks, us])
            values[method] = {
                "auroc": float(roc_auc_score(y, combined)),
                "auprc": float(average_precision_score(y, combined)),
                "ufar": float(np.mean(us < threshold)),
                "known_frr": float(np.mean(ks >= threshold)),
            }
        for metric in ("auroc", "auprc", "ufar", "known_frr"):
            rows.append({
                "iteration": iteration,
                "metric": metric,
                "m0": values["M0"][metric],
                "m1": values["M1"][metric],
                "delta_m1_minus_m0": values["M1"][metric] - values["M0"][metric],
            })
    summary = {}
    for metric in ("auroc", "auprc", "ufar", "known_frr"):
        deltas = np.asarray([row["delta_m1_minus_m0"] for row in rows if row["metric"] == metric])
        summary[metric] = {
            "delta_mean": float(np.mean(deltas)),
            "ci95_low": float(np.quantile(deltas, 0.025)),
            "ci95_high": float(np.quantile(deltas, 0.975)),
            "iterations": iterations,
            "seed": seed,
        }
    return rows, summary


def evaluate_run(dataset: str, setting: str, seed: int, opening: dict, config: dict) -> dict:
    run = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
    success = run / "SUCCESS"
    if success.is_file():
        existing = json.loads((run / "final_results.json").read_text(encoding="utf-8"))
        if existing["final_test_opening_code_hash"] != opening["evaluation_code_hash"]:
            raise RuntimeError(f"completed result has wrong opening hash: {run}")
        return existing
    protocol_dir = STAGE_ROOT / "protocol" / dataset
    data_dir = STAGE_ROOT / "artifacts" / dataset / "protocol" / setting
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((run / "pretest_freeze.json").read_text(encoding="utf-8"))
    known_x, known_y = load_npz(data_dir / "known_test.npz")
    unknown_x, unknown_y = load_npz(data_dir / "unknown_test.npz")
    if array_digest(known_x, known_y) != freeze["known_test_array_sha256_frozen_not_opened"]:
        raise RuntimeError("Known Test array differs from pretest freeze")
    if array_digest(unknown_x, unknown_y) != freeze["unknown_test_array_sha256_frozen_not_opened"]:
        raise RuntimeError("Unknown Test array differs from pretest freeze")
    label_map = json.loads((protocol_dir / "canonical_label_map.json").read_text(encoding="utf-8"))[
        "canonical_class_to_label"
    ]
    inverse = {int(value): key for key, value in label_map.items()}
    known_names = list(protocol["known_classes"])
    known_labels = [int(label_map[name]) for name in known_names]
    known_set = TrafficImages(known_x, known_y, known_labels, train=False)
    # Unknown labels are retained outside the dataset; images alone enter the encoder.
    dummy_unknown = np.full(len(unknown_y), known_labels[0], dtype=np.int64)
    unknown_set = TrafficImages(unknown_x, dummy_unknown, known_labels, train=False)
    loader_args = {"batch_size": int(config["training"]["eval_batch_size"]), "shuffle": False,
                   "num_workers": 0, "pin_memory": True}
    known_loader = DataLoader(known_set, **loader_args)
    unknown_loader = DataLoader(unknown_set, **loader_args)
    CorrectedOpenDetectNet, _, _, _ = import_open_detect(config)
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        config["training"]["architecture"], 1, int(config["training"]["latent_dim"]),
        len(known_labels), 1, 1,
    ).to(device)
    checkpoint = torch.load(run / "model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    known_mu, known_logvar, known_true_reindexed = collect_mu_logvar(model, known_loader, device)
    unknown_mu, unknown_logvar, _ = collect_mu_logvar(model, unknown_loader, device)
    centroids = np.load(run / "des_centroids.npy", allow_pickle=False)
    m0_known, m0_known_pred = native_scores(model, known_mu, known_logvar, device)
    m0_unknown, m0_unknown_pred = native_scores(model, unknown_mu, unknown_logvar, device)
    m1_known, m1_known_pred = des_scores(known_mu, centroids)
    m1_unknown, m1_unknown_pred = des_scores(unknown_mu, centroids)
    thresholds = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
    metrics = {
        "M0_OPEN_DETECT_NATIVE": method_metrics(
            "M0", m0_known, m0_unknown, m0_known_pred, known_true_reindexed,
            thresholds["M0_OPEN_DETECT_NATIVE"]["threshold"],
        ),
        "M1_DES_V0": method_metrics(
            "M1", m1_known, m1_unknown, m1_known_pred, known_true_reindexed,
            thresholds["M1_DES_V0"]["threshold"],
        ),
    }
    per_unknown = []
    absorption = []
    score_pairs = {
        "M0_OPEN_DETECT_NATIVE": (m0_known, m0_unknown, m0_unknown_pred),
        "M1_DES_V0": (m1_known, m1_unknown, m1_unknown_pred),
    }
    for method, (known_scores, unknown_scores, unknown_pred) in score_pairs.items():
        threshold = metrics[method]["threshold"]
        for label in sorted(map(int, np.unique(unknown_y))):
            mask = unknown_y == label
            class_scores = unknown_scores[mask]
            y = np.concatenate([np.zeros(len(known_scores), dtype=np.int8), np.ones(int(mask.sum()), dtype=np.int8)])
            scores = np.concatenate([known_scores, class_scores])
            per_unknown.append({
                "dataset": dataset, "setting": setting, "seed": seed, "method": method,
                "unknown_label": label, "unknown_class": inverse[label], "n": int(mask.sum()),
                "ufar": float(np.mean(class_scores < threshold)),
                "mean_anomaly_score": float(np.mean(class_scores)),
                "median_anomaly_score": float(np.median(class_scores)),
                "auroc_vs_all_known": float(roc_auc_score(y, scores)),
            })
            accepted = mask & (unknown_scores < threshold)
            accepted_total = int(accepted.sum())
            for predicted_index, predicted_name in enumerate(known_names):
                count = int(np.sum(accepted & (unknown_pred == predicted_index)))
                absorption.append({
                    "dataset": dataset, "setting": setting, "seed": seed, "method": method,
                    "true_unknown_class": inverse[label], "predicted_known_class": predicted_name,
                    "true_unknown_samples": int(mask.sum()), "accepted_unknown_samples": accepted_total,
                    "count": count, "rate_all_unknown_class": count / int(mask.sum()),
                    "share_accepted": count / accepted_total if accepted_total else 0.0,
                })
    bootstrap_rows, bootstrap_summary = paired_bootstrap(
        m0_known, m0_unknown, m1_known, m1_unknown,
        metrics["M0_OPEN_DETECT_NATIVE"]["threshold"],
        metrics["M1_DES_V0"]["threshold"],
        int(config["bootstrap"]["iterations"]), int(config["bootstrap"]["seed"]),
    )
    identity = {"dataset": dataset, "setting": setting, "seed": seed}
    write_csv(run / "per_unknown_results.csv", per_unknown)
    write_csv(run / "absorption_matrix.csv", absorption)
    write_csv(run / "paired_bootstrap.csv", [{**identity, **row} for row in bootstrap_rows])
    np.savez_compressed(
        run / "final_test_sample_outputs.npz",
        known_labels_original=known_y, unknown_labels_original=unknown_y,
        known_mu=known_mu, known_logvar=known_logvar, unknown_mu=unknown_mu, unknown_logvar=unknown_logvar,
        m0_known_scores=m0_known, m0_unknown_scores=m0_unknown,
        m1_known_scores=m1_known, m1_unknown_scores=m1_unknown,
        m0_known_predictions_reindexed=m0_known_pred, m0_unknown_predictions_reindexed=m0_unknown_pred,
        m1_known_predictions_reindexed=m1_known_pred, m1_unknown_predictions_reindexed=m1_unknown_pred,
    )
    deltas = {
        metric: metrics["M1_DES_V0"][metric] - metrics["M0_OPEN_DETECT_NATIVE"][metric]
        for metric in ("auroc", "auprc", "ufar", "known_frr", "known_macro_f1", "known_accuracy")
    }
    result = {
        **identity,
        "status": "FINAL_TEST_COMPLETE",
        "methods": metrics,
        "delta_m1_minus_m0": deltas,
        "bootstrap": bootstrap_summary,
        "known_classes": known_names,
        "unknown_classes": protocol["unknown_classes"],
        "split_policy": protocol["split_policy"],
        "group_aware_not_feasible_classes": protocol["group_aware_not_feasible_classes"],
        "checkpoint_sha256": sha256_file(run / "model_best.pt"),
        "des_centroids_sha256": sha256_file(run / "des_centroids.npy"),
        "final_test_opening_code_hash": opening["evaluation_code_hash"],
        "post_test_method_modification": False,
        "forbidden_methods_run": False,
    }
    write_json(run / "final_results.json", result)
    with (run / "RESULTS.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## One-shot final test result\n\n")
        handle.write(f"- M0 AUROC/AUPRC/UFAR/FRR: `{metrics['M0_OPEN_DETECT_NATIVE']['auroc']:.6f}` / `{metrics['M0_OPEN_DETECT_NATIVE']['auprc']:.6f}` / `{metrics['M0_OPEN_DETECT_NATIVE']['ufar']:.6f}` / `{metrics['M0_OPEN_DETECT_NATIVE']['known_frr']:.6f}`\n")
        handle.write(f"- M1 AUROC/AUPRC/UFAR/FRR: `{metrics['M1_DES_V0']['auroc']:.6f}` / `{metrics['M1_DES_V0']['auprc']:.6f}` / `{metrics['M1_DES_V0']['ufar']:.6f}` / `{metrics['M1_DES_V0']['known_frr']:.6f}`\n")
        handle.write(f"- Delta AUROC (M1-M0): `{deltas['auroc']:+.6f}`\n")
        handle.write("- Post-test method modification: `NO`\n")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "success"
    manifest["final_results"] = result
    artifact_names = sorted({
        *(item["path"] for item in manifest["artifacts"]),
        "final_results.json", "per_unknown_results.csv", "absorption_matrix.csv",
        "paired_bootstrap.csv", "final_test_sample_outputs.npz", "RESULTS.md",
    })
    manifest["artifacts"] = [
        {"path": name, "sha256": sha256_file(run / name)} for name in artifact_names
    ]
    write_json(run / "manifest.json", manifest)
    success.write_text("STAGE12_ONE_SHOT_TEST_SUCCESS\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-one-shot-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_one_shot_test:
        raise RuntimeError("explicit --confirm-one-shot-test is required")
    config = load_config()
    ready = verify_all_ready(config)
    opening = record_opening(config, ready)
    results = []
    for position, identity in enumerate(expected_runs(config), start=1):
        dataset, setting, seed = identity
        print(f"test_open_progress={position}/{len(ready)} run={dataset}/{setting}/{seed}", flush=True)
        try:
            results.append(evaluate_run(dataset, setting, seed, opening, config))
        except BaseException as exc:
            run = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
            write_json(run / "FINAL_TEST_FAILURE.json", {
                "status": "FAILED_AFTER_TEST_OPENING",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "method_change_or_rerun_forbidden": True,
                "same_frozen_code_resume_only": True,
                "opening_code_hash": opening["evaluation_code_hash"],
            })
            raise
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("aggregate_results.py"))], cwd=STAGE_ROOT, check=True)
    print(json.dumps({"status": "ALL_FINAL_TESTS_COMPLETE", "runs": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
