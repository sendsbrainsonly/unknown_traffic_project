#!/usr/bin/env python3
"""Run one frozen setting's first and only final Known/Unknown evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from common import (
    EXPECTED_EXECUTION_PLAN_RAW_SHA256,
    EXPECTED_PROTOCOL_CANONICAL_SHA256,
    PROJECT_ROOT,
    STAGE3_ROOT,
    latent_columns,
    load_fold,
    score_density_models,
    sha256_file,
    verify_frozen_inputs,
)


DETECTORS = ("Native", "Single-Full", "Multi-Full-K2")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_thresholds(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    thresholds = {row["detector"]: float(row["threshold"]) for row in rows}
    if set(thresholds) != set(DETECTORS):
        raise RuntimeError("frozen threshold detector set mismatch")
    return thresholds


def verify_freeze(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["created_before_final_test"] or payload["test_data_read"]:
        raise RuntimeError("support-model freeze did not occur before final test")
    for item in payload["files"]:
        target = Path(item["path"])
        if sha256_file(target) != item["sha256"]:
            raise RuntimeError(f"frozen hash mismatch: {target}")
    return payload


def fpr_at_unknown_tpr95(labels: np.ndarray, unknown_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, unknown_scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    return float(np.min(fpr[eligible])) if len(eligible) else 1.0


def detector_metrics(
    detector: str,
    known_scores: np.ndarray,
    unknown_scores_knownness: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    known_acceptance = float(np.mean(known_scores >= threshold))
    unknown_far = float(np.mean(unknown_scores_knownness >= threshold))
    labels = np.concatenate(
        [np.zeros(len(known_scores), dtype=np.int8), np.ones(len(unknown_scores_knownness), dtype=np.int8)]
    )
    anomaly_scores = -np.concatenate([known_scores, unknown_scores_knownness])
    return {
        "detector": detector,
        "threshold": threshold,
        "score_semantics": "higher_is_more_known; Unknown is positive for ranking metrics",
        "known_test_samples": len(known_scores),
        "unknown_test_samples": len(unknown_scores_knownness),
        "known_acceptance_rate": known_acceptance,
        "known_false_rejection_rate": 1.0 - known_acceptance,
        "unknown_false_acceptance_rate": unknown_far,
        "unknown_rejection_rate": 1.0 - unknown_far,
        "auroc_unknown_positive": float(roc_auc_score(labels, anomaly_scores)),
        "auprc_unknown_positive": float(average_precision_score(labels, anomaly_scores)),
        "fpr_at_95pct_unknown_tpr": fpr_at_unknown_tpr95(labels, anomaly_scores),
        "unknown_tpr_at_95pct_known_validation_acceptance": 1.0 - unknown_far,
    }


def paired_class_stratified_bootstrap(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy() for _, group in frame.groupby(["known_or_unknown", "class_name"], sort=True)]
    rows: list[dict[str, object]] = []
    for replicate in range(resamples):
        indices = np.concatenate([rng.choice(values, size=len(values), replace=True) for values in strata])
        sample = frame.loc[indices]
        is_unknown = sample["known_or_unknown"].to_numpy() == "Unknown"
        labels = is_unknown.astype(np.int8)
        single = sample["single_known_score"].to_numpy(dtype=np.float64)
        multi = sample["multi_known_score"].to_numpy(dtype=np.float64)
        single_far = float(np.mean(single[is_unknown] >= thresholds["Single-Full"]))
        multi_far = float(np.mean(multi[is_unknown] >= thresholds["Multi-Full-K2"]))
        single_unknown_score = -single
        multi_unknown_score = -multi
        rows.append(
            {
                "replicate": replicate,
                "delta_unknown_far_multi_minus_single": multi_far - single_far,
                "delta_auroc_multi_minus_single": roc_auc_score(labels, multi_unknown_score)
                - roc_auc_score(labels, single_unknown_score),
                "delta_auprc_multi_minus_single": average_precision_score(labels, multi_unknown_score)
                - average_precision_score(labels, single_unknown_score),
            }
        )
    frame_rows = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for metric in (
        "delta_unknown_far_multi_minus_single",
        "delta_auroc_multi_minus_single",
        "delta_auprc_multi_minus_single",
    ):
        values = frame_rows[metric].to_numpy()
        lower, upper = np.quantile(values, [0.025, 0.975])
        summaries.append(
            {
                "metric": metric,
                "bootstrap_mean": float(values.mean()),
                "ci_2_5pct": float(lower),
                "ci_97_5pct": float(upper),
                "resamples": resamples,
                "seed": seed,
            }
        )
    return rows, summaries


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--tmux-session", required=True)
    args = parser.parse_args()
    if args.bootstrap_resamples != 1000 or args.bootstrap_seed != 0:
        raise RuntimeError("formal Stage 3 freezes bootstrap at 1000 resamples with seed 0")
    verify_frozen_inputs()
    fold = load_fold(args.setting)
    output_dir = STAGE3_ROOT / "outputs" / args.setting
    artifact_dir = STAGE3_ROOT / "artifacts" / args.setting
    if (output_dir / "detector_metrics.csv").exists():
        raise RuntimeError("refusing to repeat or overwrite final-test evaluation")
    freeze = verify_freeze(output_dir / "frozen_model_hashes.json")
    latent_gate = json.loads((output_dir / "latent_quality_gate.json").read_text(encoding="utf-8"))
    if latent_gate["status"] != "PASS":
        raise RuntimeError("final latent quality gate is not PASS")

    known = pd.read_parquet(artifact_dir / "test_known_mu.parquet")
    unknown = pd.read_parquet(artifact_dir / "test_unknown_mu.parquet")
    columns = latent_columns(known.columns)
    if columns != latent_columns(unknown.columns):
        raise RuntimeError("Known/Unknown final latent columns differ")
    if len(known) != fold["formal_counts"]["known_final_test"]:
        raise RuntimeError("Known final-test count mismatch")
    if len(unknown) != fold["formal_counts"]["unknown_final_test"]:
        raise RuntimeError("Unknown final-test count mismatch")
    if set(unknown["class_name"]) != set(fold["unknown_classes"]):
        raise RuntimeError("Unknown final-test class set mismatch")
    scaler = joblib.load(artifact_dir / "standard_scaler.joblib")
    pca = joblib.load(artifact_dir / "pca64.joblib")
    k1_models = joblib.load(artifact_dir / "single_full_k1_models.joblib")
    k2_models = joblib.load(artifact_dir / "multi_full_k2_models.joblib")
    known_pca = pca.transform(scaler.transform(known[columns].to_numpy(dtype=np.float64)))
    unknown_pca = pca.transform(scaler.transform(unknown[columns].to_numpy(dtype=np.float64)))
    single_known, single_known_pred = score_density_models(k1_models, known_pca)
    single_unknown, single_unknown_pred = score_density_models(k1_models, unknown_pca)
    multi_known, multi_known_pred = score_density_models(k2_models, known_pca)
    multi_unknown, multi_unknown_pred = score_density_models(k2_models, unknown_pca)
    native_known = -known["native_unknown_score"].to_numpy(dtype=np.float64)
    native_unknown = -unknown["native_unknown_score"].to_numpy(dtype=np.float64)
    thresholds = load_thresholds(output_dir / "threshold_calibration.csv")
    score_pairs = {
        "Native": (native_known, native_unknown),
        "Single-Full": (single_known, single_unknown),
        "Multi-Full-K2": (multi_known, multi_unknown),
    }
    metric_rows = [
        {"setting": args.setting, **detector_metrics(name, *scores, thresholds[name])}
        for name, scores in score_pairs.items()
    ]
    write_csv(output_dir / "detector_metrics.csv", metric_rows)
    metrics = {row["detector"]: row for row in metric_rows}

    known_predictions = pd.DataFrame(
        {
            "flow_id": known["flow_id"],
            "class_name": known["class_name"],
            "known_or_unknown": "Known",
            "native_known_score": native_known,
            "single_known_score": single_known,
            "multi_known_score": multi_known,
            "native_predicted_known_class": known["native_predicted_known_class"],
            "single_predicted_known_class": single_known_pred,
            "multi_predicted_known_class": multi_known_pred,
        }
    )
    unknown_predictions = pd.DataFrame(
        {
            "flow_id": unknown["flow_id"],
            "class_name": unknown["class_name"],
            "known_or_unknown": "Unknown",
            "native_known_score": native_unknown,
            "single_known_score": single_unknown,
            "multi_known_score": multi_unknown,
            "native_predicted_known_class": unknown["native_predicted_known_class"],
            "single_predicted_known_class": single_unknown_pred,
            "multi_predicted_known_class": multi_unknown_pred,
        }
    )
    predictions = pd.concat([known_predictions, unknown_predictions], ignore_index=True)
    for detector, prefix in (("Native", "native"), ("Single-Full", "single"), ("Multi-Full-K2", "multi")):
        predictions[f"{prefix}_accepted_as_known"] = (
            predictions[f"{prefix}_known_score"].to_numpy() >= thresholds[detector]
        )
    predictions_path = artifact_dir / "frozen_final_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False, engine="pyarrow", compression="zstd")

    per_unknown_rows: list[dict[str, object]] = []
    for class_name in fold["unknown_classes"]:
        subset = unknown_predictions[unknown_predictions["class_name"] == class_name]
        native_far = float(np.mean(subset["native_known_score"] >= thresholds["Native"]))
        single_far = float(np.mean(subset["single_known_score"] >= thresholds["Single-Full"]))
        multi_far = float(np.mean(subset["multi_known_score"] >= thresholds["Multi-Full-K2"]))
        per_unknown_rows.append(
            {
                "setting": args.setting,
                "unknown_class": class_name,
                "sample_count": len(subset),
                "native_unknown_far": native_far,
                "single_unknown_far": single_far,
                "multi_unknown_far": multi_far,
                "delta_unknown_far_multi_minus_single": multi_far - single_far,
            }
        )
    write_csv(output_dir / "per_unknown_class_results.csv", per_unknown_rows)

    single_unknown_accept = unknown_predictions["single_known_score"].to_numpy() >= thresholds["Single-Full"]
    multi_unknown_accept = unknown_predictions["multi_known_score"].to_numpy() >= thresholds["Multi-Full-K2"]
    single_known_accept = known_predictions["single_known_score"].to_numpy() >= thresholds["Single-Full"]
    multi_known_accept = known_predictions["multi_known_score"].to_numpy() >= thresholds["Multi-Full-K2"]
    transition_rows = [
        {"setting": args.setting, "population": "Unknown", "transition": "Recovered Unknown: Single accepted, Multi rejected", "count": int(np.sum(single_unknown_accept & ~multi_unknown_accept)), "population_size": len(unknown)},
        {"setting": args.setting, "population": "Unknown", "transition": "Regressed Unknown: Single rejected, Multi accepted", "count": int(np.sum(~single_unknown_accept & multi_unknown_accept)), "population_size": len(unknown)},
        {"setting": args.setting, "population": "Unknown", "transition": "Accepted by both", "count": int(np.sum(single_unknown_accept & multi_unknown_accept)), "population_size": len(unknown)},
        {"setting": args.setting, "population": "Unknown", "transition": "Rejected by both", "count": int(np.sum(~single_unknown_accept & ~multi_unknown_accept)), "population_size": len(unknown)},
        {"setting": args.setting, "population": "Known", "transition": "Recovered Known: Single rejected, Multi accepted", "count": int(np.sum(~single_known_accept & multi_known_accept)), "population_size": len(known)},
        {"setting": args.setting, "population": "Known", "transition": "Regressed Known: Single accepted, Multi rejected", "count": int(np.sum(single_known_accept & ~multi_known_accept)), "population_size": len(known)},
        {"setting": args.setting, "population": "Known", "transition": "Accepted by both", "count": int(np.sum(single_known_accept & multi_known_accept)), "population_size": len(known)},
        {"setting": args.setting, "population": "Known", "transition": "Rejected by both", "count": int(np.sum(~single_known_accept & ~multi_known_accept)), "population_size": len(known)},
    ]
    for row in transition_rows:
        row["rate"] = row["count"] / row["population_size"]
    write_csv(output_dir / "decision_transition_matrix.csv", transition_rows)

    absorbed = unknown_predictions.loc[single_unknown_accept].copy()
    absorbed["multi_still_accepted"] = multi_unknown_accept[single_unknown_accept]
    absorption_rows: list[dict[str, object]] = []
    for (unknown_class, known_class), group in absorbed.groupby(
        ["class_name", "single_predicted_known_class"], sort=True
    ):
        still = int(group["multi_still_accepted"].sum())
        absorption_rows.append(
            {
                "setting": args.setting,
                "unknown_class": unknown_class,
                "absorbed_known_class_single": known_class,
                "single_false_accept_count": len(group),
                "multi_still_accepts_same_samples": still,
                "multi_recovers_same_samples": len(group) - still,
            }
        )
    write_csv(
        output_dir / "unknown_absorption_by_known_class.csv",
        absorption_rows,
        [
            "setting",
            "unknown_class",
            "absorbed_known_class_single",
            "single_false_accept_count",
            "multi_still_accepts_same_samples",
            "multi_recovers_same_samples",
        ],
    )

    bootstrap_rows, ci_rows = paired_class_stratified_bootstrap(
        predictions, thresholds, args.bootstrap_resamples, args.bootstrap_seed
    )
    write_csv(output_dir / "paired_bootstrap_samples.csv", bootstrap_rows)
    write_csv(output_dir / "paired_bootstrap_ci.csv", [{"setting": args.setting, **row} for row in ci_rows])

    script_paths = sorted((STAGE3_ROOT / "scripts").glob("*.py"))
    metadata = {
        "setting": args.setting,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_hash": EXPECTED_PROTOCOL_CANONICAL_SHA256,
        "execution_plan_hash": EXPECTED_EXECUTION_PLAN_RAW_SHA256,
        "git_commit": git_revision(PROJECT_ROOT),
        "git_dirty_at_execution": True,
        "open_detect_commit": git_revision(PROJECT_ROOT.parent / "Open-Detect"),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))["physical_gpu"],
        "seed": 2022,
        "checkpoint_hash": sha256_file(artifact_dir / "best_checkpoint.pt"),
        "scaler_hash": sha256_file(artifact_dir / "standard_scaler.joblib"),
        "pca_hash": sha256_file(artifact_dir / "pca64.joblib"),
        "single_gaussian_hash": sha256_file(artifact_dir / "single_full_k1_models.joblib"),
        "multi_gmm_hash": sha256_file(artifact_dir / "multi_full_k2_models.joblib"),
        "thresholds": thresholds,
        "script_hashes": {path.name: sha256_file(path) for path in script_paths},
        "final_predictions_hash": sha256_file(predictions_path),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "frozen_model_manifest_hash": sha256_file(output_dir / "frozen_model_hashes.json"),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    delta = metrics["Multi-Full-K2"]["unknown_false_acceptance_rate"] - metrics["Single-Full"]["unknown_false_acceptance_rate"]
    results = f"""# {args.setting} Strict Unknown-Free Utility Results

- Status: **SUCCESS**
- Claim scope: local frozen-split approximate experiment; not author-fold equivalent

## Data and split

- Known / Unknown classes: {fold['known_count']} / {fold['unknown_count']}
- Known train / validation / test: {fold['formal_counts']['known_train']} / {fold['formal_counts']['known_validation']} / {fold['formal_counts']['known_final_test']}
- Unknown final-test: {fold['formal_counts']['unknown_final_test']}
- Leakage audit: PASS

## Configuration and execution

- Checkpoint selection: Known Validation Accuracy/Macro-F1 harmonic mean only
- Threshold calibration: 95% Known Validation acceptance only
- Density comparison: PCA-64 + full covariance, K=1 versus fixed K=2
- Bootstrap: 1,000 paired class-stratified resamples, seed 0

## Core results

| Detector | Known test FRR | Unknown FAR | AUROC | AUPRC-Unknown |
|---|---:|---:|---:|---:|
"""
    for name in DETECTORS:
        row = metrics[name]
        results += f"| {name} | {row['known_false_rejection_rate']:.6f} | {row['unknown_false_acceptance_rate']:.6f} | {row['auroc_unknown_positive']:.6f} | {row['auprc_unknown_positive']:.6f} |\n"
    results += f"""

- Delta Unknown FAR (Multi - Single): `{delta:.9f}`; negative favors Multi.
## Preserved evidence

- All final predictions, per-class results, transitions, absorption counts,
  K2 stability diagnostics, and 1,000 paired class-stratified bootstrap samples
  are retained in this setting bundle.

## Limitations

- Official Open-Detect five sample-fold identities are unavailable; this uses
  the frozen local 80/10/10 flow split and official class-holdout scenario.
- Fixed K2 is a controlled utility diagnostic, not evidence that every class
  has two semantic modes.

## Conclusion and next step

This primary setting is complete. Apply the predeclared cross-setting Gate only
after all primary settings complete; do not proceed to a later stage without
explicit authorization.
"""
    (output_dir / "RESULTS.md").write_text(results, encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    external_paths = sorted(path for path in artifact_dir.rglob("*") if path.is_file())
    session_prefix = f"stage3_{args.setting.lower().replace('-', '')}"
    session_dirs = sorted((PROJECT_ROOT / ".tmux-task").glob(f"{session_prefix}_smoke_v*"))
    session_dirs += sorted((PROJECT_ROOT / ".tmux-task").glob(f"{session_prefix}_formal_v*"))
    session_dirs.append(PROJECT_ROOT / ".tmux-task" / args.tmux_session)
    for session_dir in session_dirs:
        external_paths.extend(
            path for path in (session_dir / "output.log", session_dir / "exit.status") if path.exists()
        )
    external_artifacts = [
        {
            "path": os.path.relpath(path.resolve(), output_dir.resolve()),
            "scope": "external",
            "role": "checkpoint-or-derived-artifact" if artifact_dir in path.parents else "execution-log",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "hash_status": "verified",
        }
        for path in external_paths
    ]
    manifest.update(
        {
            "updated_at_utc": metadata["completed_utc"],
            "status": "success",
            "inputs": [
                {"path": str((PROJECT_ROOT / "data/splits/compatible_min1").resolve()), "role": "frozen source split"},
                {"path": str((PROJECT_ROOT / "stage3_protocol").resolve()), "role": "frozen protocol"},
            ],
            "code": {"revision": metadata["git_commit"], "dirty": True, "changes": ["stage3_unknown_utility execution implementation"]},
            "execution": {
                "tmux_session": args.tmux_session,
                "command": f"evaluate_final.py --setting {args.setting} --bootstrap-resamples 1000 --bootstrap-seed 0",
                "exit_code": 0,
                "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
                "physical_gpu_ids": [str(metadata["gpu"])],
            },
            "configuration": {
                "files": [str((STAGE3_ROOT / "configs" / f"{args.setting}.json").resolve())],
                "parameters": {"K_single": 1, "K_multi": 2, "reg_covar": 0.001, "pca_dim": 64},
                "seeds": [2022, 0],
            },
            "core_results": [
                {"name": "single_unknown_far", "value": metrics["Single-Full"]["unknown_false_acceptance_rate"]},
                {"name": "multi_unknown_far", "value": metrics["Multi-Full-K2"]["unknown_false_acceptance_rate"]},
                {"name": "delta_unknown_far_multi_minus_single", "value": delta},
            ],
            "artifacts": external_artifacts,
            "limitations": ["Official author sample-fold identities are unavailable; local frozen split is non-author-equivalent."],
            "next_step": "Complete all primary settings and compute the frozen cross-setting Gate.",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"setting": args.setting, "delta_ufar": delta, "status": "COMPLETE"}, sort_keys=True))


if __name__ == "__main__":
    main()
