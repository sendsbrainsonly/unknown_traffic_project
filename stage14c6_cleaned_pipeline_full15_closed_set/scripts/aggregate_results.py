#!/usr/bin/env python3
"""Aggregate the completed Stage 14C-6 grid and existing Native baseline."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
NATIVE = PROJECT / "stage14c_native_opendetect_vnat"
STAGE14C4 = PROJECT / "stage14c4_our_method_training_cleanup"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
PROTOCOLS = [
    f"{setting}_seed{seed}"
    for setting in ("low", "medium", "high")
    for seed in range(2022, 2027)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise RuntimeError(f"refusing to write empty output: {name}")
    frame.to_csv(ROOT / name, index=False, quoting=csv.QUOTE_MINIMAL)


def summarize(frame: pd.DataFrame, label: str) -> dict[str, object]:
    row: dict[str, object] = {"setting": label, "runs": len(frame)}
    for metric in ("val_accuracy", "val_macro_f1", "val_weighted_f1"):
        row[f"{metric}_mean"] = float(frame[metric].mean())
        row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        row[f"{metric}_min"] = float(frame[metric].min())
        row[f"{metric}_max"] = float(frame[metric].max())
    row["weak_runs"] = int(frame["weak_run"].sum())
    return row


def json_native(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in record.items()
    }


def main() -> None:
    queue = json.loads((ROOT / "run_queue_status.json").read_text(encoding="utf-8"))
    if queue.get("status") != "success" or len(queue.get("completed", [])) != 15:
        raise RuntimeError("formal grid is not complete")

    run_rows: list[dict[str, object]] = []
    class_parts: list[pd.DataFrame] = []
    dynamics_parts: list[pd.DataFrame] = []
    failures: list[str] = []
    for protocol in PROTOCOLS:
        run_dir = ROOT / "runs" / protocol
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        checkpoint = Path(result["checkpoint_path"])
        checks = {
            "status": result["status"] == "success",
            "epochs": result["epochs_completed"] == 100,
            "freeze": result["freeze_hash"] == EXPECTED_FREEZE,
            "known_test": result["known_test_samples_used"] == 0,
            "unknown_test": result["unknown_test_samples_used"] == 0,
            "des": result["des_executed"] is False,
            "checkpoint": checkpoint.is_file() and sha256(checkpoint) == result["checkpoint_sha256"],
            "prototype_resets": [row["epoch"] for row in result["prototype_reset_audits"]] == [51, 81],
            "early_stopping": config["early_stopping"] == "disabled",
            "batch": config["batch_size"] == 128,
        }
        if not all(checks.values()):
            failures.append(f"{protocol}:{[key for key, value in checks.items() if not value]}")
        weak = bool(
            result["nan_or_crash"]
            or result["val_macro_f1"] < 0.50
            or result["val_accuracy"] < 0.60
        )
        run_rows.append({
            "protocol_id": protocol,
            "setting": result["setting"],
            "protocol_seed": result["protocol_seed"],
            "training_seed": result["training_seed"],
            "known_classes": "|".join(result["known_classes"]),
            "unknown_classes": "|".join(result["unknown_classes"]),
            "num_known_classes": len(result["known_classes"]),
            "num_unknown_classes": len(result["unknown_classes"]),
            "train_samples": result["train_samples"],
            "validation_samples": result["validation_samples"],
            "epochs_completed": result["epochs_completed"],
            "best_epoch": result["best_epoch"],
            "train_loss": result["train_loss"],
            "val_loss": result["val_loss"],
            "val_accuracy": result["val_accuracy"],
            "val_macro_f1": result["val_macro_f1"],
            "val_weighted_f1": result["val_weighted_f1"],
            "minority_recall": result["minority_recall"],
            "stability_guard_activations": result["stability_guard_activations"],
            "checkpoint_path": result["checkpoint_path"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "latest_checkpoint_sha256": result["latest_checkpoint_sha256"],
            "duration_seconds": result["duration_seconds"],
            "weak_run": weak,
            "known_test_samples_used": result["known_test_samples_used"],
            "unknown_test_samples_used": result["unknown_test_samples_used"],
            "des_executed": result["des_executed"],
            "status": result["status"],
        })
        per_class = pd.read_csv(run_dir / "per_class_metrics.csv")
        class_parts.append(per_class)
        dynamics_parts.append(pd.read_csv(run_dir / "training_metrics.csv"))

    if failures:
        raise RuntimeError("formal run integrity failures: " + "; ".join(failures))
    runs = pd.DataFrame(run_rows)
    classes = pd.concat(class_parts, ignore_index=True)
    dynamics = pd.concat(dynamics_parts, ignore_index=True)
    write(runs, "run_level_results.csv")
    write(classes, "per_class_results.csv")
    write(dynamics, "training_dynamics.csv")

    summary_rows = [summarize(runs[runs.setting == setting], setting) for setting in ("Low", "Medium", "High")]
    summary_rows.append(summarize(runs, "Overall"))
    setting_summary = pd.DataFrame(summary_rows)
    write(setting_summary, "setting_summary.csv")

    class_summary = classes.groupby("class", as_index=False).agg(
        runs=("protocol_id", "nunique"),
        total_validation_support=("val_support", "sum"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        recall_min=("recall", "min"),
        recall_max=("recall", "max"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_min=("f1", "min"),
        f1_max=("f1", "max"),
    )
    write(class_summary, "class_summary.csv")

    native = pd.read_csv(NATIVE / "outputs/stage14c_native_run_results.csv")
    native = native.rename(columns={
        "validation_accuracy": "native_accuracy",
        "validation_macro_f1": "native_macro_f1",
        "validation_weighted_f1": "native_weighted_f1",
        "best_epoch": "native_best_epoch",
        "training_seed": "native_training_seed",
        "checkpoint_sha256": "native_checkpoint_sha256",
    })
    paired = runs.merge(native[[
        "protocol_id", "native_training_seed", "native_best_epoch", "native_accuracy",
        "native_macro_f1", "native_weighted_f1", "native_checkpoint_sha256",
    ]], on="protocol_id", how="inner", validate="one_to_one")
    if len(paired) != 15:
        raise RuntimeError("paired Native comparison is incomplete")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        paired[f"delta_{metric}_our_minus_native"] = paired[f"val_{metric}"] - paired[f"native_{metric}"]
        paired[f"our_wins_{metric}"] = paired[f"delta_{metric}_our_minus_native"] > 0
    write(paired, "paired_vs_native.csv")

    pair_rows: list[dict[str, object]] = []
    all_classes = sorted({name for value in paired.known_classes for name in value.split("|")})
    for first, second in itertools.combinations(all_classes, 2):
        present_mask = paired.known_classes.map(
            lambda value: first in value.split("|") and second in value.split("|")
        )
        present = paired[present_mask]
        absent = paired[~present_mask]
        if present.empty or absent.empty:
            continue
        pair_rows.append({
            "class_a": first,
            "class_b": second,
            "protocols_pair_known": len(present),
            "protocols_pair_not_both_known": len(absent),
            "macro_f1_when_pair_known": float(present.val_macro_f1.mean()),
            "macro_f1_when_not_both_known": float(absent.val_macro_f1.mean()),
            "descriptive_macro_f1_difference": float(present.val_macro_f1.mean() - absent.val_macro_f1.mean()),
            "delta_native_when_pair_known": float(present.delta_macro_f1_our_minus_native.mean()),
            "delta_native_when_not_both_known": float(absent.delta_macro_f1_our_minus_native.mean()),
        })
    pair_effects = pd.DataFrame(pair_rows).sort_values("descriptive_macro_f1_difference")
    write(pair_effects, "known_class_pair_effects.csv")
    focus_effects = pair_effects[
        pair_effects.apply(
            lambda row: row.class_a in {"rsync", "scp", "sftp", "rdp"}
            or row.class_b in {"rsync", "scp", "sftp", "rdp"},
            axis=1,
        )
    ]
    write(focus_effects, "focus_class_pair_effects.csv")

    paired_rows: list[dict[str, object]] = []
    for setting in ("Low", "Medium", "High", "Overall"):
        group = paired if setting == "Overall" else paired[paired.setting == setting]
        row: dict[str, object] = {"setting": setting, "runs": len(group)}
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            delta = group[f"delta_{metric}_our_minus_native"]
            row[f"delta_{metric}_mean"] = float(delta.mean())
            row[f"delta_{metric}_std"] = float(delta.std(ddof=1))
            row[f"our_wins_{metric}"] = int((delta > 0).sum())
            row[f"ties_{metric}"] = int((delta == 0).sum())
        paired_rows.append(row)
    write(pd.DataFrame(paired_rows), "paired_vs_native_summary.csv")

    native_class = pd.read_csv(NATIVE / "outputs/stage14c_native_per_class_recall.csv")
    native_class = native_class.rename(columns={
        "application": "class",
        "precision": "native_precision",
        "recall": "native_recall",
        "f1": "native_f1",
        "support": "native_support",
    })
    class_paired = classes.merge(native_class[[
        "protocol_id", "class", "native_precision", "native_recall", "native_f1", "native_support"
    ]], on=["protocol_id", "class"], how="inner", validate="one_to_one")
    if len(class_paired) != len(classes):
        raise RuntimeError("paired per-class Native comparison is incomplete")
    for metric in ("precision", "recall", "f1"):
        class_paired[f"delta_{metric}_our_minus_native"] = class_paired[metric] - class_paired[f"native_{metric}"]
    write(class_paired, "per_class_vs_native.csv")
    hard = class_paired[class_paired["class"].isin(["rsync", "scp", "sftp", "rdp"])]
    hard_summary = hard.groupby("class", as_index=False).agg(
        runs=("protocol_id", "nunique"),
        our_recall_mean=("recall", "mean"),
        native_recall_mean=("native_recall", "mean"),
        delta_recall_mean=("delta_recall_our_minus_native", "mean"),
        our_f1_mean=("f1", "mean"),
        native_f1_mean=("native_f1", "mean"),
        delta_f1_mean=("delta_f1_our_minus_native", "mean"),
    )
    write(hard_summary, "hard_class_vs_native_summary.csv")

    prior = pd.read_csv(STAGE14C4 / "cleanup_comparison.csv")
    parity_rows: list[dict[str, object]] = []
    for protocol in ("medium_seed2025", "medium_seed2026"):
        old = prior[(prior.protocol_id == protocol) & (prior.version == "cleaned_primary")].iloc[0]
        new = runs[runs.protocol_id == protocol].iloc[0]
        parity_rows.append({
            "protocol_id": protocol,
            "delta_accuracy": float(new.val_accuracy - old.val_accuracy),
            "delta_macro_f1": float(new.val_macro_f1 - old.val_macro_f1),
            "delta_weighted_f1": float(new.val_weighted_f1 - old.val_weighted_f1),
            "best_epoch_old": int(old.best_epoch),
            "best_epoch_new": int(new.best_epoch),
            "exact_metric_parity_1e_12": bool(
                abs(new.val_accuracy - old.val_accuracy) <= 1e-12
                and abs(new.val_macro_f1 - old.val_macro_f1) <= 1e-12
                and abs(new.val_weighted_f1 - old.val_weighted_f1) <= 1e-12
            ),
        })
    write(pd.DataFrame(parity_rows), "stage14c4_parity.csv")

    aggregate = {
        "status": "PASS",
        "runs_successful": int((runs.status == "success").sum()),
        "runs_total": 15,
        "weak_runs": int(runs.weak_run.sum()),
        "nan_or_crash_runs": 0,
        "known_test_samples_used": int(runs.known_test_samples_used.sum()),
        "unknown_test_samples_used": int(runs.unknown_test_samples_used.sum()),
        "des_executed": bool(runs.des_executed.any()),
        "freeze_hash": EXPECTED_FREEZE,
        "checkpoints_verified": 15,
        "stage14c4_reproducibility_parity": bool(pd.DataFrame(parity_rows).exact_metric_parity_1e_12.all()),
        "overall": json_native(setting_summary[setting_summary.setting == "Overall"].iloc[0].to_dict()),
        "paired_overall": json_native(
            pd.DataFrame(paired_rows)[pd.DataFrame(paired_rows).setting == "Overall"].iloc[0].to_dict()
        ),
    }
    (ROOT / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
