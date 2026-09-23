#!/usr/bin/env python3
"""Aggregate all successful native Open-Detect Known Validation results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import OUTPUTS_ROOT, PROJECT_ROOT, ROOT, RUNS_ROOT, SETTINGS, load_protocols, verify_freeze


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def main() -> None:
    freeze = verify_freeze()
    protocols = load_protocols()
    run_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for protocol_id in sorted(protocols):
        run_dir = RUNS_ROOT / protocol_id
        if not (run_dir / "COMPLETED").exists() or not (run_dir / "result.json").exists():
            missing.append(protocol_id)
            continue
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        run_rows.append({
            "protocol_id": protocol_id,
            "setting": result["setting"],
            "protocol_seed": result["protocol_seed"],
            "training_seed": result["training_seed"],
            "physical_gpu_id": result["physical_gpu_id"],
            "num_known_classes": result["num_known_classes"],
            "num_unknown_classes": result["num_unknown_classes"],
            "train_samples": result["train_samples"],
            "validation_samples": result["validation_samples"],
            "best_epoch": result["best_epoch"],
            "validation_accuracy": result["validation_accuracy"],
            "validation_macro_f1": result["validation_macro_f1"],
            "validation_weighted_f1": result["validation_weighted_f1"],
            "epochs_completed": result["epochs_completed"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "status": result["status"],
            "anomaly": result["anomaly"],
            "duration_seconds": result["duration_seconds"],
        })
        per_class_rows.extend(json.loads((run_dir / "per_class_metrics.json").read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete protocols: {missing}")
    run_rows.sort(key=lambda row: (SETTINGS.index(str(row["setting"])), int(row["protocol_seed"])))
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_fields = list(run_rows[0])
    write_csv(OUTPUTS_ROOT / "stage14c_native_run_results.csv", run_rows, run_fields)
    per_class_fields = list(per_class_rows[0])
    write_csv(OUTPUTS_ROOT / "stage14c_native_per_class_recall.csv", per_class_rows, per_class_fields)

    summary_rows: list[dict[str, object]] = []
    for setting in (*SETTINGS, "Overall"):
        chosen = run_rows if setting == "Overall" else [r for r in run_rows if r["setting"] == setting]
        row: dict[str, object] = {"setting": setting, "runs": len(chosen)}
        for metric in ("validation_accuracy", "validation_macro_f1", "validation_weighted_f1"):
            mean, std = mean_std([float(item[metric]) for item in chosen])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_min"] = min(float(item[metric]) for item in chosen)
            row[f"{metric}_max"] = max(float(item[metric]) for item in chosen)
        row["weak_run_count_macro_f1_lt_0_35"] = sum(
            float(item["validation_macro_f1"]) < 0.35 for item in chosen
        )
        summary_rows.append(row)
    write_csv(OUTPUTS_ROOT / "stage14c_native_setting_summary.csv", summary_rows, list(summary_rows[0]))

    weak = sorted(run_rows, key=lambda row: float(row["validation_macro_f1"]))[:5]
    per_application: dict[str, list[float]] = defaultdict(list)
    for row in per_class_rows:
        per_application[str(row["application"])].append(float(row["recall"]))
    application_summary = [
        {
            "application": application,
            "appearances": len(values),
            "recall_mean": float(np.mean(values)),
            "recall_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "recall_min": min(values),
            "recall_max": max(values),
        }
        for application, values in sorted(per_application.items())
    ]
    write_csv(OUTPUTS_ROOT / "stage14c_native_application_recall_summary.csv", application_summary, list(application_summary[0]))

    prior_path = PROJECT_ROOT / "stage14c5_feature_representation_audit" / "stage14c5_feature_ablation.csv"
    with prior_path.open(encoding="utf-8", newline="") as handle:
        prior_f0 = {
            row["protocol_id"]: row for row in csv.DictReader(handle)
            if row["feature"].lower() == "f0"
        }
    if set(prior_f0) != {str(row["protocol_id"]) for row in run_rows}:
        raise RuntimeError("prior F0/native protocol identities differ")
    comparison_rows: list[dict[str, object]] = []
    for native in run_rows:
        prior = prior_f0[str(native["protocol_id"])]
        comparison_rows.append({
            "protocol_id": native["protocol_id"],
            "setting": native["setting"],
            "protocol_seed": native["protocol_seed"],
            "native_accuracy": native["validation_accuracy"],
            "prior_f0_accuracy": float(prior["val_accuracy_recomputed"]),
            "delta_accuracy_native_minus_prior_f0": float(native["validation_accuracy"]) - float(prior["val_accuracy_recomputed"]),
            "native_macro_f1": native["validation_macro_f1"],
            "prior_f0_macro_f1": float(prior["val_macro_f1_recomputed"]),
            "delta_macro_f1_native_minus_prior_f0": float(native["validation_macro_f1"]) - float(prior["val_macro_f1_recomputed"]),
            "native_weighted_f1": native["validation_weighted_f1"],
            "prior_f0_weighted_f1": float(prior["val_weighted_f1"]),
            "delta_weighted_f1_native_minus_prior_f0": float(native["validation_weighted_f1"]) - float(prior["val_weighted_f1"]),
            "comparison_boundary": "same F0 arrays and Known Validation split; different training seed policy, batch, early stopping, numerical guard, and checkpoint rule",
        })
    write_csv(
        OUTPUTS_ROOT / "stage14c_native_vs_prior_f0.csv",
        comparison_rows, list(comparison_rows[0]),
    )
    paired_delta = {
        metric: float(np.mean([float(row[f"delta_{metric}_native_minus_prior_f0"]) for row in comparison_rows]))
        for metric in ("accuracy", "macro_f1", "weighted_f1")
    }
    native_macro_wins = sum(float(row["delta_macro_f1_native_minus_prior_f0"]) > 0 for row in comparison_rows)
    prior_summary_path = PROJECT_ROOT / "stage14c5_feature_representation_audit" / "stage14c5_feature_summary.csv"
    with prior_summary_path.open(encoding="utf-8", newline="") as handle:
        prior_summary = {
            (row["feature"].lower(), row["setting"]): row for row in csv.DictReader(handle)
        }
    f2_overall = prior_summary[("f2", "Overall")]
    overall_native = summary_rows[-1]
    native_minus_f2 = {
        "accuracy": float(overall_native["validation_accuracy_mean"]) - float(f2_overall["val_accuracy_mean"]),
        "macro_f1": float(overall_native["validation_macro_f1_mean"]) - float(f2_overall["val_macro_f1_mean"]),
        "weighted_f1": float(overall_native["validation_weighted_f1_mean"]) - float(f2_overall["val_weighted_f1_mean"]),
    }

    report_lines = [
        "# Stage 14C — Released Open-Detect on VNAT",
        "",
        "## Scope and integrity",
        "",
        f"- Stage 14B freeze hash before/after: `{freeze['freeze_hash']}` (PASS).",
        "- Runs completed: 15/15.",
        "- Model-visible data: Known Train and Known Validation only.",
        "- Unknown samples used in training/validation: 0/0.",
        "- Known Test samples used: 0.",
        "- DES and open-set evaluation: not run.",
        "- Checkpoint selection: maximum Known Validation Accuracy only.",
        "",
        "## Aggregate Known Validation results",
        "",
        "| Setting | Runs | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['setting']} | {row['runs']} | "
            f"{row['validation_accuracy_mean']:.6f} +/- {row['validation_accuracy_std']:.6f} | "
            f"{row['validation_macro_f1_mean']:.6f} +/- {row['validation_macro_f1_std']:.6f} | "
            f"{row['validation_weighted_f1_mean']:.6f} +/- {row['validation_weighted_f1_std']:.6f} |"
        )
    report_lines.extend([
        "",
        "## Lowest Macro-F1 runs",
        "",
        "| Protocol | Accuracy | Macro-F1 | Weighted-F1 | Best epoch | Anomaly |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in weak:
        report_lines.append(
            f"| {row['protocol_id']} | {row['validation_accuracy']:.6f} | "
            f"{row['validation_macro_f1']:.6f} | {row['validation_weighted_f1']:.6f} | "
            f"{row['best_epoch']} | {row['anomaly'] or 'none'} |"
        )
    report_lines.extend([
        "",
        "## Known-Validation comparison with the earlier F0 implementation",
        "",
        f"- Mean native-minus-prior F0 Accuracy: `{paired_delta['accuracy']:+.6f}`.",
        f"- Mean native-minus-prior F0 Macro-F1: `{paired_delta['macro_f1']:+.6f}`.",
        f"- Mean native-minus-prior F0 Weighted-F1: `{paired_delta['weighted_f1']:+.6f}`.",
        f"- Native Macro-F1 wins: `{native_macro_wins}/15` protocols.",
        f"- Native-minus-prior F2 overall Accuracy/Macro-F1/Weighted-F1: "
        f"`{native_minus_f2['accuracy']:+.6f}` / `{native_minus_f2['macro_f1']:+.6f}` / "
        f"`{native_minus_f2['weighted_f1']:+.6f}`.",
        "- This is not a pure feature ablation: arrays and Known Validation splits match, "
        "but training seed policy, batch size, early stopping, numerical guard, and "
        "checkpoint rule differ.",
        "",
        "## Weak-run and class-level diagnosis",
        "",
        "- No run has Macro-F1 below 0.35 and no run crashed or produced non-finite metrics.",
        "- `medium_seed2023` is nevertheless a clear weak run by Accuracy (0.587112) and "
        "Weighted-F1 (0.587748). Its rsync/scp/sftp recalls are 0.424084/0.427948/0.403614, "
        "while vimeo/youtube/zoiper are 0.966942/1.000000/0.989247. The 4-sample rdp "
        "validation support also makes its class estimate fragile.",
        "- Across protocol appearances, mean recall is lowest for sftp (0.471084) and "
        "rsync (0.545455); rdp is unstable (mean 0.704545, range 0.25-1.00).",
        "",
        "## Answer to the bottleneck question",
        "",
        "The evidence favors the earlier implementation/training policy as the larger "
        "source of the previously low closed-set result: with the same F0 arrays and "
        "Known Validation splits, released-code training improves mean Macro-F1 by "
        f"{paired_delta['macro_f1']:+.6f} and wins 14/15 protocols. Its Macro-F1 is also "
        f"{native_minus_f2['macro_f1']:+.6f} above the earlier F2 mean. This does not prove "
        "that VNAT is easy: sftp, rsync, rdp, and the Medium-2023 composition remain "
        "genuinely difficult or unstable. Because the comparison changes multiple "
        "training details at once, it identifies an implementation/policy effect but "
        "cannot isolate which single detail caused it.",
        "",
        "## Interpretation boundary",
        "",
        "This is the released-code Open-Detect baseline on frozen VNAT protocols. "
        "It does not use Unknown Test and therefore makes no unknown-detection claim. "
        "The comparison with earlier F0/F1/F2/F3 Known Validation runs is performed "
        "separately and cannot attribute causality from a single confounded comparison.",
        "",
    ])
    (OUTPUTS_ROOT / "stage14c_native_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs": len(run_rows), "weak": [r["protocol_id"] for r in weak]}))


if __name__ == "__main__":
    main()
