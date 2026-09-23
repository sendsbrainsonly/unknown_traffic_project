#!/usr/bin/env python3
"""Aggregate all 60 Known-Validation feature-ablation runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURES, ROOT, RUNS_ROOT, protocol_ids, sha256_file, verify_frozen_inputs


def mean_std(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def weighted_f1_from_per_class(part: pd.DataFrame) -> float:
    """Reconstruct sklearn-style weighted F1 from validation class supports."""
    support = part["val_support"].to_numpy(dtype=np.float64)
    if support.sum() <= 0:
        raise RuntimeError("per-class metrics contain no validation support")
    return float(np.average(part["f1"].to_numpy(dtype=np.float64), weights=support))


def main() -> None:
    frozen = verify_frozen_inputs()
    results = []
    per_class_parts = []
    for feature in FEATURES:
        for protocol_id in protocol_ids():
            run_dir = RUNS_ROOT / feature / protocol_id
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                raise RuntimeError(f"missing result: {result_path}")
            row = json.loads(result_path.read_text())
            if row["status"] != "success" or row["unknown_samples_used"] != 0 or row["known_test_samples_used"] != 0:
                raise RuntimeError(f"invalid result boundary: {result_path}")
            part = pd.read_csv(run_dir / "per_class_metrics.csv")
            row["val_weighted_f1"] = weighted_f1_from_per_class(part)
            results.append(row)
            per_class_parts.append(part)
    run_df = pd.DataFrame(results).sort_values(["feature", "setting", "seed"])
    if len(run_df) != 60:
        raise RuntimeError("expected 60 run results")
    run_df.to_csv(ROOT / "stage14c5_feature_ablation.csv", index=False)
    class_df = pd.concat(per_class_parts, ignore_index=True).sort_values(["feature", "protocol_id", "local_label"])
    class_df.to_csv(ROOT / "stage14c5_per_class_metrics.csv", index=False)

    summary_rows = []
    for feature in FEATURES:
        subset = run_df[run_df.feature == feature]
        for setting in ["Low", "Medium", "High", "Overall"]:
            group = subset if setting == "Overall" else subset[subset.setting == setting]
            acc_mean, acc_std = mean_std(group.val_accuracy)
            f1_mean, f1_std = mean_std(group.val_macro_f1)
            weighted_f1_mean, weighted_f1_std = mean_std(group.val_weighted_f1)
            minority_mean, minority_std = mean_std(group.minority_recall)
            summary_rows.append({
                "feature": feature, "setting": setting, "runs": len(group),
                "val_accuracy_mean": acc_mean, "val_accuracy_std": acc_std,
                "val_macro_f1_mean": f1_mean, "val_macro_f1_std": f1_std,
                "val_weighted_f1_mean": weighted_f1_mean,
                "val_weighted_f1_std": weighted_f1_std,
                "minority_recall_mean": minority_mean, "minority_recall_std": minority_std,
                "weak_run_count_macro_f1_lt_0_35": int((group.val_macro_f1 < 0.35).sum()),
                "worst_val_macro_f1": float(group.val_macro_f1.min()),
                "worst_val_weighted_f1": float(group.val_weighted_f1.min()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ROOT / "stage14c5_feature_summary.csv", index=False)

    paired_rows = []
    baseline = run_df[run_df.feature == "f0"].set_index("protocol_id")
    for feature in FEATURES[1:]:
        candidate = run_df[run_df.feature == feature].set_index("protocol_id")
        for protocol_id in protocol_ids():
            paired_rows.append({
                "feature": feature, "protocol_id": protocol_id,
                "setting": candidate.loc[protocol_id, "setting"], "seed": int(candidate.loc[protocol_id, "seed"]),
                "delta_val_accuracy": float(candidate.loc[protocol_id, "val_accuracy"] - baseline.loc[protocol_id, "val_accuracy"]),
                "delta_val_macro_f1": float(candidate.loc[protocol_id, "val_macro_f1"] - baseline.loc[protocol_id, "val_macro_f1"]),
                "delta_val_weighted_f1": float(candidate.loc[protocol_id, "val_weighted_f1"] - baseline.loc[protocol_id, "val_weighted_f1"]),
                "delta_minority_recall": float(candidate.loc[protocol_id, "minority_recall"] - baseline.loc[protocol_id, "minority_recall"]),
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(ROOT / "stage14c5_paired_vs_f0.csv", index=False)

    applications = sorted(class_df["class"].unique())
    app_to_index = {name: index for index, name in enumerate(applications)}
    for feature in FEATURES:
        pooled = np.zeros((len(applications), len(applications)), dtype=np.int64)
        for protocol_id in protocol_ids():
            path = ROOT / "confusion_matrices" / feature / f"{protocol_id}.csv"
            matrix_df = pd.read_csv(path, index_col=0)
            for truth_name in matrix_df.index:
                for pred_name in matrix_df.columns:
                    pooled[app_to_index[truth_name], app_to_index[pred_name]] += int(matrix_df.loc[truth_name, pred_name])
        pd.DataFrame(pooled, index=applications, columns=applications).rename_axis("true\\predicted").to_csv(ROOT / "confusion_matrices" / f"{feature}_pooled.csv")

    stability = {}
    for feature in FEATURES:
        per_setting_std = summary_df[(summary_df.feature == feature) & (summary_df.setting != "Overall")].val_macro_f1_std
        overall = summary_df[(summary_df.feature == feature) & (summary_df.setting == "Overall")].iloc[0]
        stability[feature] = {
            "mean_within_setting_seed_std": float(per_setting_std.mean()),
            "overall_mean_macro_f1": float(overall.val_macro_f1_mean),
            "worst_macro_f1": float(overall.worst_val_macro_f1),
        }
    most_stable = min(FEATURES, key=lambda feature: stability[feature]["mean_within_setting_seed_std"])
    best_non_f0 = max(FEATURES[1:], key=lambda feature: float(summary_df[(summary_df.feature == feature) & (summary_df.setting == "Overall")].iloc[0].val_macro_f1_mean))
    best_pairs = paired_df[paired_df.feature == best_non_f0]
    overall_f0 = summary_df[(summary_df.feature == "f0") & (summary_df.setting == "Overall")].iloc[0]
    overall_best = summary_df[(summary_df.feature == best_non_f0) & (summary_df.setting == "Overall")].iloc[0]
    f1_gain = float(overall_best.val_macro_f1_mean - overall_f0.val_macro_f1_mean)
    minority_gain = float(overall_best.minority_recall_mean - overall_f0.minority_recall_mean)
    wins = int((best_pairs.delta_val_macro_f1 > 0).sum())
    std_reductions = 0
    setting_deltas = {}
    for setting in ("Low", "Medium", "High"):
        b = summary_df[(summary_df.feature == "f0") & (summary_df.setting == setting)].iloc[0]
        c = summary_df[(summary_df.feature == best_non_f0) & (summary_df.setting == setting)].iloc[0]
        setting_deltas[setting] = float(c.val_macro_f1_mean - b.val_macro_f1_mean)
        std_reductions += int(c.val_macro_f1_std < b.val_macro_f1_std)
    primary_bottleneck = f1_gain >= 0.05 and wins >= 12 and minority_gain >= 0.05 and std_reductions >= 2
    recommend_freeze = f1_gain >= 0.03 and wins >= 10 and min(setting_deltas.values()) >= -0.01

    weak_table = run_df[(run_df.setting == "Medium") & (run_df.seed.isin([2024, 2025]))][["feature", "seed", "val_accuracy", "val_macro_f1", "val_weighted_f1", "minority_recall"]]
    lines = [
        "# Stage 14C.5 — VNAT Feature Representation Audit",
        "",
        "## Data boundary",
        "",
        f"- Stage 14B freeze hash: `{frozen['freeze_hash']}`.",
        "- Model-visible samples: frozen Known Train and Known Validation only.",
        "- Unknown Test samples used: **0**.",
        "- Known Test samples used: **0**.",
        "- Unknown Detection executed: **NO**.",
        "",
        "## Mean ± std across seeds",
        "",
        summary_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Macro-F1 gives every application equal weight and remains the primary decision metric. "
        "Weighted-F1 weights each application by Known-Validation support and is reported alongside it "
        "to reveal whether aggregate performance is dominated by large classes.",
        "",
        "## Medium-2024/2025 weak-run check",
        "",
        weak_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Stability",
        "",
        pd.DataFrame([{"feature": key, **value} for key, value in stability.items()]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Pre-registered decision",
        "",
        f"- Best non-F0 by overall mean Macro-F1: **{best_non_f0.upper()}**.",
        f"- Overall ΔMacro-F1 versus F0: **{f1_gain:+.6f}**.",
        f"- Paired Macro-F1 wins versus F0: **{wins}/15**.",
        f"- Overall minority-recall delta: **{minority_gain:+.6f}**.",
        f"- Settings with lower seed std: **{std_reductions}/3**.",
        f"- Most stable by mean within-setting seed std: **{most_stable.upper()}**.",
        f"- Feature representation is the primary bottleneck: **{'YES' if primary_bottleneck else 'NO'}**.",
        f"- Recommend freezing a new feature configuration before formal encoder retraining: **{'YES — ' + best_non_f0.upper() if recommend_freeze else 'NO'}**.",
        "",
        "The primary-bottleneck conclusion follows the thresholds frozen in README.md before training. A NO does not mean features are irrelevant; it means the observed improvement did not satisfy all four pre-registered conditions.",
    ]
    (ROOT / "stage14c5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {
        "status": "PASS", "run_count": 60, "feature_count": 4,
        "unknown_test_samples_used": 0, "known_test_samples_used": 0,
        "best_non_f0": best_non_f0, "most_stable": most_stable,
        "primary_feature_bottleneck": primary_bottleneck, "recommend_new_feature_freeze": recommend_freeze,
        "stage14c5_feature_ablation_sha256": sha256_file(ROOT / "stage14c5_feature_ablation.csv"),
        "stage14c5_report_sha256": sha256_file(ROOT / "stage14c5_report.md"),
    }
    (ROOT / "aggregation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
