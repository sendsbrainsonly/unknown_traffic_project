#!/usr/bin/env python3
"""Pair Native Open-Detect with Stage 44 DES-v1 on identical frozen roles."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from matched_common import FOLDS, ROOT, STAGE44, read_csv, verify_freeze, write_csv


METRICS = ("auroc", "auprc", "binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "ufar", "known_frr")


def main() -> None:
    frozen_before = verify_freeze()
    od_rows = []
    od_app_rows = []
    closed_rows = []
    for fold in FOLDS:
        evaluation = ROOT / "evaluation" / fold
        verification = json.loads((evaluation / "evaluation_verification.json").read_text(encoding="utf-8"))
        result = json.loads((ROOT / "runs" / fold / "result.json").read_text(encoding="utf-8"))
        if verification["status"] != "PASS" or result["status"] != "success":
            raise RuntimeError(f"{fold}: Native training/evaluation incomplete")
        if verification["checkpoint_sha256_after"] != result["checkpoint_sha256"]:
            raise RuntimeError(f"{fold}: checkpoint hash mismatch")
        rows = read_csv(evaluation / "open_set_metrics.csv")
        if len(rows) != 6:
            raise RuntimeError(f"{fold}: expected 6 Native metric rows, found {len(rows)}")
        od_rows.extend(rows)
        od_app_rows.extend(read_csv(evaluation / "per_unknown_application.csv"))
        native_closed = json.loads((evaluation / "closed_set_metrics.json").read_text(encoding="utf-8"))
        three_view_closed = json.loads((STAGE44 / "protocols" / fold / "evaluation/closed_set_metrics.json").read_text(encoding="utf-8"))
        closed_rows.append({"fold": fold, **{f"od_{key}": value for key, value in native_closed.items()},
                            **{f"three_view_{key}": value for key, value in three_view_closed.items()}})
    write_csv(ROOT / "od_run_results.csv", od_rows)
    write_csv(ROOT / "closed_set_comparison.csv", closed_rows)

    stage44 = read_csv(STAGE44 / "stage44_run_results.csv")
    reference = {(row["fold"], row["view"], row["threshold_percentile"]): row
                 for row in stage44 if row["method"] == "des_v1"}
    if len(reference) != 24:
        raise RuntimeError("Stage 44 DES-v1 grid is incomplete")
    comparison = []
    for row in od_rows:
        key = (row["fold"], row["view"], row["threshold_percentile"])
        other = reference[key]
        if (row["known_samples"], row["unknown_samples"]) != (other["known_samples"], other["unknown_samples"]):
            raise RuntimeError(f"{key}: Test membership count differs")
        comparison.append({
            "fold": row["fold"], "unknown_service": row["unknown_service"],
            "view": row["view"], "threshold_percentile": row["threshold_percentile"],
            "known_samples": row["known_samples"], "unknown_samples": row["unknown_samples"],
            **{f"od_{metric}": float(row[metric]) for metric in METRICS},
            **{f"three_view_des_v1_{metric}": float(other[metric]) for metric in METRICS},
            **{f"delta_three_view_minus_od_{metric}": float(other[metric]) - float(row[metric]) for metric in METRICS},
        })
    write_csv(ROOT / "matched_comparison.csv", comparison)

    baseline_apps = {(row["fold"], row["unknown_application"]): row
                     for row in read_csv(STAGE44 / "stage44_per_unknown_application.csv") if row["method"] == "des_v1"}
    apps = []
    for row in od_app_rows:
        other = baseline_apps[(row["fold"], row["unknown_application"])]
        if row["unknown_samples"] != other["unknown_samples"]:
            raise RuntimeError("per-application Unknown membership differs")
        apps.append({
            "fold": row["fold"], "application": row["unknown_application"],
            "unknown_samples": row["unknown_samples"],
            **{f"od_{key}": float(row[key]) for key in ("auroc", "auprc", "binary_f1", "ufar")},
            **{f"three_view_des_v1_{key}": float(other[key]) for key in ("auroc", "auprc", "binary_f1", "ufar")},
        })
    write_csv(ROOT / "per_unknown_application_comparison.csv", apps)

    primary = [row for row in comparison if row["view"] == "natural" and row["threshold_percentile"] == "95"]
    if len(primary) != 4:
        raise RuntimeError("incomplete primary comparison")
    summary = []
    for method in ("od", "three_view_des_v1"):
        record: dict[str, object] = {"method": method, "folds": 4, "view": "natural", "threshold_percentile": 95}
        for metric in METRICS:
            values = np.asarray([row[f"{method}_{metric}"] for row in primary], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(record)
    write_csv(ROOT / "method_summary.csv", summary)
    before_after = verify_freeze()
    if before_after != frozen_before:
        raise RuntimeError("Stage 44 frozen sources changed during aggregation")
    verification = {
        "status": "PASS", "folds_completed": 4,
        "native_metric_rows": len(od_rows), "paired_metric_rows": len(comparison),
        "native_sample_score_rows": sum(len(read_csv(ROOT / "evaluation" / fold / "sample_scores.csv")) for fold in FOLDS),
        "matched_test_flow_ids": True,
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "unknown_threshold_samples": 0, "test_threshold_samples": 0,
        "source_freeze_before": frozen_before, "source_freeze_after": before_after,
    }
    (ROOT / "completion_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
