#!/usr/bin/env python3
"""Aggregate the four completed Stage44 folds without refitting anything."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("communication", "file_transfer", "remote_access", "streaming")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_rows: list[dict[str, object]] = []
    closed_rows: list[dict[str, object]] = []
    per_application: list[dict[str, object]] = []
    for fold in FOLDS:
        evaluation = ROOT / "protocols" / fold / "evaluation"
        verification = json.loads((evaluation / "evaluation_verification.json").read_text())
        if verification["status"] != "PASS" or verification["unknown_training_samples"] != 0:
            raise RuntimeError(f"evaluation verification failed: {fold}")
        rows = read_csv(evaluation / "open_set_metrics.csv")
        if len(rows) != 24:
            raise RuntimeError(f"unexpected metric rows: {fold}/{len(rows)}")
        run_rows.extend(rows)
        closed = json.loads((evaluation / "closed_set_metrics.json").read_text())
        closed_rows.append({"fold": fold, **closed})
        per_application.extend(read_csv(evaluation / "per_unknown_application.csv"))
    write_csv(ROOT / "stage44_run_results.csv", run_rows)
    write_csv(ROOT / "stage44_closed_set_results.csv", closed_rows)
    write_csv(ROOT / "stage44_per_unknown_application.csv", per_application)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["method"]), str(row["view"]), str(row["threshold_percentile"]))].append(row)
    metrics = ("auroc", "auprc", "binary_accuracy", "binary_f1", "ufar", "known_frr")
    summary = []
    for key, rows in sorted(grouped.items()):
        record: dict[str, object] = {
            "method": key[0], "view": key[1], "threshold_percentile": key[2], "folds": len(rows)
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        summary.append(record)
    write_csv(ROOT / "stage44_method_summary.csv", summary)

    primary = [
        row for row in run_rows
        if row["view"] == "natural" and row["threshold_percentile"] == "95"
    ]
    lines = [
        "# Stage 44 Results — VNAT Coarse Service-Level LOSO Open Set",
        "",
        "Status: complete only after all four fold verification files pass.",
        "",
        "## Closed-set Known Test",
        "",
        "| Unknown Service | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in closed_rows:
        lines.append(
            f"| {row['fold']} | {float(row['accuracy']):.6f} | "
            f"{float(row['macro_f1']):.6f} | {float(row['weighted_f1']):.6f} |"
        )
    lines += [
        "", "## Open-set natural prevalence, Known-Val P95", "",
        "| Unknown Service | Method | AUROC | AUPRC | Binary F1 | UFAR | Known FRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda item: (str(item["fold"]), str(item["method"]))):
        lines.append(
            f"| {row['fold']} | {row['method']} | {float(row['auroc']):.6f} | "
            f"{float(row['auprc']):.6f} | {float(row['binary_f1']):.6f} | "
            f"{float(row['ufar']):.6f} | {float(row['known_frr']):.6f} |"
        )
    lines += [
        "", "## Boundaries", "",
        "- All normalization, supports and thresholds use Known Train/Validation only.",
        "- This is a single-seed, previously developed VNAT diagnostic, not untouched external validation.",
        "- Group/capture-disjoint splitting causes real imbalance, especially Remote-Access and Streaming; see `split_audit.csv`.",
        "- Natural and fixed 1:1 results are both retained; Binary F1 is prevalence-dependent.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    completion = {
        "status": "PASS",
        "folds_completed": len(FOLDS),
        "run_metric_rows": len(run_rows),
        "expected_run_metric_rows": 96,
        "unknown_training_samples": 0,
        "unknown_validation_samples": 0,
        "test_threshold_samples": 0,
        "protocol_freeze_status": json.loads((ROOT / "protocol_freeze_verification.json").read_text())["status"],
    }
    if len(run_rows) != 96 or completion["protocol_freeze_status"] != "PASS":
        raise RuntimeError(f"completion mismatch: {completion}")
    (ROOT / "completion_verification.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
