#!/usr/bin/env python3
"""Retrospective A-2 1:1 evaluation of frozen Stage40 sample scores.

Sample selection uses only role and flow ID.  Existing encoder, anomaly scores,
and validation thresholds are read-only.  The labeled-Test Youden threshold is
reported separately and must not be interpreted as an independent test result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "stage40_ustc_cic_open_set/ustc_a2/detection"
OD_SUMMARY = ROOT.parent / "Open-Detect/artifacts/paper-reproduction/v6-local-pcap-fivefold-corrected-v1/campaign_summary.json"
OD_POOL = ROOT.parent / "Open-Detect/artifacts/v6-local-pcap-fivefold-splits-v1/campaign_manifest.json"
METHODS = ("msp", "energy", "centroid", "des_v1")
SEED = 2022
SOURCE_FILES = (SOURCE / "sample_scores.csv", SOURCE / "calibration.json", SOURCE / "open_set_results.csv", OD_SUMMARY, OD_POOL)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(method: str, policy: str, threshold: float, truth: np.ndarray,
            score: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(truth, pred, labels=[0, 1]).ravel())
    return {
        "dataset": "USTC-TFC2016", "setting": "A-2", "model": "Stage40 frozen three-view",
        "subset": "hash2022_558_known_558_unknown", "method": method,
        "threshold_policy": policy, "threshold": float(threshold),
        "known_test": int((truth == 0).sum()), "unknown_test": int((truth == 1).sum()),
        "binary_accuracy": float(accuracy_score(truth, pred)),
        "unknown_precision": float(precision_score(truth, pred, zero_division=0)),
        "unknown_recall": float(recall_score(truth, pred, zero_division=0)),
        "unknown_binary_f1": float(f1_score(truth, pred, zero_division=0)),
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "ufar": fn / (fn + tp), "known_frr": fp / (tn + fp),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "test_labels_used_for_threshold": int(policy == "released_code_test_oracle_youden"),
    }


def main() -> None:
    source_hashes_before = {str(path): digest(path) for path in SOURCE_FILES}
    with (SOURCE / "sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    if len(sample_rows) != 4891 or len({r["flow_id"] for r in sample_rows}) != len(sample_rows):
        raise RuntimeError("Frozen Stage40 score ID/count mismatch")
    known = [r for r in sample_rows if r["role"] == "known_test" and r["is_unknown"] == "0"]
    unknown = [r for r in sample_rows if r["role"] == "unknown_test" and r["is_unknown"] == "1"]
    if len(known) != 4333 or len(unknown) != 558:
        raise RuntimeError("Unexpected frozen Stage40 role counts")

    # Freeze a score-blind subset: 2022-salted SHA256 order of Known flow IDs.
    # All Unknown Test IDs are retained.  This approximates only the 1:1 class
    # prevalence of OD v6; it is not a flow-matched, five-fold evaluation.
    def rank(row: dict) -> str:
        return hashlib.sha256(f"stage40-a2-balanced-{SEED}:{row['flow_id']}".encode()).hexdigest()

    chosen_known = sorted(known, key=lambda r: (rank(r), r["flow_id"]))[: len(unknown)]
    selected = sorted(chosen_known + unknown, key=lambda r: (r["role"], r["flow_id"]))
    selection_rows = [
        {"flow_id": r["flow_id"], "role": r["role"], "class_name": r["class_name"],
         "selection_seed": SEED, "selection_rule": "all_unknown_and_lowest_sha256_known"}
        for r in selected
    ]
    csv_write(OUT / "selection_manifest.csv", selection_rows)
    sample_id_hash = hashlib.sha256("\n".join(r["flow_id"] for r in selected).encode()).hexdigest()

    truth = np.asarray([int(r["is_unknown"]) for r in selected], dtype=np.int64)
    if len(selected) != 1116 or int(truth.sum()) != 558:
        raise RuntimeError("Balanced subset count mismatch")
    calibration = json.loads((SOURCE / "calibration.json").read_text(encoding="utf-8"))
    existing = list(csv.DictReader((SOURCE / "open_set_results.csv").open(newline="", encoding="utf-8")))
    if {r["method"] for r in existing} != set(METHODS):
        raise RuntimeError("Unexpected frozen method list")

    rows: list[dict] = []
    for method in METHODS:
        score = np.asarray([float(r[f"score_{method}"]) for r in selected], dtype=np.float64)
        if not np.all(np.isfinite(score)):
            raise RuntimeError(f"Nonfinite score: {method}")
        p95 = float(calibration["thresholds"][method])
        formal = (score > p95).astype(np.int64)
        stored = np.asarray([int(r[f"reject_{method}"]) for r in selected], dtype=np.int64)
        if not np.array_equal(formal, stored):
            raise RuntimeError(f"Stored P95 decisions differ: {method}")
        rows.append(metrics(method, "known_validation_p95", p95, truth, score, formal))
        fpr, tpr, thresholds = roc_curve(truth, score)
        oracle = float(thresholds[int(np.argmax(tpr - fpr))])
        rows.append(metrics(method, "released_code_test_oracle_youden", oracle, truth,
                            score, (score >= oracle).astype(np.int64)))
    csv_write(OUT / "balanced_results.csv", rows)

    known_truth = [r["class_name"] for r in chosen_known]
    known_pred = [r["predicted_known"] for r in chosen_known]
    closed = {
        "known_test": len(chosen_known), "known_classes_present": len(set(known_truth)),
        "accuracy": accuracy_score(known_truth, known_pred),
        "weighted_f1": f1_score(known_truth, known_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(known_truth, known_pred, average="macro", zero_division=0),
    }
    (OUT / "balanced_closed_set.json").write_text(json.dumps(closed, indent=2) + "\n", encoding="utf-8")
    counts = [
        {"role": role, "class_name": cls, "samples": count}
        for (role, cls), count in sorted(Counter((r["role"], r["class_name"]) for r in selected).items())
    ]
    csv_write(OUT / "class_support.csv", counts)

    od = json.loads(OD_SUMMARY.read_text(encoding="utf-8"))
    od_a2 = next(r for r in od["metrics"] if r["scenario"] == "A-2")
    od_pool = json.loads(OD_POOL.read_text(encoding="utf-8"))
    context = {
        "paper_scenario": "A-2: 17 Known / Geodo-Htbot-Tinba Unknown",
        "paper_table_v_accuracy_mean": 0.8922, "paper_table_v_f1_mean": 0.8910,
        "od_v6_local_model_runs": 5, "od_v6_pool_size": od_pool["sample_count"],
        "od_v6_test_per_run": "600 Known + 600 Unknown; five different local folds",
        "od_v6_p95_binary_accuracy_mean": od_a2["validation_threshold_accuracy_mean"],
        "od_v6_p95_unknown_f1_mean": od_a2["validation_threshold_f1_mean"],
        "od_v6_auroc_mean": od_a2["auroc_mean"],
        "od_v6_oracle_binary_accuracy_mean": od_a2["oracle_accuracy_mean"],
        "od_v6_oracle_unknown_f1_mean": od_a2["oracle_f1_mean"],
        "current_test_per_run": "558 Known + all 558 frozen Unknown; one frozen Stage40 model",
        "exact_sample_identity_proven": False,
        "reason": "OD v6 has separate 34,585-image local pool/folds and hashed provenance IDs; Stage40 has a different flow-ID and source-split pipeline, and only 558 scored Unknown Test samples",
        "paired_method_comparison_allowed": False,
    }
    (OUT / "comparison_context.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    source_hashes_after = {str(path): digest(path) for path in SOURCE_FILES}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Frozen source changed during balanced evaluation")
    audit = {
        "status": "PASS", "selection_score_blind": True, "selection_seed": SEED,
        "selected_id_sha256": sample_id_hash, "selected_ids_unique": len({r["flow_id"] for r in selected}) == 1116,
        "source_hashes_unchanged": True, "source_sha256": source_hashes_before,
        "model_training": False, "new_inference": False, "threshold_p95_reused": True,
        "oracle_uses_test_labels": True, "same_flow_as_od_v6": "NOT_ESTABLISHED",
        "metric_rows": len(rows), "closed": closed,
    }
    (OUT / "completion_verification.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "closed": closed, "results": rows, "context": context}, indent=2))


if __name__ == "__main__":
    main()
