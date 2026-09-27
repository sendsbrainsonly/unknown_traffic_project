#!/usr/bin/env python3
"""Read-only metric replay of frozen Stage40 USTC scores.

The validation-P95 rows are the formal operating point.  The Test-oracle
rows reproduce the released Open-Detect code's threshold-selection rule only
as a retrospective, non-independent diagnostic.
"""

from __future__ import annotations

import csv
import hashlib
import json
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


PROJECT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SOURCE = PROJECT / "stage40_ustc_cic_open_set/ustc_a2/detection"
METHODS = ("msp", "energy", "centroid", "des_v1")
PROTECTED = ("sample_scores.csv", "open_set_results.csv", "calibration.json", "known_test_closed_set.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_row(method: str, policy: str, threshold: float, truth: np.ndarray,
               scores: np.ndarray, predicted: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    return {
        "dataset": "USTC-TFC2016", "protocol": "A-2", "method": method,
        "threshold_policy": policy, "threshold": float(threshold),
        "known_test": int(np.count_nonzero(truth == 0)),
        "unknown_test": int(np.count_nonzero(truth == 1)),
        "unknown_prevalence": float(np.mean(truth)),
        "accuracy_binary": float(accuracy_score(truth, predicted)),
        "precision_unknown": float(precision_score(truth, predicted, zero_division=0)),
        "recall_unknown": float(recall_score(truth, predicted, zero_division=0)),
        "f1_unknown_binary": float(f1_score(truth, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
        "ufar": float(fn / (tp + fn)),
        "known_frr": float(fp / (tn + fp)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "test_labels_used_for_threshold": policy == "released_code_test_oracle_youden",
    }


def main() -> None:
    before = {name: sha256(SOURCE / name) for name in PROTECTED}
    with (SOURCE / "sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    with (SOURCE / "open_set_results.csv").open(newline="", encoding="utf-8") as handle:
        existing = {row["method"]: row for row in csv.DictReader(handle)}
    calibration = json.loads((SOURCE / "calibration.json").read_text(encoding="utf-8"))
    closed_existing = json.loads((SOURCE / "known_test_closed_set.json").read_text(encoding="utf-8"))

    ids = [row["flow_id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample ID in frozen score file")
    roles = {row["role"] for row in samples}
    if roles != {"known_test", "unknown_test"}:
        raise RuntimeError(f"Unexpected roles: {roles}")
    truth = np.asarray([int(row["is_unknown"]) for row in samples], dtype=np.int64)
    if any((row["role"] == "unknown_test") != bool(int(row["is_unknown"])) for row in samples):
        raise RuntimeError("Role and unknown flag mismatch")

    known = [row for row in samples if row["role"] == "known_test"]
    y_known = [row["class_name"] for row in known]
    p_known = [row["predicted_known"] for row in known]
    closed = {
        "dataset": "USTC-TFC2016", "protocol": "A-2",
        "known_test": len(known), "num_known_classes": len(set(y_known)),
        "accuracy": float(accuracy_score(y_known, p_known)),
        "precision_weighted": float(precision_score(y_known, p_known, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_known, p_known, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_known, p_known, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_known, p_known, average="macro", zero_division=0)),
    }
    for key, old_key in (("accuracy", "accuracy"), ("f1_weighted", "weighted_f1"), ("f1_macro", "macro_f1")):
        if not np.isclose(closed[key], float(closed_existing[old_key]), atol=1e-12):
            raise RuntimeError(f"Closed-set replay mismatch: {key}")

    results = []
    checks = []
    for method in METHODS:
        scores = np.asarray([float(row[f"score_{method}"]) for row in samples], dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"Nonfinite score: {method}")
        threshold = float(calibration["thresholds"][method])
        official_pred = (scores > threshold).astype(np.int64)
        stored_pred = np.asarray([int(row[f"reject_{method}"]) for row in samples], dtype=np.int64)
        if not np.array_equal(official_pred, stored_pred):
            raise RuntimeError(f"Frozen P95 decisions mismatch: {method}")
        formal = metric_row(method, "known_validation_p95", threshold, truth, scores, official_pred)
        for key, previous in (("auroc", "auroc"), ("auprc", "auprc"),
                              ("ufar", "ufar"), ("known_frr", "known_frr"),
                              ("f1_unknown_binary", "binary_f1")):
            if not np.isclose(formal[key], float(existing[method][previous]), atol=1e-12):
                raise RuntimeError(f"Frozen result replay mismatch: {method}/{key}")
        results.append(formal)

        # Exact released-code rule in Open-Detect/code/test.py: roc_curve on
        # labeled Known+Unknown Test, first max(tpr-fpr), decision >= threshold.
        fpr, tpr, thresholds = roc_curve(truth, scores)
        oracle_threshold = float(thresholds[int(np.argmax(tpr - fpr))])
        oracle_pred = (scores >= oracle_threshold).astype(np.int64)
        results.append(metric_row(method, "released_code_test_oracle_youden",
                                  oracle_threshold, truth, scores, oracle_pred))
        checks.append({"method": method, "stored_p95_decisions_equal": True,
                       "existing_formal_metrics_replayed": True})

    with (OUT / "open_detect_metric_supplement.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (OUT / "closed_set_metrics.json").write_text(json.dumps(closed, indent=2) + "\n", encoding="utf-8")
    after = {name: sha256(SOURCE / name) for name in PROTECTED}
    if before != after:
        raise RuntimeError("Frozen Stage40 input hashes changed")
    audit = {
        "status": "PASS", "frozen_input_hashes_unchanged": True,
        "source_sha256": before, "sample_id_unique": True,
        "samples": len(samples), "known_test": len(known),
        "unknown_test": int(truth.sum()), "method_checks": checks,
        "model_training": False, "threshold_refit_for_formal_metrics": False,
        "test_oracle_is_non_independent_retrospective_only": True,
        "cic_open_set_scores_available": False,
    }
    (OUT / "completion_verification.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"closed": closed, "results": results, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
