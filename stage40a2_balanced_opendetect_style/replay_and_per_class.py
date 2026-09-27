#!/usr/bin/env python3
"""Independent count/threshold replay and per-Unknown-class detection summary."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "stage40_ustc_cic_open_set/ustc_a2/detection/sample_scores.csv"
METHODS = ("msp", "energy", "centroid", "des_v1")


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    selection = rows(HERE / "selection_manifest.csv")
    results = rows(HERE / "balanced_results.csv")
    all_scores = {r["flow_id"]: r for r in rows(SOURCE)}
    if len(selection) != 1116 or len({r["flow_id"] for r in selection}) != 1116:
        raise RuntimeError("Selected IDs not 1,116 unique")
    sample = [all_scores[r["flow_id"]] for r in selection]
    if any((a["role"], a["class_name"]) != (b["role"], b["class_name"])
           for a, b in zip(selection, sample)):
        raise RuntimeError("Selection labels do not match frozen source")
    if Counter(r["role"] for r in sample) != {"known_test": 558, "unknown_test": 558}:
        raise RuntimeError("Selected role counts are not balanced")
    unknown_classes = sorted({r["class_name"] for r in sample if r["role"] == "unknown_test"})
    if unknown_classes != ["Geodo", "Htbot", "Tinba"]:
        raise RuntimeError("A-2 Unknown classes mismatch")

    by_key = {(r["method"], r["threshold_policy"]): r for r in results}
    if len(by_key) != 8:
        raise RuntimeError("Expected four methods and two policies")
    class_results = []
    for method in METHODS:
        scores = np.asarray([float(r[f"score_{method}"]) for r in sample], dtype=float)
        truth = np.asarray([int(r["role"] == "unknown_test") for r in sample], dtype=int)
        fpr, tpr, thresholds = roc_curve(truth, scores)
        for policy in ("known_validation_p95", "released_code_test_oracle_youden"):
            r = by_key[(method, policy)]
            threshold = float(r["threshold"])
            if policy == "released_code_test_oracle_youden":
                if not np.isclose(threshold, thresholds[int(np.argmax(tpr - fpr))], atol=1e-12):
                    raise RuntimeError(f"Oracle threshold mismatch: {method}")
                pred = scores >= threshold
            else:
                pred = scores > threshold
                stored = np.asarray([int(s[f"reject_{method}"]) for s in sample], dtype=bool)
                if not np.array_equal(pred, stored):
                    raise RuntimeError(f"P95 frozen decisions mismatch: {method}")
            tp = int(np.sum(pred & (truth == 1)))
            fp = int(np.sum(pred & (truth == 0)))
            tn = int(np.sum(~pred & (truth == 0)))
            fn = int(np.sum(~pred & (truth == 1)))
            for key, actual in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
                if int(r[key]) != actual:
                    raise RuntimeError(f"Confusion count mismatch: {method}/{policy}/{key}")
            expected_acc = (tp + tn) / len(truth)
            expected_f1 = 2 * tp / (2 * tp + fp + fn)
            if not np.isclose(expected_acc, float(r["binary_accuracy"]), atol=1e-12):
                raise RuntimeError(f"Accuracy mismatch: {method}/{policy}")
            if not np.isclose(expected_f1, float(r["unknown_binary_f1"]), atol=1e-12):
                raise RuntimeError(f"F1 mismatch: {method}/{policy}")
            if not np.isclose(roc_auc_score(truth, scores), float(r["auroc"]), atol=1e-12):
                raise RuntimeError(f"AUROC mismatch: {method}/{policy}")
            if not np.isclose(average_precision_score(truth, scores), float(r["auprc"]), atol=1e-12):
                raise RuntimeError(f"AUPRC mismatch: {method}/{policy}")
            for cls in unknown_classes:
                mask = np.asarray([s["role"] == "known_test" or s["class_name"] == cls for s in sample])
                y = truth[mask]
                s = scores[mask]
                p = pred[mask]
                class_results.append({
                    "method": method, "threshold_policy": policy,
                    "unknown_class": cls, "known_test": int((y == 0).sum()),
                    "unknown_test": int((y == 1).sum()),
                    "unknown_detected": int(np.sum(p & (y == 1))),
                    "unknown_accepted": int(np.sum(~p & (y == 1))),
                    "unknown_recall": float(np.mean(p[y == 1])),
                    "ufar": float(np.mean(~p[y == 1])),
                    "auroc": float(roc_auc_score(y, s)),
                    "auprc": float(average_precision_score(y, s)),
                })
    write_rows(HERE / "per_unknown_class.csv", class_results)
    selection_hash = hashlib.sha256("\n".join(r["flow_id"] for r in selection).encode()).hexdigest()
    first_audit = json.loads((HERE / "completion_verification.json").read_text(encoding="utf-8"))
    if selection_hash != first_audit["selected_id_sha256"]:
        raise RuntimeError("Frozen selection hash mismatch")
    audit = {
        "status": "PASS", "selected_id_sha256": selection_hash,
        "balanced_rows": len(selection), "methods": list(METHODS),
        "policies": ["known_validation_p95", "released_code_test_oracle_youden"],
        "metric_rows_replayed": len(results), "per_unknown_class_rows": len(class_results),
        "no_training_or_new_inference": True,
    }
    (HERE / "independent_replay.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
