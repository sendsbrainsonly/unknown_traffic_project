#!/usr/bin/env python3
"""Pair old and balanced CIC predictions on the exact same Test flow IDs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from prepare_balanced import AUDIT, CLASSES, MANIFEST, ROOT, SOURCE, sha, verified_rows

NEW = ROOT / "cicids2017/runs/ustc/A-2/known_test_evaluation"
OLD = SOURCE / "cicids2017/runs/ustc/A-2/known_test_evaluation"


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {r["flow_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise RuntimeError(f"duplicate predictions in {path}")
    return by_id


def metrics(truth: list[str], prediction: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=CLASSES, average="weighted", zero_division=0)),
    }


def main() -> None:
    balanced = verified_rows()
    test = {r["flow_id"]: r["label"] for r in balanced if r["split"] == "test"}
    if len(test) != 2536:
        raise RuntimeError(f"unexpected balanced Test count: {len(test)}")
    old_report = json.loads((OLD / "results.json").read_text())
    new_report = json.loads((NEW / "results.json").read_text())
    if old_report["status"] != "PASS" or new_report["status"] != "PASS":
        raise RuntimeError("Test result not PASS")
    old = load_predictions(OLD / "sample_predictions.csv")
    new = load_predictions(NEW / "sample_predictions.csv")
    if set(new) != set(test) or not set(test).issubset(set(old)):
        raise RuntimeError("old/new matched Test IDs differ")
    ids = sorted(test)
    if any(old[uid]["true_class"] != test[uid] or new[uid]["true_class"] != test[uid] for uid in ids):
        raise RuntimeError("matched Test label mismatch")
    truth = [test[uid] for uid in ids]
    old_pred = [old[uid]["predicted_class"] for uid in ids]
    new_pred = [new[uid]["predicted_class"] for uid in ids]
    old_score = metrics(truth, old_pred)
    new_score = metrics(truth, new_pred)
    reported = {name: float(new_report[name]) for name in new_score}
    if any(abs(new_score[name] - reported[name]) > 1e-12 for name in reported):
        raise RuntimeError("new Test report replay failed")
    old_counts = Counter(truth)
    per_class = []
    for name, pred in (("original", old_pred), ("balanced", new_pred)):
        p, r, f, support = precision_recall_fscore_support(truth, pred, labels=CLASSES, zero_division=0)
        per_class.extend({"model": name, "class": label, "precision": float(p[i]),
                          "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                         for i, label in enumerate(CLASSES))
    with (ROOT / "matched_test_per_class.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "class", "precision", "recall", "f1", "support"))
        writer.writeheader()
        writer.writerows(per_class)
    comparison = {
        "status": "PASS", "claim_scope": "matched-subset development comparison",
        "balanced_test_samples": len(ids), "balanced_test_class_counts": dict(old_counts),
        "original_full_test_samples": old_report["known_test_samples"],
        "original_full_test": {name: old_report[name] for name in old_score},
        "original_on_balanced_test": old_score,
        "balanced_model_on_balanced_test": new_score,
        "balanced_minus_original_same_test": {name: new_score[name] - old_score[name] for name in old_score},
        "old_predictions_sha256": sha(OLD / "sample_predictions.csv"),
        "new_predictions_sha256": sha(NEW / "sample_predictions.csv"),
        "balanced_manifest_sha256": sha(MANIFEST),
        "source_manifest_sha256": json.loads(AUDIT.read_text())["source_manifest_sha256"],
        "unknown_samples_used": 0, "test_threshold_or_checkpoint_selection": 0,
    }
    (ROOT / "matched_test_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison))


if __name__ == "__main__":
    main()
