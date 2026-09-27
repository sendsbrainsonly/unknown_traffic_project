#!/usr/bin/env python3
"""Read-only scientific replay; writes only new Stage37 verification artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from run_joint import HERE, PROJECT, CLASSES, EXPECTED, MANIFEST, FROZEN_MANIFEST_SHA
from run_joint import joint


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if joint.sha(MANIFEST) != FROZEN_MANIFEST_SHA:
        raise RuntimeError("Stage20 freeze drift")
    rows = []
    audits = {}
    for dataset in CLASSES:
        run = HERE / "runs" / dataset / "seed2022_joint_e2e20"
        test = run / "known_test_evaluation"
        summary = json.loads((run / "train_summary.json").read_text())
        result = json.loads((test / "results.json").read_text())
        if summary["status"] != "PASS" or result["status"] != "PASS":
            raise RuntimeError("missing PASS status")
        if joint.sha(run / "model_best.pt") != summary["checkpoint_sha256"]:
            raise RuntimeError("checkpoint SHA drift")
        if result["checkpoint_sha256"] != summary["checkpoint_sha256"]:
            raise RuntimeError("test checkpoint provenance drift")
        own = read_csv(test / "sample_predictions.csv")
        reference = read_csv(PROJECT / "stage32_three_dataset_coarse_single_seed/runs" /
                             dataset / "known_test_evaluation/sample_predictions.csv")
        a = {r["flow_id"]: r for r in own}
        b = {r["flow_id"]: r for r in reference}
        if len(a) != EXPECTED[dataset][2] or set(a) != set(b):
            raise RuntimeError("joint/staged Test flow-ID mismatch")
        ordered = sorted(a)
        classes = CLASSES[dataset]
        truth = np.asarray([classes.index(a[uid]["true_class"]) for uid in ordered])
        predicted = np.asarray([classes.index(a[uid]["predicted_class"]) for uid in ordered])
        staged = np.asarray([classes.index(b[uid]["pred_coarse"]) for uid in ordered])
        if any(a[uid]["true_class"] != b[uid]["true_coarse"] for uid in ordered):
            raise RuntimeError("joint/staged Test truth mismatch")
        for key, computed in (("accuracy", accuracy_score(truth, predicted)),
                              ("macro_f1", f1_score(truth, predicted, average="macro")),
                              ("weighted_f1", f1_score(truth, predicted, average="weighted"))):
            if abs(result["known_test"][key] - computed) > 1e-12:
                raise RuntimeError(f"joint metric replay failed: {dataset}/{key}")
        stored_confusion = np.load(test / "confusion_matrix.npy", allow_pickle=False)
        if not np.array_equal(stored_confusion, confusion_matrix(truth, predicted,
                             labels=np.arange(len(classes)))):
            raise RuntimeError("joint confusion replay failed")
        own_correct, staged_correct = predicted == truth, staged == truth
        paired = {"joint_only_correct": int(np.sum(own_correct & ~staged_correct)),
                  "staged_only_correct": int(np.sum(~own_correct & staged_correct)),
                  "both_correct": int(np.sum(own_correct & staged_correct)),
                  "both_wrong": int(np.sum(~own_correct & ~staged_correct))}
        metric = {"dataset": dataset, "test_samples": len(ordered),
                  "joint_accuracy": accuracy_score(truth, predicted),
                  "staged_accuracy": accuracy_score(truth, staged),
                  "joint_macro_f1": f1_score(truth, predicted, average="macro"),
                  "staged_macro_f1": f1_score(truth, staged, average="macro"),
                  "joint_weighted_f1": f1_score(truth, predicted, average="weighted"),
                  "staged_weighted_f1": f1_score(truth, staged, average="weighted"),
                  **paired}
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            metric[f"delta_joint_minus_staged_{name}"] = metric[f"joint_{name}"] - metric[f"staged_{name}"]
        rows.append(metric)
        audits[dataset] = {"status": "PASS", "test_sample_ids_match_stage32": True,
                           "test_true_labels_match_stage32": True,
                           "checkpoint_sha256": summary["checkpoint_sha256"],
                           "metric_confusion_replay": True, "paired": paired}
    with (HERE / "paired_comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "completion_verification.json").write_text(json.dumps({
        "status": "PASS", "stage20_manifest_sha256": FROZEN_MANIFEST_SHA,
        "datasets": audits, "paired_rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
