#!/usr/bin/env python3
"""Independent score-level replay of every matched Open-Detect metric row."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from matched_common import FOLDS, ROOT, STAGE44, load_protocols, read_csv, role_rows, sha256, verify_freeze


KEYS = ("auroc", "auprc", "binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "ufar", "known_frr", "known_acceptance")


def select(ids: list[str], count: int, namespace: str) -> list[int]:
    ranked = sorted(range(len(ids)), key=lambda index: hashlib.sha256(
        ("2022:" + namespace + ":" + ids[index]).encode("utf-8")
    ).digest())
    return sorted(ranked[:count])


def replay(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    decision = score > threshold
    known = truth == 0
    unknown = truth == 1
    return {
        "auroc": roc_auc_score(truth, score),
        "auprc": average_precision_score(truth, score),
        "binary_accuracy": accuracy_score(truth, decision),
        "binary_precision": precision_score(truth, decision, zero_division=0),
        "binary_recall": recall_score(truth, decision, zero_division=0),
        "binary_f1": f1_score(truth, decision, zero_division=0),
        "ufar": np.mean(~decision[unknown]),
        "known_frr": np.mean(decision[known]),
        "known_acceptance": np.mean(~decision[known]),
    }


def equal(value: float, expected: str | float, name: str) -> float:
    difference = abs(float(value) - float(expected))
    if not np.isfinite(difference) or difference > 1e-10:
        raise AssertionError(f"{name}: replay={value} saved={expected}")
    return difference


def main() -> None:
    frozen = verify_freeze()
    protocols = load_protocols()
    result_rows = read_csv(ROOT / "od_run_results.csv")
    paired = read_csv(ROOT / "matched_comparison.csv")
    if len(result_rows) != 24 or len(paired) != 24:
        raise AssertionError("incomplete Native or paired metric grid")
    max_difference = 0.0
    score_count = 0
    for fold in FOLDS:
        roles = role_rows(fold)
        output = ROOT / "evaluation" / fold
        verification = json.loads((output / "evaluation_verification.json").read_text(encoding="utf-8"))
        checkpoint = ROOT / "runs" / fold / "best_checkpoint.pt"
        if verification["status"] != "PASS" or sha256(checkpoint) != verification["checkpoint_sha256_before"]:
            raise AssertionError(f"{fold}: checkpoint/evaluation verification failed")
        samples = read_csv(output / "sample_scores.csv")
        score_count += len(samples)
        baseline = read_csv(STAGE44 / "protocols" / fold / "evaluation/sample_scores.csv")
        selected = {role: [row for row in samples if row["role"] == role]
                    for role in ("known_validation", "known_test", "unknown_test")}
        for role, rows in selected.items():
            ids = [row["flow_uid"] for row in rows]
            if ids != [row["flow_uid"] for row in roles[role]] or ids != [row["flow_uid"] for row in baseline if row["role"] == role]:
                raise AssertionError(f"{fold}: {role} flow-ID mismatch")
        classes = list(protocols[fold]["known_applications"])
        known_y = [row["service"] for row in selected["known_test"]]
        known_pred = [row["predicted_known"] for row in selected["known_test"]]
        closed = json.loads((output / "closed_set_metrics.json").read_text(encoding="utf-8"))
        for key, value in (
            ("accuracy", accuracy_score(known_y, known_pred)),
            ("macro_f1", f1_score(known_y, known_pred, labels=classes, average="macro", zero_division=0)),
            ("weighted_f1", f1_score(known_y, known_pred, labels=classes, average="weighted", zero_division=0)),
        ):
            max_difference = max(max_difference, equal(value, closed[key], f"{fold}:closed:{key}"))
        for row in (item for item in result_rows if item["fold"] == fold):
            percentile = int(row["threshold_percentile"])
            validation = np.asarray([float(item["score_od_native"]) for item in selected["known_validation"]])
            threshold = float(np.percentile(validation, percentile, method="higher"))
            max_difference = max(max_difference, equal(threshold, row["threshold"], f"{fold}:P{percentile}:threshold"))
            known = selected["known_test"]
            unknown = selected["unknown_test"]
            if row["view"] == "balanced_1to1":
                count = min(len(known), len(unknown))
                known = [known[index] for index in select([item["flow_uid"] for item in known], count, f"{fold}:known")]
                unknown = [unknown[index] for index in select([item["flow_uid"] for item in unknown], count, f"{fold}:unknown")]
            elif row["view"] != "natural":
                raise AssertionError(f"{fold}: unexpected view")
            if len(known) != int(row["known_samples"]) or len(unknown) != int(row["unknown_samples"]):
                raise AssertionError(f"{fold}: evaluation count mismatch")
            truth = np.r_[np.zeros(len(known), dtype=np.int64), np.ones(len(unknown), dtype=np.int64)]
            score = np.asarray([float(item["score_od_native"]) for item in known + unknown])
            observed = replay(truth, score, threshold)
            for key in KEYS:
                max_difference = max(max_difference, equal(observed[key], row[key], f"{fold}:{row['view']}:P{percentile}:{key}"))
            if percentile == 95:
                for item in samples:
                    if int(float(item["score_od_native"]) > threshold) != int(item["prediction_od_native_p95"]):
                        raise AssertionError(f"{fold}: saved P95 decision mismatch")
        matching = [item for item in paired if item["fold"] == fold]
        if len(matching) != 6:
            raise AssertionError(f"{fold}: paired row count mismatch")
        for row in matching:
            native = next(item for item in result_rows if (item["fold"], item["view"], item["threshold_percentile"]) ==
                          (fold, row["view"], row["threshold_percentile"]))
            reference = next(item for item in read_csv(STAGE44 / "protocols" / fold / "evaluation/open_set_metrics.csv")
                             if (item["method"], item["view"], item["threshold_percentile"]) ==
                             ("des_v1", row["view"], row["threshold_percentile"]))
            for key in ("auroc", "auprc", "binary_f1", "ufar", "known_frr"):
                max_difference = max(max_difference, equal(float(reference[key]) - float(native[key]),
                                                           row[f"delta_three_view_minus_od_{key}"], f"{fold}:paired:{key}"))
    if verify_freeze() != frozen:
        raise AssertionError("frozen source changed during independent replay")
    output = {
        "status": "PASS", "folds": 4, "replayed_native_metric_rows": 24,
        "verified_paired_rows": 24, "replayed_score_rows": score_count,
        "verified_checkpoint_hashes": 4, "max_absolute_metric_difference": max_difference,
        "matched_train_val_test_unknown_flow_ids": True,
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "unknown_threshold_samples": 0, "test_threshold_samples": 0,
    }
    (ROOT / "independent_replay_verification.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
