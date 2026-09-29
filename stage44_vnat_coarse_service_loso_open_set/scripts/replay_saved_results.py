#!/usr/bin/env python3
"""Independently replay Stage 44 metrics from frozen per-sample outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
FOLDS = ("communication", "file_transfer", "remote_access", "streaming")
METHODS = ("msp", "energy", "centroid", "des_v1")
METRICS = ("auroc", "auprc", "binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "ufar", "known_frr", "known_acceptance")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick(ids: list[str], count: int, namespace: str) -> np.ndarray:
    ranked = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"2022:{namespace}:{ids[i]}".encode("utf-8")
    ).digest())
    return np.asarray(sorted(ranked[:count]), dtype=np.int64)


def replay_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores > threshold
    known = truth == 0
    unknown = truth == 1
    return {
        "auroc": roc_auc_score(truth, scores),
        "auprc": average_precision_score(truth, scores),
        "binary_accuracy": accuracy_score(truth, predicted),
        "binary_precision": precision_score(truth, predicted, zero_division=0),
        "binary_recall": recall_score(truth, predicted, zero_division=0),
        "binary_f1": f1_score(truth, predicted, zero_division=0),
        "ufar": np.mean(~predicted[unknown]),
        "known_frr": np.mean(predicted[known]),
        "known_acceptance": np.mean(~predicted[known]),
    }


def check_close(actual: float, expected: str | float, label: str) -> float:
    difference = abs(float(actual) - float(expected))
    if not np.isfinite(difference) or difference > 1e-10:
        raise AssertionError(f"{label}: observed={actual} saved={expected}")
    return difference


def main() -> None:
    aggregate = rows(ROOT / "stage44_run_results.csv")
    if len(aggregate) != 96:
        raise AssertionError(f"expected 96 aggregate rows, found {len(aggregate)}")
    max_difference = 0.0
    replayed_rows = 0
    score_rows_total = 0
    verified_checkpoint_hashes = 0
    fold_counts: dict[str, dict[str, int]] = {}

    for fold in FOLDS:
        fold_dir = ROOT / "protocols" / fold
        protocol = json.loads((fold_dir / "protocol.json").read_text(encoding="utf-8"))
        verification = json.loads((fold_dir / "evaluation/evaluation_verification.json").read_text(encoding="utf-8"))
        role_path = fold_dir / "role_manifest.csv"
        if sha256(role_path) != protocol["role_manifest_sha256"] or sha256(role_path) != verification["role_manifest_sha256"]:
            raise AssertionError(f"{fold}: role-manifest hash changed")
        membership = rows(role_path)
        if len(membership) != len({row["flow_uid"] for row in membership}):
            raise AssertionError(f"{fold}: duplicate flow UID")
        by_role = {role: sorted((r for r in membership if r["role"] == role), key=lambda r: r["flow_uid"])
                   for role in ("known_train", "known_validation", "known_test", "unknown_test")}
        if {role: len(value) for role, value in by_role.items()} != protocol["counts"]:
            raise AssertionError(f"{fold}: role counts differ from frozen protocol")
        for key in ("group_id", "capture_id"):
            sets = [{r[key] for r in by_role[role]} for role in ("known_train", "known_validation", "known_test")]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError(f"{fold}: {key} crosses Known splits")
        if any(r["service"] == fold for role in ("known_train", "known_validation", "known_test") for r in by_role[role]):
            raise AssertionError(f"{fold}: Unknown service used as Known")
        if {r["service"] for r in by_role["unknown_test"]} != {fold}:
            raise AssertionError(f"{fold}: Unknown Test service mismatch")

        samples = rows(fold_dir / "evaluation/sample_scores.csv")
        score_rows_total += len(samples)
        if len(samples) != verification["score_rows"]:
            raise AssertionError(f"{fold}: score-row count mismatch")
        score_by_role = {role: [r for r in samples if r["role"] == role]
                         for role in ("known_validation", "known_test", "unknown_test")}
        for role, score_rows in score_by_role.items():
            if [r["flow_uid"] for r in score_rows] != [r["flow_uid"] for r in by_role[role]]:
                raise AssertionError(f"{fold}: score/membership ID mismatch for {role}")
        fold_counts[fold] = {role: len(value) for role, value in by_role.items()}

        with np.load(fold_dir / "evaluation/representations/known_test.npz", allow_pickle=False) as values:
            if values["flow_ids"].tolist() != [r["flow_uid"] for r in by_role["known_test"]]:
                raise AssertionError(f"{fold}: closed-set logits/membership ID mismatch")
            predicted = values["logits"].argmax(axis=1)
        truth = np.asarray([protocol["known_services"].index(r["service"]) for r in by_role["known_test"]])
        closed = json.loads((fold_dir / "evaluation/closed_set_metrics.json").read_text(encoding="utf-8"))
        for key, observed in (("accuracy", accuracy_score(truth, predicted)),
                              ("macro_f1", f1_score(truth, predicted, average="macro", zero_division=0)),
                              ("weighted_f1", f1_score(truth, predicted, average="weighted", zero_division=0))):
            max_difference = max(max_difference, check_close(observed, closed[key], f"{fold}: closed {key}"))

        saved = rows(fold_dir / "evaluation/open_set_metrics.csv")
        if len(saved) != 24:
            raise AssertionError(f"{fold}: expected 24 metric rows")
        for record in saved:
            method = record["method"]
            if method not in METHODS:
                raise AssertionError(f"{fold}: unexpected method {method}")
            percentile = int(record["threshold_percentile"])
            val_scores = np.asarray([float(r[f"score_{method}"]) for r in score_by_role["known_validation"]])
            threshold = float(np.percentile(val_scores, percentile, method="higher"))
            max_difference = max(max_difference, check_close(threshold, record["threshold"], f"{fold}:{method}:P{percentile}"))

            known = score_by_role["known_test"]
            unknown = score_by_role["unknown_test"]
            if record["view"] == "balanced_1to1":
                count = min(len(known), len(unknown))
                known = [known[i] for i in pick([r["flow_uid"] for r in known], count, f"{fold}:known")]
                unknown = [unknown[i] for i in pick([r["flow_uid"] for r in unknown], count, f"{fold}:unknown")]
            elif record["view"] != "natural":
                raise AssertionError(f"{fold}: unexpected view {record['view']}")
            if len(known) != int(record["known_samples"]) or len(unknown) != int(record["unknown_samples"]):
                raise AssertionError(f"{fold}: evaluation membership count mismatch")
            binary_truth = np.r_[np.zeros(len(known), dtype=np.int64), np.ones(len(unknown), dtype=np.int64)]
            score = np.asarray([float(r[f"score_{method}"]) for r in known + unknown])
            recalculated = replay_metrics(binary_truth, score, threshold)
            for key in METRICS:
                max_difference = max(max_difference, check_close(recalculated[key], record[key], f"{fold}:{method}:{record['view']}:P{percentile}:{key}"))
            if percentile == 95:
                for row in score_by_role["known_validation"] + score_by_role["known_test"] + score_by_role["unknown_test"]:
                    predicted_unknown = int(float(row[f"score_{method}"]) > threshold)
                    if predicted_unknown != int(row[f"prediction_{method}_p95"]):
                        raise AssertionError(f"{fold}:{method}: saved P95 prediction mismatch")
            replayed_rows += 1

        for path, expected in verification["checkpoint_hashes"].items():
            if sha256(PROJECT / path) != expected:
                raise AssertionError(f"{fold}: checkpoint hash mismatch: {path}")
            verified_checkpoint_hashes += 1

    result = {
        "status": "PASS",
        "folds": len(FOLDS),
        "replayed_metric_rows": replayed_rows,
        "score_rows": score_rows_total,
        "verified_checkpoint_hashes": verified_checkpoint_hashes,
        "max_absolute_metric_difference": max_difference,
        "fold_role_counts": fold_counts,
        "unknown_training_samples": 0,
        "unknown_validation_samples": 0,
        "test_threshold_samples": 0,
        "source": "saved scores, frozen membership and Known Validation only",
    }
    output = ROOT / "independent_replay_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
