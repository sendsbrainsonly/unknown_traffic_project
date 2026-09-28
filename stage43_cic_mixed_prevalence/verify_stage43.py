#!/usr/bin/env python3
"""Independent replay for Stage 43 frozen-score mixtures."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


OUT = Path(__file__).resolve().parent
PROTOCOL = OUT / "mixed_protocol.json"
METHODS = ("msp", "energy", "centroid", "des_v1")
METRICS = (
    "auroc", "auprc", "ufar", "known_frr", "known_acceptance",
    "binary_accuracy", "balanced_accuracy", "mcc",
    "unknown_precision", "unknown_recall", "binary_f1",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    decision = score > threshold
    known = truth == 0
    unknown = truth == 1
    return {
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "ufar": float(np.mean(~decision[unknown])),
        "known_frr": float(np.mean(decision[known])),
        "known_acceptance": float(np.mean(~decision[known])),
        "binary_accuracy": float(accuracy_score(truth, decision)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, decision)),
        "mcc": float(matthews_corrcoef(truth, decision)),
        "unknown_precision": float(precision_score(truth, decision, zero_division=0)),
        "unknown_recall": float(recall_score(truth, decision, zero_division=0)),
        "binary_f1": float(f1_score(truth, decision, zero_division=0)),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "PASS":
        raise RuntimeError("protocol not PASS")
    if digest(OUT / "run_stage43.py") != protocol["runner_sha256"]:
        raise RuntimeError("runner hash drift")
    if digest(Path(__file__)) != protocol["verifier_sha256"]:
        raise RuntimeError("verifier hash drift")
    lookup: dict[str, dict[str, str]] = {}
    known_loaded = False
    for source in protocol["source_files"]:
        score_path = Path(source["score_path"])
        result_path = Path(source["result_path"])
        if digest(score_path) != source["score_sha256"] or digest(result_path) != source["result_sha256"]:
            raise RuntimeError(f"source hash drift: {source['source_key']}")
        with score_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["role"] == "known_test":
                    if known_loaded:
                        continue
                    lookup[row["flow_id"]] = row
                else:
                    if row["flow_id"] in lookup:
                        raise RuntimeError(f"duplicate lookup ID: {row['flow_id']}")
                    lookup[row["flow_id"]] = row
        known_loaded = True
    scenario_map = {(row["experiment"], row["setting"]): row for row in protocol["scenarios"]}
    with (OUT / "run_level_metrics.csv").open(newline="") as handle:
        recorded = {
            (row["experiment"], row["setting"], int(row["seed"]), row["method"]): row
            for row in csv.DictReader(handle)
        }
    checked_mixtures = 0
    checked_metrics = 0
    max_error = 0.0
    with (OUT / "mixture_sample_manifest.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        keyfunc = lambda row: (row["experiment"], row["setting"], int(row["seed"]))
        for key, group in itertools.groupby(reader, key=keyfunc):
            rows = list(group)
            experiment, setting, seed = key
            scenario = scenario_map[(experiment, setting)]
            if len(rows) != protocol["total_samples_per_mixture"]:
                raise RuntimeError(f"mixture size drift: {key}")
            ids = [row["flow_id"] for row in rows]
            if len(ids) != len(set(ids)) or any(flow_id not in lookup for flow_id in ids):
                raise RuntimeError(f"membership ID failure: {key}")
            expected_known = sum(scenario["known_quotas"].values())
            expected_unknown = sum(scenario["unknown_quotas"].values())
            if Counter(row["role"] for row in rows) != Counter({"known_test": expected_known, "unknown_test": expected_unknown}):
                raise RuntimeError(f"role quota failure: {key}")
            known_counts = Counter(row["class_name"] for row in rows if row["role"] == "known_test")
            unknown_counts = Counter(row["class_name"] for row in rows if row["role"] == "unknown_test")
            if known_counts != Counter(scenario["known_quotas"]) or unknown_counts != Counter(scenario["unknown_quotas"]):
                raise RuntimeError(f"class quota failure: {key}")
            source_rows = [lookup[flow_id] for flow_id in ids]
            truth = np.asarray([int(row["is_unknown"]) for row in source_rows])
            for method in METHODS:
                scores = np.asarray([float(row[f"score_{method}"]) for row in source_rows])
                replay = metric(truth, scores, float(protocol["thresholds"][method]))
                saved = recorded[(experiment, setting, seed, method)]
                for name in METRICS:
                    error = abs(float(saved[name]) - replay[name])
                    max_error = max(max_error, error)
                    if error > 1e-12:
                        raise RuntimeError(f"metric mismatch {key}/{method}/{name}: {error}")
                checked_metrics += 1
            checked_mixtures += 1
    expected_mixtures = len(protocol["scenarios"]) * len(protocol["resampling_seeds"])
    if checked_mixtures != expected_mixtures or checked_metrics != expected_mixtures * len(METHODS):
        raise RuntimeError("completion cardinality failure")
    completion = json.loads((OUT / "run_completion.json").read_text())
    output = {
        "status": "PASS",
        "source_hashes_unchanged": True,
        "runner_and_verifier_hashes_unchanged": True,
        "mixtures_replayed": checked_mixtures,
        "metric_rows_replayed": checked_metrics,
        "membership_rows_checked": checked_mixtures * protocol["total_samples_per_mixture"],
        "maximum_absolute_metric_error": max_error,
        "final_gate": completion["final_gate"],
        "unknown_fitting_count": 0,
        "test_fitting_count": 0,
        "encoder_training_count": 0,
        "pcap_reads": 0,
    }
    tmp = OUT / "completion_verification.json.tmp"
    tmp.write_text(json.dumps(output, indent=2) + "\n")
    tmp.replace(OUT / "completion_verification.json")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
