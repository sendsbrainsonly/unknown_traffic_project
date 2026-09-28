#!/usr/bin/env python3
"""Frozen evaluation of slowloris using Stage40 BENIGN+PortScan representation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from stage42s_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress, read_csv, write_csv, write_json_new

STAGE40 = PROJECT / "stage40_ustc_cic_open_set"
sys.path.insert(0, str(STAGE40))
import score_frozen as s40  # noqa: E402

SOURCE_UNIT = STAGE40 / "unknown_slowhttptest"
RUN = SOURCE_UNIT / "runs/ustc/A-2"
SOURCE_DETECTION = SOURCE_UNIT / "detection"
OUT = ROOT / UNIT / "detection"
CLASSES = ["BENIGN", "PortScan"]
METHODS = ("msp", "energy", "centroid", "des_v1")


def metric(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict:
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
        "unknown_precision": float(precision_score(truth, decision, zero_division=0)),
        "unknown_recall": float(recall_score(truth, decision, zero_division=0)),
        "binary_f1": float(f1_score(truth, decision, zero_division=0)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen three-view inference")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "PASS" or digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("candidate protocol drift")
    calibration_path = SOURCE_DETECTION / "calibration.json"
    source_scores_path = SOURCE_DETECTION / "sample_scores.csv"
    if digest(calibration_path) != protocol["known_calibration_sha256"] or \
       digest(source_scores_path) != protocol["known_test_sample_scores_sha256"]:
        raise RuntimeError("Stage40 Known evidence changed")
    calibration = json.loads(calibration_path.read_text())
    if calibration["unknown_fitting_count"] or calibration["test_fitting_count"]:
        raise RuntimeError("calibration is not Known-only")
    manifest = read_csv(MANIFEST)
    ids = [row["flow_id"] for row in manifest]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or len(ids) != protocol["unknown_test"]:
        raise RuntimeError("candidate manifest ID order/cardinality failed")
    base = ROOT / UNIT / "input_caches/ustc/A-2"
    tf_cache, mfr_cache = base / "tf_fig_test", base / "yatc_mfr_test"
    for cache in (tf_cache, mfr_cache):
        audit = json.loads((cache / "cache_audit.json").read_text())
        if audit["status"] != "PASS" or audit["unknown_samples_used_for_fitting"]:
            raise RuntimeError(f"candidate cache audit failed: {cache}")
    if np.load(tf_cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist() != ids:
        raise RuntimeError("candidate TF cache IDs drift")
    if np.load(mfr_cache / "known_test_flow_ids.npy", allow_pickle=False).astype(str).tolist() != ids:
        raise RuntimeError("candidate MFR cache IDs drift")

    OUT.mkdir(parents=True, exist_ok=False)
    progress("RUNNING", "frozen_gpu_inference", unknown_test=len(ids))
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    adapters, head, scaler, hashes, _ = s40.frozen_model(RUN, CLASSES, device)
    if hashes != calibration["checkpoint_hashes"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    features = s40.extract_eval_legacy_mfr(RUN, CLASSES, ids, tf_cache, mfr_cache,
                                           "known_test_mfr.npy", device)
    h, logits = s40.fused(features, device, adapters, head, scaler)
    with np.load(SOURCE_DETECTION / "known_train.npz", allow_pickle=False) as data:
        centers, neighbor = s40.support_models(data["h"], data["labels"], CLASSES)
    raw = s40.raw_scores(h, logits, centers, neighbor)
    s40.add_des_v1(raw, calibration["median"], calibration["mad"])

    known_rows = [row for row in read_csv(source_scores_path) if row["role"] == "known_test"]
    if len(known_rows) != 2272 or {row["class_name"] for row in known_rows} != set(CLASSES):
        raise RuntimeError("Stage40 Known Test score population drift")
    unknown_rows = []
    predicted = logits.argmax(1)
    for index, row in enumerate(manifest):
        item = {"flow_id": row["flow_id"], "role": "unknown_test",
                "class_name": row["class_name"], "is_unknown": 1,
                "predicted_known": CLASSES[int(predicted[index])],
                "balanced_selected": 0}
        for name in METHODS:
            item[f"score_{name}"] = float(raw[name][index])
            item[f"reject_{name}"] = int(raw[name][index] > calibration["thresholds"][name])
        unknown_rows.append(item)
    ranked = sorted(range(len(unknown_rows)), key=lambda i: (
        hashlib.sha256(f"2022|balanced|{unknown_rows[i]['flow_id']}".encode()).digest(),
        unknown_rows[i]["flow_id"]))[:len(known_rows)]
    for index in ranked:
        unknown_rows[index]["balanced_selected"] = 1
    combined = []
    for row in known_rows:
        item = {key: row[key] for key in ("flow_id", "role", "class_name", "is_unknown", "predicted_known")}
        item["balanced_selected"] = 1
        for name in METHODS:
            item[f"score_{name}"] = float(row[f"score_{name}"])
            item[f"reject_{name}"] = int(row[f"reject_{name}"])
        combined.append(item)
    combined.extend(unknown_rows)
    combined.sort(key=lambda row: row["flow_id"])
    write_csv(OUT / "sample_scores.csv", combined)

    results = []
    for view in ("natural", "balanced_1to1"):
        selected = combined if view == "natural" else [row for row in combined if int(row["balanced_selected"])]
        truth = np.asarray([int(row["is_unknown"]) for row in selected], dtype=np.int64)
        for name in METHODS:
            score = np.asarray([float(row[f"score_{name}"]) for row in selected])
            results.append({"dataset": "CIC-IDS-2017", "unit": UNIT, "view": view,
                            "method": name, "known_test": int((truth == 0).sum()),
                            "unknown_test": int((truth == 1).sum()),
                            "unknown_prevalence": float(truth.mean()),
                            "threshold": float(calibration["thresholds"][name]),
                            **metric(truth, score, calibration["thresholds"][name])})
    write_csv(OUT / "open_set_results.csv", results)
    source_closed = json.loads((SOURCE_DETECTION / "known_test_closed_set.json").read_text())
    write_json_new(OUT / "known_test_closed_set.json", source_closed)
    hashes_after = s40.frozen_model(RUN, CLASSES, torch.device("cpu"))[3]
    if hashes_after != hashes:
        raise RuntimeError("checkpoint changed during candidate inference")
    write_json_new(OUT / "evaluation_audit.json", {
        "status": "PASS", "unit": UNIT, "known_test": len(known_rows),
        "unknown_test": len(unknown_rows), "unknown_fit_count": 0,
        "test_fit_count": 0, "threshold_source": "Stage40 Known Validation P95",
        "checkpoint_hashes_unchanged": True, "checkpoint_hashes": hashes,
        "candidate_manifest_sha256": protocol["candidate_manifest_sha256"],
        "known_calibration_sha256": protocol["known_calibration_sha256"],
        "known_test_sample_scores_sha256": protocol["known_test_sample_scores_sha256"],
        "balanced_unknown_selection": "SHA256(seed=2022, flow_id), score-blind",
    })
    (OUT / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    progress("RUNNING", "evaluation_complete", unknown_test=len(unknown_rows))
    print(json.dumps({"status": "PASS", "results": results}), flush=True)


if __name__ == "__main__":
    main()
