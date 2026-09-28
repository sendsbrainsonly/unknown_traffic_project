#!/usr/bin/env python3
"""Independent replay for the first Stage42-S candidate."""
from __future__ import annotations

import json
import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from stage42s_common import MANIFEST, PROTOCOL, ROOT, UNIT, digest, progress, read_csv, replace_json

METHODS = ("msp", "energy", "centroid", "des_v1")


def close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("candidate manifest changed")
    out = ROOT / UNIT / "detection"
    audit = json.loads((out / "evaluation_audit.json").read_text())
    if audit["status"] != "PASS" or audit["unknown_fit_count"] or audit["test_fit_count"]:
        raise RuntimeError("evaluation audit failed")
    saved = read_csv(out / "sample_scores.csv")
    manifest_ids = {row["flow_id"] for row in read_csv(MANIFEST)}
    saved_unknown = {row["flow_id"] for row in saved if row["role"] == "unknown_test"}
    if saved_unknown != manifest_ids or len(saved) != 2272 + len(manifest_ids):
        raise RuntimeError("saved sample population mismatch")
    reported = {(row["view"], row["method"]): row for row in read_csv(out / "open_set_results.csv")}
    for view in ("natural", "balanced_1to1"):
        selected = saved if view == "natural" else [row for row in saved if int(row["balanced_selected"])]
        truth = np.asarray([int(row["is_unknown"]) for row in selected])
        for method in METHODS:
            score = np.asarray([float(row[f"score_{method}"]) for row in selected])
            threshold = float(reported[view, method]["threshold"])
            decision = score > threshold
            actual = {
                "auroc": roc_auc_score(truth, score),
                "auprc": average_precision_score(truth, score),
                "ufar": np.mean(~decision[truth == 1]),
                "known_frr": np.mean(decision[truth == 0]),
                "known_acceptance": np.mean(~decision[truth == 0]),
                "binary_accuracy": accuracy_score(truth, decision),
                "unknown_precision": precision_score(truth, decision, zero_division=0),
                "unknown_recall": recall_score(truth, decision, zero_division=0),
                "binary_f1": f1_score(truth, decision, zero_division=0),
            }
            if any(not close(value, reported[view, method][name]) for name, value in actual.items()):
                raise RuntimeError(f"metric replay failed: {view}/{method}")
            if any(int(row[f"reject_{method}"]) != int(flag)
                   for row, flag in zip(selected, decision, strict=True)):
                raise RuntimeError(f"decision replay failed: {view}/{method}")
    payload = {"status": "PASS", "unit": UNIT, "sample_rows": len(saved),
               "known_test": 2272, "unknown_test": len(manifest_ids),
               "views": ["natural", "balanced_1to1"], "methods": list(METHODS),
               "candidate_manifest_unchanged": True, "metrics_replayed": 8,
               "unknown_fit_count": 0, "test_fit_count": 0}
    replace_json(out / "independent_verification.json", payload)
    progress("COMPLETE", "independent_verification_pass", verification=payload)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
