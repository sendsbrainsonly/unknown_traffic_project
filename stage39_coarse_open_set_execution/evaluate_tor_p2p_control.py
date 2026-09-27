"""Replay the prespecified Tor P2P-only service task from frozen Stage38 scores."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from freeze_protocols import PROJECT, ROOT, sha, write_csv, write_json

METHODS = (("C0", "msp"), ("C1", "energy"), ("D0", "centroid"), ("D1", "des_v1"))


def main():
    unit = ROOT / "settings/iscx_tor/p2p_only"
    frozen = json.loads((unit / "protocol.json").read_text())
    if frozen["status"] != "FROZEN_PRETRAIN" or not frozen["checkpoint_reuse"]:
        raise RuntimeError("P2P-only source protocol is not frozen")
    role_path = unit / "role_manifest.csv"
    if sha(role_path) != frozen["role_manifest_sha256"]:
        raise RuntimeError("P2P-only role manifest changed")
    roles = {r["sample_id"]: r for r in csv.DictReader(role_path.open())}
    source = PROJECT / "stage38_three_view_des_open_set_transfer/runs/iscx_tor/medium_seed2022/detection"
    verification = json.loads((source / "completion_verification.json").read_text())
    calibration = json.loads((source / "calibration.json").read_text())
    if verification["status"] != "PASS" or calibration["status"] != "PASS":
        raise RuntimeError("Stage38 Tor scores or calibration not verified")
    source_path = source / "sample_scores.csv"
    with source_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    wanted = [r for r in rows if roles[r["flow_id"]]["role"] in
              ("known_test", "unknown_eval", "auxiliary_known_service_unseen_app")]
    counts = Counter(roles[r["flow_id"]]["role"] for r in wanted)
    if dict(counts) != {"known_test": 1113, "unknown_eval": 2000,
                        "auxiliary_known_service_unseen_app": 2000}:
        raise RuntimeError(f"P2P-only Test roles mismatch: {counts}")
    outputs, decisions = [], []
    for score_id, name in METHODS:
        selected = [r for r in wanted if roles[r["flow_id"]]["role"] !=
                    "auxiliary_known_service_unseen_app"]
        truth = np.asarray([int(roles[r["flow_id"]]["role"] == "unknown_eval") for r in selected])
        values = np.asarray([float(r["score_" + name]) for r in selected])
        threshold = float(calibration["thresholds"][name])
        label = np.asarray([int(r["prediction_" + name]) for r in selected])
        if not np.array_equal(label, (values > threshold).astype(int)):
            raise RuntimeError(f"frozen threshold replay mismatch: {name}")
        known = truth == 0
        unknown = truth == 1
        auxiliary = [r for r in wanted if roles[r["flow_id"]]["role"] ==
                     "auxiliary_known_service_unseen_app"]
        accepted = [int(r["prediction_" + name]) == 0 for r in auxiliary]
        correct = [a and r["predicted_known_class"] == roles[r["flow_id"]]["service"]
                   for a, r in zip(accepted, auxiliary)]
        outputs.append({"dataset": "iscx_tor", "heldout_service": "P2P",
            "score_id": score_id, "method": name, "auroc": roc_auc_score(truth, values),
            "auprc": average_precision_score(truth, values),
            "unknown_prevalence": float(unknown.mean()),
            "ufar": float((label[unknown] == 0).mean()),
            "known_frr": float(label[known].mean()),
            "known_test": int(known.sum()), "unknown_eval": int(unknown.sum()),
            "auxiliary_known_service_unseen_app": len(auxiliary),
            "auxiliary_acceptance": float(np.mean(accepted)),
            "auxiliary_correct_service_and_accepted": float(np.mean(correct)),
            "threshold": threshold, "checkpoint_reused": True,
            "calibration_reused": True})
        for row in wanted:
            role = roles[row["flow_id"]]["role"]
            decisions.append({"sample_id": row["flow_id"], "role": role,
                "application": row["application"], "service": roles[row["flow_id"]]["service"],
                "score_id": score_id, "raw_score": row["score_" + name],
                "threshold": threshold, "predicted_unknown": row["prediction_" + name],
                "predicted_known_class": row["predicted_known_class"]})
    write_csv(unit / "results.csv", outputs)
    write_csv(unit / "sample_decisions.csv", decisions)
    write_json(unit / "evaluation_verification.json", {"status": "PASS", "source_scores_sha256": sha(source_path),
        "source_calibration_sha256": sha(source / "calibration.json"),
        "role_manifest_sha256": sha(role_path), "models_trained": 0,
        "new_thresholds_fitted": 0, "unknown_for_fitting": 0,
        "test_for_selection": 0, "score_rows": len(outputs),
        "sample_decisions": len(decisions)})
    print(json.dumps({"status": "PASS", "control": "Tor P2P-only",
                      "scores": len(outputs), "sample_decisions": len(decisions)}), flush=True)


if __name__ == "__main__":
    main()
