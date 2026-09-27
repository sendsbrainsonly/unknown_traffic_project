#!/usr/bin/env python3
"""Independently replay Stage 27 saved logits and sample-level metrics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
ROLES = ("known_train", "known_validation", "known_test")
METHODS = ("E3", "YaTC", "F1_LinearConcat", "F2_EqualProjected", "F3_FeatureGate")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_run(run: Path) -> dict:
    info = json.loads((run / "verification.json").read_text())
    assert info["status"] == "PASS"
    assert json.loads((run / "source_hashes_before.json").read_text()) == json.loads(
        (run / "source_hashes_after.json").read_text()
    )
    for method, digest in info["checkpoints"].items():
        assert sha256(run / f"{method}_best.pt") == digest
    all_rows = read_csv(run / "sample_predictions.csv")
    metric_rows = read_csv(run / "run_metrics.csv")
    gates = read_csv(run / "gate_weights.csv")
    with np.load(run / "features.npz", allow_pickle=False) as features, np.load(
        run / "logits.npz", allow_pickle=False
    ) as logits:
        assert len(all_rows) == sum(info["samples"].values()) * len(METHODS)
        assert len(gates) == sum(info["samples"].values())
        checked = 0
        for role in ROLES:
            flow_ids = features[f"{role}_flow_ids"].astype(str).tolist()
            assert len(flow_ids) == info["samples"][role] == len(set(flow_ids))
            role_gates = [r for r in gates if r["role"] == role]
            assert [r["flow_id"] for r in role_gates] == flow_ids
            for row in role_gates:
                weights = np.array([float(row[f"weight_{branch}"]) for branch in
                                    ("trafficformer", "fig", "yatc")])
                assert np.isfinite(weights).all() and (weights >= 0).all()
                assert abs(weights.sum() - 1) < 1e-5
            for method in METHODS:
                subset = [r for r in all_rows if r["role"] == role and r["method"] == method]
                assert [r["flow_id"] for r in subset] == flow_ids
                score = logits[f"{role}_{method}"]
                assert score.shape == (len(flow_ids), score.shape[1]) and np.isfinite(score).all()
                labels = sorted({r["true_service"] for r in all_rows if r["role"] == "known_train"})
                # The frozen class order is recorded in the Stage 22 config, not inferred alphabetically.
                config = json.loads((ROOT.parent / "stage22_pretrained_trafficformer_e3_closed_set_comparison" /
                                     "config.json").read_text())
                dataset = info["dataset"]
                labels = config["datasets"][dataset]["services"]
                assert score.shape[1] == len(labels)
                assert [labels[i] for i in score.argmax(axis=1)] == [r["predicted_service"] for r in subset]
                for level in ("fine", "coarse"):
                    true_key, pred_key = (("true_service", "predicted_service") if level == "fine" else
                                          ("true_coarse", "predicted_coarse"))
                    truth = [r[true_key] for r in subset]
                    pred = [r[pred_key] for r in subset]
                    names = labels if level == "fine" else sorted(set(truth) | set(pred))
                    _, _, f1, support = precision_recall_fscore_support(
                        truth, pred, labels=names, zero_division=0
                    )
                    record = [r for r in metric_rows if r["role"] == role and
                              r["method"] == method and r["level"] == level]
                    assert len(record) == 1
                    stored = record[0]
                    assert int(stored["samples"]) == len(flow_ids)
                    assert abs(float(stored["accuracy"]) - accuracy_score(truth, pred)) < 1e-12
                    assert abs(float(stored["macro_f1"]) - f1.mean()) < 1e-12
                    assert abs(float(stored["weighted_f1"]) - np.average(f1, weights=support)) < 1e-12
                    checked += 1
    return {"dataset": info["dataset"], "seed": info["seed"], "samples": info["samples"],
            "metric_rows_replayed": checked, "checkpoint_hashes": "PASS",
            "predictions_from_logits": "PASS", "gate_weights": "PASS"}


def main() -> None:
    checked = [check_run(ROOT / "runs" / dataset / f"seed{seed}")
               for dataset in ("iscx_vpn", "iscx_tor") for seed in (2022, 2023)]
    output = {"status": "PASS", "runs": checked, "total_metric_rows_replayed": sum(
        item["metric_rows_replayed"] for item in checked)}
    (ROOT / "independent_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
