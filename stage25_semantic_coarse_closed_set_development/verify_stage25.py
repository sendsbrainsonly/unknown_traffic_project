#!/usr/bin/env python3
"""Independent replay of Stage25 predictions and frozen-head numeric arrays."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from run_stage25 import source_files

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "runs"


def rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replay_metric(group: list[dict], level: str, labels: list[str]) -> tuple[float, float, float]:
    true = [r[f"true_{level}"] for r in group]
    pred = [r[f"pred_{level}"] for r in group]
    _p, _r, f1, support = precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    return float(accuracy_score(true, pred)), float(np.mean(f1)), float(np.average(f1, weights=support))


def main() -> None:
    before = json.loads((OUT / "protected_hashes_before.json").read_text())
    after = json.loads((OUT / "protected_hashes_after.json").read_text())
    assert before == after and len(before) == 48
    paths = source_files()
    assert set(paths) == set(before)
    for name, expected in before.items():
        assert checksum(paths[name]) == expected, name

    remap_predictions = rows("remap_predictions.csv")
    remap_results = rows("remap_run_results.csv")
    assert len(remap_predictions) == 61964 and len(remap_results) == 224
    index = {}
    for row in remap_predictions:
        key = row["dataset"], row["seed"], row["method"], row["role"], row["flow_id"]
        assert key not in index
        index[key] = row
        assert row["true_coarse"] == ("Communication" if row["true_fine"] in ("Chat", "Email", "VoIP") else row["true_fine"])
        assert row["pred_coarse"] == ("Communication" if row["pred_fine"] in ("Chat", "Email", "VoIP") else row["pred_fine"])
        assert int(row["retained_ge2"]) == int(int(row["packet_count_capped8"]) >= 2)
    checked_metrics = 0
    for result in remap_results:
        base = (result["dataset"], result["seed"], result["method"], result["role"])
        group = [r for r in remap_predictions if (r["dataset"], r["seed"], r["method"], r["role"]) == base]
        if result["scope"] == "ge2":
            group = [r for r in group if int(r["retained_ge2"])]
        assert len(group) == int(result["samples"])
        level = result["label_level"]
        # A class absent from a conditional slice remains in the original task label set.
        all_group = [r for r in remap_predictions if (r["dataset"], r["seed"], r["method"], r["role"]) == base]
        labels = sorted({r[f"true_{level}"] for r in all_group})
        got = replay_metric(group, level, labels)
        for key, value in zip(("accuracy", "macro_f1", "weighted_f1"), got):
            assert abs(value - float(result[key])) < 1e-12, (base, result["scope"], level, key)
        checked_metrics += 1

    head_predictions = rows("coarse_head_predictions.csv")
    head_results = rows("coarse_head_results.csv")
    assert len(head_results) == 32
    head_index = {(r["dataset"], r["seed"], r["encoder"], r["role"], r["flow_id"]): r
                  for r in head_predictions}
    assert len(head_index) == len(head_predictions)
    replayed_heads = 0
    checked_head_decisions = 0
    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            with np.load(STAGE22 / dataset / f"seed{seed}" / "representations.npz", allow_pickle=False) as rep:
                for encoder in ("E1", "E3"):
                    head_path = OUT / "heads" / dataset / f"seed{seed}" / encoder / "coarse_head.npz"
                    with np.load(head_path, allow_pickle=False) as head:
                        coef, bias = head["coef"], head["intercept"]
                        classes, mean, scale = head["classes"], head["mean"], head["scale"]
                        for role in ("known_validation", "known_test"):
                            ids = rep[f"{role}_flow_ids"].tolist()
                            x = rep[f"{role}_{encoder.lower()}_z"].astype(np.float64)
                            predicted = classes[np.argmax(((x-mean)/scale) @ coef.T + bias, axis=1)]
                            for flow_id, label in zip(ids, predicted.tolist(), strict=True):
                                assert head_index[(dataset, str(seed), encoder, role, flow_id)]["pred_coarse"] == label
                                checked_head_decisions += 1
                    replayed_heads += 1
    result = {"status": "PASS", "protected_hashes": len(before), "remap_metric_rows_replayed": checked_metrics,
              "remap_predictions_checked": len(remap_predictions), "coarse_heads_replayed": replayed_heads,
              "coarse_head_decisions_checked": checked_head_decisions}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
