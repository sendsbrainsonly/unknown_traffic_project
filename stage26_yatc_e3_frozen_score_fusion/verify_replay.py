#!/usr/bin/env python3
"""Independently replay Stage26 decisions from saved frozen logits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
STAGE22_CONFIG = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "config.json"


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z -= z.max(1, keepdims=True)
    a = np.exp(z)
    return a / a.sum(1, keepdims=True)


def score(rows: list[dict], level: str, method: str, labels: list[str]) -> tuple[float, float, float]:
    true = [r[f"true_{level}"] for r in rows]
    pred = [r[f"pred_{method}_{level}"] for r in rows]
    _p, _r, f1, support = precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    return float(accuracy_score(true, pred)), float(np.mean(f1)), float(np.average(f1, weights=support))


def main() -> None:
    config = json.loads(STAGE22_CONFIG.read_text())
    decision_count = metric_count = source_count = 0
    for dataset in ("iscx_vpn", "iscx_tor"):
        services = config["datasets"][dataset]["services"]
        coarse_labels = sorted({"Communication" if x in ("Chat", "Email", "VoIP") else x for x in services})
        coarse_idx = {label: [i for i, name in enumerate(services)
                              if ("Communication" if name in ("Chat", "Email", "VoIP") else name) == label]
                      for label in coarse_labels}
        for seed in (2022, 2023):
            run = OUT / "runs" / dataset / f"seed{seed}"
            assert (run / "SUCCESS").is_file()
            before = json.loads((run / "source_hashes_before.json").read_text())
            after = json.loads((run / "source_hashes_after.json").read_text())
            assert before == after
            e3 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "runs" / dataset / f"seed{seed}" / "E3_model_best.pt"
            yatc = PROJECT / "stage23_closed_set_method_table" / "runs" / "yatc_stage20" / dataset / f"seed{seed}_formal" / "model_best.pt"
            assert sha(e3) == before["e3_checkpoint"] and sha(yatc) == before["yatc_checkpoint"]
            source_count += len(before)
            samples = read(run / "sample_predictions.csv")
            recorded_metrics = read(run / "run_metrics.csv")
            with np.load(run / "frozen_logits.npz", allow_pickle=False) as z:
                for role in ("known_validation", "known_test"):
                    ids = z[f"{role}_flow_ids"].tolist()
                    by_id = {r["flow_id"]: r for r in samples if r["role"] == role}
                    assert len(ids) == len(by_id) and set(ids) == set(by_id)
                    p_e3, p_yatc = softmax(z[f"{role}_e3_logits"]), softmax(z[f"{role}_yatc_logits"])
                    p_fusion = (p_e3+p_yatc)/2
                    for method, p in (("E3", p_e3), ("YaTC", p_yatc), ("Fusion", p_fusion)):
                        fine = [services[i] for i in p.argmax(1)]
                        cp = np.stack([p[:, coarse_idx[label]].sum(1) for label in coarse_labels], axis=1)
                        coarse = [coarse_labels[i] for i in cp.argmax(1)]
                        for flow_id, f, c in zip(ids, fine, coarse, strict=True):
                            row = by_id[flow_id]
                            assert row[f"pred_{method}_fine"] == f and row[f"pred_{method}_coarse"] == c
                            decision_count += 1
                    for scope in ("all", "ge2"):
                        subset = [by_id[i] for i in ids if scope == "all" or int(by_id[i]["retained_ge2"])]
                        for level, labels in (("fine", services), ("coarse", coarse_labels)):
                            for method in ("E3", "YaTC", "Fusion"):
                                rec = [r for r in recorded_metrics if r["role"] == role and r["scope"] == scope
                                       and r["level"] == level and r["method"] == method]
                                assert len(rec) == 1 and int(rec[0]["samples"]) == len(subset)
                                got = score(subset, level, method, labels)
                                for key, val in zip(("accuracy", "macro_f1", "weighted_f1"), got):
                                    assert abs(val-float(rec[0][key])) < 1e-12, (dataset, seed, role, scope, level, method, key)
                                metric_count += 1
    result = {"status": "PASS", "source_hash_entries_checked": source_count,
              "probability_decisions_replayed": decision_count, "metric_rows_replayed": metric_count,
              "fusion_weight": [0.5, 0.5], "unknown_usage": 0, "weight_updates": 0}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
