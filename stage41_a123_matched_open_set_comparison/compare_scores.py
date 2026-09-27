#!/usr/bin/env python3
"""Replay same-flow Open-Detect versus frozen three-view DES-v1 test scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from freeze_matched_protocol import PROJECT, ROOT, digest


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    decision = score > threshold
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "accuracy": float(accuracy_score(y, decision)),
            "unknown_precision": float(precision_score(y, decision, zero_division=0)),
            "unknown_recall": float(recall_score(y, decision, zero_division=0)),
            "unknown_f1": float(f1_score(y, decision, zero_division=0)),
            "ufar": float((~decision[y == 1]).mean()),
            "known_frr": float(decision[y == 0].mean())}


def oracle(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    # sklearn's ROC point uses score >= threshold. The common comparison
    # decision below is score > threshold, so move one representable float
    # lower to reproduce that exact ROC point without changing the P95 rule.
    return float(np.nextafter(thresholds[int(np.argmax(tpr - fpr))], -np.inf))


def main() -> None:
    if (ROOT / "completion_verification.json").exists():
        raise FileExistsError("Stage41 comparison already completed")
    protocol_path = ROOT / "matched_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "PASS" or digest(Path(protocol["source_manifest"])) != protocol["source_sha256"]:
        raise RuntimeError("common source protocol hash drift")
    balanced_path = ROOT / "balanced_test_ids.csv"
    if digest(balanced_path) != "e1e7a982328abf87fde192d5f395ff5bf1b913ddc29ea3146a4fbabec4db100e":
        raise RuntimeError("pre-frozen balanced subset changed")
    balanced = rows(balanced_path)
    run_rows, delta_rows, class_rows = [], [], []
    baseline_hashes = {}
    for setting in ("A-1", "A-2", "A-3"):
        unit = protocol["units"][setting]
        role_path = Path(unit["role_manifest"])
        if digest(role_path) != unit["role_manifest_sha256"]:
            raise RuntimeError(f"{setting}: role manifest drift")
        role = {r["flow_id"]: r for r in rows(role_path) if r["role"] in ("known_test", "unknown_test")}
        od_path = ROOT / setting / "od_sample_scores.csv"
        od = {r["flow_id"]: r for r in rows(od_path)}
        own_base = (PROJECT / "stage40_ustc_cic_open_set/ustc_a2/detection"
                    if setting == "A-2" else ROOT / setting / "detection")
        own_path = own_base / "sample_scores.csv"
        own = {r["flow_id"]: r for r in rows(own_path)}
        if set(od) != set(own) or set(od) != set(role):
            raise RuntimeError(f"{setting}: sample-level method flow IDs differ")
        if any(od[uid]["class_name"] != role[uid]["class_name"] or
               own[uid]["class_name"] != role[uid]["class_name"] or
               od[uid]["role"] != role[uid]["role"] or
               own[uid]["role"] != role[uid]["role"] for uid in role):
            raise RuntimeError(f"{setting}: method label/role mismatch")
        od_val = rows(ROOT / setting / "od_validation_scores.csv")
        if {r["flow_id"] for r in od_val} != {
                r["flow_id"] for r in rows(role_path) if r["role"] == "known_validation"}:
            raise RuntimeError(f"{setting}: Known Validation score IDs differ")
        od_threshold = float(np.percentile(
            [float(r["score_od_native"]) for r in od_val], 95, method="higher"))
        own_cal = json.loads((own_base / "calibration.json").read_text())
        calibration_role_hash = (unit["role_manifest_sha256"] if setting != "A-2" else
            json.loads((PROJECT / "stage40_ustc_cic_open_set/protocols.json").read_text())
            ["units"][0]["role_manifest_sha256"])
        if own_cal["role_manifest_sha256"] != calibration_role_hash:
            raise RuntimeError(f"{setting}: three-view calibration role hash differs")
        own_threshold = float(own_cal["thresholds"]["des_v1"])
        methods = {"Open-Detect Native": (od, "score_od_native", od_threshold),
                   "Three-view DES-v1": (own, "score_des_v1", own_threshold)}
        balanced_ids = {r["flow_id"] for r in balanced if r["setting"] == setting}
        if len(balanced_ids) != 2 * unit["role_counts"]["unknown_test"]:
            raise RuntimeError(f"{setting}: malformed balanced subset")
        measured = {}
        for population, subset in (("natural", sorted(role)),
                                   ("balanced_1to1", sorted(balanced_ids))):
            y = np.asarray([int(role[uid]["role"] == "unknown_test") for uid in subset])
            if set(y.tolist()) != {0, 1}:
                raise RuntimeError(f"{setting}: empty test group")
            for method, (table, column, threshold) in methods.items():
                s = np.asarray([float(table[uid][column]) for uid in subset], dtype=np.float64)
                if not np.isfinite(s).all():
                    raise RuntimeError(f"{setting}/{method}: nonfinite score")
                fixed = metrics(y, s, threshold)
                chosen = oracle(y, s)
                tuned = metrics(y, s, chosen)
                common = {"scenario": setting, "population": population, "method": method,
                          "known_test": int((y == 0).sum()), "unknown_test": int((y == 1).sum()),
                          "threshold_p95_known_val": threshold,
                          "threshold_test_oracle_youden": chosen,
                          "auroc": fixed["auroc"], "auprc": fixed["auprc"]}
                run_rows.append({**common,
                    **{f"p95_{k}": fixed[k] for k in ("accuracy", "unknown_precision", "unknown_recall",
                                                         "unknown_f1", "ufar", "known_frr")},
                    **{f"oracle_{k}": tuned[k] for k in ("accuracy", "unknown_precision", "unknown_recall",
                                                           "unknown_f1", "ufar", "known_frr")}})
                measured[population, method] = {**fixed, "oracle": tuned}
        for population in ("natural", "balanced_1to1"):
            od_m = measured[population, "Open-Detect Native"]
            own_m = measured[population, "Three-view DES-v1"]
            delta_rows.append({"scenario": setting, "population": population,
                               **{f"delta_{key}": own_m[key] - od_m[key] for key in
                                  ("auroc", "auprc", "accuracy", "unknown_f1", "ufar", "known_frr")},
                               "delta_oracle_accuracy": own_m["oracle"]["accuracy"] - od_m["oracle"]["accuracy"],
                               "delta_oracle_unknown_f1": own_m["oracle"]["unknown_f1"] - od_m["oracle"]["unknown_f1"]})
        known_ids = sorted(uid for uid in role if role[uid]["role"] == "known_test")
        for unknown_name in unit["unknown_classes"]:
            subset = known_ids + sorted(uid for uid in role if role[uid]["class_name"] == unknown_name)
            y = np.asarray([int(role[uid]["role"] == "unknown_test") for uid in subset])
            for method, (table, column, threshold) in methods.items():
                s = np.asarray([float(table[uid][column]) for uid in subset])
                result = metrics(y, s, threshold)
                class_rows.append({"scenario": setting, "unknown_class": unknown_name,
                                   "method": method, "known_test": len(known_ids),
                                   "unknown_test": int(y.sum()), "auroc": result["auroc"],
                                   "auprc": result["auprc"], "ufar_p95": result["ufar"]})
        od_run = "od_run_2gpu" if setting == "A-1" else "od_run"
        baseline_hashes[setting] = {"od_scores": digest(od_path), "threeview_scores": digest(own_path),
                                    "od_checkpoint": digest(ROOT / setting / od_run / "model_best.pt")}
    write_csv(ROOT / "comparison_run_results.csv", run_rows)
    write_csv(ROOT / "comparison_paired.csv", delta_rows)
    write_csv(ROOT / "per_unknown_class.csv", class_rows)
    report = ["# Stage41 matched-flow Open-Detect vs three-view DES-v1", "",
              "Single seed 2022, same USTC flow IDs and source Train/Val/Test partitions for both methods.",
              "The 1:1 Test subset was frozen by flow ID before score inspection. Primary operating point",
              "is Known Validation P95; Test-Youden is explicitly retrospective/oracle, not deployable.", "",
              "| Scenario | Known/Unknown classes | Balanced Test Known/Unknown | OD AUROC | Ours AUROC | ΔAUROC | OD F1@P95 | Ours F1@P95 | ΔF1 | OD UFAR | Ours UFAR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for setting in ("A-1", "A-2", "A-3"):
        od_r = next(r for r in run_rows if r["scenario"] == setting and
                    r["population"] == "balanced_1to1" and r["method"] == "Open-Detect Native")
        our_r = next(r for r in run_rows if r["scenario"] == setting and
                     r["population"] == "balanced_1to1" and r["method"] == "Three-view DES-v1")
        unit = protocol["units"][setting]
        report.append(f"| {setting} | {len(unit['known_classes'])}/{len(unit['unknown_classes'])} | "
                      f"{od_r['known_test']}/{od_r['unknown_test']} | {od_r['auroc']:.4f} | "
                      f"{our_r['auroc']:.4f} | {our_r['auroc']-od_r['auroc']:+.4f} | "
                      f"{od_r['p95_unknown_f1']:.4f} | {our_r['p95_unknown_f1']:.4f} | "
                      f"{our_r['p95_unknown_f1']-od_r['p95_unknown_f1']:+.4f} | "
                      f"{od_r['p95_ufar']:.4f} | {our_r['p95_ufar']:.4f} |")
    report.extend(["", "The full natural Test prevalence and retrospective oracle-threshold",
                   "metrics are in `comparison_run_results.csv`. This local 10% flow subset is",
                   "not the authors' exact five-fold dataset/protocol. Our method includes",
                   "external pretrained TrafficFormer/YaTC branches; OD is trained from scratch.",
                   "A-1 has only 85 Unknown Test flows, so its single-seed estimate is fragile."])
    (ROOT / "comparison_report.md").write_text("\n".join(report) + "\n")
    verification = {"status": "PASS", "scenarios": 3, "methods": 2,
                    "same_flow_test_ids": True, "unknown_train_or_validation_usage": 0,
                    "threshold_source": "Known Validation P95; oracle only retrospective",
                    "balanced_test_ids_sha256": digest(balanced_path),
                    "matched_protocol_sha256": digest(protocol_path), "score_hashes": baseline_hashes}
    (ROOT / "completion_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "balanced": [r for r in run_rows
                      if r["population"] == "balanced_1to1"]}), flush=True)


if __name__ == "__main__":
    main()
