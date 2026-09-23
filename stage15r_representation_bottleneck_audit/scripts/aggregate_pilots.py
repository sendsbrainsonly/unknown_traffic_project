#!/usr/bin/env python3
"""Aggregate Stage 15R pilot validation results and apply the frozen gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import CONFIG_PATH, ROOT, read_csv, read_json, write_csv, write_json


def main() -> None:
    config = read_json(CONFIG_PATH)
    expected = {(item["dataset"], item["protocol_id"]) for item in config["pilot_protocols"]}
    rows = []
    result_paths = sorted((ROOT / "pilot_runs").glob("e[0-3]/*/*/result.json"))
    for path in result_paths:
        result = read_json(path)
        rows.append({
            "experiment": result["experiment"], "dataset": result["dataset"], "protocol_id": result["protocol_id"],
            "setting": result.get("setting", ""), "status": result["status"], "known_train_samples": result["known_train_samples"],
            "known_validation_samples": result["known_validation_samples"], "validation_accuracy": result["validation_accuracy"],
            "validation_macro_f1": result["validation_macro_f1"], "validation_weighted_f1": result["validation_weighted_f1"],
            "train_loss": result.get("train_loss", ""), "validation_loss": result.get("validation_loss", ""),
            "best_epoch": result.get("best_epoch", ""), "epochs_completed": result.get("epochs_completed", ""),
            "runtime_seconds": result.get("runtime_seconds", ""), "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes", ""),
            "known_test_samples_used": result["known_test_samples_used"], "unknown_test_samples_used": result["unknown_test_samples_used"],
            "checkpoint_sha256": result.get("checkpoint_sha256", ""), "result_path": str(path.resolve()),
        })
    if not rows:
        raise RuntimeError("no pilot results")
    write_csv(ROOT / "pilot_closed_set_results.csv", rows)
    baseline = {(row["dataset"], row["protocol_id"]): row for row in rows if row["experiment"] == "E0"}
    paired = []
    for row in rows:
        if row["experiment"] == "E0":
            continue
        key = (row["dataset"], row["protocol_id"])
        native = baseline[key]
        paired.append({
            "experiment": row["experiment"], "dataset": row["dataset"], "protocol_id": row["protocol_id"],
            "delta_accuracy": float(row["validation_accuracy"]) - float(native["validation_accuracy"]),
            "delta_macro_f1": float(row["validation_macro_f1"]) - float(native["validation_macro_f1"]),
            "delta_weighted_f1": float(row["validation_weighted_f1"]) - float(native["validation_weighted_f1"]),
            "candidate_validation_macro_f1": row["validation_macro_f1"], "native_validation_macro_f1": native["validation_macro_f1"],
        })
    write_csv(ROOT / "paired_vs_native.csv", paired)
    per_class = []
    confusion_rows = []
    for result_path in result_paths:
        run = result_path.parent
        per_class.extend(read_csv(run / "per_class_results.csv"))
        result = read_json(result_path)
        names = result["class_names"]
        matrix = np.load(run / "validation_confusion_matrix.npy", allow_pickle=False)
        for true_index, true_name in enumerate(names):
            for pred_index, pred_name in enumerate(names):
                count = int(matrix[true_index, pred_index])
                if count:
                    confusion_rows.append({"experiment": result["experiment"], "dataset": result["dataset"], "protocol_id": result["protocol_id"], "true_class": true_name, "predicted_class": pred_name, "count": count, "true_class_support": int(matrix[true_index].sum()), "row_fraction": float(count / matrix[true_index].sum())})
    write_csv(ROOT / "per_class_results.csv", per_class)
    write_csv(ROOT / "confusion_analysis.csv", confusion_rows)
    gate_config = config["pilot_gate"]
    gate = {"expected_protocols": len(expected), "available_e0_protocols": len(baseline), "candidates": {}, "full15_candidates": []}
    for experiment in ("E1", "E2", "E3"):
        values = [row for row in paired if row["experiment"] == experiment]
        deltas = np.asarray([float(row["delta_macro_f1"]) for row in values], dtype=np.float64)
        complete = {(row["dataset"], row["protocol_id"]) for row in values} == expected
        summary = {
            "complete": complete, "protocols": len(values), "mean_delta_macro_f1": float(deltas.mean()) if len(deltas) else None,
            "positive_protocols": int(np.sum(deltas > 0)) if len(deltas) else 0,
            "negative_protocols": int(np.sum(deltas < 0)) if len(deltas) else 0,
            "worst_delta_macro_f1": float(deltas.min()) if len(deltas) else None,
        }
        summary["clear_gate"] = bool(complete and summary["mean_delta_macro_f1"] >= gate_config["clear_candidate_mean_delta_macro_f1"] and summary["positive_protocols"] >= gate_config["minimum_positive_protocols"] and summary["worst_delta_macro_f1"] > gate_config["maximum_allowed_worst_delta_macro_f1"])
        summary["diagnostic_signal"] = bool(complete and summary["mean_delta_macro_f1"] >= gate_config["diagnostic_candidate_mean_delta_macro_f1"] and summary["positive_protocols"] >= gate_config["diagnostic_minimum_positive_protocols"])
        gate["candidates"][experiment] = summary
        if summary["clear_gate"]:
            gate["full15_candidates"].append(experiment)
    gate["full15_status"] = "ELIGIBLE" if gate["full15_candidates"] else ("NOT_RUN_GATE_FAILED" if all(value["complete"] for value in gate["candidates"].values()) else "PENDING_PILOTS")
    write_json(ROOT / "pilot_gate.json", gate)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
