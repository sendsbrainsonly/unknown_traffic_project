#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import ROOT, pilot_specs, protocol_rows, read_json, sha256_file, write_json


def check(name: str, passed: bool, detail) -> dict:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def main() -> None:
    checks = []
    config = read_json(ROOT / "config.json")
    checks.append(check("config_remains_frozen_before_training", config["status"] == "FROZEN_BEFORE_TRAINING", config["status"]))
    caches = []
    for dataset in ("ustc", "iscx_vpn", "iscx_tor", "vnat"):
        audit = read_json(ROOT / "sequence_cache" / dataset / "cache_audit.json")
        caches.append(audit)
    checks.append(check("all_sequence_caches_pass", all(x["status"] == "PASS" and x["legacy_first8_exact_parity"] for x in caches), [{"dataset": x["dataset"], "status": x["status"], "flows": x["target_flows"]} for x in caches]))
    parity = read_json(ROOT / "legacy_t8_parity" / "parity_summary.json")
    checks.append(check("legacy_t8_inference_parity", parity["status"] == "PASS" and len(parity["runs"]) == 5, parity["status"]))

    run_details = []
    all_runs_pass = True
    zero_test_usage = True
    checkpoints_valid = True
    predictions_valid = True
    for window in (8, 16, 32):
        for spec in pilot_specs():
            dataset, protocol_id = spec["dataset"], spec["protocol_id"]
            directory = ROOT / "runs" / f"T{window}" / dataset / protocol_id
            result_path = directory / "result.json"
            if not result_path.exists():
                all_runs_pass = False
                run_details.append({"window": window, "dataset": dataset, "protocol_id": protocol_id, "status": "MISSING"})
                continue
            result = read_json(result_path)
            _, train, val = protocol_rows(dataset, protocol_id)
            checkpoint = Path(result["checkpoint_path"])
            checkpoint_ok = checkpoint.is_file() and sha256_file(checkpoint) == result["checkpoint_sha256"]
            checkpoints_valid &= checkpoint_ok
            prediction_path = directory / "validation_predictions.npz"
            prediction_ok = False
            if prediction_path.exists():
                with np.load(prediction_path, allow_pickle=False) as data:
                    prediction_ok = len(data["sample_ids"]) == len(val) == len(data["true_labels"]) == len(data["predicted_labels"])
            predictions_valid &= prediction_ok
            valid = result["status"] == "PASS" and result["epochs_completed"] == 100 and result.get("mask_safe") is True and result["known_train_samples"] == len(train) and result["known_validation_samples"] == len(val)
            all_runs_pass &= valid
            zero_test_usage &= result["known_test_samples_used"] == 0 and result["unknown_test_samples_used"] == 0
            run_details.append({"window": window, "dataset": dataset, "protocol_id": protocol_id, "status": result["status"], "train": len(train), "validation": len(val), "checkpoint_ok": checkpoint_ok, "prediction_ok": prediction_ok})
    checks.append(check("formal_runs_15_of_15", all_runs_pass and len(run_details) == 15, run_details))
    checks.append(check("known_test_and_unknown_test_usage_zero", zero_test_usage, {"known_test": 0, "unknown_test": 0}))
    checks.append(check("checkpoint_hashes_valid", checkpoints_valid, "15 formal best checkpoints"))
    checks.append(check("validation_prediction_counts_valid", predictions_valid, "15 formal prediction files"))

    required = [
        "window_pilot_results.csv", "window_paired_comparison.csv", "window_per_class_results.csv",
        "window_confusion_analysis.csv", "window_error_complementarity.csv",
        "window_coverage_vs_performance.csv", "stage15f1a_report.md", "RESULTS.md",
    ]
    checks.append(check("required_outputs_present", all((ROOT / name).is_file() for name in required), required))
    checks.append(check("t64_not_run", not (ROOT / "runs" / "T64").exists(), "T64 directory absent"))
    after = read_json(ROOT / "frozen_asset_hashes_after.json")
    checks.append(check("protected_assets_unchanged", after["comparison"]["status"] == "PASS", after["comparison"]))
    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    payload = {
        "status": overall,
        "stage": "Stage 15F-1A",
        "formal_runs_expected": 15,
        "formal_runs_observed": sum(1 for x in run_details if x["status"] == "PASS"),
        "known_test_samples_used": 0 if zero_test_usage else "NONZERO",
        "unknown_test_samples_used": 0 if zero_test_usage else "NONZERO",
        "checks": checks,
    }
    write_json(ROOT / "completion_verification.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if overall != "PASS":
        raise SystemExit("completion verification failed")


if __name__ == "__main__":
    main()
