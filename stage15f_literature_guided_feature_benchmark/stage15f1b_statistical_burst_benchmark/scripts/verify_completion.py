#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import ROOT, cache_dir, pilot_specs, protocol_rows, read_csv, read_json, run_dir, sha256_file, write_json


REQUIRED = [
    "feature_parity_audit.md", "feature_definitions.json", "feature_availability.csv", "feature_missingness.csv",
    "statistical_pilot_results.csv", "statistical_group_ablation.csv", "burst_incremental_results.csv",
    "early16_vs_fullflow.csv", "per_class_results.csv", "confusion_analysis.csv",
    "native_behavior_complementarity.csv", "feature_importance.csv", "stage15f1b_report.md", "RESULTS.md",
]


def check(name: str, condition: bool, detail):
    return {"check": name, "status": "PASS" if condition else "FAIL", "detail": detail}


def main() -> None:
    checks = []
    config = read_json(ROOT / "config.json")
    checks.append(check("config_frozen_before_formal_training", config["status"] == "FROZEN_BEFORE_FORMAL_TRAINING", config["status"]))
    parity = read_json(ROOT / "s12_parity_summary.json")
    checks.append(check("s12_parity_5_of_5", parity["status"] == "PASS" and len(parity["runs"]) == 5, {"status": parity["status"], "runs": len(parity["runs"])}))
    cache_details = []
    for dataset in ("ustc", "vnat", "iscx_vpn", "iscx_tor"):
        audit = read_json(cache_dir(dataset) / "cache_audit.json")
        legacy = read_json(cache_dir(dataset) / "legacy_s12_audit.json")
        ok = audit["status"] == "PASS" and audit["reference_parity"]["status"] == "PASS" and audit["target_flows"] == audit["recovered_flows"] and legacy["status"] == "PASS" and legacy["flows"] == audit["target_flows"]
        cache_details.append({"dataset": dataset, "status": "PASS" if ok else "FAIL", "flows": audit["target_flows"]})
    checks.append(check("behavior_caches_and_legacy_s12", all(row["status"] == "PASS" for row in cache_details), cache_details))
    grid = read_json(ROOT / "grid_status.json")
    checks.append(check("formal_trained_grid_40_of_40", grid["status"] == "PASS" and len(grid["completed"]) == 40 and not grid["failed"], {"status": grid["status"], "completed": len(grid["completed"]), "failed": len(grid["failed"])}))
    run_details = []
    expected = []
    for scenario, feature_sets in config["observation_scenarios"].items():
        for feature_set in feature_sets:
            for spec in pilot_specs():
                expected.append((scenario, feature_set, spec["dataset"], spec["protocol_id"]))
    for scenario, feature_set, dataset, protocol_id in expected:
        directory = run_dir(scenario, feature_set, dataset, protocol_id)
        result = read_json(directory / "result.json")
        classes, _train, val = protocol_rows(dataset, protocol_id)
        with np.load(directory / "validation_predictions.npz", allow_pickle=False) as data:
            prediction_count = len(data["sample_ids"])
        checkpoint_ok = sha256_file(Path(result["checkpoint_path"])) == result["checkpoint_sha256"]
        ok = result["status"] == "PASS" and result["known_test_samples_used"] == 0 and result["unknown_test_samples_used"] == 0 and prediction_count == len(val) and checkpoint_ok and len(result["feature_names"]) == result["feature_count"]
        run_details.append({"scenario": scenario, "feature_set": feature_set, "dataset": dataset, "protocol_id": protocol_id, "status": "PASS" if ok else "FAIL", "validation": len(val), "features": result["feature_count"]})
    checks.append(check("formal_runs_45_of_45", len(run_details) == 45 and all(row["status"] == "PASS" for row in run_details), run_details))
    checks.append(check("known_test_and_unknown_test_usage_zero", all(read_json(run_dir(s, f, d, p) / "result.json")["known_test_samples_used"] == 0 and read_json(run_dir(s, f, d, p) / "result.json")["unknown_test_samples_used"] == 0 for s, f, d, p in expected), {"known_test": 0, "unknown_test": 0}))
    outputs = {name: (ROOT / name).is_file() for name in REQUIRED}
    checks.append(check("required_outputs_present", all(outputs.values()), outputs))
    hashes = read_json(ROOT / "frozen_asset_hashes_after.json")
    checks.append(check("protected_assets_unchanged", hashes["comparison"]["status"] == "PASS", hashes["comparison"]))
    forbidden = {
        "full15": (ROOT / "full15").exists(), "stage15f1c": (ROOT.parent / "stage15f1c").exists(),
        "open_set": (ROOT / "open_set_evaluation").exists(), "byte_behavior_model": (ROOT / "byte_behavior_model").exists(),
    }
    checks.append(check("forbidden_next_stages_not_run", not any(forbidden.values()), forbidden))
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    payload = {"stage": "Stage 15F-1B", "status": status, "formal_runs_expected": 45, "formal_runs_observed": len(run_details), "known_test_samples_used": 0, "unknown_test_samples_used": 0, "checks": checks}
    write_json(ROOT / "completion_verification.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit("Stage 15F-1B completion verification failed")


if __name__ == "__main__":
    main()
