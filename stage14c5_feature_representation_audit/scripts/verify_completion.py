#!/usr/bin/env python3
"""Final fail-closed verification for the complete Stage 14C.5 bundle."""

from __future__ import annotations

import json

import pandas as pd

from common import FEATURES, NEW_FEATURES, RAW_CACHE, ROOT, RUNS_ROOT, json_dump, protocol_ids, sha256_file, verify_frozen_inputs


def main() -> None:
    frozen = verify_frozen_inputs()
    cache = json.loads((RAW_CACHE / "cache_audit.json").read_text())
    input_audit = json.loads((ROOT / "input_verification.json").read_text())
    aggregation = json.loads((ROOT / "aggregation_audit.json").read_text())
    failures: list[str] = []
    results = []
    checkpoint_hashes = {}
    for feature in FEATURES:
        for protocol_id in protocol_ids():
            run_dir = RUNS_ROOT / feature / protocol_id
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                failures.append(f"missing result {feature}/{protocol_id}")
                continue
            result = json.loads(result_path.read_text())
            results.append(result)
            if result["status"] != "success" or result["unknown_samples_used"] != 0 or result["known_test_samples_used"] != 0:
                failures.append(f"boundary/status failure {feature}/{protocol_id}")
            if not (run_dir / "per_class_metrics.csv").is_file():
                failures.append(f"missing per-class {feature}/{protocol_id}")
            if not (ROOT / "confusion_matrices" / feature / f"{protocol_id}.csv").is_file():
                failures.append(f"missing confusion {feature}/{protocol_id}")
            if feature in NEW_FEATURES:
                checkpoint_path = ROOT / "checkpoints" / f"{feature}_{protocol_id}_best.pt"
                actual = sha256_file(checkpoint_path)
                if actual != result["checkpoint_sha256"]:
                    failures.append(f"checkpoint hash mismatch {feature}/{protocol_id}")
                checkpoint_hashes[f"{feature}:{protocol_id}"] = actual
    table = pd.read_csv(ROOT / "stage14c5_feature_ablation.csv")
    per_class = pd.read_csv(ROOT / "stage14c5_per_class_metrics.csv")
    expected_grid = {(feature, protocol_id) for feature in FEATURES for protocol_id in protocol_ids()}
    observed_grid = set(zip(table.feature, table.protocol_id))
    if len(table) != 60 or observed_grid != expected_grid:
        failures.append("60-run aggregate grid mismatch")
    if len(per_class) == 0:
        failures.append("empty per-class aggregate")
    if cache["status"] != "PASS" or cache["packet_count_mismatches"] != 0:
        failures.append("raw cache audit failed")
    if cache["unknown_test_rows_loaded"] != 0 or cache["known_test_rows_loaded"] != 0:
        failures.append("raw cache test boundary failure")
    if cache["stage14c_summary_sha256"] != frozen["stage14c_summary_sha256"]:
        failures.append("Stage 14C summary changed after audit start")
    if input_audit["status"] != "PASS" or input_audit["checked_feature_protocol_inputs"] != 45:
        failures.append("input verification failed")
    if aggregation["status"] != "PASS" or aggregation["run_count"] != 60:
        failures.append("aggregation audit failed")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "formal_run_results": len(results),
        "expected_formal_run_results": 60,
        "new_checkpoint_count": len(checkpoint_hashes),
        "expected_new_checkpoint_count": 45,
        "per_class_rows": len(per_class),
        "unknown_test_samples_used": int(table.unknown_samples_used.sum()),
        "known_test_samples_used": int(table.known_test_samples_used.sum()),
        "stage14b_freeze_hash": frozen["freeze_hash"],
        "stage14c_summary_unchanged": cache["stage14c_summary_sha256"] == frozen["stage14c_summary_sha256"],
        "failures": failures,
        "interrupted_attempts_excluded": ["f3_low_seed2026_gpu3_attempt1"],
        "stage14c5_feature_ablation_sha256": sha256_file(ROOT / "stage14c5_feature_ablation.csv"),
        "stage14c5_per_class_metrics_sha256": sha256_file(ROOT / "stage14c5_per_class_metrics.csv"),
        "stage14c5_report_sha256": sha256_file(ROOT / "stage14c5_report.md"),
        "checkpoint_hashes": checkpoint_hashes,
    }
    json_dump(ROOT / "final_verification.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS" or len(results) != 60 or len(checkpoint_hashes) != 45:
        raise RuntimeError("Stage 14C.5 completion verification failed")


if __name__ == "__main__":
    main()
