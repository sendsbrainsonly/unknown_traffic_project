#!/usr/bin/env python3
"""Independent completion verifier for DQ-4."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dq4_common import OUT, SEEDS, hash_protected_inputs, read_json, sha256_file, write_json


REQUIRED = [
    "dq4_source_parity.md", "dq4_known_membership_audit.csv", "dq4_subset_manifest.csv",
    "dq4_subset_counts.csv", "dq4_subset_class_capture_support.csv",
    "dq4_trafficformer_eligibility_audit.md", "dq4_cate_matching_audit.md",
    "dq4_feasibility_gate.md", "dq4_evaluation_population_effect.csv",
    "dq4_removed_flow_error_analysis.csv", "dq4_training_configs.json", "dq4_training_results.csv",
    "dq4_within_population_results.csv", "dq4_matched_evaluation_results.csv",
    "dq4_full_validation_stress_test.csv", "dq4_per_service_metrics.csv",
    "dq4_capture_level_results.csv", "dq4_paired_error_analysis.csv",
    "dq4_selection_overlap_analysis.csv", "dq4_performance_gap_attribution.md", "RESULTS.md",
    "protected_asset_hashes_before.json", "protected_asset_hashes_after.json", "aggregate_summary.json",
]


def main() -> None:
    checks = {}
    checks["required_outputs"] = all((OUT / name).is_file() and (OUT / name).stat().st_size > 0 for name in REQUIRED)
    manifest = pd.read_csv(OUT / "dq4_subset_manifest.csv")
    checks["mother_3065_unique"] = len(manifest) == 3065 and manifest["flow_id"].nunique() == 3065
    checks["mother_split_2730_335"] = manifest["split_role"].value_counts().to_dict() == {"known_train": 2730, "known_validation": 335}
    expected = {"A": 3065, "B": 2101, "C_PARENT": 1551, "C_FINAL": 1551, "D": 1440}
    checks["subset_counts"] = {name: int(manifest[name].sum()) for name in expected} == expected
    checks["c_parent_final_identical"] = manifest["C_PARENT"].equals(manifest["C_FINAL"])
    checks["all_six_services_each_split"] = all(
        manifest[(manifest[subset] == 1) & (manifest["split_role"] == role)]["service"].nunique() == 6
        for subset in ("A", "B", "C_FINAL", "D") for role in ("known_train", "known_validation")
    )
    successes = []
    hashes = []
    for model in ("M-B", "M-C", "M-D"):
        for seed in SEEDS:
            run = OUT / "runs" / model / f"seed{seed}"
            successes.append((run / "SUCCESS").is_file())
            if (run / "result.json").is_file() and (run / "model_best.pt").is_file():
                result = read_json(run / "result.json")
                hashes.append(sha256_file(run / "model_best.pt") == result["checkpoint_sha256"])
            else:
                hashes.append(False)
    checks["nine_new_runs_success"] = all(successes) and len(successes) == 9
    checks["nine_checkpoint_hashes"] = all(hashes) and len(hashes) == 9
    training = pd.read_csv(OUT / "dq4_training_results.csv")
    checks["twelve_model_seed_rows"] = len(training) == 12 and set(training["model"]) == {"M-A", "M-B", "M-C", "M-D"}
    checks["test_usage_zero"] = int(training["known_test_feature_values_used"].sum()) == 0 and int(training["unknown_test_feature_values_used"].sum()) == 0
    before = read_json(OUT / "protected_asset_hashes_before.json")
    after = read_json(OUT / "protected_asset_hashes_after.json")
    live = hash_protected_inputs()
    checks["protected_hashes_unchanged"] = before == after == live
    summary = read_json(OUT / "aggregate_summary.json")
    checks["persistent_weak_label_limit"] = "WEAK_CAPTURE_LABEL" in summary["persistent_limitations"]
    checks["persistent_capture_limit"] = "CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE" in summary["persistent_limitations"]
    checks["no_future_stage_outputs"] = not any(OUT.glob("*dq5*")) and not any(OUT.glob("*open_set*"))
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status, "checks": checks, "new_training_runs": 9, "reused_m_a_runs": 3,
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
        "dq5_to_dq7": "NOT_RUN", "trafficformer_training": "NOT_RUN",
        "tfe_gnn_training": "NOT_RUN", "byte_behavior": "NOT_RUN", "open_set": "NOT_RUN",
        "des_h1_modified": False,
    }
    write_json(OUT / "completion_verification.json", payload)
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
