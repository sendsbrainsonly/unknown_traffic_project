#!/usr/bin/env python3
"""Independent completion checks for the Stage 15F-DQ interim bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        "cross_project_pcap_manifest.csv", "cross_project_flow_manifest.csv", "flow_matching_audit.md",
        "label_mapping.json", "sample_eligibility_summary.csv", "cate_matching_analysis.csv",
        "short_flow_error_analysis.csv", "fine_to_coarse_error_analysis.csv", "granularity_feasibility.md",
        "controlled_protocol_matrix.csv", "paper_protocol_reproduction.csv", "fair_matched_protocol_results.csv",
        "filter_effect_analysis.csv", "domain_mixing_analysis.csv", "split_sensitivity_analysis.csv",
        "performance_gap_attribution.md", "RESULTS.md", "completion_verification.json", "manifest.json",
        "frozen_asset_hashes_before.json", "frozen_asset_hashes_after.json",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    assert not missing, f"missing required files: {missing}"

    before = json.loads((OUT / "frozen_asset_hashes_before.json").read_text(encoding="utf-8"))
    after = json.loads((OUT / "frozen_asset_hashes_after.json").read_text(encoding="utf-8"))
    assert before["file_count"] == after["file_count"] == 208
    assert before["files"] == after["files"], "frozen input hashes changed"

    pcaps = csv_rows("cross_project_pcap_manifest.csv")
    shared = [r for r in pcaps if r["shared_all_three"].lower() == "true"]
    assert len(shared) == 31
    assert all(len(r["pcap_sha256"]) == 64 for r in shared)

    flow_audit = csv_rows("flow_reconstruction_audit.csv")
    assert len(flow_audit) == 31
    assert all(r["trafficformer_exact_multiset_parity"].lower() == "true" for r in flow_audit)

    flows = csv_rows("cross_project_flow_manifest.csv")
    native_a = [r for r in flows if r["native_pool_membership"].lower() == "true"]
    assert len(flows) == 23645
    assert len(native_a) == 4224
    assert Counter(r["cate_match_status"] for r in native_a) == Counter({"MATCHED": 2720, "UNMATCHED": 1504})

    matrix = {r["stage"]: r for r in csv_rows("controlled_protocol_matrix.csv")}
    assert matrix["DQ-0"]["status"] == "PASS"
    assert matrix["DQ-3"]["status"] == "BLOCKED_MAPPING_AMBIGUITY"
    assert all(matrix[f"DQ-{i}"]["status"] == "NOT_RUN" for i in range(4, 8))
    assert all(int(r["test_values_used"]) == 0 for r in matrix.values())

    for name in ["filter_effect_analysis.csv", "domain_mixing_analysis.csv", "split_sensitivity_analysis.csv"]:
        rows = csv_rows(name)
        assert len(rows) == 1 and rows[0]["status"] == "NOT_RUN"
        assert rows[0]["known_test_samples_used"] == "0"
        assert rows[0]["unknown_test_samples_used"] == "0"

    fair = {r["experiment"]: r for r in csv_rows("fair_matched_protocol_results.csv")}
    assert fair["F3"]["status"] == "NOT_RUN_GATE_FAILED"
    assert fair["DQ-7"]["status"] == "NOT_RUN"

    completion = json.loads((OUT / "completion_verification.json").read_text(encoding="utf-8"))
    assert completion["frozen_hash_status"] == "PASS"
    assert completion["known_test_samples_used"] == 0
    assert completion["unknown_test_samples_used"] == 0
    assert not completion["new_model_training"]
    assert all(completion["required_files_present"].values())

    report = (OUT / "performance_gap_attribution.md").read_text(encoding="utf-8")
    for token in ["STOP_RELATED_BRANCH_MAPPING_AMBIGUITY", "Known Test values used: 0", "Unknown Test values used: 0", "DQ-4--DQ-7"]:
        assert token in report, f"report missing token: {token}"

    print(json.dumps({
        "verification": "PASS",
        "required_files": len(required),
        "frozen_assets": before["file_count"],
        "shared_pcaps": len(shared),
        "flow_manifest_rows": len(flows),
        "native_a_rows": len(native_a),
        "dq3": matrix["DQ-3"]["status"],
        "dq4_to_dq7": "NOT_RUN",
        "known_test_used": 0,
        "unknown_test_used": 0,
    }, indent=2))


if __name__ == "__main__":
    from collections import Counter
    main()
