#!/usr/bin/env python3
"""Independent validation for Stage 15F-DQ-3R outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]


def rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        "service_label_provenance.csv",
        "service_label_mapping_audit.md",
        "service_train_val_support.csv",
        "capture_group_feasibility.md",
        "coarse_open_set_semantics.md",
        "dq3r_report.md",
        "dq3f_preregistered_protocol.md",
        "completion_verification.json",
        "RESULTS.md",
        "manifest.json",
        "capture_service_mapping_audit.csv",
        "service_open_set_overlap.csv",
        "service_label_status_definitions.json",
        "dq3f_preregistered_config.json",
        "frozen_asset_hashes_before.json",
        "frozen_asset_hashes_after.json",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    assert not missing, missing

    before = json.loads((OUT / "frozen_asset_hashes_before.json").read_text(encoding="utf-8"))
    after = json.loads((OUT / "frozen_asset_hashes_after.json").read_text(encoding="utf-8"))
    assert before["file_count"] == after["file_count"] == 17
    assert before["files"] == after["files"]

    provenance = rows("service_label_provenance.csv")
    assert len(provenance) == 3065
    assert len({r["flow_id"] for r in provenance}) == 3065
    assert Counter(r["split_role"] for r in provenance) == Counter({"known_train": 2730, "known_validation": 335})
    assert set(r["service_label_status"] for r in provenance) == {"WEAK_CAPTURE_LABEL"}
    assert set(r["service_label"] for r in provenance) == {"Chat", "Email", "File-Transfer", "P2P", "Streaming", "VoIP"}
    assert all(r["flow_id"] and r["capture_id"] and r["service_label_source"] for r in provenance)

    definitions = json.loads((OUT / "service_label_status_definitions.json").read_text(encoding="utf-8"))
    assert set(definitions) == {"VERIFIED_DEFINITION", "WEAK_CAPTURE_LABEL", "AMBIGUOUS", "UNAVAILABLE"}

    captures = rows("capture_service_mapping_audit.csv")
    assert len(captures) == 31
    assert all(r["project_mapping_agreement"] == "True" for r in captures)
    app_services: dict[str, set[str]] = defaultdict(set)
    for row in captures:
        app_services[row["application_label"]].add(row["stage12_service"])
    assert app_services["Facebook"] == {"Chat", "VoIP"}
    assert app_services["Hangouts"] == {"Chat", "VoIP"}
    assert app_services["Skype"] == {"Chat", "File-Transfer", "VoIP"}

    support = {r["service"]: r for r in rows("service_train_val_support.csv")}
    assert len(support) == 6
    assert sum(int(r["train_flows"]) for r in support.values()) == 2730
    assert sum(int(r["validation_flows"]) for r in support.values()) == 335
    assert all(int(r["train_flows"]) > 0 and int(r["validation_flows"]) > 0 for r in support.values())
    assert support["P2P"]["total_captures"] == "1"
    assert support["P2P"]["capture_group_train_val_feasible"] == "False"

    semantics = rows("service_open_set_overlap.csv")
    assert {r["setting"] for r in semantics} == {"low", "medium", "high"}
    assert all(r["strict_service_unknown_reusable"] == "False" for r in semantics)
    assert all(r["service_overlap"] for r in semantics)

    config = json.loads((OUT / "dq3f_preregistered_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "PREREGISTERED_NOT_RUN"
    assert config["same_flow_ids_f1_f3"] is True
    assert config["known_test_feature_values_used"] == 0
    assert config["unknown_test_feature_values_used"] == 0
    assert config["group_generalization_claim_allowed"] is False

    completion = json.loads((OUT / "completion_verification.json").read_text(encoding="utf-8"))
    assert completion["decision"] == "SERVICE_TASK_FEASIBLE_WITH_WEAK_LABELS"
    assert completion["model_training_runs"] == 0
    assert completion["known_test_feature_values_used"] == 0
    assert completion["unknown_test_feature_values_used"] == 0
    assert completion["byte_behavior_started"] is False
    assert completion["des_h1_modified"] is False
    assert completion["prior_experiment_outputs_modified"] is False
    assert completion["frozen_asset_hash_status"] == "PASS"
    assert completion["frozen_asset_count"] == 17

    forbidden_suffixes = {".pt", ".pth", ".bin", ".ckpt", ".npy", ".npz"}
    unexpected_models = [str(p) for p in OUT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden_suffixes]
    assert not unexpected_models, unexpected_models

    report = (OUT / "dq3r_report.md").read_text(encoding="utf-8")
    for token in ["SERVICE_TASK_FEASIBLE_WITH_WEAK_LABELS", "Known Test feature values read: 0", "Unknown Test feature values read: 0"]:
        assert token in report

    print(json.dumps({
        "verification": "PASS",
        "required_files": len(required),
        "frozen_inputs": 17,
        "provenance_rows": len(provenance),
        "train_validation": [2730, 335],
        "capture_mapping_agreement": "31/31",
        "services": sorted(support),
        "group_blocker": "P2P_ONE_CAPTURE",
        "decision": completion["decision"],
        "dq3f": "PREREGISTERED_NOT_RUN",
        "test_feature_values_used": 0,
        "training_runs": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
