#!/usr/bin/env python3
"""Independent structural and numerical checks for Stage 14A.5 outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DECISIONS = {
    "eligible": {"sftp", "skype"},
    "borderline": {"rdp", "ssh", "youtube", "zoiper"},
    "excluded": {"netflix", "rsync", "scp", "vimeo"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = (
        ROOT / "vnat_group_statistics.csv",
        ROOT / "vnat_split_feasibility.csv",
        ROOT / "stage14a5_split_audit.md",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)

    groups = read_csv(ROOT / "vnat_group_statistics.csv")
    splits = read_csv(ROOT / "vnat_split_feasibility.csv")
    quarantined = read_csv(ROOT / "outputs" / "quarantined_duplicate_flows.csv")
    details = json.loads((ROOT / "outputs" / "audit_details.json").read_text(encoding="utf-8"))
    hashes = json.loads((ROOT / "outputs" / "input_hashes.json").read_text(encoding="utf-8"))

    assert len(groups) == 91
    assert len(splits) == 10
    assert len(quarantined) == 4
    assert {row["application"] for row in quarantined} == {"rsync", "sftp"}
    assert sum(int(row["quarantined_flow_count"]) for row in groups) == 4
    assert sum(int(row["flow_count"]) for row in groups) == 23450
    assert sum(int(row["raw_flow_count"]) for row in groups) == 23454

    group_ids_by_application: dict[str, set[str]] = {}
    for row in groups:
        if row["usable_for_split"] == "True":
            group_ids_by_application.setdefault(row["application"], set()).add(row["group_id"])
    for row in splits:
        observed = set()
        for split in ("train", "val", "test"):
            ids = {value for value in row[f"{split}_group_ids"].split(";") if value}
            assert observed.isdisjoint(ids), (row["application"], split)
            observed.update(ids)
        if int(row["usable_group_count"]) >= 3:
            assert observed == group_ids_by_application[row["application"]]
            assert int(row["train_flows"]) + int(row["val_flows"]) + int(row["test_flows"]) == int(row["flow_count_after_quarantine"])

    for decision, expected in EXPECTED_DECISIONS.items():
        actual = {row["application"] for row in splits if row["decision"] == decision}
        assert actual == expected, (decision, actual)
    by_name = {row["application"]: row for row in splits}
    assert int(by_name["rdp"]["train_flows"]) == 16
    assert by_name["rdp"]["train_k10_satisfied"] == "True"
    assert int(by_name["scp"]["min_split_flows"]) == 1
    assert int(by_name["zoiper"]["usable_group_count"]) == 3
    assert float(by_name["ssh"]["max_group_share"]) > 0.83
    retained = EXPECTED_DECISIONS["eligible"] | EXPECTED_DECISIONS["borderline"]
    assert sum(int(by_name[name]["val_flows"]) for name in retained) == 3125

    assert details["raw_stage14a_flows"] == 23454
    assert details["quarantined_cross_application_flow_rows"] == 4
    assert details["flows_after_quarantine"] == 23450
    for forbidden in ("model_training", "unknown_setting_generated", "open_detect_run", "des_run"):
        assert details[forbidden] is False
    assert len(hashes["inputs"]) == 2

    print(json.dumps({
        "verification": "PASS",
        "group_rows": len(groups),
        "split_rows": len(splits),
        "flows_before_quarantine": 23454,
        "quarantined_rows": 4,
        "flows_after_quarantine": 23450,
        "retained_global_validation_flows": 3125,
        "decisions": {key: sorted(value) for key, value in EXPECTED_DECISIONS.items()},
        "forbidden_actions": "all false",
    }, indent=2))


if __name__ == "__main__":
    main()
