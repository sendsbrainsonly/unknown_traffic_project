#!/usr/bin/env python3
"""Independent structural and invariant checks for the suitability audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

REQUIRED = {
    "dataset_inventory.csv",
    "pcap_inventory.csv",
    "pcap_sample_audit.csv",
    "table_schema_audit.csv",
    "archive_inventory.csv",
    "class_inventory.csv",
    "class_balance.csv",
    "open_set_class_suitability.csv",
    "opendetect_input_suitability.csv",
    "flow_pcap_mapping_audit.csv",
    "leakage_risk_audit.csv",
    "unknown_free_protocol_suitability.csv",
    "openness_suitability.csv",
    "dataset_ranking.csv",
    "inventory_summary.md",
    "audit_summary.md",
    "run_metadata.json",
}
DATASETS = {"CipherSpectrum", "CSTNET-TLS1.3", "CICDDoS2019"}


def read_csv(name: str):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    missing = sorted(REQUIRED - {path.name for path in OUT.iterdir() if path.is_file()})
    assert not missing, missing
    metadata = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["read_only_dataset_audit"] is True
    assert metadata["bulk_pcap_parsing"] is False
    assert metadata["models_trained"] is False
    assert metadata["unknown_detection_run"] is False
    assert metadata["splits_created"] is False
    assert metadata["archives_extracted"] is False

    inventory = read_csv("dataset_inventory.csv")
    counts = {dataset: sum(row["dataset"] == dataset for row in inventory) for dataset in DATASETS}
    for dataset in DATASETS:
        assert counts[dataset] == metadata["totals"][dataset]["files"]
        byte_sum = sum(int(row["size_bytes"]) for row in inventory if row["dataset"] == dataset)
        assert byte_sum == metadata["totals"][dataset]["logical_bytes"]

    pcap = read_csv("pcap_inventory.csv")
    valid_counts = {
        dataset: sum(row["dataset"] == dataset and row["macos_sidecar"] == "False" and row["readable_magic"] == "True" for row in pcap)
        for dataset in DATASETS
    }
    assert valid_counts == metadata["valid_pcap_counts"]
    assert valid_counts["CipherSpectrum"] > 0
    assert valid_counts["CSTNET-TLS1.3"] > 0
    assert valid_counts["CICDDoS2019"] == 0

    ranking = read_csv("dataset_ranking.csv")
    assert [row["dataset"] for row in ranking] == ["CSTNET-TLS1.3", "CipherSpectrum", "CICDDoS2019"]
    assert all(0 <= int(row["total_score"]) <= 20 for row in ranking)
    assert {row["dataset"] for row in read_csv("open_set_class_suitability.csv")} == DATASETS
    assert {row["dataset"] for row in read_csv("opendetect_input_suitability.csv")} == DATASETS

    openness = read_csv("openness_suitability.csv")
    for row in openness:
        candidate_classes = int(row["candidate_classes"])
        for field in ("low_unknown_ratio", "medium_unknown_ratio", "high_unknown_ratio"):
            known, unknown = (int(value) for value in row[field].split("/"))
            assert known + unknown == candidate_classes, (row["dataset"], field, row[field])

    open_set = {row["dataset"]: row for row in read_csv("open_set_class_suitability.csv")}
    primary = read_csv("class_balance.csv")
    for dataset in DATASETS:
        observed_classes = len({row["class_name"] for row in primary if row["dataset"] == dataset})
        assert int(open_set[dataset]["num_candidate_classes"]) == observed_classes

    class_inventory = read_csv("class_inventory.csv")
    cipher_primary = [
        row for row in class_inventory
        if row["dataset"] == "CipherSpectrum" and row["label_level"] == "website_domain"
    ]
    assert cipher_primary
    assert all("LABEL_PROVENANCE_UNCLEAR" in row["label_provenance"] for row in cipher_primary)

    summary = (OUT / "audit_summary.md").read_text(encoding="utf-8")
    for dataset in DATASETS:
        assert dataset in summary
    assert "No training" in summary

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## CipherSpectrum",
        "## CSTNET-TLS1.3",
        "## CICDDoS2019",
        "## Recommended Experimental Role",
    ):
        assert heading in readme
    print("PASS", {"inventory_rows": len(inventory), "valid_pcaps": valid_counts, "scores": {r['dataset']: r['total_score'] for r in ranking}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
