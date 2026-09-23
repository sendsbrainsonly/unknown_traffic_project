#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    required = [
        "README.md", "EXPERIMENT_INDEX.md", "outputs/RESULTS.md", "outputs/manifest.json",
        "outputs/summary/dataset_path_resolution.csv", "outputs/summary/four_dataset_readiness_matrix.csv", "outputs/summary/audit_summary.json",
        "outputs/ustc/class_inventory.csv", "outputs/ustc/file_integrity.csv", "outputs/ustc/duplicate_summary.csv", "outputs/ustc/existing_split_summary.md", "outputs/ustc/readiness.md",
        "outputs/cipherspectrum/source_class_counts.csv", "outputs/cipherspectrum/mix_provenance_audit.csv", "outputs/cipherspectrum/mix_duplicate_pairs.csv", "outputs/cipherspectrum/mix_provenance_summary.md", "outputs/cipherspectrum/grouping_audit.csv", "outputs/cipherspectrum/raw_input_manifest.csv", "outputs/cipherspectrum/domain_sni_leakage.csv", "outputs/cipherspectrum/endpoint_input_audit.md", "outputs/cipherspectrum/calibration_capacity.csv",
        "outputs/cstnet/frozen_protocol_verification.md", "outputs/cstnet/calibration_capacity.csv", "outputs/cstnet/input_leakage_summary.md",
        "outputs/cicids2017/class_inventory.csv", "outputs/cicids2017/class_day_matrix.csv", "outputs/cicids2017/class_pcap_matrix.csv", "outputs/cicids2017/class_hour_distribution.csv", "outputs/cicids2017/endpoint_port_shortcut.csv", "outputs/cicids2017/raw_input_manifest.csv", "outputs/cicids2017/input_endpoint_audit.md", "outputs/cicids2017/class_eligibility.csv", "outputs/cicids2017/split_feasibility.md",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    assert not missing, f"missing required outputs: {missing}"

    paths = rows(OUT / "summary/dataset_path_resolution.csv")
    assert len(paths) == 4 and all(row["exists"] == "True" for row in paths)
    ustc = rows(OUT / "ustc/class_inventory.csv")
    assert len(ustc) == 20
    assert sum(int(row["flow_count"]) for row in ustc) == 489_101
    assert next(row for row in ustc if row["class_name"] == "SMB")["source_pcap_count"] == "2"
    ustc_integrity = rows(OUT / "ustc/file_integrity.csv")
    canonical = [row for row in ustc_integrity if row["is_canonical_extracted"] == "True"]
    assert len(canonical) == 24 and all(row["readable"] == "True" and int(row["packet_count"]) > 0 for row in canonical)

    cipher_counts = rows(OUT / "cipherspectrum/source_class_counts.csv")
    official = [row for row in cipher_counts if row["corpus_status"] == "OFFICIAL40"]
    assert len({row["class_name"] for row in official}) == 40
    assert sum(int(row["pcap_count"]) for row in official) == 160_200
    assert any(row["class_name"] == "getpocket.com" and row["corpus_status"] == "NON_PRIMARY_EXTRA_CLASS" for row in cipher_counts)
    cipher_fingerprints = rows(OUT / "cipherspectrum/all_pcap_fingerprints.csv")
    assert len(cipher_fingerprints) == 164_207
    mix = json.loads((OUT / "cipherspectrum/mix_provenance_summary.json").read_text(encoding="utf-8"))
    assert mix["mix_status"] in {"INDEPENDENT", "DERIVED_OR_DUPLICATED", "PARTIALLY_OVERLAPPING", "UNCERTAIN"}
    cipher_inputs = rows(OUT / "cipherspectrum/raw_input_manifest.csv")
    assert len(cipher_inputs) <= 400 and len({row["class_name"] for row in cipher_inputs}) <= 20
    assert all(row["seed"] == "20260913" for row in cipher_inputs)

    cstnet = (OUT / "cstnet/frozen_protocol_verification.md").read_text(encoding="utf-8")
    assert "PASS (10/10)" in cstnet and "46,372" in cstnet and "119/1" in cstnet

    cic = rows(OUT / "cicids2017/class_inventory.csv")
    assert len(cic) == 15 and sum(int(row["strict_flow_count"]) for row in cic) == 2_087_440
    cic_inputs = rows(OUT / "cicids2017/raw_input_manifest.csv")
    assert len(cic_inputs) <= 200 and len({row["label"] for row in cic_inputs}) <= 10
    assert all(row["seed"] == "20260913" for row in cic_inputs)
    matrix = rows(OUT / "summary/four_dataset_readiness_matrix.csv")
    assert len(matrix) == 4
    assert all(row["final_readiness"] in {"READY", "CONDITIONALLY_READY", "NOT_READY"} for row in matrix)

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["execution"]["physical_gpu_ids"] == []
    bad_hashes = []
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file() or digest(path) != artifact["sha256"]:
            bad_hashes.append(artifact["path"])
    assert not bad_hashes, f"artifact hash mismatch: {bad_hashes}"
    print(json.dumps({
        "verification": "PASS", "required_files": len(required), "artifact_hashes": len(manifest["artifacts"]),
        "ustc_flows": 489101, "cipher_official40": 160200, "cstnet_pcaps": 46372, "cicids_strict": 2087440,
    }, indent=2))


if __name__ == "__main__":
    main()
