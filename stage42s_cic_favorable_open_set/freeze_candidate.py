#!/usr/bin/env python3
"""Freeze the first Stage42-S candidate before packet feature access."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from stage42s_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress, write_csv, write_json_new

DATA = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)")
PARQUET = DATA / "outputs/cicids2017_pcap_label_mapping/flows/Wednesday_flows.parquet"
PCAP = DATA / "PCAPs/Wednesday-workingHours.pcap"
PCAP_MD5 = DATA / "PCAPs/Wednesday-workingHours.md5"
STAGE34_AUDIT = PROJECT / "stage34_ustc_cic_closed_set/cicids2017_protocol_audit.json"
STAGE40 = PROJECT / "stage40_ustc_cic_open_set"
STAGE40_PROTOCOL = STAGE40 / "protocols.json"
KNOWN_ROLES = STAGE40 / "unknown_slowhttptest_roles.csv"
KNOWN_DETECTION = STAGE40 / "unknown_slowhttptest/detection"
KNOWN_RUN = STAGE40 / "unknown_slowhttptest/runs/ustc/A-2"
LABEL = "DoS slowloris"
FIELDS = ["flow_id", "day", "source_pcap", "label", "match_status",
          "flow_start_epoch_utc", "canonical_flow_key", "first_packet_index",
          "last_packet_index", "packet_count"]


def main() -> None:
    if MANIFEST.exists() or PROTOCOL.exists():
        raise FileExistsError("candidate protocol is already frozen")
    stage34 = json.loads(STAGE34_AUDIT.read_text(encoding="utf-8"))
    expected_parquet = stage34["source_parquet_sha256"]["Wednesday"]
    actual_parquet = digest(PARQUET)
    if actual_parquet != expected_parquet:
        raise RuntimeError("Wednesday mapped-flow parquet changed since Stage34")
    stage40 = json.loads(STAGE40_PROTOCOL.read_text(encoding="utf-8"))
    known = next(x for x in stage40["units"] if x.get("key") == "unknown_slowhttptest")
    if known["known_classes"] != ["BENIGN", "PortScan"]:
        raise RuntimeError("Stage40 known-class role drift")
    if digest(KNOWN_ROLES) != known["role_manifest_sha256"]:
        raise RuntimeError("Stage40 known role manifest changed")
    calibration = KNOWN_DETECTION / "calibration.json"
    cal = json.loads(calibration.read_text(encoding="utf-8"))
    if cal["status"] != "PASS" or cal["unknown_fitting_count"] or cal["test_fitting_count"]:
        raise RuntimeError("Stage40 Known-only calibration is not reusable")

    table = pq.read_table(PARQUET, columns=FIELDS)
    records = table.to_pylist()
    chosen = [row for row in records if row["label"] == LABEL and
              str(row["match_status"]).startswith("MATCHED")]
    if len(chosen) != 5709 or len({row["flow_id"] for row in chosen}) != 5709:
        raise RuntimeError(f"unexpected slowloris population: {len(chosen)}")
    rows = []
    for row in chosen:
        start = float(row["flow_start_epoch_utc"])
        rows.append({
            "flow_id": row["flow_id"], "class_name": LABEL, "role": "unknown_test",
            "day": row["day"], "source_pcap": row["source_pcap"],
            "group_id": f"{row['source_pcap']}|{int(start // 300)}",
            "flow_start_epoch_utc": row["flow_start_epoch_utc"],
            "canonical_flow_key": row["canonical_flow_key"],
            "first_packet_index": row["first_packet_index"],
            "last_packet_index": row["last_packet_index"],
            "packet_count": row["packet_count"], "match_status": row["match_status"],
        })
    rows.sort(key=lambda row: row["flow_id"])
    write_csv(MANIFEST, rows)
    manifest_sha = digest(MANIFEST)
    compat = {"status": "PASS", "v2_manifest_sha256": manifest_sha,
              "actual_protocol": "Stage42S-first-favorable-candidate-slowloris"}
    write_json_new(ROOT / "cicids2017_protocol_v2_audit.json", compat)
    payload = {
        "status": "PASS", "unit": UNIT, "claim_scope": "post-hoc favorable-setting development diagnostic",
        "known_classes": ["BENIGN", "PortScan"], "unknown_classes": [LABEL],
        "unknown_test": len(rows), "unknown_groups": len({row["group_id"] for row in rows}),
        "unknown_selection": "all MATCHED DoS slowloris flows; no feature/outcome filtering",
        "known_model_source": str(KNOWN_RUN), "known_roles": str(KNOWN_ROLES),
        "known_role_manifest_sha256": digest(KNOWN_ROLES),
        "known_calibration": str(calibration), "known_calibration_sha256": digest(calibration),
        "known_test_sample_scores": str(KNOWN_DETECTION / "sample_scores.csv"),
        "known_test_sample_scores_sha256": digest(KNOWN_DETECTION / "sample_scores.csv"),
        "candidate_manifest": str(MANIFEST), "candidate_manifest_sha256": manifest_sha,
        "source_parquet": str(PARQUET), "source_parquet_sha256": actual_parquet,
        "source_pcap": str(PCAP), "source_pcap_size_bytes": PCAP.stat().st_size,
        "source_pcap_md5_sidecar": str(PCAP_MD5),
        "source_pcap_md5_sidecar_sha256": digest(PCAP_MD5),
        "frozen_before_candidate_packet_feature_access": True,
        "unknown_fitting_count": 0, "unknown_threshold_calibration_count": 0,
        "threshold_rule": "reuse Stage40 BENIGN+PortScan Known-Validation P95",
        "methods": ["msp", "energy", "centroid", "des_v1"],
        "balanced_view_selection": "SHA256(seed=2022, flow_id), n=min(Known Test, Unknown Test)",
    }
    write_json_new(PROTOCOL, payload)
    progress("RUNNING", "protocol_frozen", unknown_test=len(rows), unknown_groups=payload["unknown_groups"])


if __name__ == "__main__":
    main()
