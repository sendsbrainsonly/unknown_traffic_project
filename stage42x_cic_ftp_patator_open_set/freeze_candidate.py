#!/usr/bin/env python3
"""Freeze all matched Tuesday FTP-Patator flows before packet-feature access."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from stage42x_common import (MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest,
                             progress, write_json_new, write_manifest_new)

DATA = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)")
PARQUET = DATA / "outputs/cicids2017_pcap_label_mapping/flows/Tuesday_flows.parquet"
PCAP = DATA / "PCAPs/Tuesday-WorkingHours.pcap"
PCAP_MD5 = DATA / "PCAPs/Tuesday-WorkingHours.md5"
STAGE34 = PROJECT / "stage34_ustc_cic_closed_set"
STAGE40 = PROJECT / "stage40_ustc_cic_open_set"
STAGE42S = PROJECT / "stage42s_cic_favorable_open_set"
STAGE42W = PROJECT / "stage42w_cic_bot_open_set"
KNOWN_ROLES = STAGE40 / "unknown_slowhttptest_roles.csv"
KNOWN_DETECTION = STAGE40 / "unknown_slowhttptest/detection"
KNOWN_RUN = STAGE40 / "unknown_slowhttptest/runs/ustc/A-2"
LABEL = "FTP-Patator"
EXPECTED_COUNT = 3985
FIELDS = ["flow_id", "day", "source_pcap", "label", "match_status",
          "flow_start_epoch_utc", "canonical_flow_key", "first_packet_index",
          "last_packet_index", "packet_count"]
REUSED_CODE = ("stage42s_common.py", "build_unknown_cache.py",
               "evaluate_candidate.py", "verify_candidate.py")


def main() -> None:
    if MANIFEST.exists() or PROTOCOL.exists():
        raise FileExistsError("FTP-Patator candidate protocol is already frozen")
    previous = json.loads((STAGE42W / "queue_progress.json").read_text(encoding="utf-8"))
    if previous["status"] != "COMPLETE" or previous["verification"]["status"] != "PASS":
        raise RuntimeError("fifth candidate is not independently complete")
    stage34 = json.loads((STAGE34 / "cicids2017_protocol_audit.json").read_text(encoding="utf-8"))
    source_hash = digest(PARQUET)
    if source_hash != stage34["source_parquet_sha256"]["Tuesday"]:
        raise RuntimeError("Tuesday mapped-flow parquet changed since Stage34")
    stage40 = json.loads((STAGE40 / "protocols.json").read_text(encoding="utf-8"))
    known = next(x for x in stage40["units"] if x.get("key") == "unknown_slowhttptest")
    if known["known_classes"] != ["BENIGN", "PortScan"] or digest(KNOWN_ROLES) != known["role_manifest_sha256"]:
        raise RuntimeError("Stage40 Known-role drift")
    calibration_path = KNOWN_DETECTION / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration["status"] != "PASS" or calibration["unknown_fitting_count"] or calibration["test_fitting_count"]:
        raise RuntimeError("Known-only calibration is not reusable")
    chosen = [row for row in pq.read_table(PARQUET, columns=FIELDS).to_pylist()
              if row["label"] == LABEL and str(row["match_status"]).startswith("MATCHED")]
    if len(chosen) != EXPECTED_COUNT or len({row["flow_id"] for row in chosen}) != EXPECTED_COUNT:
        raise RuntimeError(f"unexpected matched FTP-Patator population: {len(chosen)}")
    if {row["source_pcap"] for row in chosen} != {PCAP.name} or {row["day"] for row in chosen} != {"Tuesday"}:
        raise RuntimeError("FTP-Patator metadata does not resolve to the frozen Tuesday PCAP")
    with KNOWN_ROLES.open(newline="", encoding="utf-8") as handle:
        role_ids = {row["flow_id"] for row in csv.DictReader(handle)}
    if role_ids.intersection(row["flow_id"] for row in chosen):
        raise RuntimeError("candidate Unknown overlaps a frozen Stage40 role")
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
    write_manifest_new(rows)
    manifest_hash = digest(MANIFEST)
    payload = {
        "status": "PASS", "unit": UNIT,
        "claim_scope": "post-hoc favorable-setting development diagnostic",
        "candidate_order_basis": "sixth candidate: largest not-yet-evaluated matched attack label after the Friday candidates; selected from label/count metadata before FTP-Patator packet-feature access",
        "known_classes": ["BENIGN", "PortScan"], "unknown_classes": [LABEL],
        "unknown_test": len(rows), "unknown_groups": len({row["group_id"] for row in rows}),
        "unknown_selection": "all MATCHED FTP-Patator flows; no feature/outcome filtering",
        "known_model_source": str(KNOWN_RUN), "known_roles": str(KNOWN_ROLES),
        "known_role_manifest_sha256": digest(KNOWN_ROLES),
        "known_calibration": str(calibration_path), "known_calibration_sha256": digest(calibration_path),
        "known_test_sample_scores": str(KNOWN_DETECTION / "sample_scores.csv"),
        "known_test_sample_scores_sha256": digest(KNOWN_DETECTION / "sample_scores.csv"),
        "candidate_manifest": str(MANIFEST), "candidate_manifest_sha256": manifest_hash,
        "source_parquet": str(PARQUET), "source_parquet_sha256": source_hash,
        "source_pcap": str(PCAP), "source_pcap_size_bytes": PCAP.stat().st_size,
        "source_pcap_md5_sidecar": str(PCAP_MD5),
        "source_pcap_md5_sidecar_sha256": digest(PCAP_MD5),
        "reused_stage42s_code_sha256": {name: digest(STAGE42S / name) for name in REUSED_CODE},
        "tuesday_cache_adapter_sha256": digest(ROOT / "build_tuesday_cache.py"),
        "frozen_before_candidate_packet_feature_access": True,
        "unknown_fitting_count": 0, "unknown_threshold_calibration_count": 0,
        "threshold_rule": "reuse Stage40 BENIGN+PortScan Known-Validation P95",
        "methods": ["msp", "energy", "centroid", "des_v1"],
        "balanced_view_selection": "SHA256(seed=2022, flow_id), n=min(Known Test, Unknown Test)",
    }
    write_json_new(PROTOCOL, payload)
    write_json_new(ROOT / "cicids2017_protocol_v2_audit.json", {
        "status": "PASS", "v2_manifest_sha256": manifest_hash,
        "actual_protocol": "Stage42X-sixth-favorable-candidate-FTP-Patator",
    })
    progress("RUNNING", "protocol_frozen", unknown_test=len(rows), unknown_groups=payload["unknown_groups"])


if __name__ == "__main__":
    main()
