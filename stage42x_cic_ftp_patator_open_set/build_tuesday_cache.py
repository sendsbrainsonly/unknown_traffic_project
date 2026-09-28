#!/usr/bin/env python3
"""Build unchanged packet views for frozen Tuesday FTP-Patator Unknown flows."""
from __future__ import annotations

import csv
import json
import sys

sys.dont_write_bytecode = True

from stage42x_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress, replace_json

SOURCE = PROJECT / "stage34_ustc_cic_closed_set"
STAGE40 = PROJECT / "stage40_ustc_cic_open_set"
sys.path.insert(0, str(SOURCE))
import cic_build_inputs as builder  # noqa: E402


def rows_for_test(_phase):
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = row.pop("class_name")
        row["split"] = "test"
    return rows


def frozen_gate():
    detection = STAGE40 / "unknown_slowhttptest/detection"
    calibration_path = detection / "calibration.json"
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if digest(calibration_path) != protocol["known_calibration_sha256"]:
        raise RuntimeError("Known calibration drift before FTP-Patator feature extraction")
    if calibration["unknown_fitting_count"] or calibration["test_fitting_count"]:
        raise RuntimeError("Known calibration was not Unknown/Test free")
    return calibration


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (protocol["status"] != "PASS" or digest(MANIFEST) != protocol["candidate_manifest_sha256"]
            or digest(ROOT / "build_tuesday_cache.py") != protocol["tuesday_cache_adapter_sha256"]):
        raise RuntimeError("candidate freeze or Tuesday cache adapter drift")
    output = ROOT / UNIT
    builder.ROOT = ROOT
    builder.CIC = output
    builder.MANIFEST = MANIFEST
    builder.protocol.DAYS = ("Tuesday",)
    builder.read_manifest = rows_for_test
    builder.frozen_head_gate = frozen_gate
    old_argv = sys.argv
    sys.argv = [str(builder.__file__), "--phase", "test"]
    progress("RUNNING", "unknown_packet_cache", unknown_test=protocol["unknown_test"])
    try:
        builder.main()
    finally:
        sys.argv = old_argv
    base = output / "input_caches/ustc/A-2"
    audit = {
        "status": "PASS", "dataset": "CIC-IDS-2017", "unit": UNIT,
        "role": "unknown_test", "flows": protocol["unknown_test"],
        "candidate_manifest_sha256": protocol["candidate_manifest_sha256"],
        "source_parquet_sha256": protocol["source_parquet_sha256"],
        "feature_formulas": "unchanged Stage34 TrafficFormer/FIG/YaTC packet views",
        "max_trafficformer_packets": 5, "max_graph_packets": 30,
        "max_yatc_packets": 5,
        "unknown_samples_transformed": protocol["unknown_test"],
        "unknown_samples_used_for_fitting": 0,
        "unknown_samples_used_for_calibration": 0,
    }
    for path in (base / "tf_fig_test", base / "yatc_mfr_test"):
        replace_json(path / "cache_audit.json", audit)
    progress("RUNNING", "unknown_cache_complete", unknown_test=protocol["unknown_test"])


if __name__ == "__main__":
    main()
