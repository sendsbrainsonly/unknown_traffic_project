#!/usr/bin/env python3
"""Build unchanged TF/FIG/YaTC packet views for frozen slowloris Unknown flows."""
from __future__ import annotations

import json
import sys

from stage42s_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress, replace_json

SOURCE = PROJECT / "stage34_ustc_cic_closed_set"
STAGE40 = PROJECT / "stage40_ustc_cic_open_set"
sys.path.insert(0, str(SOURCE))
import cic_build_inputs as builder  # noqa: E402


def rows_for_test(_phase):
    import csv
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = row.pop("class_name")
        row["split"] = "test"
    return rows


def frozen_gate():
    detection = STAGE40 / "unknown_slowhttptest/detection"
    calibration = json.loads((detection / "calibration.json").read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if digest(detection / "calibration.json") != protocol["known_calibration_sha256"]:
        raise RuntimeError("Known calibration drift before Unknown feature extraction")
    if calibration["unknown_fitting_count"] or calibration["test_fitting_count"]:
        raise RuntimeError("Known calibration was not Unknown/Test free")
    return calibration


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "PASS" or digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("candidate freeze failed")
    output = ROOT / UNIT
    builder.ROOT = ROOT
    builder.CIC = output
    builder.MANIFEST = MANIFEST
    builder.protocol.DAYS = ("Wednesday",)
    builder.read_manifest = rows_for_test
    builder.frozen_head_gate = frozen_gate
    old = sys.argv
    sys.argv = [str(builder.__file__), "--phase", "test"]
    progress("RUNNING", "unknown_packet_cache", unknown_test=protocol["unknown_test"])
    try:
        builder.main()
    finally:
        sys.argv = old
    base = output / "input_caches/ustc/A-2"
    accurate = {
        "status": "PASS", "dataset": "CIC-IDS-2017", "unit": UNIT,
        "role": "unknown_test", "flows": protocol["unknown_test"],
        "candidate_manifest_sha256": protocol["candidate_manifest_sha256"],
        "source_parquet_sha256": protocol["source_parquet_sha256"],
        "feature_formulas": "unchanged Stage34 TrafficFormer/FIG/YaTC packet views",
        "max_trafficformer_packets": 5, "max_graph_packets": 30, "max_yatc_packets": 5,
        "unknown_samples_transformed": protocol["unknown_test"],
        "unknown_samples_used_for_fitting": 0, "unknown_samples_used_for_calibration": 0,
    }
    for path in (base / "tf_fig_test", base / "yatc_mfr_test"):
        replace_json(path / "cache_audit.json", accurate)
    progress("RUNNING", "unknown_cache_complete", unknown_test=protocol["unknown_test"])


if __name__ == "__main__":
    main()
