#!/usr/bin/env python3
"""Enforce the score-blind frozen 1:1 view after unchanged natural scoring."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True

from stage42w_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, replace_json

sys.path.insert(0, str(PROJECT / "stage42s_cic_favorable_open_set"))
import evaluate_candidate as base  # noqa: E402

SELECTED = ROOT / "balanced_known_manifest.csv"
AMENDMENT = ROOT / "balanced_view_implementation_audit.json"
OUT = ROOT / UNIT / "detection"
METHODS = ("msp", "energy", "centroid", "des_v1")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def replace_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    audit = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if (digest(MANIFEST) != protocol["candidate_manifest_sha256"]
            or digest(SELECTED) != audit["balanced_known_manifest_sha256"]
            or digest(ROOT / "fix_balanced_view.py") != audit["fix_code_sha256"]):
        raise RuntimeError("frozen balanced-view evidence drift")
    score_path = OUT / "sample_scores.csv"
    score_fields, rows = read_csv(score_path)
    known_fields, known_rows = read_csv(SELECTED)
    if known_fields != ["flow_id"]:
        raise RuntimeError("unexpected frozen Known selection schema")
    known_ids = {row["flow_id"] for row in known_rows}
    if len(known_ids) != 1228:
        raise RuntimeError("frozen Known selection size drift")
    if len(rows) != 2272 + 1228:
        raise RuntimeError("saved score population drift")
    for row in rows:
        if row["role"] == "known_test":
            row["balanced_selected"] = int(row["flow_id"] in known_ids)
        elif row["role"] == "unknown_test":
            row["balanced_selected"] = 1
        else:
            raise RuntimeError("unexpected Test role")
    selected = [row for row in rows if int(row["balanced_selected"])]
    truth = np.asarray([int(row["is_unknown"]) for row in selected], dtype=np.int64)
    if len(selected) != 2456 or int(truth.sum()) != 1228:
        raise RuntimeError("balanced 1:1 population not achieved")
    result_path = OUT / "open_set_results.csv"
    result_fields, results = read_csv(result_path)
    for row in results:
        if row["view"] != "balanced_1to1":
            continue
        if row["method"] not in METHODS:
            raise RuntimeError("unexpected method")
        score = np.asarray([float(item[f"score_{row['method']}"]) for item in selected])
        row.update({"known_test": 1228, "unknown_test": 1228, "unknown_prevalence": 0.5,
                    **base.metric(truth, score, float(row["threshold"]))})
    if len(results) != 8:
        raise RuntimeError("expected eight method-by-view metric rows")
    replace_csv(score_path, score_fields, rows)
    replace_csv(result_path, result_fields, results)
    evaluation_path = OUT / "evaluation_audit.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.update({
        "balanced_known_manifest_sha256": audit["balanced_known_manifest_sha256"],
        "balanced_known_test": 1228, "balanced_unknown_test": 1228,
        "balanced_view_implementation_audit": str(AMENDMENT),
    })
    replace_json(evaluation_path, evaluation)
    print(json.dumps({"status": "PASS", "balanced_known_test": 1228,
                      "balanced_unknown_test": 1228, "metrics_corrected": 4}), flush=True)


if __name__ == "__main__":
    main()
