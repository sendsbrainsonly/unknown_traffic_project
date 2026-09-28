#!/usr/bin/env python3
"""Check the human-facing GoldenEye result table against frozen score outputs."""
from __future__ import annotations

import csv
import json
import math

from stage42t_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest

METRICS = ("auroc", "auprc", "ufar", "known_frr", "binary_f1")


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("candidate manifest hash changed")
    if len(rows(MANIFEST)) != 7441:
        raise RuntimeError("candidate manifest count changed")
    base = PROJECT / "stage42s_cic_favorable_open_set"
    for name, expected in protocol["reused_stage42s_code_sha256"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"reused code changed: {name}")
    detection = ROOT / UNIT / "detection"
    saved = {(r["view"], r["method"]): r for r in rows(detection / "open_set_results.csv")}
    summary = {(r["view"], r["method"]): r for r in rows(ROOT / "candidate_results.csv")}
    if saved.keys() != summary.keys() or len(saved) != 8:
        raise RuntimeError("result row identities changed")
    for key in saved:
        source, stated = saved[key], summary[key]
        if (int(source["known_test"]), int(source["unknown_test"])) != (
            int(stated["known_test"]), int(stated["unknown_test"])
        ):
            raise RuntimeError(f"population mismatch: {key}")
        for metric in METRICS:
            if not math.isclose(float(source[metric]), float(stated[metric]), abs_tol=1e-12):
                raise RuntimeError(f"summary metric mismatch: {key}/{metric}")
        passes = (float(source["auroc"]) >= 0.95 and float(source["auprc"]) >= 0.80
                  and float(source["ufar"]) <= 0.10 and float(source["known_frr"]) <= 0.10)
        if stated["strict_screen"] != ("PASS" if passes else "FAIL"):
            raise RuntimeError(f"screen mismatch: {key}")
    audit = json.loads((detection / "evaluation_audit.json").read_text(encoding="utf-8"))
    replay = json.loads((detection / "independent_verification.json").read_text(encoding="utf-8"))
    if (audit["status"] != "PASS" or not audit["checkpoint_hashes_unchanged"]
            or audit["unknown_fit_count"] or audit["test_fit_count"]
            or replay["status"] != "PASS" or replay["metrics_replayed"] != 8
            or replay["sample_rows"] != 9713):
        raise RuntimeError("frozen evaluation integrity failed")
    print(json.dumps({"status": "PASS", "summary_rows": len(summary),
                      "score_rows": replay["sample_rows"], "unknown_fit_count": 0,
                      "test_fit_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
