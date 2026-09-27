#!/usr/bin/env python3
"""Pre-training v2 amendment: retain all 5,096 Slowhttptest flows.

The v1 metadata-only pilot left 13 Slowhttptest Test flows after 10% sampling.
Group assignments and large-class samples remain exactly frozen; this script
adds all Slowhttptest flows in their already assigned group role.  No packet
features, validation predictions or Test outcomes are consulted.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import cic_protocol as v1

ROOT = v1.ROOT
V1 = ROOT / "cicids2017_three_class_10pct_manifest.csv"
V2 = ROOT / "cicids2017_three_class_manifest_v2.csv"
GROUPS = ROOT / "cicids2017_group_assignments.csv"


def main():
    if V2.exists():
        raise FileExistsError(V2)
    with GROUPS.open(newline="", encoding="utf-8") as f:
        roles = {r["group_id"]: r["split"] for r in csv.DictReader(f)}
    with V1.open(newline="", encoding="utf-8") as f:
        large = [r for r in csv.DictReader(f) if r["label"] != "DoS Slowhttptest"]
    slow = []
    for row in v1.rows():
        if row["label"] == "DoS Slowhttptest":
            row["group_id"] = v1.group(row)
            row["split"] = roles[row["group_id"]]
            slow.append(row)
    if len(slow) != 5096:
        raise RuntimeError(f"Slowhttptest count drift: {len(slow)}")
    selected = sorted(large + slow, key=lambda r: r["flow_id"])
    if len(selected) != len({r["flow_id"] for r in selected}):
        raise RuntimeError("duplicate CIC flow IDs in v2")
    with V2.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(*v1.FIELDS, "group_id", "split"))
        writer.writeheader()
        writer.writerows(selected)
    counts = Counter((r["label"], r["split"]) for r in selected)
    if counts[("DoS Slowhttptest", "test")] < 100:
        raise RuntimeError("v2 still has too few Slowhttptest Test flows")
    report = {"status": "PASS", "amendment_reason": "v1 Slowhttptest Test count 13 after 10% sampling",
              "basis": "pre-training metadata-only sample-support audit",
              "large_class_sample_source": str(V1), "large_class_sample_source_sha256": v1.digest(V1),
              "group_assignments": str(GROUPS), "group_assignments_sha256": v1.digest(GROUPS),
              "sample_rates": {"BENIGN": .1, "PortScan": .1, "DoS Slowhttptest": 1.0},
              "sample_counts": {f"{label}|{split}": count for (label, split), count in sorted(counts.items())},
              "v2_manifest_sha256": v1.digest(V2), "unknown_used": 0,
              "test_feature_values_read": 0, "test_outcomes_read": 0}
    (ROOT / "cicids2017_protocol_v2_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "counts": report["sample_counts"]}), flush=True)


if __name__ == "__main__":
    main()
