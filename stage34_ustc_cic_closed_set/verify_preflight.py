#!/usr/bin/env python3
"""Read-only Stage34 split, sampling and pre-Test leakage checks."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import cic_protocol
import ustc_runner

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ua = json.loads((ROOT / "ustc_sample_audit.json").read_text())
    urows = read(ustc_runner.MANIFEST)
    if ustc_runner.sha(ustc_runner.MANIFEST) != ua["sample_manifest_sha256"]:
        raise RuntimeError("USTC sample hash drift")
    if len(urows) != 43331 or len({r["flow_id"] for r in urows}) != len(urows):
        raise RuntimeError("USTC cardinality/uniqueness failed")
    ucounts = Counter(r["original_split"] for r in urows)
    if ucounts != {"train": 34665, "val": 4333, "test": 4333}:
        raise RuntimeError(f"USTC split counts drift: {ucounts}")
    trainval = {r["flow_id"] for r in urows if r["original_split"] != "test"}
    ids = np.load(ROOT / "input_caches/ustc/A-2/tf_fig/flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if set(ids) != trainval or len(ids) != len(trainval):
        raise RuntimeError("USTC TF/FIG cache ID mismatch")
    for split, role in (("train", "known_train"), ("val", "known_validation")):
        mids = np.load(ROOT / f"input_caches/ustc/A-2/yatc_mfr/{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if set(mids) != {r["flow_id"] for r in urows if r["original_split"] == split}:
            raise RuntimeError(f"USTC MFR cache ID mismatch: {role}")
    cv = json.loads((ROOT / "cicids2017_protocol_v2_audit.json").read_text())
    crows = read(ROOT / "cicids2017_three_class_manifest_v2.csv")
    if cic_protocol.digest(ROOT / "cicids2017_three_class_manifest_v2.csv") != cv["v2_manifest_sha256"]:
        raise RuntimeError("CIC v2 manifest hash drift")
    group_roles = defaultdict(set)
    class_roles = defaultdict(set)
    for row in crows:
        if row["group_id"] != f"{row['source_pcap']}|{int(float(row['flow_start_epoch_utc'])//300)}":
            raise RuntimeError("CIC group ID mismatch")
        group_roles[row["group_id"]].add(row["split"])
        class_roles[row["label"]].add(row["split"])
    if any(len(values) != 1 for values in group_roles.values()):
        raise RuntimeError("CIC group leakage")
    if set(class_roles) != set(cic_protocol.CLASSES) or any(values != {"train", "validation", "test"} for values in class_roles.values()):
        raise RuntimeError("CIC class/split coverage failed")
    counts = Counter((r["label"], r["split"]) for r in crows)
    if counts[("DoS Slowhttptest", "test")] != 132 or sum(n for (c, _), n in counts.items() if c == "DoS Slowhttptest") != 5096:
        raise RuntimeError("CIC small-class preservation failed")
    if (ROOT / "cicids2017/input_caches/ustc/A-2/tf_fig_test").exists() or \
       (ROOT / "input_caches/ustc/A-2/tf_fig_test").exists():
        raise RuntimeError("Test cache appeared before preflight")
    report = {"status": "PASS", "ustc_flows": len(urows), "cic_flows": len(crows),
              "cic_groups_in_sample": len(group_roles), "cic_counts": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())},
              "unknown_samples_used": 0, "test_feature_values_loaded": 0}
    (ROOT / "preflight_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
