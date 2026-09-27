#!/usr/bin/env python3
"""Read-only protocol and three-view readiness audit before Stage31 training."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
CELLS = (("iscx_vpn", "medium_seed2022"), ("iscx_tor", "medium_seed2022"),
         ("vnat", "medium_seed2025"), ("vnat", "medium_seed2026"), ("ustc", "A-2"))
AUDIT = PROJECT / "stage15r_representation_bottleneck_audit" / "dataset_protocol_audit.csv"
SOURCES = {
    "iscx_vpn": PROJECT / "stage12_dual_external_validation" / "protocol" / "iscx_vpn" / "split_manifest.csv",
    "iscx_tor": PROJECT / "stage12_dual_external_validation" / "protocol" / "iscx_tor" / "split_manifest.csv",
    "vnat": PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv",
    "ustc": PROJECT / "stage3_protocol" / "split_manifest.csv",
}
RAW = {
    "iscx_vpn": PROJECT / "stage12_dual_external_validation" / "protocol" / "iscx_vpn" / "preprocessing_manifest.json",
    "iscx_tor": PROJECT / "stage12_dual_external_validation" / "protocol" / "iscx_tor" / "preprocessing_manifest.json",
    "vnat": PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_open_set_protocol.json",
    "ustc": PROJECT / "data" / "flows",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (OUT / "input_readiness.csv").exists():
        raise RuntimeError("preflight already ran")
    if not AUDIT.is_file():
        raise RuntimeError(f"missing frozen prior audit: {AUDIT}")
    with AUDIT.open(newline="", encoding="utf-8") as handle:
        old = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in old:
        grouped[(row["dataset"], row["protocol_id"])].append(row)
    rows = []
    source_hashes = {"stage15r_protocol_audit": sha(AUDIT)}
    for dataset, protocol in CELLS:
        selected = grouped[(dataset, protocol)]
        if not selected:
            raise RuntimeError(f"missing frozen protocol in Stage15R: {dataset}/{protocol}")
        known = [row for row in selected if row["class_role"].lower() == "known"]
        if not known:
            raise RuntimeError(f"no Known classes: {dataset}/{protocol}")
        train = sum(int(r["known_train_samples"]) for r in known)
        val = sum(int(r["known_validation_samples"]) for r in known)
        test = sum(int(r["known_test_samples"]) for r in known)
        if min(train, val, test) <= 0 or any(min(int(r[x]) for x in
            ("known_train_samples", "known_validation_samples", "known_test_samples")) <= 0 for r in known):
            raise RuntimeError(f"empty Known role/class: {dataset}/{protocol}")
        source = SOURCES[dataset]
        if not source.is_file():
            raise RuntimeError(f"missing frozen split source: {source}")
        source_hashes[str(source.relative_to(PROJECT))] = sha(source)
        raw = RAW[dataset]
        if not raw.exists() or (dataset == "ustc" and not any(raw.glob("*.pkl"))):
            raise RuntimeError(f"missing raw input provenance: {raw}")
        # Stage30 aligned arrays were built exclusively for Stage20 ISCX full-Service,
        # so they cannot be silently reused for these Stage15R frozen protocols.
        rows.append({"dataset": dataset, "protocol_id": protocol,
                     "known_classes": len(known), "known_train": train,
                     "known_validation": val, "known_test": test,
                     "split_manifest": str(source.relative_to(PROJECT)),
                     "split_sha256": source_hashes[str(source.relative_to(PROJECT))],
                     "raw_provenance_present": True,
                     "matched_three_view_features_for_this_protocol": False,
                     "needs_trafficformer_generation": True,
                     "needs_graph_generation": True,
                     "needs_yatc_generation": True,
                     "test_feature_values_loaded": 0,
                     "unknown_feature_values_loaded": 0})
    write_csv(OUT / "input_readiness.csv", rows)
    (OUT / "source_hashes_before.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    result = {"status": "INPUT_GENERATION_REQUIRED", "frozen_protocols": len(rows),
              "datasets": len({r["dataset"] for r in rows}),
              "matched_three_view_protocols": 0,
              "test_feature_values_loaded": 0, "unknown_feature_values_loaded": 0,
              "raw_provenance_available": True}
    (OUT / "preflight_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
