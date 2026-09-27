#!/usr/bin/env python3
"""Freeze Stage38B coarse labels and training inputs without reading Test features."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S12 = PROJECT / "stage12_dual_external_validation/protocol"
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
EXPECTED = {"iscx_vpn": (12910, 1611, 1621, 6000),
            "iscx_tor": (8876, 1107, 1113, 4000)}
ROLES = ("known_train", "known_validation", "known_test", "unknown_test")
CATEGORY = {"Chat": "Communication", "Email": "Communication",
            "VoIP": "Communication", "Audio": "Streaming",
            "Video": "Streaming", "Streaming": "Streaming",
            "File-Transfer": "File-Transfer", "P2P": "P2P",
            "Browsing": "Browsing"}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def audit(dataset: str) -> tuple[dict, list[dict]]:
    home = S12 / dataset
    path = home / "split_manifest.csv"
    freeze = json.loads((home / "protocol_freeze.json").read_text())
    unknown = set(json.loads((home / "unknown_class_protocol.json").read_text())
                  ["settings"]["medium"]["unknown_classes"])
    if freeze["status"] != "PROTOCOL_FROZEN_PRETRAIN" or sha(path) != freeze["split_manifest_sha256"]:
        raise RuntimeError(f"{dataset}: Stage12 frozen split hash mismatch")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["setting"] == "medium"]
    counts = Counter(row["role"] for row in rows)
    if tuple(counts[role] for role in ROLES) != EXPECTED[dataset]:
        raise RuntimeError(f"{dataset}: frozen role counts changed: {counts}")
    ids = [row["flow_id_sha256"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{dataset}: flow IDs overlap between roles")
    if any((row["canonical_class"] in unknown) != (row["role"] == "unknown_test") for row in rows):
        raise RuntimeError(f"{dataset}: Unknown class leakage")
    if any(row["official_category"] not in CATEGORY for row in rows):
        raise RuntimeError(f"{dataset}: unmapped source category")
    expected_known = sorted(row["flow_id_sha256"] for row in rows
                            if row["role"] in ("known_train", "known_validation"))
    cache = S31 / "input_caches" / dataset
    tf = cache / "tf_fig"
    mfr = cache / "yatc_mfr"
    audits = [json.loads((folder / "cache_audit.json").read_text()) for folder in (tf, mfr)]
    if any(item["status"] != "PASS" or item["split_sha256"] != sha(path) for item in audits):
        raise RuntimeError(f"{dataset}: Stage31 Known-only cache audit mismatch")
    tf_ids = np.load(tf / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    mfr_ids = sorted(np.load(mfr / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
                     for role in ("known_train", "known_validation"))
    mfr_ids = sorted(uid for block in mfr_ids for uid in block)
    if tf_ids != expected_known or mfr_ids != expected_known:
        raise RuntimeError(f"{dataset}: Known Train/Val cache flow-ID mismatch")
    support = Counter((row["role"], CATEGORY[row["official_category"]]) for row in rows
                      if row["role"] != "unknown_test")
    classes = sorted({name for role, name in support if role == "known_train"})
    if any(support[role, name] <= 0 for name in classes
           for role in ("known_train", "known_validation", "known_test")):
        raise RuntimeError(f"{dataset}: empty Known class in a role")
    if dataset == "iscx_vpn" and len(classes) != 4 or dataset == "iscx_tor" and len(classes) != 4:
        raise RuntimeError(f"{dataset}: unexpected number of Known coarse classes")
    source_categories: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["role"] != "unknown_test":
            source_categories[row["canonical_class"]].add(row["official_category"])
    support_rows = [{"dataset": dataset, "role": role, "coarse_class": name,
                     "flow_count": support[role, name],
                     "label_source": "Stage12 capture-derived official_category"}
                    for role in ("known_train", "known_validation", "known_test")
                    for name in classes]
    protected = [path, home / "protocol_freeze.json", home / "unknown_class_protocol.json",
                 tf / "cache_audit.json", mfr / "cache_audit.json",
                 tf / "flow_ids.npy", mfr / "known_train_flow_ids.npy",
                 mfr / "known_validation_flow_ids.npy"]
    protected += [tf / f"{name}.npy" for name in
                  ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask")]
    protected += [mfr / f"{role}_mfr.npy" for role in ("known_train", "known_validation")]
    return {"dataset": dataset, "status": "PASS", "protocol": "Stage12 medium_seed2022",
            "role_counts": {role: counts[role] for role in ROLES},
            "known_coarse_classes": classes, "unknown_applications": sorted(unknown),
            "multiple_categories_per_known_application":
                {app: sorted(values) for app, values in source_categories.items() if len(values) > 1},
            "label_granularity": "capture-derived category metadata, not verified per-flow activity",
            "test_feature_values_loaded": 0, "unknown_feature_values_loaded": 0,
            "source_hashes": {str(item.relative_to(PROJECT)): sha(item) for item in protected}}, support_rows


def main() -> None:
    audits, support = [], []
    for dataset in EXPECTED:
        item, rows = audit(dataset)
        audits.append(item)
        support.extend(rows)
    output = ROOT / "stage38b_known_label_audit.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"status": "PASS", "datasets": audits,
                   "unknown_training_samples": 0, "unknown_validation_samples": 0,
                   "test_feature_values_loaded": 0}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (ROOT / "stage38b_known_label_support.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(support[0]))
        writer.writeheader()
        writer.writerows(support)
    print(json.dumps({"status": "PASS", "datasets":
        [{"dataset": item["dataset"], "counts": item["role_counts"],
          "classes": item["known_coarse_classes"]} for item in audits]}), flush=True)


if __name__ == "__main__":
    main()
