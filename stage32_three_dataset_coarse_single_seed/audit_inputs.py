#!/usr/bin/env python3
"""Stage32 prefit Known-only three-view and historical-alignment audit."""
from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np

from common import (COARSE_CLASSES, DATASETS, F2_INPUT, F2_RUN, ROOT, S25,
                    TRAINING_SEED, VIEWS, coarse, expected_rows, freeze_sources,
                    load_known, write_json)


def iscx_baseline_coverage(dataset: str, role: str) -> dict:
    expected = dict(expected_rows(dataset, role))
    files = ((S25 / "remap_predictions.csv", "method"),
             (S25 / "coarse_head_predictions.csv", "encoder"))
    coverage = {}
    for path, key in files:
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset
                    and r["seed"] == str(TRAINING_SEED) and r[key] == "E3"
                    and r["role"] == role]
        ids = [r["flow_id"] for r in rows]
        if len(ids) != len(expected) or set(ids) != set(expected):
            raise RuntimeError(f"historical E3 {path.name}/{dataset}/{role} coverage mismatch")
        if any(r["true_fine"] != expected[r["flow_id"]] or
               r["true_coarse"] != coarse(dataset, expected[r["flow_id"]]) for r in rows):
            raise RuntimeError(f"historical E3 {path.name}/{dataset}/{role} truth mismatch")
        coverage[path.name] = len(rows)
    return coverage


def vnat_val_baseline_coverage() -> dict:
    expected = dict(expected_rows("vnat", "known_validation"))
    path = F2_INPUT / "input_manifest.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "validation"]
    rows.sort(key=lambda r: int(r["local_index"]))
    ids = [r["flow_uid"] for r in rows]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise RuntimeError("VNAT F2 historical validation flow-ID mismatch")
    if [int(r["local_index"]) for r in rows] != list(range(len(rows))):
        raise RuntimeError("VNAT F2 historical local-index sequence invalid")
    uid_array = np.load(F2_INPUT / "validation_flow_uids.npy", allow_pickle=False).astype(str).tolist()
    labels = np.load(F2_INPUT / "validation_labels.npy", allow_pickle=False)
    prediction = np.load(F2_RUN / "validation_predictions.npy", allow_pickle=False)
    if uid_array != ids or labels.shape != prediction.shape or len(prediction) != len(rows):
        raise RuntimeError("VNAT F2 historical prediction ordering/shape mismatch")
    config = json.loads((F2_RUN / "training_config.json").read_text())
    classes = config["known_classes"]
    if any(expected[r["flow_uid"]] != r["application"] or
           classes[int(labels[i])] != r["application"] or
           int(labels[i]) != int(r["local_label"]) for i, r in enumerate(rows)):
        raise RuntimeError("VNAT F2 historical label/UID mismatch")
    if prediction.min() < 0 or prediction.max() >= len(classes):
        raise RuntimeError("VNAT F2 historical prediction out of class range")
    return {"validation_samples": len(rows), "prediction_file": str(F2_RUN / "validation_predictions.npy"),
            "training_seed_different_from_stage32": int(config["training_seed"]) != TRAINING_SEED,
            "test_pairing": "UNAVAILABLE_UNLESS_FROZEN_CHECKPOINT_REPLAY_PASSES"}


def main() -> None:
    if (ROOT / "preflight.json").exists() or (ROOT / "frozen_source_hashes_before.json").exists():
        raise RuntimeError("Stage32 preflight already exists; refusing overwrite")
    audit = []
    all_ids = {}
    for dataset in DATASETS:
        roles = {}
        for role in ("known_train", "known_validation"):
            bundle = load_known(dataset, role)
            ids = bundle["flow_ids"].astype(str).tolist()
            all_ids[(dataset, role)] = set(ids)
            counts = Counter(COARSE_CLASSES[dataset][int(i)] for i in bundle["labels"])
            roles[role] = {"samples": len(ids), "class_counts": dict(counts),
                           "views": {view: list(bundle[view].shape) for view in VIEWS}}
        if all_ids[(dataset, "known_train")] & all_ids[(dataset, "known_validation")]:
            raise RuntimeError(f"{dataset}: Train/Val overlap")
        test = expected_rows(dataset, "known_test")  # metadata only; do not read Test feature values
        test_ids = {uid for uid, _ in test}
        if test_ids & (all_ids[(dataset, "known_train")] | all_ids[(dataset, "known_validation")]):
            raise RuntimeError(f"{dataset}: Test ID overlap")
        roles["known_test_metadata_only"] = {"samples": len(test),
            "class_counts": dict(Counter(coarse(dataset, fine) for _, fine in test))}
        if dataset != "vnat":
            historical = {role: iscx_baseline_coverage(dataset, role)
                          for role in ("known_validation", "known_test")}
        else:
            historical = vnat_val_baseline_coverage()
        audit.append({"dataset": dataset, "coarse_classes": COARSE_CLASSES[dataset],
                      "roles": roles, "historical_alignment": historical})
    hashes = freeze_sources()
    write_json(ROOT / "frozen_source_hashes_before.json", hashes)
    write_json(ROOT / "preflight.json", {"status": "PASS", "units": len(audit), "rows": audit,
        "train_seed": TRAINING_SEED, "test_feature_values_loaded": 0,
        "unknown_feature_values_loaded": 0, "source_hash_count": len(hashes)})
    print(json.dumps({"status": "PASS", "units": len(audit), "source_hash_count": len(hashes),
                      "test_feature_values_loaded": 0}), flush=True)


if __name__ == "__main__":
    main()
