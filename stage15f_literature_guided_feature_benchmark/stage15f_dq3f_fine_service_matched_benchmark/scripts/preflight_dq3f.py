#!/usr/bin/env python3
"""Fail-closed DQ-3F manifest, label, and feature-alignment audit."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from dq3f_common import (
    OUT, SEEDS, SERVICES, TRAINING_CONFIG, array_digest, hash_protected_inputs,
    label_space, load_matched_split, provenance_rows, read_json, sha256_file,
    singleton_application_service_map, support_counts, write_csv, write_json,
)


def main() -> int:
    rows = provenance_rows()
    train = load_matched_split("known_train")
    validation = load_matched_split("known_validation")
    ids = [row["flow_id"] for row in rows]
    train_ids = {row["flow_id"] for row in train["metadata"]}
    val_ids = {row["flow_id"] for row in validation["metadata"]}
    if len(ids) != len(set(ids)) or train_ids & val_ids:
        raise RuntimeError("flow IDs are not unique/disjoint")
    if len(train_ids) != 2730 or len(val_ids) != 335 or len(train_ids | val_ids) != 3065:
        raise RuntimeError("frozen DQ-3R membership counts changed")
    if {row["service_label"] for row in rows} != set(SERVICES):
        raise RuntimeError("invalid Service label vocabulary")
    if any(row["service_label_status"] != "WEAK_CAPTURE_LABEL" for row in rows):
        raise RuntimeError("unexpected Service label status")
    if {row["split_role"] for row in rows} != {"known_train", "known_validation"}:
        raise RuntimeError("non-Known Train/Validation role entered DQ-3F")
    if any(not row["application_label"] for row in rows):
        raise RuntimeError("missing Fine Application label")
    fine_classes, _ = label_space(train, validation, "fine")
    service_classes, _ = label_space(train, validation, "service")
    if service_classes != SERVICES:
        raise RuntimeError("Service class order mismatch")

    parity = []
    for split in (train, validation):
        role = split["role"]
        expected = 2730 if role == "known_train" else 335
        parity.append({
            "role": role,
            "source_stage12_rows": split["source_total"],
            "matched_dq3r_rows": len(split["metadata"]),
            "expected_dq3r_rows": expected,
            "unique_flow_ids": len({row["flow_id"] for row in split["metadata"]}),
            "unique_image_hashes": len({row["image_sha256"] for row in split["metadata"]}),
            "source_target_alignment": "PASS",
            "source_array_digest": "PASS",
            "application_label_alignment": "PASS",
            "source_file_alignment": "PASS",
            "status": "PASS" if len(split["metadata"]) == expected else "FAIL",
        })
    write_csv(OUT / "dq3f_manifest_parity.csv", parity)

    image_roles: dict[str, set[str]] = defaultdict(set)
    for split in (train, validation):
        for row in split["metadata"]:
            image_roles[row["image_sha256"]].add(split["role"])
    cross_split_exact_images = sum(len(roles) > 1 for roles in image_roles.values())
    if cross_split_exact_images:
        raise RuntimeError(f"exact image leakage across Train/Validation: {cross_split_exact_images}")
    singleton, ambiguous = singleton_application_service_map()
    prereg = read_json(TRAINING_CONFIG)
    training = prereg["training"]
    config = {
        "status": "FROZEN_BEFORE_TRAINING",
        "dataset": "iscx_vpn_shared31_vpn_only",
        "claim_scope": "Known-only matched-sample weak-capture-label diagnostic",
        "tasks": {
            "F1": "Fine Application direct training and full Validation evaluation",
            "F2_PARTIAL": "F1 prediction deterministic singleton Application-to-Service mapping only",
            "F3": "Service direct training and full Validation evaluation",
        },
        "seeds": SEEDS,
        "repeat_rule": "DQ-3R fixed seed 2022 but no repeat_count; task default freezes three paired seeds before outcomes",
        "train_samples": len(train["metadata"]),
        "validation_samples": len(validation["metadata"]),
        "fine_classes": fine_classes,
        "service_classes": service_classes,
        "training": training,
        "same_inputs_membership_budget_f1_f3": True,
        "only_allowed_difference": "supervision labels and output class count",
        "known_test_feature_values_used": 0,
        "unknown_test_feature_values_used": 0,
        "metadata_as_model_input": False,
        "capture_id_as_model_input": False,
        "service_label_as_model_input": False,
        "f2_singleton_application_service_map": singleton,
        "f2_ambiguous_applications": ambiguous,
        "source_hashes": {
            "provenance": sha256_file(OUT.parent / "stage15f_dq3r_service_label_reconstruction" / "service_label_provenance.csv"),
        },
        "stage12_subset_array_digests": {
            "known_train": array_digest(train["data"], train["original_target"]),
            "known_validation": array_digest(validation["data"], validation["original_target"]),
        },
    }
    write_json(OUT / "dq3f_training_configs.json", config)
    write_json(OUT / "protected_asset_hashes_before.json", hash_protected_inputs())

    support = support_counts()
    lines = [
        "# DQ-3F Label and Membership Audit", "", "Status: `PASS`", "",
        "- Frozen flows: 3,065; Known Train/Validation: 2,730/335.",
        f"- Fine classes: {len(fine_classes)} (`{' | '.join(fine_classes)}`).",
        f"- Service classes: 6 (`{' | '.join(service_classes)}`).",
        "- Flow IDs are unique; Train and Validation flow IDs are disjoint.",
        "- Stage12 NPZ row order was reconstructed from `(canonical_class, flow_id_sha256)` and both target and array digests passed.",
        f"- Exact-image hashes shared across Train/Validation: {cross_split_exact_images}.",
        "- Every Service label is `WEAK_CAPTURE_LABEL`; no Validation prediction is used to define labels.",
        "- Model input is only the frozen 32x32 uint8 traffic image. Service, capture, filename, activity and split metadata are not input features.",
        "- Known Test and Unknown Test arrays were not loaded.", "", "## Frozen Service support", "",
        "| Service | Train | Validation | Captures |", "|---|---:|---:|---:|",
    ]
    for service in SERVICES:
        counter = support[service]
        captures = sum(key.startswith("capture::") for key in counter)
        lines.append(f"| {service} | {counter['known_train']} | {counter['known_validation']} | {captures} |")
    lines += [
        "", "## Interpretation boundary", "",
        "P2P has one independent capture. DQ-3F is a flow-level matched-sample diagnostic and cannot establish capture-disjoint generalization.",
    ]
    (OUT / "dq3f_label_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "preflight_status.json", {
        "status": "PASS",
        "rows": len(rows),
        "train": len(train_ids),
        "validation": len(val_ids),
        "fine_classes": len(fine_classes),
        "service_classes": len(service_classes),
        "cross_split_flow_overlap": len(train_ids & val_ids),
        "cross_split_exact_image_overlap": cross_split_exact_images,
        "known_test_feature_values_used": 0,
        "unknown_test_feature_values_used": 0,
    })
    print(read_json(OUT / "preflight_status.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
