#!/usr/bin/env python3
"""Freeze the six Service LOSO protocols before any Stage 16S training."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

from stage16s_common import (
    CONFIG, DQ3F, MANIFEST, METHODS, OUT, SEEDS, SERVICES, array_hash, base_splits,
    canonical_hash, protected_hashes, protocol_id, read_json, sha256_file, write_csv,
    write_json,
)


SPLIT_SEED = 16001


def deterministic_validation_test_split(base: dict) -> dict[str, str]:
    """Split old Known Validation 1:1, keeping identical byte images together."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (image, meta) in enumerate(zip(base["data"], base["metadata"])):
        grouped[(meta["service_label"], array_hash(image))].append(index)
    assignment: dict[str, str] = {}
    for service in SERVICES:
        groups = [(digest, indices) for (label, digest), indices in grouped.items() if label == service]
        groups.sort(key=lambda item: hashlib.sha256(f"{SPLIT_SEED}|{service}|{item[0]}".encode()).hexdigest())
        target = sum(len(indices) for _, indices in groups) // 2
        validation_count = 0
        for digest, indices in groups:
            role = "known_validation" if validation_count < target else "known_test"
            validation_count += len(indices) if role == "known_validation" else 0
            for index in indices:
                assignment[base["metadata"][index]["flow_id"]] = role
        counts = Counter(
            assignment[base["metadata"][index]["flow_id"]]
            for _, indices in groups for index in indices
        )
        if not counts["known_validation"] or not counts["known_test"]:
            raise RuntimeError(f"cannot split {service}: {counts}")
    return assignment


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {MANIFEST}")
    before = protected_hashes()
    write_json(OUT / "protected_asset_hashes_before.json", before)
    splits = base_splits()
    train = splits["known_train"]
    old_validation = splits["known_validation"]
    val_test = deterministic_validation_test_split(old_validation)

    records = []
    base_records = []
    for base_role, split in splits.items():
        for index, (image, meta) in enumerate(zip(split["data"], split["metadata"])):
            base_records.append({
                "base_role": base_role,
                "base_row_index": index,
                "image_sha256": array_hash(image),
                "flow_id": meta["flow_id"],
                "service_label": meta["service_label"],
                "application_label": meta["application_label"],
                "capture_id": meta["capture_id"],
                "source_file": meta["source_file"],
            })
    if len(base_records) != 3065 or len({row["flow_id"] for row in base_records}) != 3065:
        raise RuntimeError("base population identity failure")

    protocol_summary = []
    audit_rows = []
    for unknown_service in SERVICES:
        pid = protocol_id(unknown_service)
        classes = sorted(set(SERVICES) - {unknown_service})
        class_to_local = {name: index for index, name in enumerate(classes)}
        role_counters = Counter()
        role_rows: dict[str, list[dict]] = defaultdict(list)
        for base in base_records:
            if base["service_label"] == unknown_service:
                role = "unknown_test"
                class_role = "unknown"
                local_label = -1
            elif base["base_role"] == "known_train":
                role = "known_train"
                class_role = "known"
                local_label = class_to_local[base["service_label"]]
            else:
                role = val_test[base["flow_id"]]
                class_role = "known"
                local_label = class_to_local[base["service_label"]]
            row = {
                "protocol_id": pid,
                "unknown_service": unknown_service,
                "known_services": "|".join(classes),
                "role": role,
                "class_role": class_role,
                "flow_id": base["flow_id"],
                "service_label": base["service_label"],
                "service_label_status": "WEAK_CAPTURE_LABEL",
                "local_label": local_label,
                "application_label": base["application_label"],
                "capture_id": base["capture_id"],
                "source_file": base["source_file"],
                "base_role": base["base_role"],
                "base_row_index": base["base_row_index"],
                "image_sha256": base["image_sha256"],
                "role_row_index": 0,
                "protocol_scope": "FLOW_LEVEL_WEAK_CAPTURE_LABEL",
                "protocol_limitation": "PROTOCOL_LIMITED_SINGLE_CAPTURE" if unknown_service == "P2P" else "CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE",
            }
            row["role_row_index"] = len(role_rows[role])
            role_rows[role].append(row)
            role_counters[role] += 1
        for role in ("known_train", "known_validation", "known_test", "unknown_test"):
            records.extend(role_rows[role])
        if sum(role_counters.values()) != 3065:
            raise RuntimeError(f"{pid}: population not conserved")

        train_services = Counter(row["service_label"] for row in role_rows["known_train"])
        val_services = Counter(row["service_label"] for row in role_rows["known_validation"])
        test_services = Counter(row["service_label"] for row in role_rows["known_test"])
        if set(train_services) != set(classes) or set(val_services) != set(classes) or set(test_services) != set(classes):
            raise RuntimeError(f"{pid}: missing Known Service in a role")
        if min(train_services.values()) < 10:
            raise RuntimeError(f"{pid}: kNN-10 support failure")
        role_sets = {role: {row["flow_id"] for row in values} for role, values in role_rows.items()}
        if sum(len(values) for values in role_sets.values()) != len(set().union(*role_sets.values())):
            raise RuntimeError(f"{pid}: flow overlap across roles")
        image_sets = {role: {row["image_sha256"] for row in values} for role, values in role_rows.items()}
        cross_image_pairs = []
        roles = list(image_sets)
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1:]:
                overlap = image_sets[left] & image_sets[right]
                if overlap:
                    cross_image_pairs.append(f"{left}:{right}:{len(overlap)}")
        if cross_image_pairs:
            raise RuntimeError(f"{pid}: exact-image overlap {cross_image_pairs}")
        protocol_summary.append({
            "protocol_id": pid,
            "unknown_service": unknown_service,
            "known_services": classes,
            "counts": dict(role_counters),
            "minimum_known_train_class_support": min(train_services.values()),
            "minimum_known_validation_class_support": min(val_services.values()),
            "minimum_known_test_class_support": min(test_services.values()),
            "unknown_capture_count": len({row["capture_id"] for row in role_rows["unknown_test"]}),
            "status": "PROTOCOL_LIMITED" if unknown_service == "P2P" else "FEASIBLE_FLOW_LEVEL",
        })
        for role, values in role_rows.items():
            by_service = Counter(row["service_label"] for row in values)
            for service, count in sorted(by_service.items()):
                audit_rows.append({
                    "protocol_id": pid,
                    "unknown_service": unknown_service,
                    "role": role,
                    "service": service,
                    "flows": count,
                    "captures": len({row["capture_id"] for row in values if row["service_label"] == service}),
                    "flow_id_unique": len({row["flow_id"] for row in values if row["service_label"] == service}) == count,
                    "exact_image_cross_role_overlap": 0,
                    "unknown_used_for_fit": 0,
                })

    fields = [
        "protocol_id", "unknown_service", "known_services", "role", "class_role", "flow_id",
        "service_label", "service_label_status", "local_label", "application_label", "capture_id",
        "source_file", "base_role", "base_row_index", "image_sha256", "role_row_index",
        "protocol_scope", "protocol_limitation",
    ]
    write_csv(MANIFEST, records, fields)
    write_csv(OUT / "known_unknown_membership_audit.csv", audit_rows)

    training = read_json(DQ3F / "dq3f_training_configs.json")["training"]
    config = {
        "status": "FROZEN_BEFORE_TRAINING",
        "protocol_version": "stage16s_loso_service_v1",
        "protocol_split_seed": SPLIT_SEED,
        "protocol_definition": "old Known Train retained as Known Train; old Known Validation split 1:1 into new Known Validation/Test by deterministic exact-image groups; held-out Service uses all 3065-pool rows as Unknown Test",
        "services": list(SERVICES),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "training": training,
        "od_native": {
            "score": "min_y KL[N(mu,diag(exp(clamped_logvar))) || N(prototype_y,I)]",
            "score_direction": "larger_is_more_unknown",
            "known_classifier": "argmin_y 0.5*squared_Euclidean(mu, learned_prototype_y)",
        },
        "des_v0": {"score": "min_y squared_Euclidean(mu, Known-Train empirical centroid_y)"},
        "des_v1": {"k": 10, "global_weight": 0.5, "local_weight": 0.5, "normalization": "Known Validation median/MAD", "epsilon": 1e-12},
        "h1": {"definition": "max(ECDF_KnownVal(OD), ECDF_KnownVal(DES-v0))", "ecdf_ties": "searchsorted side=right"},
        "threshold": {"split": "Known Validation only", "quantile": 0.95, "numpy_method": "higher", "unknown_if": "score >= threshold"},
        "checkpoint_selection": "Known Validation accuracy only",
        "known_test_feature_values_used_before_freeze": 0,
        "unknown_test_feature_values_used_before_freeze": 0,
        "capture_generalization": "NOT_IDENTIFIABLE",
        "protocols": protocol_summary,
        "manifest_sha256": sha256_file(MANIFEST),
        "manifest_canonical_membership_sha256": canonical_hash(
            f"{row['protocol_id']}|{row['role']}|{row['flow_id']}|{row['service_label']}" for row in records
        ),
    }
    write_json(CONFIG, config)

    method_doc = """# Method identity and lineage

| Method | Shared encoder/classifier | Frozen score implementation |
|---|---|---|
| OD-Native | Per-protocol corrected Open-Detect checkpoint | minimum learned-prototype KL, including clamped posterior variance |
| DES-v0 | Same checkpoint and native Known classifier | minimum squared Euclidean distance from posterior mean to Known-Train empirical class centroid |
| DES-v1 | Same checkpoint and native Known classifier | 0.5 global centroid z-score + 0.5 predicted-centroid-class kNN-10 z-score; median/MAD fit on Known Validation |
| H1 | Same checkpoint and native Known classifier | max of Known-Val empirical percentiles for OD-Native and DES-v0 |

Code lineage is reused from Stage 14D (`stage14d_common.py`,
`evaluate_protocol.py`) and Stage 15B (`run_stage15b.py`). No detector formula,
k, fusion weight, score direction, or threshold rule was searched or changed.
DQ-3F establishes that the task model is corrected Open-Detect; Stage 16S does
not relabel that encoder as a new method.
"""
    (OUT / "method_identity_and_lineage.md").write_text(method_doc, encoding="utf-8")
    reuse_doc = f"""# Existing implementation reuse audit

- Corrected Open-Detect trainer: `{sha256_file(DQ3F / 'scripts' / 'train_dq3f.py')}`
- Corrected model: `{sha256_file(DQ3F.parents[2] / 'Open-Detect' / 'reproduction' / 'corrected_model.py')}`
- Stage 14D DES implementation: `{sha256_file(OUT.parent / 'stage14d_vnat_frozen_open_set_evaluation' / 'scripts' / 'stage14d_common.py')}`
- Stage 15B H1 implementation: `{sha256_file(OUT.parent / 'stage15b_known_only_hybrid_detector' / 'scripts' / 'run_stage15b.py')}`

The five-Known-Service model is newly trained from random initialization for
each LOSO protocol/seed because every existing six-Service checkpoint has seen
the held-out Service. No six-Service checkpoint is used as initialization.
"""
    (OUT / "existing_implementation_reuse_audit.md").write_text(reuse_doc, encoding="utf-8")
    feasibility_lines = [
        "# Service-level open-set feasibility", "",
        "All six LOSO rounds are executable as **flow-level weak-label diagnostics**.", "",
        "| Unknown Service | Unknown flows | Unknown captures | Status |", "|---|---:|---:|---|",
    ]
    for item in protocol_summary:
        feasibility_lines.append(
            f"| {item['unknown_service']} | {item['counts']['unknown_test']} | {item['unknown_capture_count']} | {item['status']} |"
        )
    feasibility_lines.extend([
        "", "P2P has one capture and is retained only as `PROTOCOL_LIMITED`; it cannot support a cross-capture claim.",
        "All rounds share captures across flow-level roles, so `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE` applies globally.",
    ])
    (OUT / "service_open_set_feasibility.md").write_text("\n".join(feasibility_lines) + "\n", encoding="utf-8")
    write_json(OUT / "preflight_verification.json", {
        "status": "PASS",
        "protocols": 6,
        "manifest_rows": len(records),
        "all_services_feasible_flow_level": True,
        "p2p": "PROTOCOL_LIMITED_SINGLE_CAPTURE",
        "exact_image_cross_role_overlap": 0,
        "flow_id_cross_role_overlap": 0,
        "unknown_fit_samples": 0,
        "known_test_used_for_selection": 0,
        "manifest_sha256": sha256_file(MANIFEST),
    })


if __name__ == "__main__":
    main()
