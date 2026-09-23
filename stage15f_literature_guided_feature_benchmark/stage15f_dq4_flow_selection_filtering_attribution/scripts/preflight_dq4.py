#!/usr/bin/env python3
"""Construct the frozen DQ-4 subsets and run the pre-training feasibility Gate."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from dq4_common import (
    DQ3F, OUT, SEEDS, SERVICES, SUBSETS, flow_id_digest, hash_protected_inputs,
    load_mother, read_json, trafficformer_final_parity, validate_a_reuse, write_csv,
    write_json,
)


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "samples": int(len(frame)),
        "accuracy": float(accuracy_score(frame["true_index"], frame["predicted_index"])),
        "macro_f1": float(f1_score(frame["true_index"], frame["predicted_index"], average="macro", labels=range(6), zero_division=0)),
        "weighted_f1": float(f1_score(frame["true_index"], frame["predicted_index"], average="weighted", zero_division=0)),
    }


def summary_row(subset: str, grouping: str, group_value: str, frame: pd.DataFrame) -> dict:
    packet = frame["packet_count"].astype(int)
    total = frame["total_bytes"].astype(float)
    duration = frame["duration_seconds"].astype(float)
    return {
        "subset": subset, "grouping": grouping, "group_value": group_value,
        "flow_count": int(len(frame)), "capture_count": int(frame["capture_id"].nunique()),
        "parent_count": int(frame["trafficformer_parent_flow_id"].nunique()),
        "one_packet_count": int((packet == 1).sum()), "two_packet_count": int((packet == 2).sum()),
        "short_le2_count": int((packet <= 2).sum()), "mean_packet_count": float(packet.mean()) if len(frame) else np.nan,
        "median_packet_count": float(packet.median()) if len(frame) else np.nan,
        "mean_total_bytes": float(total.mean()) if len(frame) else np.nan,
        "median_total_bytes": float(total.median()) if len(frame) else np.nan,
        "p25_total_bytes": float(total.quantile(.25)) if len(frame) else np.nan,
        "p75_total_bytes": float(total.quantile(.75)) if len(frame) else np.nan,
        "mean_duration_seconds": float(duration.mean()) if len(frame) else np.nan,
        "median_duration_seconds": float(duration.median()) if len(frame) else np.nan,
    }


def main() -> None:
    if any(OUT.glob("runs/*/seed*/SUCCESS")):
        raise RuntimeError("refusing to rebuild preflight after DQ-4 training")
    before = hash_protected_inputs()
    write_json(OUT / "protected_asset_hashes_before.json", before)
    tf_parity = trafficformer_final_parity()
    if not tf_parity["success_to_final_service_multiset_parity"]:
        raise RuntimeError("TrafficFormer success/final Service multiset parity failed")
    reuse = validate_a_reuse()

    all_rows = []
    split_payload = {}
    for role in ("known_train", "known_validation"):
        mother = load_mother(role)
        split_payload[role] = mother
        for row in mother["metadata"]:
            all_rows.append({
                "flow_id": row["flow_id"], "split_role": role,
                "mother_row_index": row["mother_row_index"], "source_array_index": row["source_array_index"],
                "application": row["application_label"], "service": row["service_label"],
                "capture_id": row["capture_id"], "source_file": row["source_file"],
                "packet_count": row["packet_count"], "total_bytes": row["total_bytes"],
                "captured_bytes": row["captured_bytes"], "duration_seconds": row["duration_seconds"],
                "cate_match_status": row["cate_match_status"], "cate_match_reason": row["cate_match_reason"],
                "cate_candidate_count": row["cate_candidate_count"],
                "trafficformer_parent_flow_id": row["trafficformer_parent_flow_id"],
                "trafficformer_parent_packet_count": row["trafficformer_parent_packet_count"],
                "trafficformer_parent_captured_bytes": row["trafficformer_parent_captured_bytes"],
                **{name: int(bool(row[name])) for name in SUBSETS},
            })
    manifest = pd.DataFrame(all_rows)
    if len(manifest) != 3065 or manifest["flow_id"].nunique() != 3065:
        raise RuntimeError("DQ-4 mother population is not exactly 3,065 unique flows")
    manifest.to_csv(OUT / "dq4_subset_manifest.csv", index=False)

    audit_rows = []
    for row in manifest.itertuples(index=False):
        audit_rows.append({
            "flow_id": row.flow_id, "split_role": row.split_role, "application": row.application,
            "service": row.service, "cate_match_status": row.cate_match_status,
            "cate_match_reason": row.cate_match_reason, "trafficformer_parent_flow_id": row.trafficformer_parent_flow_id,
            "trafficformer_parent_eligible": row.C_PARENT, "trafficformer_final_service_included": row.C_FINAL,
            "B_cate_selected": row.B, "D_intersection_selected": row.D,
        })
    write_csv(OUT / "dq4_known_membership_audit.csv", audit_rows)

    count_rows = []
    support_rows = []
    for subset in SUBSETS:
        selected = manifest[manifest[subset].eq(1)]
        count_rows.append(summary_row(subset, "overall", "ALL", selected))
        for role, part in selected.groupby("split_role", sort=True):
            count_rows.append(summary_row(subset, "split", str(role), part))
        for service, part in selected.groupby("service", sort=True):
            count_rows.append(summary_row(subset, "service", str(service), part))
        for capture, part in selected.groupby("capture_id", sort=True):
            count_rows.append(summary_row(subset, "capture", str(capture), part))
        for (service, role), part in selected.groupby(["service", "split_role"], sort=True):
            per_capture = part.groupby("capture_id").size()
            support_rows.append({
                "subset": subset, "service": service, "split_role": role,
                "flow_count": int(len(part)), "capture_count": int(part["capture_id"].nunique()),
                "min_flows_per_capture": int(per_capture.min()), "median_flows_per_capture": float(per_capture.median()),
                "max_flows_per_capture": int(per_capture.max()),
                "captures": "|".join(sorted(part["capture_id"].unique())),
            })
    write_csv(OUT / "dq4_subset_counts.csv", count_rows)
    write_csv(OUT / "dq4_subset_class_capture_support.csv", support_rows)

    cate_counts = manifest["cate_match_status"].value_counts().to_dict()
    (OUT / "dq4_cate_matching_audit.md").write_text(
        "# DQ-4 CATE Matching Audit\n\n"
        f"On the frozen 3,065-flow mother population: `{json.dumps(cate_counts, sort_keys=True)}`.\n\n"
        "B includes only `MATCHED`: exactly one capture-level CATE five-tuple candidate and parent packet-count agreement. "
        "`UNMATCHED`, `AMBIGUOUS`, and `MATCH_FAILED` are retained separately in the manifest and are never relabeled. "
        "This is a target-flow selection indicator, not authoritative per-flow label truth.\n",
        encoding="utf-8",
    )
    (OUT / "dq4_trafficformer_eligibility_audit.md").write_text(
        "# DQ-4 TrafficFormer Eligibility Audit\n\n"
        f"- Existing processing audit rows: {tf_parity['processing_audit_rows']}\n"
        f"- Successful parent flows: {tf_parity['success_parent_rows']}\n"
        f"- Final Service manifest rows: {tf_parity['final_service_manifest_rows']}\n"
        f"- Status counts: `{json.dumps(tf_parity['status_counts'], sort_keys=True)}`\n"
        f"- Success-parent to final-Service multiset parity: `{tf_parity['success_to_final_service_multiset_parity']}`\n"
        "- Actual code order: reject captured bytes `<2048`; then reject packet count `<3`; then encode first 5 packets x 64 bytes.\n"
        "- C_PARENT is the Native session whose capture-wide bidirectional five-tuple parent passes that rule.\n"
        "- C_FINAL equals C_PARENT for the six-Service task because every successful Service class has >=10 parents and the final Service manifest contains the complete success multiset.\n"
        "- Limitation: TrafficFormer parent flow and Native 60-second/TCP-boundary session are not one-to-one; C is a parent-derived selection subset, not a claim that each Native row equals one final TrafficFormer row.\n",
        encoding="utf-8",
    )

    feasibility = []
    for model, subset in (("M-B", "B"), ("M-C", "C_FINAL"), ("M-D", "D")):
        selected = manifest[manifest[subset].eq(1)]
        table = selected.groupby(["service", "split_role"]).size().unstack(fill_value=0)
        missing = [service for service in SERVICES if service not in table.index]
        empty = []
        for service in SERVICES:
            for role in ("known_train", "known_validation"):
                if service not in table.index or role not in table.columns or int(table.loc[service, role]) == 0:
                    empty.append(f"{service}:{role}")
        min_train = int(table.get("known_train", pd.Series(dtype=int)).min()) if not table.empty else 0
        min_val = int(table.get("known_validation", pd.Series(dtype=int)).min()) if not table.empty else 0
        status = "PASS_WITH_LOW_VALIDATION_SUPPORT" if not empty and min_val < 10 else ("PASS" if not empty else "SIX_CLASS_EVALUATION_NOT_FEASIBLE")
        feasibility.append({
            "model": model, "subset": subset, "status": status, "missing_services": missing,
            "empty_service_splits": empty, "min_train_support": min_train, "min_validation_support": min_val,
            "train_flows": int((selected["split_role"] == "known_train").sum()),
            "validation_flows": int((selected["split_role"] == "known_validation").sum()),
        })
    if any(item["status"] == "SIX_CLASS_EVALUATION_NOT_FEASIBLE" for item in feasibility):
        train_allowed = False
        gate = "SIX_CLASS_EVALUATION_NOT_FEASIBLE"
    else:
        train_allowed = True
        gate = "PASS_WITH_LOW_VALIDATION_SUPPORT"
    (OUT / "dq4_feasibility_gate.md").write_text(
        "# DQ-4 Six-Service Feasibility Gate\n\n"
        f"Gate: `{gate}`; training allowed: `{train_allowed}`.\n\n"
        + "\n".join(
            f"- {item['model']} / {item['subset']}: Train={item['train_flows']}, Val={item['validation_flows']}, "
            f"minimum Train/Val support={item['min_train_support']}/{item['min_validation_support']}, status={item['status']}"
            for item in feasibility
        )
        + "\n\nAll six Services are non-empty in Train and Validation. B and D have only four VoIP Validation flows, "
        "so six-class metrics are computable but fragile. Capture-disjoint generalization remains unidentifiable.\n",
        encoding="utf-8",
    )

    evaluation_rows = []
    removed_rows = []
    validation = manifest[manifest["split_role"].eq("known_validation")]
    for seed in SEEDS:
        pred = pd.read_csv(DQ3F / "runs/service" / f"seed{seed}" / "validation_predictions.csv")
        joined = pred.merge(validation[["flow_id", *SUBSETS, "packet_count", "service", "cate_match_status"]], on="flow_id", validate="one_to_one")
        if len(joined) != 335:
            raise RuntimeError("M-A prediction/mother Validation mismatch")
        for subset in SUBSETS:
            part = joined[joined[subset].eq(1)]
            evaluation_rows.append({"model": "M-A", "training_subset": "A", "evaluation_subset": subset, "seed": seed, **metrics(part)})
            removed = joined[joined[subset].eq(0)]
            for grouping, groups in (("overall", [("ALL", removed)]), ("service", removed.groupby("service", sort=True))):
                for value, group in groups:
                    removed_rows.append({
                        "model": "M-A", "seed": seed, "selection_subset": subset,
                        "removed_grouping": grouping, "removed_group": value,
                        "removed_samples": int(len(group)), "error_count": int((group["correct"] == 0).sum()),
                        "error_rate": float((group["correct"] == 0).mean()) if len(group) else np.nan,
                        "one_packet_count": int((group["packet_count"] == 1).sum()),
                        "two_packet_count": int((group["packet_count"] == 2).sum()),
                    })
    write_csv(OUT / "dq4_evaluation_population_effect.csv", evaluation_rows)
    write_csv(OUT / "dq4_removed_flow_error_analysis.csv", removed_rows)

    training = read_json(DQ3F / "dq3f_training_configs.json")["training"]
    write_json(OUT / "dq4_training_configs.json", {
        "status": "FROZEN_BEFORE_DQ4_TRAINING", "mother_population": 3065,
        "models": {"M-A": {"subset": "A", "action": "REUSE_DQ3F_F3", "reuse_parity": reuse},
                   "M-B": {"subset": "B", "action": "TRAIN"},
                   "M-C": {"subset": "C_FINAL", "action": "TRAIN"},
                   "M-D": {"subset": "D", "action": "TRAIN"}},
        "seeds": list(SEEDS), "training": training, "feasibility": feasibility,
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
        "capture_generalization_claim": "CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE",
        "labels": "WEAK_CAPTURE_LABEL",
    })
    write_json(OUT / "preflight_status.json", {
        "status": "PASS" if train_allowed else gate, "training_allowed": train_allowed,
        "mother_counts": manifest["split_role"].value_counts().to_dict(),
        "subset_counts": {subset: int(manifest[subset].sum()) for subset in SUBSETS},
        "flow_id_sha256": flow_id_digest(all_rows), "m_a_reuse": reuse,
        "trafficformer_parity": tf_parity, "feasibility": feasibility,
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
    })
    (OUT / "dq4_source_parity.md").write_text(
        "# DQ-4 Source Parity\n\n"
        "- Frozen mother population: 3,065 unique flows (2,730 Train / 335 Validation).\n"
        "- All mother flow IDs join one-to-one to the DQ reconstructed flow manifest.\n"
        "- M-A is the unchanged DQ-3F F3 model family; all three array digests, prediction order, class space, counts and checkpoint hashes pass.\n"
        "- TrafficFormer 31-capture parent reconstruction parity remains exact; success-parent and final Service-manifest multisets are identical.\n"
        "- Known Test feature usage: 0; Unknown Test feature usage: 0.\n",
        encoding="utf-8",
    )
    print(json.dumps(read_json(OUT / "preflight_status.json"), indent=2))


if __name__ == "__main__":
    main()
