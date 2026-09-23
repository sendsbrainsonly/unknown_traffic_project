#!/usr/bin/env python3
"""Finalize the Stage 15F-DQ DQ-0--DQ-3 interim evidence bundle.

This script never opens Known Test or Unknown Test feature files.  It only
summarizes the audit CSV/JSON products already created from the shared 31
ISCX-VPN captures and frozen Known Train/Validation predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def load_hash_status() -> tuple[str, int, list[str]]:
    before_path = OUT / "frozen_asset_hashes_before.json"
    after_path = OUT / "frozen_asset_hashes_after.json"
    if not after_path.exists():
        return "PENDING", 0, []
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    bfiles = before["files"]
    afiles = after["files"]
    changed = sorted(
        set(bfiles).symmetric_difference(afiles)
        | {p for p in set(bfiles) & set(afiles) if bfiles[p] != afiles[p]}
    )
    return ("PASS" if not changed else "FAIL"), len(bfiles), changed


def main() -> None:
    pcaps = read_csv("cross_project_pcap_manifest.csv")
    eligibility = read_csv("sample_eligibility_summary.csv")
    predictions = read_csv("known_validation_shared31_predictions.csv")
    short = read_csv("short_flow_error_analysis.csv")
    cate = read_csv("cate_matching_analysis.csv")
    summary = json.loads((OUT / "dq0_dq2_summary.json").read_text(encoding="utf-8"))
    mapping = json.loads((OUT / "label_mapping.json").read_text(encoding="utf-8"))
    summary["gate"]["dq3_training_status"] = "NOT_RUN_GATE_FAILED_MAPPING_AMBIGUITY"
    summary["gate"]["dq3_feasibility_decision"] = "BLOCKED_MAPPING_AMBIGUITY"
    (OUT / "dq0_dq2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    overall_sets = {
        row["set_id"]: row
        for row in eligibility
        if row["grouping"] == "overall" and row["group_value"] == "ALL"
    }
    packet_rows = {
        row["group_value"]: row
        for row in short
        if row["dimension"] == "packet_group"
    }
    cate_val = {
        row["group_value"]: row
        for row in cate
        if row["scope"] == "KNOWN_VALIDATION_SHARED31"
        and row["dimensions"] == "cate_match_status"
    }

    shared = [row for row in pcaps if row["shared_all_three"].lower() == "true"]
    pcap_counts = {
        "native": sum(row["native_included"].lower() == "true" for row in pcaps),
        "tfe": sum(row["tfe_gnn_included"].lower() == "true" for row in pcaps),
        "tf": sum(row["trafficformer_included"].lower() == "true" for row in pcaps),
        "shared": len(shared),
    }
    flow_counts = {
        "native_all": sum(int(row["native_selected_flow_count"]) for row in pcaps),
        "native_shared": int(overall_sets["A"]["flow_count"]),
        "tfe_raw": sum(int(row["tfe_gnn_cate_flow_count"]) for row in pcaps),
        "tfe_valid": sum(int(row["tfe_gnn_paper_text_valid_flow_count"]) for row in pcaps),
        "tf_parent": sum(int(row["trafficformer_parent_flow_count"]) for row in pcaps),
        "tf_eligible": sum(int(row["trafficformer_eligible_flow_count"]) for row in pcaps),
    }

    true_fine = [r["true_application"] for r in predictions]
    pred_fine = [r["predicted_application"] for r in predictions]
    fine_acc = accuracy_score(true_fine, pred_fine)
    fine_label_set = sorted(set(true_fine))
    fine_macro = f1_score(
        true_fine,
        pred_fine,
        labels=fine_label_set,
        average="macro",
        zero_division=0,
    )
    fine_weighted = f1_score(true_fine, pred_fine, average="weighted", zero_division=0)
    mappable = [
        r for r in predictions
        if r["mapped_true_service"] != "UNMAPPED"
        and r["mapped_predicted_service"] != "UNMAPPED"
    ]
    true_coarse = [r["mapped_true_service"] for r in mappable]
    pred_coarse = [r["mapped_predicted_service"] for r in mappable]
    coarse_acc = accuracy_score(true_coarse, pred_coarse)
    coarse_macro = f1_score(true_coarse, pred_coarse, average="macro", zero_division=0)
    coarse_weighted = f1_score(true_coarse, pred_coarse, average="weighted", zero_division=0)

    hash_status, hashed_files, changed_hashes = load_hash_status()

    controlled_rows = [
        {"stage": "DQ-0", "question": "Cross-project PCAP/flow reconciliation", "data_scope": "ISCX-VPN shared 31 VPN PCAPs", "labels": "capture application and service metadata", "split": "none", "test_values_used": 0, "status": "PASS", "gate_or_reason": "31/31 TrafficFormer parent-flow multisets reproduced exactly"},
        {"stage": "DQ-1", "question": "CATE and TrafficFormer eligibility reconciliation", "data_scope": "Native A=4224 shared-PCAP sessions", "labels": "unchanged", "split": "none", "test_values_used": 0, "status": "PASS_DESCRIPTIVE", "gate_or_reason": "CATE is a selection indicator, not authoritative truth"},
        {"stage": "DQ-2", "question": "Frozen Native validation error attribution", "data_scope": "335 shared31 Known Validation samples", "labels": "Fine application plus fixed auditable service mapping", "split": "frozen medium_seed2022 Known Validation", "test_values_used": 0, "status": "PASS_DIAGNOSTIC", "gate_or_reason": "No retraining; all results descriptive"},
        {"stage": "DQ-3", "question": "Fine/coarse same-flow feasibility", "data_scope": "shared31 Known Train/Validation", "labels": "Fine application vs capture-level service", "split": "same frozen membership", "test_values_used": 0, "status": "BLOCKED_MAPPING_AMBIGUITY", "gate_or_reason": "Facebook prediction maps to Chat or VoIP; full F2 is not identifiable without a new rule"},
        {"stage": "DQ-4", "question": "Filtering effects", "data_scope": "NOT_RUN", "labels": "NOT_RUN", "split": "NOT_RUN", "test_values_used": 0, "status": "NOT_RUN", "gate_or_reason": "Stopped after DQ-3 interim gate"},
        {"stage": "DQ-5", "question": "VPN/nonVPN domain mixing", "data_scope": "NOT_RUN", "labels": "NOT_RUN", "split": "NOT_RUN", "test_values_used": 0, "status": "NOT_RUN", "gate_or_reason": "Stopped after DQ-3 interim gate"},
        {"stage": "DQ-6", "question": "Random vs capture-group split", "data_scope": "NOT_RUN", "labels": "NOT_RUN", "split": "NOT_RUN", "test_values_used": 0, "status": "NOT_RUN", "gate_or_reason": "Stopped after DQ-3 interim gate"},
        {"stage": "DQ-7", "question": "Fair matched-model comparison", "data_scope": "NOT_RUN", "labels": "NOT_RUN", "split": "NOT_RUN", "test_values_used": 0, "status": "NOT_RUN", "gate_or_reason": "Prerequisite DQ-3 gate did not pass"},
    ]
    write_csv("controlled_protocol_matrix.csv", list(controlled_rows[0]), controlled_rows)

    paper_rows = [
        {"track": "P", "method": "Native Open-Detect", "pcap_scope": "137 ISCX-VPN captures (VPN+nonVPN)", "flow_definition": "bidirectional IPv4 TCP/UDP session; 60s idle; TCP SYN/FIN/RST boundaries", "flow_count": flow_counts["native_all"], "filter": "stable-hash cap 2000/application; no packet/byte minimum", "label_granularity": "Application", "split": "frozen Stage12 flow-level membership", "result_status": "EXISTING_PROTOCOL_CHARACTERIZED_NOT_RERUN", "comparable_rank": "NO"},
        {"track": "P", "method": "TFE-GNN", "pcap_scope": "31 VPN PCAPs", "flow_definition": "CATE TCP target five-tuples", "flow_count": flow_counts["tfe_valid"], "filter": "paper-text empty-payload and >10000-packet exclusions", "label_granularity": "6 Service categories", "split": "project paper-protocol manifest; not Native split", "result_status": "EXISTING_PROTOCOL_CHARACTERIZED_NOT_RERUN", "comparable_rank": "NO"},
        {"track": "P", "method": "TrafficFormer", "pcap_scope": "31 VPN PCAPs", "flow_definition": "capture-wide bidirectional IPv4 TCP/UDP five-tuple", "flow_count": flow_counts["tf_eligible"], "filter": ">=2048 captured frame bytes then >=3 packets", "label_granularity": "Service/Application project tasks", "split": "project protocol; not Native split", "result_status": "EXISTING_PROTOCOL_CHARACTERIZED_NOT_RERUN", "comparable_rank": "NO"},
        {"track": "P", "method": "YaTC", "pcap_scope": "author/reproduction MFR assets; not proven same flow population", "flow_definition": "MFR image construction", "flow_count": "NOT_ALIGNED", "filter": "model-specific MFR preprocessing", "label_granularity": "coarse author task", "split": "not Native split", "result_status": "NOT_RUN_IN_DQ0_DQ3", "comparable_rank": "NO"},
    ]
    write_csv("paper_protocol_reproduction.csv", list(paper_rows[0]), paper_rows)

    fair_rows = [
        {"track": "F", "experiment": "F1", "model": "Native Open-Detect", "population": "shared31 Known Validation subset", "samples": len(predictions), "accuracy": fine_acc, "macro_f1": fine_macro, "weighted_f1": fine_weighted, "mapping_coverage": 1.0, "status": "EXISTING_FROZEN_PREDICTIONS_DIAGNOSTIC_ONLY", "warning": "Not a newly trained shared31-only F1"},
        {"track": "F", "experiment": "F2", "model": "Native Open-Detect mapped output", "population": "mappable subset of shared31 Known Validation", "samples": len(mappable), "accuracy": coarse_acc, "macro_f1": coarse_macro, "weighted_f1": coarse_weighted, "mapping_coverage": len(mappable) / len(predictions), "status": "PARTIAL_DIAGNOSTIC_ONLY", "warning": "14/335 rows excluded because true or predicted application is not deterministically mappable"},
        {"track": "F", "experiment": "F3", "model": "Native Service classifier", "population": "same-flow Known Train/Validation", "samples": "NOT_RUN", "accuracy": "NOT_RUN", "macro_f1": "NOT_RUN", "weighted_f1": "NOT_RUN", "mapping_coverage": "NOT_RUN", "status": "NOT_RUN_GATE_FAILED", "warning": "Full same-population F2/F3 comparison blocked by application-to-service ambiguity"},
        {"track": "F", "experiment": "DQ-7", "model": "TFE-GNN/TrafficFormer matched protocol", "population": "NOT_RUN", "samples": "NOT_RUN", "accuracy": "NOT_RUN", "macro_f1": "NOT_RUN", "weighted_f1": "NOT_RUN", "mapping_coverage": "NOT_RUN", "status": "NOT_RUN", "warning": "DQ-7 not authorized and DQ-3 prerequisite gate failed"},
    ]
    write_csv("fair_matched_protocol_results.csv", list(fair_rows[0]), fair_rows)

    not_run_fields = ["stage", "status", "reason", "known_test_samples_used", "unknown_test_samples_used"]
    for filename, stage, reason in [
        ("filter_effect_analysis.csv", "DQ-4", "NOT_RUN_AFTER_DQ3_GATE"),
        ("domain_mixing_analysis.csv", "DQ-5", "NOT_RUN_AFTER_DQ3_GATE"),
        ("split_sensitivity_analysis.csv", "DQ-6", "NOT_RUN_AFTER_DQ3_GATE"),
    ]:
        write_csv(filename, not_run_fields, [{"stage": stage, "status": "NOT_RUN", "reason": reason, "known_test_samples_used": 0, "unknown_test_samples_used": 0}])

    ambiguous = {
        app: entry["possible_capture_level_services"]
        for app, entry in mapping["application_mapping"].items()
        if entry["status"] != "MAPPED"
    }
    granularity = f"""# DQ-3 Granularity Feasibility and Preregistration

## Gate outcome

**BLOCKED_MAPPING_AMBIGUITY.** DQ-3 was executed as a feasibility gate; no new F1/F3 model was trained.

The capture-level service label is available, and all six services have non-zero Known Train/Validation support. However, application predictions cannot be deterministically mapped to service for the complete population: `{json.dumps(ambiguous, ensure_ascii=False)}`. In the active `medium_seed2022` Known set, Facebook is present and maps to both Chat and VoIP. A majority rule, a capture-conditioned prediction map, or use of the true capture to map a predicted Facebook label would each add an unregistered decision rule.

## Same-flow support

| Task label | Train support | Validation support |
|---|---|---|
| Fine applications | `{json.dumps(summary['gate']['support']['fine_train'], ensure_ascii=False)}` | `{json.dumps(summary['gate']['support']['fine_validation'], ensure_ascii=False)}` |
| Coarse services | `{json.dumps(summary['gate']['support']['service_train'], ensure_ascii=False)}` | `{json.dumps(summary['gate']['support']['service_validation'], ensure_ascii=False)}` |

Support count is not the blocking issue. Mapping identifiability is.

## Frozen-prediction diagnostics (not F3)

- F1-like existing frozen shared31 subset: N={len(predictions)}, Accuracy={fine_acc:.6f}, Macro-F1={fine_macro:.6f}, Weighted-F1={fine_weighted:.6f}.
- Partial F2 on the N={len(mappable)} deterministically mappable rows: Accuracy={coarse_acc:.6f}, Macro-F1={coarse_macro:.6f}, Weighted-F1={coarse_weighted:.6f}.
- Complete F2 coverage: {pct(len(mappable) / len(predictions))}; 14 rows are outside the deterministic mapping domain.
- Of 66 Fine errors, 22 become Coarse-correct, 39 remain Coarse-wrong, and 5 are unmappable. Thus 33.33% of all Fine errors, or 36.07% of mappable Fine errors, are within-service errors.

These numbers are hierarchical evaluation of an existing Fine model. They are not evidence for the performance of a separately trained Service classifier.

## Preregistered protocol that would be required after resolving the semantic definition

1. Freeze a service-native label definition at capture level before training.
2. Use exactly the same shared31 flow IDs and existing Known Train/Validation membership for Fine and Service tasks.
3. Do not use Known Test or Unknown Test for mapping, normalization, checkpoint selection, or model choice.
4. Specify how multi-service applications are represented. A valid option must preserve service information in the prediction label space; a single application label such as Facebook is insufficient for full F2.
5. Re-run F1 and F3 with identical architecture/training budget, then compare complete-population F2 and F3.
6. Before promoting Service to a formal task, pass DQ-6 capture-group feasibility and redefine open-set semantic isolation at service level.

## Decision

- `COARSE_TASK_SUPPORTED`: **not established**.
- `FINE_TASK_SUPPORTED`: **not adjudicated in DQ-3**.
- New training started: **NO**.
- DQ-4 through DQ-7 started: **NO**.
"""
    write_text("granularity_feasibility.md", granularity)

    flow_audit = f"""# DQ-0/DQ-1 Flow Matching Audit

## PCAP scope

- Native inventory: {pcap_counts['native']} PCAP/PCAPNG files.
- TFE-GNN scope: {pcap_counts['tfe']} VPN PCAPs.
- TrafficFormer scope: {pcap_counts['tf']} VPN PCAPs.
- Exact three-way shared scope: {pcap_counts['shared']} PCAPs.
- Live SHA256 values for the shared PCAPs matched the prior verified capture audit.

Therefore, the projects do **not** use the same complete PCAP population. TFE-GNN and TrafficFormer share 31 VPN captures; Native additionally uses 106 other VPN/nonVPN captures.

## Flow definitions and counts

| Pipeline | Definition / filter | Count |
|---|---|---:|
| Native, all 137 captures | bidirectional IPv4 TCP/UDP sessions; 60s idle; TCP SYN/FIN/RST boundaries; no packet/byte minimum | {flow_counts['native_all']} |
| Native A, shared31 only | same Native definition | {flow_counts['native_shared']} |
| TFE-GNN CATE raw targets | TCP target five-tuples | {flow_counts['tfe_raw']} |
| TFE-GNN paper-text valid | CATE plus empty-payload/anomalous-length exclusions | {flow_counts['tfe_valid']} |
| TrafficFormer parent flows | capture-wide bidirectional IPv4 TCP/UDP five-tuples | {flow_counts['tf_parent']} |
| TrafficFormer eligible | first `captured frame bytes >= 2048`, then `packets >= 3` | {flow_counts['tf_eligible']} |

Native session rows cannot be equated one-to-one with TrafficFormer parent flows because one capture-wide five-tuple can contain multiple Native sessions. This is why set C contains Native sessions whose own packet count is 1 or 2: eligibility belongs to their TrafficFormer parent, not the Native child session.

## Reconstruction validation

- TrafficFormer processing audit exact multiset parity: {summary['gate']['trafficformer_exact_parity_captures']}/31 captures.
- Native reconstruction explicitly requires outer IPv4 protocol TCP/UDP and ignores non-first fragments. This prevents inner TCP/UDP headers encapsulated in ICMP and fragment-carried ports from being falsely treated as outer flows.
- Two failed parser probes are preserved under `failed_attempts/`; they are diagnostic evidence and not part of the formal counts.

## CATE matching

- A (Native shared31): {overall_sets['A']['flow_count']} flows.
- B (A with unique packet-count-consistent CATE parent match): {overall_sets['B']['flow_count']} flows.
- C (A linked to an eligible TrafficFormer parent): {overall_sets['C']['flow_count']} flows.
- D (B intersect C): {overall_sets['D']['flow_count']} flows.
- CATE statuses in A: MATCHED={summary['gate']['cate_status_counts'].get('MATCHED', 0)}, UNMATCHED={summary['gate']['cate_status_counts'].get('UNMATCHED', 0)}, AMBIGUOUS={summary['gate']['cate_status_counts'].get('AMBIGUOUS', 0)}, MATCH_FAILED={summary['gate']['cate_status_counts'].get('MATCH_FAILED', 0)}.

`UNMATCHED` means only that no unique verified CATE parent relation was found. It is not interpreted as background traffic or label error.
"""
    write_text("flow_matching_audit.md", flow_audit)

    short_12_n = int(packet_rows["1"]["sample_count"]) + int(packet_rows["2"]["sample_count"])
    short_12_err = int(packet_rows["1"]["error_count"]) + int(packet_rows["2"]["error_count"])
    long_n = len(predictions) - short_12_n
    long_err = int(summary["prediction"]["fine_errors"]) - short_12_err
    report = f"""# Stage 15F-DQ Interim Performance-Gap Attribution

## Scope and decision

This interim report covers **DQ-0 through DQ-3 only** on the shared 31 ISCX-VPN VPN PCAPs. ISCXTor, DQ-4--DQ-7, new model training, Known Test, and Unknown Test are all `NOT_RUN`.

The interim gate is **STOP_RELATED_BRANCH_MAPPING_AMBIGUITY**. Flow reconstruction is reliable, but a full same-population Fine→Coarse evaluation is not identifiable because an application label can represent multiple services.

## DQ-0: the datasets and flow populations are not the same

- Native uses {pcap_counts['native']} captures and {flow_counts['native_all']} selected sessions.
- TFE-GNN and TrafficFormer each use the same 31 VPN captures, not Native's complete capture population.
- Within those 31 captures, Native has {flow_counts['native_shared']} sessions, TFE-GNN has {flow_counts['tfe_raw']} CATE target flows ({flow_counts['tfe_valid']} paper-text-valid), and TrafficFormer has {flow_counts['tf_parent']} capture-wide parent flows of which {flow_counts['tf_eligible']} are eligible.
- TrafficFormer reconstruction matched its stored processing audit exactly on all 31 captures.

These counts are not a method leaderboard: the unit of flow and filtering rules differ.

## DQ-1: what the other pipelines remove/select

| Set | Flows | Captures | 1-packet | 2-packet | <=2 packets |
|---|---:|---:|---:|---:|---:|
| A Native shared31 | {overall_sets['A']['flow_count']} | {overall_sets['A']['capture_count']} | {pct(float(overall_sets['A']['one_packet_ratio']))} | {pct(float(overall_sets['A']['two_packet_ratio']))} | {pct(float(overall_sets['A']['short_le2_ratio']))} |
| B CATE-matched | {overall_sets['B']['flow_count']} | {overall_sets['B']['capture_count']} | {pct(float(overall_sets['B']['one_packet_ratio']))} | {pct(float(overall_sets['B']['two_packet_ratio']))} | {pct(float(overall_sets['B']['short_le2_ratio']))} |
| C TrafficFormer-parent eligible | {overall_sets['C']['flow_count']} | {overall_sets['C']['capture_count']} | {pct(float(overall_sets['C']['one_packet_ratio']))} | {pct(float(overall_sets['C']['two_packet_ratio']))} | {pct(float(overall_sets['C']['short_le2_ratio']))} |
| D B intersect C | {overall_sets['D']['flow_count']} | {overall_sets['D']['capture_count']} | {pct(float(overall_sets['D']['one_packet_ratio']))} | {pct(float(overall_sets['D']['two_packet_ratio']))} | {pct(float(overall_sets['D']['short_le2_ratio']))} |

CATE matching selects {pct(int(overall_sets['B']['flow_count']) / int(overall_sets['A']['flow_count']))} of Native shared31 sessions. This is a selection difference, not evidence that the unmatched {int(overall_sets['A']['flow_count']) - int(overall_sets['B']['flow_count'])} sessions are mislabeled.

## DQ-2: frozen Native Known Validation diagnosis

- Shared31 subset: N={len(predictions)}, Accuracy={fine_acc:.6f}, Macro-F1={fine_macro:.6f}, Weighted-F1={fine_weighted:.6f}.
- Packet-group Accuracy: 1={float(packet_rows['1']['accuracy']):.4f}, 2={float(packet_rows['2']['accuracy']):.4f}, 3--7={float(packet_rows['3-7']['accuracy']):.4f}, 8--15={float(packet_rows['8-15']['accuracy']):.4f}, >=16={float(packet_rows['>=16']['accuracy']):.4f}.
- <=2-packet flows account for {short_12_err}/66={pct(short_12_err / 66)} of errors because they comprise {short_12_n}/{len(predictions)}={pct(short_12_n / len(predictions))} of samples. Their error rate is {pct(short_12_err / short_12_n)}, lower than {pct(long_err / long_n)} for >=3-packet samples. Thus this frozen subset does not support the claim that short flows are disproportionately difficult.
- MATCHED error rate={pct(float(cate_val['MATCHED']['error_rate']))}; UNMATCHED error rate={pct(float(cate_val['UNMATCHED']['error_rate']))}. The 1.00 percentage-point difference is descriptive and composition-confounded.
- Fine errors becoming Coarse-correct: 22/66={pct(22/66)} overall, 22/61={pct(22/61)} among mappable Fine errors.

The CSVs include packet group crossed with application, service, capture, and CATE status. Small cells remain descriptive; no causal label-noise claim is made.

## DQ-3: feasibility, not a successful coarse-task experiment

Service support is non-zero for all six services, but Facebook maps to both Chat and VoIP in the active Known classes. Across the full label inventory, Hangouts and Skype are also multi-service. Only {len(mappable)}/{len(predictions)} frozen validation rows have both true and predicted applications deterministically mappable.

The partial mapped subset improves from Fine Macro-F1 {fine_macro:.6f} to Coarse Macro-F1 {coarse_macro:.6f}, but this is neither complete-population F2 nor a trained F3. No claim that Service is the better formal task is permitted yet.

## Interim answers to the research questions

1. **Why can prior papers look better?** Already-identifiable protocol differences include fewer PCAPs, VPN-only scope, service rather than application labels, CATE target-flow selection, TrafficFormer short/byte filtering, different flow units, and different splits/preprocessing. Their reported scores cannot be directly ranked against Native.
2. **How much Fine error is within service?** 33.33% of all Fine errors and 36.07% of mappable Fine errors in the frozen shared31 validation subset.
3. **Does retrained coarse classification improve?** `NOT_RUN`; the semantic mapping gate failed before training.
4. **Can Service become the main task?** `INSUFFICIENT_EVIDENCE`; it needs a service-native label space plus capture-group feasibility.
5--9. Filtering causal effect, domain mixing, split sensitivity, and matched-model comparison are `NOT_RUN`.
10. **Current proven limitation:** protocol/task non-equivalence and mapping identifiability. A representation/model gap has not been isolated.
11. **Next task definition:** first freeze service-native labels that preserve multi-service applications; do not collapse application predictions post hoc.
12. **Ready to restart Byte--Behavior research?** No. The task/protocol definition must pass DQ-3 and DQ-6 before model conclusions are interpretable.

## Integrity

- Known Test values used: 0.
- Unknown Test values used: 0.
- New encoders trained: 0.
- Frozen input hash comparison: {hash_status} ({hashed_files} files; changed={len(changed_hashes)}).
"""
    write_text("performance_gap_attribution.md", report)
    write_text("RESULTS.md", report.replace("# Stage 15F-DQ Interim Performance-Gap Attribution", "# RESULTS — Stage 15F-DQ DQ-0 to DQ-3 Interim"))

    completion = {
        "schema_version": 1,
        "scope": "DQ-0 through DQ-3 interim only",
        "status": "COMPLETE_WITH_BLOCKED_DQ3_TRAINING_BRANCH",
        "dq0": "PASS",
        "dq1": "PASS_DESCRIPTIVE",
        "dq2": "PASS_DIAGNOSTIC",
        "dq3": "BLOCKED_MAPPING_AMBIGUITY",
        "dq4_to_dq7": "NOT_RUN",
        "iscx_tor": "NOT_RUN",
        "new_model_training": False,
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "shared_pcap_count": pcap_counts["shared"],
        "trafficformer_exact_parity_captures": summary["gate"]["trafficformer_exact_parity_captures"],
        "frozen_hash_status": hash_status,
        "frozen_hash_file_count": hashed_files,
        "frozen_hash_changed_paths": changed_hashes,
        "required_files_present": {},
    }
    required = [
        "cross_project_pcap_manifest.csv", "cross_project_flow_manifest.csv", "flow_matching_audit.md",
        "label_mapping.json", "sample_eligibility_summary.csv", "cate_matching_analysis.csv",
        "short_flow_error_analysis.csv", "fine_to_coarse_error_analysis.csv", "granularity_feasibility.md",
        "controlled_protocol_matrix.csv", "paper_protocol_reproduction.csv", "fair_matched_protocol_results.csv",
        "filter_effect_analysis.csv", "domain_mixing_analysis.csv", "split_sensitivity_analysis.csv",
        "performance_gap_attribution.md", "RESULTS.md",
    ]
    completion["required_files_present"] = {name: (OUT / name).is_file() for name in required}
    write_text("completion_verification.json", json.dumps(completion, indent=2, ensure_ascii=False))

    excluded = {"manifest.json"}
    artifacts = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        rel = path.relative_to(OUT).as_posix()
        if rel in excluded or rel.startswith("tshark_stderr/"):
            continue
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "experiment_id": "stage15f-dq-performance-gap-attribution-20260920-v1",
        "created_at_utc": "2026-09-20T04:42:08Z",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_with_blocked_branch" if hash_status == "PASS" else "interim_pending_hash_verification",
        "experiment_type": "data_protocol_audit",
        "claim_scope": "diagnostic",
        "objective": "Execute DQ-0 through DQ-3 on shared 31 ISCX-VPN captures using Known Train/Validation only.",
        "inputs": [{"scope": "frozen assets", "count": hashed_files, "hash_status": hash_status}],
        "code": {"revision": "working tree; task-local scripts listed in artifacts", "dirty": True, "changes": ["task-local audit and reporting scripts only"]},
        "execution": {"tmux_sessions": ["s15fdq_dq0_dq2_outerproto", "s15fdq_finalize_interim_20260920"], "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310", "physical_gpu_ids": []},
        "configuration": {"dataset": "ISCX-VPN", "shared_pcaps": 31, "protocol": "medium_seed2022", "allowed_roles": ["known_train", "known_validation"], "seeds": []},
        "core_results": [
            {"name": "native_shared31_flows", "value": flow_counts["native_shared"]},
            {"name": "cate_matched_flows", "value": int(overall_sets["B"]["flow_count"])},
            {"name": "shared31_validation_macro_f1", "value": fine_macro},
            {"name": "fine_errors_coarse_correct", "value": 22},
            {"name": "dq3_gate", "value": "BLOCKED_MAPPING_AMBIGUITY"},
        ],
        "artifacts": artifacts,
        "limitations": [
            "No Known Test or Unknown Test values were used.",
            "DQ-3 F1/F3 training was stopped because full Fine-to-Service prediction mapping is ambiguous.",
            "DQ-4 through DQ-7 and ISCXTor are NOT_RUN.",
            "CATE matching is not authoritative per-flow label truth.",
        ],
        "next_step": "Resolve and preregister a service-native semantic label space, then reassess the DQ-3 gate; do not start DQ-4--DQ-7 before approval.",
    }
    write_text("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({
        "status": completion["status"],
        "hash_status": hash_status,
        "required_files": sum(completion["required_files_present"].values()),
        "fine": {"accuracy": fine_acc, "macro_f1": fine_macro, "weighted_f1": fine_weighted},
        "partial_coarse": {"n": len(mappable), "accuracy": coarse_acc, "macro_f1": coarse_macro, "weighted_f1": coarse_weighted},
    }, indent=2))


if __name__ == "__main__":
    main()
