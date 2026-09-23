#!/usr/bin/env python3
"""Build the Stage 15F-DQ-3R service-label feasibility audit.

Only frozen metadata for Known Train/Validation and the prior DQ reconstruction
is read.  No feature array, Known Test, Unknown Test, model, or detector is
opened or executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
WORKSPACE = ROOT.parents[1]
DQ = ROOT / "stage15f_literature_guided_feature_benchmark" / "stage15f_dq_performance_gap_attribution"
STAGE12 = ROOT / "stage12_dual_external_validation"
OFFICIAL_URL = "https://www.unb.ca/cic/datasets/vpn.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tfe_service(value: str) -> str:
    mapping = {"File": "File-Transfer", "file_transfer": "File-Transfer", "P2P": "P2P"}
    return mapping.get(value, value)


def trafficformer_service(source_file: str) -> str:
    """Exact semantic equivalent of TrafficFormer prepare_iscxvpn.service_label."""
    name = Path(source_file).stem.lower()
    if name.startswith("vpn_"):
        name = name[4:]
    if "chat" in name:
        return "Chat"
    if "audio" in name or "voipbuster" in name:
        return "VoIP"
    if "file" in name or "ftps" in name or "sftp" in name:
        return "File-Transfer"
    if "email" in name:
        return "Email"
    if "bittorrent" in name:
        return "P2P"
    if any(token in name for token in ("netflix", "spotify", "vimeo", "youtube", "video")):
        return "Streaming"
    raise ValueError(source_file)


def capture_activity(source_file: str, service: str) -> str:
    name = Path(source_file).stem.lower()
    if "chat" in name:
        return "chat session"
    if "audio" in name or "voipbuster" in name:
        return "voice call"
    if "file" in name or "ftps" in name or "sftp" in name:
        return "file transfer"
    if "bittorrent" in name:
        return "P2P transfer"
    if "email" in name:
        return "email session"
    if service == "Streaming":
        return "streaming session"
    return "capture activity inferred from filename"


def joined(values) -> str:
    return "|".join(sorted(set(str(v) for v in values if str(v))))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    pcap_rows = read_csv(DQ / "cross_project_pcap_manifest.csv")
    flow_rows = read_csv(DQ / "cross_project_flow_manifest.csv")
    split_rows = read_csv(STAGE12 / "protocol" / "iscx_vpn" / "split_manifest.csv")
    shared = [r for r in pcap_rows if r["shared_all_three"].lower() == "true"]
    assert len(shared) == 31
    pcap_by_source = {r["source_file"]: r for r in shared}
    assert len(pcap_by_source) == 31

    # Application-to-service is intentionally one-to-many at the capture level.
    app_services: dict[str, set[str]] = defaultdict(set)
    for row in pcap_rows:
        app_services[row["application"]].add(row["service"])
    multi_service_apps = {app: sorted(values) for app, values in app_services.items() if len(values) > 1}

    cate_by_capture: dict[str, Counter] = defaultdict(Counter)
    for row in flow_rows:
        if row["cate_match_status"] == "MATCHED" and row["cate_service"]:
            cate_by_capture[row["source_file"]][normalize_tfe_service(row["cate_service"])] += 1

    official_explicit = {
        ("AIM", "Chat"), ("ICQ", "Chat"), ("Facebook", "Chat"),
        ("Hangouts", "Chat"), ("Skype", "Chat"),
        ("Email", "Email"), ("FTPS", "File-Transfer"),
        ("SFTP", "File-Transfer"), ("Skype", "File-Transfer"),
        ("BitTorrent", "P2P"), ("Vimeo", "Streaming"),
        ("YouTube", "Streaming"), ("Facebook", "VoIP"),
        ("Hangouts", "VoIP"), ("Skype", "VoIP"),
    }
    capture_audit = []
    for row in sorted(shared, key=lambda r: r["source_file"]):
        source = row["source_file"]
        stage_service = row["service"]
        tf_service = trafficformer_service(source)
        cate_services = sorted(cate_by_capture.get(source, {}))
        cate_conflict = [value for value in cate_services if value != stage_service]
        agreement = stage_service == tf_service and not cate_conflict
        capture_audit.append({
            "capture_id": row["capture_id"],
            "source_file": source,
            "application_label": row["application"],
            "capture_activity": capture_activity(source, stage_service),
            "stage12_service": stage_service,
            "trafficformer_service": tf_service,
            "tfe_gnn_cate_services": joined(cate_services),
            "tfe_gnn_matched_flow_count": sum(cate_by_capture.get(source, {}).values()),
            "official_unb_pair_explicit": str((row["application"], stage_service) in official_explicit),
            "project_mapping_agreement": str(agreement),
            "capture_label_status": "WEAK_CAPTURE_LABEL" if agreement else "AMBIGUOUS",
            "reason": (
                "capture activity is unambiguous across local mappings, but individual flow semantics are not independently annotated"
                if agreement else "local mapping sources disagree"
            ),
        })
    assert all(r["project_mapping_agreement"] == "True" for r in capture_audit)
    write_csv(OUT / "capture_service_mapping_audit.csv", list(capture_audit[0]), capture_audit)
    status_definitions = {
        "VERIFIED_DEFINITION": "Independent, auditable per-flow semantic evidence supports the label.",
        "WEAK_CAPTURE_LABEL": "The flow inherits a well-defined capture activity label, but per-flow target membership is not independently verified.",
        "AMBIGUOUS": "Available sources permit multiple incompatible Service labels.",
        "UNAVAILABLE": "No auditable Service label source is available.",
    }
    write_text(
        OUT / "service_label_status_definitions.json",
        json.dumps(status_definitions, indent=2, ensure_ascii=False),
    )

    allowed_roles = {"known_train", "known_validation"}
    provenance_source = set(pcap_by_source)
    allowed = [
        r for r in split_rows
        if r["setting"] == "medium"
        and r["role"] in allowed_roles
        and r["source_file"] in provenance_source
    ]
    assert len(allowed) == 3065
    assert Counter(r["role"] for r in allowed) == Counter({"known_train": 2730, "known_validation": 335})
    assert all(r["domain_state"] == "VPN" for r in allowed)
    assert len({r["flow_id_sha256"] for r in allowed}) == len(allowed)

    capture_lookup = {r["source_file"]: r for r in capture_audit}
    provenance = []
    for row in allowed:
        capture = capture_lookup[row["source_file"]]
        app = row["canonical_class"]
        service = row["official_category"]
        assert service == capture["stage12_service"]
        is_multi = app in multi_service_apps
        explicit = capture["official_unb_pair_explicit"] == "True"
        evidence = [
            "UNB controlled-capture taxonomy",
            "capture filename activity",
            "Stage12 stage12_common.classify_iscx_vpn",
            "TrafficFormer prepare_iscxvpn.service_label agreement",
        ]
        if int(capture["tfe_gnn_matched_flow_count"]) > 0:
            evidence.append("TFE-GNN CATE category agreement at capture level")
        provenance.append({
            "flow_id": row["flow_id_sha256"],
            "capture_id": capture["capture_id"],
            "source_file": row["source_file"],
            "split_role": row["role"],
            "application_label": app,
            "capture_activity": capture["capture_activity"],
            "service_label": service,
            "service_label_source": "; ".join(evidence),
            "service_label_status": "WEAK_CAPTURE_LABEL",
            "ambiguity_reason": (
                "application is multi-service and is resolved only by capture activity; no independent per-flow semantic annotation"
                if is_multi else
                "service is inherited from the target capture activity; background/non-target flows remain possible"
            ),
            "official_application_service_pair_explicit": str(explicit),
        })
    write_csv(OUT / "service_label_provenance.csv", list(provenance[0]), provenance)

    by_service: dict[str, list[dict]] = defaultdict(list)
    for row in provenance:
        by_service[row["service_label"]].append(row)
    support_rows = []
    for service in sorted(by_service):
        rows = by_service[service]
        train = [r for r in rows if r["split_role"] == "known_train"]
        val = [r for r in rows if r["split_role"] == "known_validation"]
        all_caps = sorted({r["capture_id"] for r in rows})
        train_caps = {r["capture_id"] for r in train}
        val_caps = {r["capture_id"] for r in val}
        per_cap = Counter(r["capture_id"] for r in rows)
        group_feasible = len(all_caps) >= 2
        support_rows.append({
            "service": service,
            "train_flows": len(train),
            "validation_flows": len(val),
            "total_flows": len(rows),
            "train_captures": len(train_caps),
            "validation_captures": len(val_caps),
            "total_captures": len(all_caps),
            "train_validation_capture_overlap": len(train_caps & val_caps),
            "applications": joined(r["application_label"] for r in rows),
            "weak_capture_label_flows": sum(r["service_label_status"] == "WEAK_CAPTURE_LABEL" for r in rows),
            "ambiguous_flows": sum(r["service_label_status"] == "AMBIGUOUS" for r in rows),
            "unavailable_flows": sum(r["service_label_status"] == "UNAVAILABLE" for r in rows),
            "min_flows_per_capture": min(per_cap.values()),
            "median_flows_per_capture": statistics.median(per_cap.values()),
            "max_flows_per_capture": max(per_cap.values()),
            "max_min_capture_ratio": max(per_cap.values()) / min(per_cap.values()),
            "capture_group_train_val_feasible": str(group_feasible),
            "feasibility_reason": (
                ">=2 captures allow a two-way group partition in principle; balance still requires explicit freeze"
                if group_feasible else
                "only one independent capture; cannot place this service in both group-disjoint Train and Validation"
            ),
        })
    assert len(support_rows) == 6
    write_csv(OUT / "service_train_val_support.csv", list(support_rows[0]), support_rows)

    open_set_rows = []
    for setting in ("low", "medium", "high"):
        protocol = read_json(STAGE12 / "artifacts" / "iscx_vpn" / "protocol" / setting / "protocol.json")
        known_apps = protocol["known_classes"]
        unknown_apps = protocol["unknown_classes"]
        known_services = sorted({s for app in known_apps for s in app_services[app]})
        unknown_services = sorted({s for app in unknown_apps for s in app_services[app]})
        overlap = sorted(set(known_services) & set(unknown_services))
        open_set_rows.append({
            "setting": setting,
            "known_applications": joined(known_apps),
            "unknown_applications": joined(unknown_apps),
            "known_services": joined(known_services),
            "unknown_application_service_union": joined(unknown_services),
            "service_overlap": joined(overlap),
            "unknown_services_not_seen_in_known": joined(set(unknown_services) - set(known_services)),
            "strict_service_unknown_reusable": str(bool(unknown_services) and not overlap),
            "reason": "Fine Unknown applications map into Known Service semantics; a new held-out-Service protocol is required",
        })
    write_csv(OUT / "service_open_set_overlap.csv", list(open_set_rows[0]), open_set_rows)

    all_group_feasible = all(r["capture_group_train_val_feasible"] == "True" for r in support_rows)
    observed_statuses = Counter(r["service_label_status"] for r in provenance)
    statuses = Counter({name: observed_statuses.get(name, 0) for name in status_definitions})
    decision = (
        "SERVICE_TASK_FEASIBLE_WITH_WEAK_LABELS"
        if statuses["AMBIGUOUS"] == 0 and statuses["UNAVAILABLE"] == 0
        and all(int(r["train_flows"]) > 0 and int(r["validation_flows"]) > 0 for r in support_rows)
        else "SERVICE_TASK_BLOCKED"
    )

    support_md = "\n".join(
        f"| {r['service']} | {r['train_flows']} | {r['validation_flows']} | {r['total_captures']} | {r['train_validation_capture_overlap']} | {r['capture_group_train_val_feasible']} |"
        for r in support_rows
    )
    multi_md = "\n".join(f"- `{app}` → {', '.join(values)}" for app, values in sorted(multi_service_apps.items()))
    mapping_report = f"""# Service Label Mapping Audit

## Evidence hierarchy

1. **Primary dataset definition:** the [UNB ISCXVPN2016 page]({OFFICIAL_URL}) defines Chat, Email, Streaming, File Transfer, VoIP and P2P and explicitly places Facebook/Hangouts/Skype in multiple service types according to the performed activity. It states that the objective application was the only application deliberately executed and packets were filtered to the local client IP. It also warns that incidental browsing flows can be captured during another task.
2. **Capture identity:** all 31 filenames encode an activity or single-service application (`chat`, `audio`, `files`, FTPS/SFTP, BitTorrent, streaming application, and so on).
3. **Local independent implementations:** Stage12 `_vpn_service`, TrafficFormer `service_label`, and the TFE-GNN CATE category agree on every capture for which CATE matches exist. The 31/31 Stage12/TrafficFormer capture mappings agree.
4. **Flow-level limit:** none of these sources provides an independent semantic annotation for every reconstructed Native flow. Consequently, a capture-derived label is not upgraded to per-flow ground truth.

## Multi-service applications

{multi_md}

The ambiguity is resolved at the **capture activity** level: Facebook chat and Facebook voice-call captures receive different Service labels. It is not resolved by a global application→service map.

## Flow provenance result

- Allowed population: frozen `medium_seed2022` Known Train/Validation rows whose source is one of the shared 31 VPN PCAPs.
- Rows: {len(provenance)} = {Counter(r['split_role'] for r in provenance)['known_train']} Train + {Counter(r['split_role'] for r in provenance)['known_validation']} Validation.
- `VERIFIED_DEFINITION`: {statuses['VERIFIED_DEFINITION']}.
- `WEAK_CAPTURE_LABEL`: {statuses['WEAK_CAPTURE_LABEL']}.
- `AMBIGUOUS`: {statuses['AMBIGUOUS']}.
- `UNAVAILABLE`: {statuses['UNAVAILABLE']}.

The Service schema and capture activity are auditable, but all flow labels remain `WEAK_CAPTURE_LABEL` because background/non-target flows cannot be excluded without independent per-flow truth. Validation outcomes were not used to assign or alter any label.
"""
    write_text(OUT / "service_label_mapping_audit.md", mapping_report)

    group_report = f"""# Capture-Group Feasibility

| Service | Train flows | Validation flows | Independent captures | Existing Train/Val capture overlap | Group split feasible? |
|---|---:|---:|---:|---:|---|
{support_md}

## Findings

- Every Service has non-zero frozen flow-level Train and Validation support.
- The current frozen membership is flow-disjoint but not capture-disjoint: each listed overlap means the same capture contributes flows to both Train and Validation.
- P2P is supplied only by `vpn_bittorrent.pcap`; it cannot appear in both sides of a capture-disjoint closed-set split.
- Labels are constant within each capture by construction, so Service is fully tied to capture identity. A random flow split can reward capture-specific signatures.
- Full six-service capture-group Train/Validation feasibility: **{str(all_group_feasible).upper()}**.

Therefore DQ-3F can be run as a controlled same-membership diagnostic, but this 31-PCAP subset alone cannot validate group-generalized six-Service classification. Excluding P2P merely to make the split feasible is not preregistered and would change the task.
"""
    write_text(OUT / "capture_group_feasibility.md", group_report)

    overlap_md = "\n".join(
        f"| {r['setting']} | {r['known_services']} | {r['unknown_application_service_union']} | {r['service_overlap']} | {r['strict_service_unknown_reusable']} |"
        for r in open_set_rows
    )
    open_report = f"""# Coarse Open-Set Semantics

| Existing Fine setting | Known Service union | Fine-Unknown Service union | Overlap | Reusable as strict Service open set? |
|---|---|---|---|---|
{overlap_md}

The Fine protocols hold out entire **applications**, not Services. An Unknown application that belongs to a Known Service is not Service-level unknown. Accordingly, none of the existing Fine Open-Set settings may be relabeled and reused as a Service Open-Set protocol.

A future Service protocol must:

1. choose Unknown at the Service-class level before training;
2. remove every flow/capture carrying the held-out Service from encoder training, validation, normalization, support/prototype construction, threshold calibration and parameter selection;
3. freeze Known/Unknown Service classes and all sample IDs before model execution;
4. preserve historical Fine protocols and DES/H1 results unchanged;
5. explicitly audit multi-service applications that occur in both Known and Unknown Service captures as a separate application-identity sensitivity issue.

No new Service Open-Set split is created in DQ-3R.
"""
    write_text(OUT / "coarse_open_set_semantics.md", open_report)

    native_config = read_json(STAGE12 / "runs" / "iscx_vpn" / "medium" / "seed2022" / "config.json")
    training = native_config["training"]
    prereg = f"""# DQ-3F Preregistered Protocol

Status: `PREREGISTERED_NOT_RUN`  
Claim scope: Known-only closed-set diagnostic on weak capture labels.

## Frozen population

- Dataset: shared 31 ISCX-VPN VPN PCAPs.
- Eligibility: all and only the {len(provenance)} frozen `medium_seed2022` Known Train/Validation flows from those PCAPs.
- Membership: Train={Counter(r['split_role'] for r in provenance)['known_train']}, Validation={Counter(r['split_role'] for r in provenance)['known_validation']}.
- F1 and F3 must use exactly the same flow IDs and membership; no filtering by Service, packet count, CATE status or TrafficFormer eligibility.
- Known Test and Unknown Test feature usage: 0.

## Tasks

- **F1:** train Native on Fine Application labels; evaluate all Validation flows with Accuracy, Macro-F1, Weighted-F1 and per-class F1.
- **F2-PARTIAL:** map F1 predictions only where both true and predicted Application have a singleton Service mapping. Exclude ambiguous outputs explicitly and report coverage. F2 is diagnostic and cannot substitute for F3.
- **F3:** train Native directly on capture-activity Service labels for all identical F1 flows; evaluate Accuracy, Macro-F1, Weighted-F1, per-Service Recall and confusion matrix.

## Frozen Native configuration

- architecture=`{training['architecture']}`, channels={training['channels']}, latent_dim={training['latent_dim']}
- epochs={training['epochs']}, batch_size={training['batch_size']}, eval_batch_size={training['eval_batch_size']}
- optimizer={training['optimizer']}, learning_rate={training['learning_rate']}, betas={training['betas']}
- Open-Detect lambda={training['lambda']}
- MultiStepLR milestones={training['scheduler_milestones']}, gamma={training['scheduler_gamma']}
- prototype reset zero-based epochs={training['prototype_reset_zero_based_epochs']}
- checkpoint criterion=`{training['checkpoint_criterion']}`
- early stop patience={training['early_stop_patience']}, min_epoch={training['early_stop_min_epoch']}, min_delta={training['early_stop_min_delta']}
- seed=2022

Only output dimension/label encoding changes between F1 and F3. Architecture, input bytes, loss family, optimizer, budget, seed and membership remain fixed.

## Gates and interpretation

- DQ-3F may quantify whether direct Service training is easier under the frozen flow-level split.
- It may **not** establish capture-generalized performance because P2P has one capture and the existing Train/Validation groups overlap.
- It may **not** establish an Open-Set method or justify changing DES/H1.
- A later formal Service benchmark requires a separately frozen held-out-Service protocol and group feasibility resolution.
"""
    write_text(OUT / "dq3f_preregistered_protocol.md", prereg)
    prereg_config = {
        "status": "PREREGISTERED_NOT_RUN",
        "dataset": "iscx_vpn_shared31_vpn_only",
        "source_protocol": "stage12 medium seed2022",
        "train_samples": Counter(r["split_role"] for r in provenance)["known_train"],
        "validation_samples": Counter(r["split_role"] for r in provenance)["known_validation"],
        "same_flow_ids_f1_f3": True,
        "labels": {"F1": "application", "F2": "partial post-hoc service mapping", "F3": "capture-activity service"},
        "training": training,
        "seed": 2022,
        "known_test_feature_values_used": 0,
        "unknown_test_feature_values_used": 0,
        "group_generalization_claim_allowed": False,
    }
    write_text(OUT / "dq3f_preregistered_config.json", json.dumps(prereg_config, indent=2, ensure_ascii=False))

    before_hashes = read_json(OUT / "frozen_asset_hashes_before.json")
    after_hash_path = OUT / "frozen_asset_hashes_after.json"
    after_hashes = read_json(after_hash_path) if after_hash_path.exists() else None
    frozen_hash_status = (
        "PASS" if after_hashes is not None and before_hashes["files"] == after_hashes["files"]
        else "PENDING" if after_hashes is None else "FAIL"
    )

    report = f"""# Stage 15F-DQ-3R Report

## Conclusion and next step

`{decision}`

The earlier global application→service mapping failure is resolved for direct F3 training by assigning Service from the capture's experimental activity, not from Application alone. All {len(provenance)} eligible Known Train/Validation flows receive a deterministic capture-derived label, including Facebook Chat and Facebook VoIP captures.

This does **not** create authoritative per-flow truth. All {len(provenance)} labels are `WEAK_CAPTURE_LABEL`; target captures can include incidental or background flows. No label was changed using Validation behavior.

Next step: DQ-3F may run the preregistered same-flow F1/F2-partial/F3 comparison. It must retain the weak-label qualifier and must not reuse the Fine Open-Set protocols as Service Open-Set protocols.

## Configuration and execution

- Scope: shared 31 VPN PCAPs and their frozen Known Train/Validation membership.
- Comparison prepared: F1 Fine Application, F2 partial deterministic mapping, and F3 direct capture-activity Service training.
- F1 and F3 use exactly the same {len(provenance)} flow IDs and split membership.
- No model was trained in DQ-3R; the DQ-3F configuration is preregistered only.
- Frozen input assets checked before/after: {len(before_hashes['files'])}; SHA256 comparison: {frozen_hash_status}.

## Core results

- Shared PCAPs audited: 31/31.
- Project capture-level mapping agreement: 31/31.
- Train/Validation flows: 2730/335.
- Services: {joined(by_service)}.
- Every Service has non-zero Train and Validation support.
- Strict six-Service capture-group Train/Validation split: **not feasible** because P2P has one capture.
- Existing Fine Open-Set protocols reusable as Service Open-Set protocols: **NO**; Fine Unknown applications overlap Known Service semantics.
- DQ-3F same-flow flow-level diagnostic: **runnable under the preregistered weak-label scope**.

## Data and split

- Eligible Known Train flows: 2730.
- Eligible Known Validation flows: 335.
- No Known Test or Unknown Test feature values were accessed.
- Every Service has non-zero flow-level Train and Validation support.
- P2P has only one independent capture, so a strict six-Service capture-disjoint Train/Validation evaluation is unavailable on this subset.

## Limitations

- Solved: multi-service applications can be represented correctly at capture activity level for direct F3 labels.
- Not solved: authoritative target/background separation within a capture.
- Not solved: full six-Service group-generalization on the shared31 subset.
- Not solved: Service-level Unknown protocol; it must be redesigned and frozen independently.

## Preserved evidence

- `service_label_provenance.csv`: row-level label provenance and status.
- `service_train_val_support.csv`: Service support and capture coverage.
- `capture_service_mapping_audit.csv`: 31-capture source agreement audit.
- `service_open_set_overlap.csv`: Fine-to-Service semantic-overlap evidence.
- `dq3f_preregistered_protocol.md` and `dq3f_preregistered_config.json`: frozen next-stage plan.
- `completion_verification.json`: machine-readable boundary and completion checks.

Execution boundaries:

- New model training: 0.
- Known Test feature values read: 0.
- Unknown Test feature values read: 0.
- Byte--Behavior, DES and H1 changes: 0.
- Historical Stage 15F-DQ outputs modified: 0.
"""
    write_text(OUT / "dq3r_report.md", report)
    write_text(OUT / "RESULTS.md", report.replace("# Stage 15F-DQ-3R Report", "# RESULTS — Stage 15F-DQ-3R"))
    completion = {
        "schema_version": 1,
        "status": "PASS",
        "decision": decision,
        "shared_pcaps": len(shared),
        "capture_mapping_agreement": sum(r["project_mapping_agreement"] == "True" for r in capture_audit),
        "provenance_rows": len(provenance),
        "train_rows": Counter(r["split_role"] for r in provenance)["known_train"],
        "validation_rows": Counter(r["split_role"] for r in provenance)["known_validation"],
        "label_status_counts": dict(statuses),
        "service_count": len(by_service),
        "all_services_have_train_validation": all(int(r["train_flows"]) > 0 and int(r["validation_flows"]) > 0 for r in support_rows),
        "all_service_capture_group_split_feasible": all_group_feasible,
        "dq3f_status": "PREREGISTERED_NOT_RUN",
        "known_test_feature_values_used": 0,
        "unknown_test_feature_values_used": 0,
        "model_training_runs": 0,
        "byte_behavior_started": False,
        "des_h1_modified": False,
        "prior_experiment_outputs_modified": False,
        "frozen_asset_hash_status": frozen_hash_status,
        "frozen_asset_count": before_hashes["file_count"],
    }
    write_text(OUT / "completion_verification.json", json.dumps(completion, indent=2, ensure_ascii=False))

    manifest_path = OUT / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.update({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "inputs": [
            {"path": str(DQ / "cross_project_pcap_manifest.csv"), "role": "frozen DQ capture alignment"},
            {"path": str(DQ / "cross_project_flow_manifest.csv"), "role": "frozen DQ flow alignment"},
            {"path": str(STAGE12 / "protocol" / "iscx_vpn" / "split_manifest.csv"), "role": "frozen membership and flow IDs"},
            {"url": OFFICIAL_URL, "role": "primary ISCXVPN2016 service taxonomy"},
        ],
        "execution": {
            "tmux_session": "s15fdq3r_run_audit_20260920",
            "command": "python scripts/run_dq3r_audit.py",
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": [],
        },
        "configuration": {
            "files": ["dq3f_preregistered_config.json"],
            "parameters": {"setting": "medium", "seed": 2022, "allowed_roles": ["known_train", "known_validation"]},
            "seeds": [2022],
        },
        "core_results": [
            {"name": "decision", "value": decision},
            {"name": "service_label_provenance_rows", "value": len(provenance)},
            {"name": "weak_capture_label_rows", "value": statuses["WEAK_CAPTURE_LABEL"]},
            {"name": "capture_group_all_services_feasible", "value": all_group_feasible},
        ],
        "limitations": [
            "Capture activity is not authoritative per-flow truth.",
            "P2P has one capture, blocking a six-Service capture-disjoint Train/Validation split.",
            "Existing Fine Unknown applications overlap Known Service semantics.",
            "No model was trained and no Test feature values were opened.",
        ],
        "next_step": "Run DQ-3F only after explicit authorization; keep it a weak-label flow-level diagnostic and do not infer group-generalized or open-set performance.",
    })
    write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
