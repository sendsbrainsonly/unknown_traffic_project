#!/usr/bin/env python3
"""Freeze the Stage 14B VNAT application-held-out open-set protocol."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parents[1]
STAGE14A = PROJECT_ROOT / "stage14a_vnat_data_audit"
SCAN_ROOT = STAGE14A / "artifacts" / "pcap_scans"
OUTPUT_ROOT = STAGE_ROOT / "outputs"

MANIFEST_PATH = STAGE14A / "vnat_manifest.csv"
DUPLICATE_FLOWS_PATH = STAGE14A / "outputs" / "duplicate_flows.csv"
DUPLICATE_FILES_PATH = STAGE14A / "outputs" / "duplicate_files.csv"

APPLICATIONS = ["netflix", "rdp", "rsync", "scp", "sftp", "skype", "ssh", "vimeo", "youtube", "zoiper"]
PROTOCOL_SEEDS = [2022, 2023, 2024, 2025, 2026]
SETTING_UNKNOWN_COUNTS = {"Low": 2, "Medium": 3, "High": 4}
SETTING_ORDER = {"Low": 0, "Medium": 1, "High": 2}
MASTER_UNKNOWN_SEED = 2022


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(protocol_seed: int, application: str) -> int:
    digest = hashlib.sha256(f"{protocol_seed}:{application}".encode()).hexdigest()
    return int(digest[:16], 16)


def integrity_exclusions() -> dict[str, Any]:
    duplicate_flow_rows = read_csv(DUPLICATE_FLOWS_PATH)
    cross_application_keys = set()
    cross_application_rows = []
    for row in duplicate_flow_rows:
        peers = {value for value in row["peer_applications"].split(";") if value}
        if any(peer != row["application"] for peer in peers):
            cross_application_keys.add((row["capture_id"], row["flow_id"]))
            cross_application_rows.append(row)
    if len(cross_application_keys) != 4:
        raise AssertionError(f"Expected 4 cross-application duplicate flow rows, found {len(cross_application_keys)}")

    duplicate_pcaps = read_csv(DUPLICATE_FILES_PATH)
    retained_duplicate_captures = []
    dropped_duplicate_captures = []
    for row in duplicate_pcaps:
        captures = sorted(value for value in row["capture_ids"].split(";") if value)
        if len(captures) < 2:
            continue
        retained_duplicate_captures.append(captures[0])
        dropped_duplicate_captures.extend(captures[1:])

    manifest = read_csv(MANIFEST_PATH)
    failed_pcaps = [row["capture_id"] for row in manifest if row["file_type"] == "pcap" and row["readable"] != "True"]
    return {
        "cross_application_keys": cross_application_keys,
        "cross_application_rows": cross_application_rows,
        "retained_duplicate_captures": sorted(retained_duplicate_captures),
        "dropped_duplicate_captures": sorted(dropped_duplicate_captures),
        "failed_pcaps": sorted(failed_pcaps),
    }


def build_clean_flows(exclusions: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_csv(MANIFEST_PATH)
    pcaps = [row for row in manifest if row["file_type"] == "pcap"]
    class_state: dict[str, dict[str, Any]] = {
        application: {
            "application": application,
            "raw_readable_flow_count": 0,
            "removed_cross_application_duplicate_flows": 0,
            "removed_duplicate_pcap_flows": 0,
            "cleaned_flow_count": 0,
            "vpn_flow_count": 0,
            "nonvpn_flow_count": 0,
            "source_pcap_count": 0,
            "readable_pcap_count": 0,
            "failed_pcap_count": 0,
            "failed_pcap_ids": [],
            "clean_group_ids": set(),
            "clean_capture_ids": set(),
        }
        for application in APPLICATIONS
    }
    clean_flows = []
    seen_flow_uids = set()
    for pcap in pcaps:
        application = pcap["application"]
        state = class_state[application]
        state["source_pcap_count"] += 1
        if pcap["readable"] != "True":
            state["failed_pcap_count"] += 1
            state["failed_pcap_ids"].append(pcap["capture_id"])
            continue
        state["readable_pcap_count"] += 1
        scan = json.loads((SCAN_ROOT / f"{pcap['capture_id']}.json").read_text(encoding="utf-8"))
        flows = scan.get("flows", [])
        state["raw_readable_flow_count"] += len(flows)
        if pcap["capture_id"] in exclusions["dropped_duplicate_captures"]:
            state["removed_duplicate_pcap_flows"] += len(flows)
            continue
        for flow in flows:
            source_key = (pcap["capture_id"], flow["flow_id"])
            if source_key in exclusions["cross_application_keys"]:
                state["removed_cross_application_duplicate_flows"] += 1
                continue
            flow_uid = hashlib.sha256(f"{pcap['capture_id']}|{flow['flow_id']}".encode()).hexdigest()
            if flow_uid in seen_flow_uids:
                raise AssertionError(f"Duplicate flow_uid: {flow_uid}")
            seen_flow_uids.add(flow_uid)
            row = {
                "flow_uid": flow_uid,
                "application": application,
                "vpn_status": pcap["vpn_status"],
                "capture_id": pcap["capture_id"],
                "source_flow_id": flow["flow_id"],
                "group_id": pcap["group_id"],
                "source_group_id": pcap["source_group_id"],
                "protocol": flow["protocol"],
                "packet_count": flow["packet_count"],
                "byte_count": flow["byte_count"],
                "first_timestamp": flow["first_timestamp"],
                "last_timestamp": flow["last_timestamp"],
                "tuple_sha256": flow["tuple_sha256"],
                "content_sha256": flow["content_sha256"],
                "source_pcap_path": pcap["absolute_path"],
            }
            clean_flows.append(row)
            state["cleaned_flow_count"] += 1
            state[f"{pcap['vpn_status']}_flow_count"] += 1
            state["clean_group_ids"].add(pcap["group_id"])
            state["clean_capture_ids"].add(pcap["capture_id"])

    content_to_applications: dict[str, set[str]] = defaultdict(set)
    for row in clean_flows:
        content_to_applications[row["content_sha256"]].add(row["application"])
    cross_application_content = {
        digest: applications
        for digest, applications in content_to_applications.items()
        if len(applications) > 1
    }
    if cross_application_content:
        raise AssertionError(f"Cross-application exact duplicate content remains: {cross_application_content}")

    audit_rows = []
    by_application = defaultdict(list)
    for row in clean_flows:
        by_application[row["application"]].append(row)
    for application in APPLICATIONS:
        state = class_state[application]
        count = state["cleaned_flow_count"]
        holdout = max(1, count // 10)
        train_count = count - 2 * holdout
        retained = count >= 3 and train_count > 0
        if not retained:
            reason = "excluded: fewer than three clean flows or empty 8:1:1 partition"
        else:
            reason = "retained: clean flow count supports non-empty class-stratified 8:1:1"
        audit_rows.append({
            "application": application,
            "status": "retained" if retained else "excluded",
            "reason": reason,
            "raw_readable_flow_count": state["raw_readable_flow_count"],
            "removed_cross_application_duplicate_flows": state["removed_cross_application_duplicate_flows"],
            "removed_duplicate_pcap_flows": state["removed_duplicate_pcap_flows"],
            "corrupt_pcap_data_contribution": 0,
            "corrupt_pcap_data_note": "whole failed PCAP excluded before flow manifest; unparsed flow count is not inferable" if state["failed_pcap_count"] else "none",
            "cleaned_flow_count": count,
            "vpn_flow_count": state["vpn_flow_count"],
            "nonvpn_flow_count": state["nonvpn_flow_count"],
            "vpn_ratio": round(state["vpn_flow_count"] / count, 8) if count else 0.0,
            "source_pcap_count": state["source_pcap_count"],
            "readable_pcap_count": state["readable_pcap_count"],
            "failed_pcap_count": state["failed_pcap_count"],
            "failed_pcap_ids": ";".join(sorted(state["failed_pcap_ids"])),
            "clean_capture_count": len(state["clean_capture_ids"]),
            "clean_group_count": len(state["clean_group_ids"]),
            "known_train_flows_if_known": train_count if retained else 0,
            "known_validation_flows_if_known": holdout if retained else 0,
            "known_test_flows_if_known": holdout if retained else 0,
            "nonempty_8_1_1": retained,
        })
    return sorted(clean_flows, key=lambda row: (row["application"], row["flow_uid"])), audit_rows


def balanced_nested_unknown_schedule(applications: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    if len(applications) != 10:
        raise ValueError("Balanced Stage 14B schedule expects exactly 10 retained applications")
    base_order = sorted(applications)
    random.Random(MASTER_UNKNOWN_SEED).shuffle(base_order)
    schedule = []
    for index, seed in enumerate(PROTOCOL_SEEDS):
        low = [base_order[(2 * index) % 10], base_order[(2 * index + 1) % 10]]
        medium = low + [base_order[(2 * index + 2) % 10]]
        high = medium + [base_order[(2 * index + 3) % 10]]
        for setting, unknown in (("Low", low), ("Medium", medium), ("High", high)):
            schedule.append({
                "protocol_id": f"{setting.lower()}_seed{seed}",
                "setting": setting,
                "seed": seed,
                "unknown_applications": unknown,
                "known_applications": sorted(set(applications) - set(unknown)),
            })
    schedule.sort(key=lambda row: (SETTING_ORDER[row["setting"]], row["seed"]))
    return base_order, schedule


def known_split_map(flows: list[dict[str, Any]], protocol_seed: int) -> dict[str, str]:
    by_application: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flows:
        by_application[row["application"]].append(row)
    assignments = {}
    for application in sorted(by_application):
        rows = sorted(by_application[application], key=lambda row: row["flow_uid"])
        random.Random(stable_seed(protocol_seed, application)).shuffle(rows)
        holdout = max(1, len(rows) // 10)
        test_rows = rows[:holdout]
        validation_rows = rows[holdout:2 * holdout]
        train_rows = rows[2 * holdout:]
        if not train_rows or not validation_rows or not test_rows:
            raise AssertionError(f"Empty known split for {application}, seed={protocol_seed}")
        for row in train_rows:
            assignments[row["flow_uid"]] = "train"
        for row in validation_rows:
            assignments[row["flow_uid"]] = "validation"
        for row in test_rows:
            assignments[row["flow_uid"]] = "test"
    return assignments


def build_protocols(
    clean_flows: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_maps = {seed: known_split_map(clean_flows, seed) for seed in PROTOCOL_SEEDS}
    manifest_rows = []
    protocol_records = []
    for protocol in schedule:
        unknown = set(protocol["unknown_applications"])
        rows = []
        for flow in clean_flows:
            is_unknown = flow["application"] in unknown
            split = "unknown_test" if is_unknown else split_maps[protocol["seed"]][flow["flow_uid"]]
            row = {
                "protocol_id": protocol["protocol_id"],
                "setting": protocol["setting"],
                "protocol_seed": protocol["seed"],
                "known_class_count": len(protocol["known_applications"]),
                "unknown_class_count": len(protocol["unknown_applications"]),
                "class_role": "unknown" if is_unknown else "known",
                "split": split,
                **flow,
            }
            rows.append(row)
            manifest_rows.append(row)

        split_stats = {}
        for split in ("train", "validation", "test", "unknown_test"):
            selected = [row for row in rows if row["split"] == split]
            vpn = sum(row["vpn_status"] == "vpn" for row in selected)
            nonvpn = sum(row["vpn_status"] == "nonvpn" for row in selected)
            split_stats[split] = {
                "flows": len(selected),
                "vpn_flows": vpn,
                "nonvpn_flows": nonvpn,
                "vpn_ratio": round(vpn / len(selected), 8) if selected else 0.0,
                "applications": sorted({row["application"] for row in selected}),
                "capture_count": len({row["capture_id"] for row in selected}),
                "group_count": len({row["group_id"] for row in selected}),
            }
        known_class_counts = {}
        for application in protocol["known_applications"]:
            known_class_counts[application] = {
                split: sum(row["application"] == application and row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            }
        known_rows = [row for row in rows if row["class_role"] == "known"]
        group_splits: dict[str, set[str]] = defaultdict(set)
        capture_splits: dict[str, set[str]] = defaultdict(set)
        for row in known_rows:
            group_splits[row["group_id"]].add(row["split"])
            capture_splits[row["capture_id"]].add(row["split"])
        protocol_records.append({
            **protocol,
            "split_statistics": split_stats,
            "known_class_split_counts": known_class_counts,
            "known_group_count": len(group_splits),
            "known_groups_spanning_multiple_splits": sum(len(values) > 1 for values in group_splits.values()),
            "known_capture_count": len(capture_splits),
            "known_captures_spanning_multiple_splits": sum(len(values) > 1 for values in capture_splits.values()),
        })
    manifest_rows.sort(key=lambda row: (
        SETTING_ORDER[row["setting"]], row["protocol_seed"], row["application"], row["split"], row["flow_uid"]
    ))
    return manifest_rows, protocol_records


def strict_unknown_free_checks(
    clean_flows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    protocol_records: list[dict[str, Any]],
    exclusions: dict[str, Any],
) -> dict[str, Any]:
    by_protocol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_protocol[row["protocol_id"]].append(row)
    clean_ids = {row["flow_uid"] for row in clean_flows}
    checks = []
    all_pass = True
    for protocol in protocol_records:
        rows = by_protocol[protocol["protocol_id"]]
        known = set(protocol["known_applications"])
        unknown = set(protocol["unknown_applications"])
        row_ids = [row["flow_uid"] for row in rows]
        conditions = {
            "class_sets_disjoint": known.isdisjoint(unknown),
            "class_sets_complete": known | unknown == set(APPLICATIONS),
            "all_clean_flows_present_once": len(row_ids) == len(clean_ids) and len(set(row_ids)) == len(clean_ids),
            "unknown_only_unknown_test": all(
                (row["split"] == "unknown_test" and row["class_role"] == "unknown")
                for row in rows if row["application"] in unknown
            ),
            "known_only_known_splits": all(
                row["split"] in {"train", "validation", "test"} and row["class_role"] == "known"
                for row in rows if row["application"] in known
            ),
            "all_known_classes_nonempty_each_split": all(
                any(row["application"] == application and row["split"] == split for row in rows)
                for application in known for split in ("train", "validation", "test")
            ),
            "failed_pcaps_absent": not ({row["capture_id"] for row in rows} & set(exclusions["failed_pcaps"])),
            "duplicate_pcap_drops_absent": not ({row["capture_id"] for row in rows} & set(exclusions["dropped_duplicate_captures"])),
            "cross_application_duplicates_absent": all(
                (row["capture_id"], row["source_flow_id"]) not in exclusions["cross_application_keys"]
                for row in rows
            ),
        }
        passed = all(conditions.values())
        all_pass &= passed
        checks.append({"protocol_id": protocol["protocol_id"], "pass": passed, "conditions": conditions})
    return {
        "status": "PASS" if all_pass else "FAIL",
        "protocol_count": len(protocol_records),
        "checks": checks,
        "strict_unknown_free_definition": "unknown applications are class-held-out and their flows appear only in unknown_test",
        "capture_disjoint_known_split": False,
        "capture_disjoint_note": "not claimed; user explicitly requested flow-random Known 8:1:1 with capture/group retained only as metadata",
    }


def protocol_hash(protocol: dict[str, Any]) -> str:
    value = json.loads(json.dumps(protocol))
    value.pop("freeze_hash", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def report_markdown(
    class_audit: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> str:
    records = protocol["protocols"]
    retained = [row["application"] for row in class_audit if row["status"] == "retained"]
    excluded = [row["application"] for row in class_audit if row["status"] == "excluded"]
    lines = [
        "# Stage 14B — VNAT Open-Set Protocol Freeze",
        "",
        f"- Freeze status: **{protocol['freeze_status']}**",
        f"- Freeze hash: `{protocol['freeze_hash']}`",
        "- This stage generated data manifests only. No model, embedding, Open-Detect, DES, threshold score, or test metric was produced or consulted.",
        "- Semantic class is `application`; VPN/non-VPN is metadata only.",
        "- Known split is deterministic class-stratified flow-random 8:1:1. Capture/group metadata is retained but is not a hard split boundary, exactly as requested.",
        "- Claim boundary: Strict Unknown-Free is enforced, but Known Train/Validation/Test is **not capture-disjoint** and must not be reported as capture-generalization evidence.",
        "",
        "## 1–2. Final class audit and clean flow counts",
        "",
        "| application | status | raw readable | cross-app removed | duplicate-PCAP removed | clean flows | VPN/non-VPN | Known 8:1:1 if known | failed PCAP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in class_audit:
        lines.append(
            f"| {row['application']} | {row['status']} | {row['raw_readable_flow_count']} | "
            f"{row['removed_cross_application_duplicate_flows']} | {row['removed_duplicate_pcap_flows']} | "
            f"{row['cleaned_flow_count']} | {row['vpn_flow_count']}/{row['nonvpn_flow_count']} | "
            f"{row['known_train_flows_if_known']}/{row['known_validation_flows_if_known']}/{row['known_test_flows_if_known']} | "
            f"{row['failed_pcap_ids'] or 'none'} |"
        )
    lines.extend([
        "",
        f"- Retained: **{retained}**.",
        f"- Excluded: **{excluded if excluded else 'none'}**. All ten classes have at least 44 clean flows and support non-empty 8:1:1 when Known.",
        "- Corrupt PCAP flow contribution is zero because each failed PCAP is excluded before constructing the flow manifest. Its unparsed flow count is not inferred.",
        "",
        "## 3. Frozen setting sizes",
        "",
        "| setting | Known classes | Unknown classes | Unknown share | seeds |",
        "|---|---:|---:|---:|---|",
        "| Low | 8 | 2 | 20% | 2022–2026 |",
        "| Medium | 7 | 3 | 30% | 2022–2026 |",
        "| High | 6 | 4 | 40% | 2022–2026 |",
        "",
        "The 2/3/4 design is based on ten retained classes: it avoids a single-class Low setting, preserves at least six Known classes at High, and is not the previous 1/3/5 schedule. Unknown sets are nested within each seed and balanced across five seeds.",
        "",
        "## 4–6. Frozen Unknown applications, Known split sizes, and VPN/non-VPN",
        "",
        "| protocol | Unknown applications | Known Train/Val/Test | Train VPN/non-VPN | Val VPN/non-VPN | Test VPN/non-VPN | Unknown VPN/non-VPN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        stats = record["split_statistics"]
        lines.append(
            f"| {record['protocol_id']} | {', '.join(record['unknown_applications'])} | "
            f"{stats['train']['flows']}/{stats['validation']['flows']}/{stats['test']['flows']} | "
            f"{stats['train']['vpn_flows']}/{stats['train']['nonvpn_flows']} | "
            f"{stats['validation']['vpn_flows']}/{stats['validation']['nonvpn_flows']} | "
            f"{stats['test']['vpn_flows']}/{stats['test']['nonvpn_flows']} | "
            f"{stats['unknown_test']['vpn_flows']}/{stats['unknown_test']['nonvpn_flows']} |"
        )
    clean_counts = [int(row["cleaned_flow_count"]) for row in class_audit]
    group_overlap_counts = [int(record["known_groups_spanning_multiple_splits"]) for record in records]
    capture_overlap_counts = [int(record["known_captures_spanning_multiple_splits"]) for record in records]
    lines.extend([
        "",
        "## 7. Empty, small, and imbalanced classes",
        "",
        "- Empty classes: **0**. Classes unable to form non-empty 8:1:1: **0**.",
        f"- Class-size imbalance remains substantial: minimum **{min(clean_counts)}** flows (RDP), maximum **{max(clean_counts)}** (SSH), ratio **{max(clean_counts)/min(clean_counts):.1f}×**.",
        "- RDP is retained under the requested flow-count rule but remains statistically small: when Known it contributes only 36/4/4 Train/Validation/Test flows.",
        "- VPN domain imbalance is severe for most classes: only Zoiper has a substantial VPN fraction; several classes have fewer than 1% VPN flows. The exact per-protocol split counts are frozen above and in the JSON.",
        f"- Flow-random splitting actually places **{min(group_overlap_counts)}–{max(group_overlap_counts)} groups** and **{min(capture_overlap_counts)}–{max(capture_overlap_counts)} captures** across multiple Known splits, depending on protocol. This is permitted by the requested Stage 14B rule, but creates a capture-leakage risk for Known generalization and must be tested later as sensitivity analysis.",
        "- Vimeo has only one conservative group; its Known Train/Validation/Test samples necessarily share that group. This does not violate class-held-out Unknown-Free, but it prevents a capture-generalization claim.",
        "",
        "## 8. Strict Unknown-Free verification",
        "",
        f"- Status: **{protocol['verification']['status']}** across all {len(records)} setting × seed protocols.",
        "- Every Unknown application is absent from Known Train, Validation, Test, scaler fitting, prototype/support fitting, threshold calibration, and parameter selection.",
        "- Unknown flows occur only in `unknown_test`; all VPN and non-VPN flows of an Unknown application move together.",
        "- Frozen downstream roles: encoder/scaler/prototype/support fit on Known Train only; threshold is global Known Validation P95 only; Known Test and Unknown Test are evaluation-only.",
        "",
        "## 9. Stage 14C readiness",
        "",
        "**READY_FOR_STAGE_14C under the frozen flow-random protocol.** Stage 14C must verify the freeze hash before use. Unknown classes, seeds, 8:1:1 assignments, exclusions, and the global Known-Validation-P95 threshold rule are immutable and may not be changed after inspecting VNAT Test results.",
        "",
        "No Stage 14C training was started.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exclusions = integrity_exclusions()
    clean_flows, class_audit = build_clean_flows(exclusions)
    retained = [row["application"] for row in class_audit if row["status"] == "retained"]
    excluded = [row["application"] for row in class_audit if row["status"] == "excluded"]
    if retained != APPLICATIONS or excluded:
        raise AssertionError(f"Expected all ten applications retained after flow-level audit: retained={retained}, excluded={excluded}")

    base_order, schedule = balanced_nested_unknown_schedule(retained)
    manifest_rows, protocol_records = build_protocols(clean_flows, schedule)
    write_csv(STAGE_ROOT / "vnat_final_class_audit.csv", class_audit)
    write_csv(STAGE_ROOT / "vnat_split_manifest.csv", manifest_rows)

    split_manifest_hash = sha256_file(STAGE_ROOT / "vnat_split_manifest.csv")
    verification = strict_unknown_free_checks(clean_flows, manifest_rows, protocol_records, exclusions)
    if verification["status"] != "PASS":
        raise AssertionError("Strict Unknown-Free verification failed")
    frozen_at = dt.datetime.now(dt.timezone.utc).isoformat()
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Stage 14B — VNAT Open-Set Protocol Freeze",
        "freeze_status": "FROZEN_READY_FOR_STAGE_14C",
        "frozen_at_utc": frozen_at,
        "semantic_class": "application",
        "metadata_domains": ["vpn_status", "capture_id", "group_id"],
        "retained_applications": retained,
        "excluded_applications": excluded,
        "clean_flow_count": len(clean_flows),
        "cleaning": {
            "cross_application_duplicate_flow_rows_removed": len(exclusions["cross_application_keys"]),
            "cross_application_duplicate_flow_keys": sorted([list(value) for value in exclusions["cross_application_keys"]]),
            "failed_pcaps_excluded_whole": exclusions["failed_pcaps"],
            "exact_duplicate_pcaps_retained": exclusions["retained_duplicate_captures"],
            "exact_duplicate_pcaps_dropped": exclusions["dropped_duplicate_captures"],
        },
        "known_split": {
            "method": "per-application deterministic flow-random stratification",
            "ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "integer_rule": "validation=test=max(1,floor(class_flows/10)); train=remainder",
            "capture_group_hard_constraint": False,
            "capture_group_metadata_retained": True,
            "same_application_seed_assignment_reused_across_settings": True,
        },
        "unknown_schedule": {
            "master_seed": MASTER_UNKNOWN_SEED,
            "protocol_seeds": PROTOCOL_SEEDS,
            "base_application_order": base_order,
            "method": "balanced nested cyclic schedule",
            "settings": {
                setting: {
                    "known_class_count": len(retained) - count,
                    "unknown_class_count": count,
                    "unknown_share": count / len(retained),
                }
                for setting, count in SETTING_UNKNOWN_COUNTS.items()
            },
        },
        "downstream_data_roles": {
            "encoder_training": "known train only",
            "scaler_fitting": "known train only",
            "prototype_support_fitting": "known train only",
            "checkpoint_selection": "known validation only",
            "threshold_calibration": "global Known Validation P95 only",
            "known_test": "evaluation only",
            "unknown_test": "evaluation only; never calibration or parameter selection",
        },
        "immutability": {
            "locked_fields": [
                "retained/excluded applications",
                "integrity exclusions",
                "Unknown applications per setting and seed",
                "protocol seeds",
                "Known 8:1:1 flow assignments",
                "threshold calibration rule",
            ],
            "post_test_changes_forbidden": True,
        },
        "source_hashes": {
            str(MANIFEST_PATH.relative_to(PROJECT_ROOT)): sha256_file(MANIFEST_PATH),
            str(DUPLICATE_FLOWS_PATH.relative_to(PROJECT_ROOT)): sha256_file(DUPLICATE_FLOWS_PATH),
            str(DUPLICATE_FILES_PATH.relative_to(PROJECT_ROOT)): sha256_file(DUPLICATE_FILES_PATH),
        },
        "split_manifest": {
            "path": "vnat_split_manifest.csv",
            "rows": len(manifest_rows),
            "sha256": split_manifest_hash,
        },
        "protocols": protocol_records,
        "verification": verification,
        "forbidden_actions": {
            "model_training": False,
            "open_detect_run": False,
            "des_run": False,
            "model_result_used_for_protocol": False,
        },
    }
    protocol["freeze_hash"] = protocol_hash(protocol)
    write_json(STAGE_ROOT / "vnat_open_set_protocol.json", protocol)
    (STAGE_ROOT / "stage14b_protocol_freeze.md").write_text(report_markdown(class_audit, protocol), encoding="utf-8")
    write_json(OUTPUT_ROOT / "freeze_hashes.json", {
        "freeze_hash": protocol["freeze_hash"],
        "vnat_open_set_protocol_sha256": sha256_file(STAGE_ROOT / "vnat_open_set_protocol.json"),
        "vnat_split_manifest_sha256": split_manifest_hash,
        "vnat_final_class_audit_sha256": sha256_file(STAGE_ROOT / "vnat_final_class_audit.csv"),
        "stage14b_protocol_freeze_sha256": sha256_file(STAGE_ROOT / "stage14b_protocol_freeze.md"),
    })
    write_csv(OUTPUT_ROOT / "integrity_exclusions.csv", [
        {
            "exclusion_type": "cross_application_duplicate_flow",
            "capture_id": capture_id,
            "flow_id": flow_id,
            "action": "permanently excluded from every protocol",
        }
        for capture_id, flow_id in sorted(exclusions["cross_application_keys"])
    ] + [
        {
            "exclusion_type": "failed_pcap",
            "capture_id": capture_id,
            "flow_id": "",
            "action": "entire PCAP excluded; no parsed flow contribution",
        }
        for capture_id in exclusions["failed_pcaps"]
    ] + [
        {
            "exclusion_type": "exact_duplicate_pcap",
            "capture_id": capture_id,
            "flow_id": "",
            "action": "entire duplicate PCAP contribution excluded",
        }
        for capture_id in exclusions["dropped_duplicate_captures"]
    ])
    print(json.dumps({
        "freeze_status": protocol["freeze_status"],
        "freeze_hash": protocol["freeze_hash"],
        "retained_applications": retained,
        "excluded_applications": excluded,
        "clean_flows": len(clean_flows),
        "manifest_rows": len(manifest_rows),
        "protocols": len(protocol_records),
        "strict_unknown_free": verification["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
