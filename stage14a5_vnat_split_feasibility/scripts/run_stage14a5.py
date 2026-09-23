#!/usr/bin/env python3
"""Stage 14A.5: VNAT duplicate quarantine and split-feasibility audit."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parents[1]
STAGE14A = PROJECT_ROOT / "stage14a_vnat_data_audit"
SCAN_ROOT = STAGE14A / "artifacts" / "pcap_scans"
OUTPUT_ROOT = STAGE_ROOT / "outputs"

MANIFEST_PATH = STAGE14A / "vnat_manifest.csv"
DUPLICATE_FLOWS_PATH = STAGE14A / "outputs" / "duplicate_flows.csv"

ELIGIBLE = "eligible"
BORDERLINE = "borderline"
EXCLUDED = "excluded"


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


def quarantine_rows() -> tuple[set[tuple[str, str]], list[dict[str, str]]]:
    rows = read_csv(DUPLICATE_FLOWS_PATH)
    selected = []
    keys = set()
    for row in rows:
        peers = {value for value in row["peer_applications"].split(";") if value}
        if any(peer != row["application"] for peer in peers):
            key = (row["capture_id"], row["flow_id"])
            keys.add(key)
            selected.append({
                **row,
                "quarantine_reason": "exact packet-sequence duplicate crosses application labels",
            })
    if len(keys) != 4:
        raise AssertionError(f"Expected exactly 4 cross-application duplicate flow rows, found {len(keys)}")
    return keys, selected


def build_group_statistics(
    quarantine: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = read_csv(MANIFEST_PATH)
    pcaps = [row for row in manifest if row["file_type"] == "pcap"]
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for pcap in pcaps:
        key = (pcap["application"], pcap["group_id"])
        group = groups.setdefault(key, {
            "application": pcap["application"],
            "group_id": pcap["group_id"],
            "capture_ids": [],
            "source_group_ids": [],
            "pcap_count": 0,
            "readable_pcap_count": 0,
            "failed_pcap_count": 0,
            "raw_flow_count": 0,
            "quarantined_flow_count": 0,
            "flow_count": 0,
            "vpn_flow_count": 0,
            "nonvpn_flow_count": 0,
            "vpn_pcap_count": 0,
            "nonvpn_pcap_count": 0,
        })
        group["capture_ids"].append(pcap["capture_id"])
        group["source_group_ids"].append(pcap["source_group_id"])
        group["pcap_count"] += 1
        group[f"{pcap['vpn_status']}_pcap_count"] += 1
        if pcap["readable"] != "True":
            group["failed_pcap_count"] += 1
            continue
        group["readable_pcap_count"] += 1
        scan_path = SCAN_ROOT / f"{pcap['capture_id']}.json"
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        flows = scan.get("flows", [])
        group["raw_flow_count"] += len(flows)
        for flow in flows:
            if (pcap["capture_id"], flow["flow_id"]) in quarantine:
                group["quarantined_flow_count"] += 1
                continue
            group["flow_count"] += 1
            group[f"{pcap['vpn_status']}_flow_count"] += 1

    by_application: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        group["capture_ids"] = ";".join(sorted(group["capture_ids"]))
        group["source_group_ids"] = ";".join(sorted(group["source_group_ids"]))
        if group["vpn_flow_count"] and group["nonvpn_flow_count"]:
            composition = "both"
        elif group["vpn_flow_count"]:
            composition = "vpn_only"
        elif group["nonvpn_flow_count"]:
            composition = "nonvpn_only"
        else:
            composition = "no_usable_flows"
        group["vpn_nonvpn_composition"] = composition
        group["usable_for_split"] = group["flow_count"] > 0
        by_application[group["application"]].append(group)

    rows: list[dict[str, Any]] = []
    for application in sorted(by_application):
        usable = [row for row in by_application[application] if row["flow_count"] > 0]
        counts = [int(row["flow_count"]) for row in usable]
        total = sum(counts)
        minimum = min(counts) if counts else 0
        maximum = max(counts) if counts else 0
        mean = statistics.fmean(counts) if counts else 0.0
        cv = statistics.pstdev(counts) / mean if len(counts) > 1 and mean else 0.0
        max_share = maximum / total if total else 0.0
        ratio = maximum / minimum if minimum else math.inf
        severe = bool(max_share >= 0.80 or ratio >= 100.0)
        for row in sorted(by_application[application], key=lambda item: item["group_id"]):
            rows.append({
                **row,
                "application_usable_group_count": len(usable),
                "application_flow_count": total,
                "application_min_group_flows": minimum,
                "application_max_group_flows": maximum,
                "application_max_min_ratio": "inf" if math.isinf(ratio) else round(ratio, 6),
                "application_group_cv": round(cv, 6),
                "application_max_group_share": round(max_share, 6),
                "severe_group_imbalance": severe,
            })
    return rows, by_application


def partition_objective(bins: list[list[dict[str, Any]]]) -> tuple[int, int, float]:
    sums = [sum(int(item["flow_count"]) for item in bucket) for bucket in bins]
    total = sum(sums)
    return (-min(sums), max(sums) - min(sums), sum(abs(value - total / 3.0) for value in sums))


def label_bins(bins: list[list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        bins,
        key=lambda bucket: (
            -sum(int(item["flow_count"]) for item in bucket),
            ";".join(sorted(str(item["group_id"]) for item in bucket)),
        ),
    )
    return {"train": ordered[0], "val": ordered[1], "test": ordered[2]}


def exhaustive_maximin(groups: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    best_bins: list[list[dict[str, Any]]] | None = None
    best_key: tuple[Any, ...] | None = None
    evaluated = 0
    # Fix the first group to bin 0 to remove three-way label symmetry.
    for suffix in itertools.product(range(3), repeat=len(groups) - 1):
        assignment = (0, *suffix)
        if set(assignment) != {0, 1, 2}:
            continue
        bins = [[], [], []]
        for group, index in zip(groups, assignment):
            bins[index].append(group)
        evaluated += 1
        signature = tuple(
            sorted(tuple(sorted(str(item["group_id"]) for item in bucket)) for bucket in bins)
        )
        key = (*partition_objective(bins), signature)
        if best_key is None or key < best_key:
            best_key = key
            best_bins = bins
    if best_bins is None:
        raise ValueError("A non-empty three-way partition does not exist")
    return label_bins(best_bins), evaluated


def heuristic_maximin(groups: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bins: list[list[dict[str, Any]]] = [[], [], []]
    for group in sorted(groups, key=lambda item: (-int(item["flow_count"]), str(item["group_id"]))):
        index = min(range(3), key=lambda value: (sum(int(x["flow_count"]) for x in bins[value]), value))
        bins[index].append(group)

    # Deterministic single-move and pair-swap local improvement.
    while True:
        current = partition_objective(bins)
        best = current
        best_bins = None
        for source in range(3):
            if len(bins[source]) <= 1:
                continue
            for target in range(3):
                if source == target:
                    continue
                for item in bins[source]:
                    proposal = [list(bucket) for bucket in bins]
                    proposal[source].remove(item)
                    proposal[target].append(item)
                    score = partition_objective(proposal)
                    if score < best:
                        best, best_bins = score, proposal
        for left in range(3):
            for right in range(left + 1, 3):
                for left_item in bins[left]:
                    for right_item in bins[right]:
                        proposal = [list(bucket) for bucket in bins]
                        proposal[left].remove(left_item)
                        proposal[right].remove(right_item)
                        proposal[left].append(right_item)
                        proposal[right].append(left_item)
                        score = partition_objective(proposal)
                        if score < best:
                            best, best_bins = score, proposal
        if best_bins is None:
            break
        bins = best_bins
    return label_bins(bins)


def split_summary(application: str, all_groups: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted(
        [group for group in all_groups if int(group["flow_count"]) > 0],
        key=lambda item: str(item["group_id"]),
    )
    counts = [int(group["flow_count"]) for group in groups]
    total = sum(counts)
    minimum = min(counts) if counts else 0
    maximum = max(counts) if counts else 0
    ratio = maximum / minimum if minimum else math.inf
    max_share = maximum / total if total else 0.0
    mean = statistics.fmean(counts) if counts else 0.0
    cv = statistics.pstdev(counts) / mean if len(counts) > 1 and mean else 0.0
    row: dict[str, Any] = {
        "application": application,
        "usable_group_count": len(groups),
        "flow_count_after_quarantine": total,
        "quarantined_flow_count": sum(int(group["quarantined_flow_count"]) for group in all_groups),
        "min_group_flows": minimum,
        "max_group_flows": maximum,
        "max_min_group_ratio": "inf" if math.isinf(ratio) else round(ratio, 6),
        "group_cv": round(cv, 6),
        "max_group_share": round(max_share, 6),
        "severe_group_imbalance": bool(max_share >= 0.80 or ratio >= 100.0),
        "partition_objective": "maximize min(train,val,test) flows; then minimize range",
        "partition_method": "not_feasible",
        "partition_optimality": "not_applicable",
        "assignments_evaluated": 0,
        "train_group_count": 0,
        "val_group_count": 0,
        "test_group_count": 0,
        "train_flows": 0,
        "val_flows": 0,
        "test_flows": 0,
        "min_split_flows": 0,
        "train_k10_satisfied": False,
        "val_expected_p95_tail_count": 0.0,
        "val_empirical_quantile_resolution": "not_available",
        "val_p95_count_support": "not_available",
        "train_group_ids": "",
        "val_group_ids": "",
        "test_group_ids": "",
    }
    if len(groups) < 3:
        row["decision"] = EXCLUDED
        row["decision_reason"] = f"only {len(groups)} usable groups; non-empty group-disjoint Train/Val/Test is impossible"
        return row

    if len(groups) <= 10:
        partition, evaluated = exhaustive_maximin(groups)
        row["partition_method"] = "exact_enumeration"
        row["partition_optimality"] = "proven_maximin_for_observed_group_counts"
        row["assignments_evaluated"] = evaluated
    else:
        partition = heuristic_maximin(groups)
        row["partition_method"] = "deterministic_LPT_plus_local_improvement"
        row["partition_optimality"] = "feasible_lower_bound_not_global_proof"

    split_counts = {}
    for split in ("train", "val", "test"):
        bucket = partition[split]
        split_counts[split] = sum(int(item["flow_count"]) for item in bucket)
        row[f"{split}_group_count"] = len(bucket)
        row[f"{split}_flows"] = split_counts[split]
        row[f"{split}_group_ids"] = ";".join(sorted(str(item["group_id"]) for item in bucket))
    row["min_split_flows"] = min(split_counts.values())
    row["train_k10_satisfied"] = split_counts["train"] >= 10
    row["val_expected_p95_tail_count"] = round(split_counts["val"] * 0.05, 2)
    row["val_empirical_quantile_resolution"] = round(1.0 / split_counts["val"], 8)
    if split_counts["val"] >= 400:
        row["val_p95_count_support"] = "count_adequate_at_least_20_expected_tail_samples"
    elif split_counts["val"] >= 100:
        row["val_p95_count_support"] = "limited_5_to_19_expected_tail_samples"
    else:
        row["val_p95_count_support"] = "sparse_fewer_than_5_expected_tail_samples"

    reasons = []
    if int(row["min_split_flows"]) < 10:
        row["decision"] = EXCLUDED
        reasons.append(f"best possible minimum split has only {row['min_split_flows']} flows")
    else:
        if len(groups) == 3:
            reasons.append("exactly three groups gives one group per split and no reassignment robustness")
        if total < 100:
            reasons.append(f"only {total} total flows despite a technically feasible split")
        if max_share >= 0.80:
            reasons.append(f"largest group contains {max_share:.1%} of class flows")
        if 3 < len(groups) <= 4:
            reasons.append(f"only {len(groups)} usable groups limits split robustness")
        row["decision"] = BORDERLINE if reasons else ELIGIBLE
    if application == "rdp":
        reasons.append(f"DES-v1 k=10 training requirement {'is' if row['train_k10_satisfied'] else 'is not'} satisfied ({row['train_flows']} train flows)")
    row["decision_reason"] = "; ".join(reasons) if reasons else "adequate group count and maximin split support under this audit"
    return row


def report_markdown(
    groups: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    quarantine_rows_list: list[dict[str, str]],
) -> str:
    eligible = [row["application"] for row in splits if row["decision"] == ELIGIBLE]
    borderline = [row["application"] for row in splits if row["decision"] == BORDERLINE]
    excluded = [row["application"] for row in splits if row["decision"] == EXCLUDED]
    retained = eligible + borderline
    retained_rows = [row for row in splits if row["application"] in retained]
    global_val = sum(int(row["val_flows"]) for row in retained_rows)
    global_tail = global_val * 0.05

    lines = [
        "# Stage 14A.5 — VNAT Split Feasibility Audit",
        "",
        "## Scope and decision rules",
        "",
        "- This is a data/split audit only. No model was trained; no Unknown class or Low/Medium/High setting was generated; Open-Detect and DES were not run.",
        "- Four cross-application duplicate flow rows are quarantined before every count: two from `nonvpn_rsync_newcapture1` and two from `nonvpn_sftp_newcapture2`.",
        "- Input `group_id` is frozen from Stage 14A: VPN/non-VPN same-number conservative linkage plus exact-PCAP duplicate union.",
        "- No Train/Val/Test ratio was specified. Therefore the audit uses a maximin witness: maximize the smallest split flow count, then minimize split range. This tests whether any defensible three-way allocation exists; it is not a frozen Stage 14B split.",
        "- Up to 10 groups are exhaustively enumerated. Skype (56 groups) uses deterministic longest-processing-time allocation plus local moves/swaps, so its result is a feasible lower bound rather than a global optimality proof.",
        "- Decision rule derived after profiling: fewer than three usable groups or best maximin split below 10 flows is `excluded`; technically feasible classes with exactly three groups, fewer than 100 total flows, at most four usable groups, or a single group containing at least 80% are `borderline`; the rest are `eligible`.",
        "- P95 support labels are sample-count diagnostics, not a theorem about score stability: fewer than 5 expected upper-tail samples is sparse, 5–19 limited, and at least 20 count-adequate. Actual threshold stability cannot be established without validation scores.",
        "",
        "## Split feasibility by application",
        "",
        "| application | groups | flows | group min/max | max group share | train/val/test flows | min split | train k=10 | val P95 tail | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in splits:
        lines.append(
            f"| {row['application']} | {row['usable_group_count']} | {row['flow_count_after_quarantine']} | "
            f"{row['min_group_flows']}/{row['max_group_flows']} | {float(row['max_group_share']):.1%} | "
            f"{row['train_flows']}/{row['val_flows']}/{row['test_flows']} | {row['min_split_flows']} | "
            f"{row['train_k10_satisfied']} | {row['val_expected_p95_tail_count']} | **{row['decision']}** |"
        )
    lines.extend([
        "",
        "## Group imbalance findings",
        "",
        "- The CSV marks severe imbalance when the largest group contributes at least 80% of class flows or the observed max/min group ratio is at least 100. This is a diagnostic flag, not an automatic exclusion rule.",
        "- Severe raw group imbalance is present for `rsync`, `scp`, `sftp`, `ssh`, and the single-group `vimeo`. For Rsync/SCP it produces unusably tiny held-out splits; for SSH it creates dominant-capture dependence.",
        "- SFTP is retained despite a 709:1 raw group range because seven groups permit five smaller groups to be combined into a 313-flow held-out split; its maximin support is therefore materially better than Rsync/SCP.",
        "- `vnat_group_statistics.csv` records every `group_id`, its VPN/non-VPN flow counts, capture membership, failed-PCAP count, quarantine count, and class-level imbalance metrics.",
        "",
        "## Targeted checks",
        "",
    ])
    by_name = {row["application"]: row for row in splits}
    rdp = by_name["rdp"]
    scp = by_name["scp"]
    zoiper = by_name["zoiper"]
    ssh = by_name["ssh"]
    lines.extend([
        f"- **RDP / DES-v1 k=10:** PASS at the count level: the maximin witness has {rdp['train_flows']} training flows. It remains borderline because only {rdp['flow_count_after_quarantine']} total flows produce {rdp['val_flows']}/{rdp['test_flows']} held-out flows and fewer than one expected P95-tail validation sample.",
        f"- **SCP:** too fragile and excluded. Its three usable groups yield {scp['train_flows']}/{scp['val_flows']}/{scp['test_flows']}; the best possible minimum is only {scp['min_split_flows']} flow.",
        f"- **Zoiper:** balanced in flow volume ({zoiper['train_flows']}/{zoiper['val_flows']}/{zoiper['test_flows']}) but borderline because exactly three groups force one group per split; there is no alternative grouping robustness.",
        f"- **SSH:** extreme concentration is confirmed: the largest group holds {float(ssh['max_group_share']):.1%} of flows. A count-rich split exists ({ssh['train_flows']}/{ssh['val_flows']}/{ssh['test_flows']}), but its semantics depend strongly on assigning the dominant capture to Train, so SSH is borderline.",
        f"- **Known Validation P95:** retaining eligible plus borderline classes gives {global_val:,} validation flows and about {global_tail:.1f} expected upper-tail observations. This is count-adequate for one global P95, but class composition is highly uneven and score-level/bootstrap stability remains untested.",
        "",
        "## Classification",
        "",
        f"- `eligible`: **{', '.join(eligible)}**.",
        f"- `borderline`: **{', '.join(borderline)}**.",
        f"- `excluded`: **{', '.join(excluded)}**.",
    ])
    for row in splits:
        lines.append(f"  - `{row['application']}`: {row['decision_reason']}.")
    lines.extend([
        "",
        "## Required final conclusion",
        "",
        f"1. 最终推荐保留哪些 application：核心 eligible 为 **{eligible}**；为维持可用类别规模，可将 **{borderline}** 作为带显式限制的候选一并带入 Stage 14B，但不得在该阶段掩盖其脆弱性。",
        f"2. 哪些需要排除及原因：**{excluded}**。Netflix/Vimeo 无法形成三路非空 group split；Rsync/SCP 即使采用最优 maximin 分组，最小 held-out split 也只有 2/1 条 flow。",
        "3. 是否可以正式进入 Stage 14B Protocol Freeze：**CONDITIONAL YES**。Stage 14B 只能从上述 eligible + 明确标记的 borderline 集合中冻结协议，必须保持 4 条跨 application duplicate flow 永久隔离；不得重新纳入 excluded 类，也不得把本次 maximin witness 误称为已经冻结的最终协议。",
        "",
        f"Quarantined rows: {len(quarantine_rows_list)}. No Unknown setting was generated.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    quarantine, quarantine_detail = quarantine_rows()
    group_rows, by_application = build_group_statistics(quarantine)
    split_rows = [
        split_summary(application, by_application[application])
        for application in sorted(by_application)
    ]

    write_csv(STAGE_ROOT / "vnat_group_statistics.csv", group_rows)
    write_csv(STAGE_ROOT / "vnat_split_feasibility.csv", split_rows)
    write_csv(OUTPUT_ROOT / "quarantined_duplicate_flows.csv", quarantine_detail)
    write_json(OUTPUT_ROOT / "input_hashes.json", {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            str(MANIFEST_PATH.relative_to(PROJECT_ROOT)): sha256_file(MANIFEST_PATH),
            str(DUPLICATE_FLOWS_PATH.relative_to(PROJECT_ROOT)): sha256_file(DUPLICATE_FLOWS_PATH),
        },
    })
    write_json(OUTPUT_ROOT / "audit_details.json", {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_stage14a_flows": sum(int(row["flow_count_after_quarantine"]) + int(row["quarantined_flow_count"]) for row in split_rows),
        "quarantined_cross_application_flow_rows": len(quarantine),
        "flows_after_quarantine": sum(int(row["flow_count_after_quarantine"]) for row in split_rows),
        "applications": len(split_rows),
        "eligible": [row["application"] for row in split_rows if row["decision"] == ELIGIBLE],
        "borderline": [row["application"] for row in split_rows if row["decision"] == BORDERLINE],
        "excluded": [row["application"] for row in split_rows if row["decision"] == EXCLUDED],
        "model_training": False,
        "unknown_setting_generated": False,
        "open_detect_run": False,
        "des_run": False,
    })
    (STAGE_ROOT / "stage14a5_split_audit.md").write_text(
        report_markdown(group_rows, split_rows, quarantine_detail),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS_WITH_QUALITY_LIMITATIONS",
        "quarantined_rows": len(quarantine),
        "flows_after_quarantine": sum(int(row["flow_count_after_quarantine"]) for row in split_rows),
        "eligible": [row["application"] for row in split_rows if row["decision"] == ELIGIBLE],
        "borderline": [row["application"] for row in split_rows if row["decision"] == BORDERLINE],
        "excluded": [row["application"] for row in split_rows if row["decision"] == EXCLUDED],
    }, indent=2))


if __name__ == "__main__":
    main()
