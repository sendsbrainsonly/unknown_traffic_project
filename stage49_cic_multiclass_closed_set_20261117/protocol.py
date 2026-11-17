#!/usr/bin/env python3
"""Freeze the preregistered eleven-class CIC task using metadata only."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT.parents[1] / "Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)"
FLOW_ROOT = DATA / "outputs/cicids2017_pcap_label_mapping/flows"
AUDIT_ROOT = PROJECT / "outputs/cicids2017_existing_mapping_audit"
MANIFEST = ROOT / "split_manifest.csv"
ROLES = ("train", "validation", "test")
SEED = 2022
CLASSES = tuple(sorted(("BENIGN", "Bot", "DDoS", "DoS Hulk", "DoS Slowhttptest",
                       "DoS slowloris", "FTP-Patator", "PortScan", "SSH-Patator",
                       "Web Attack - Brute Force", "Web Attack - XSS")))
FIELDS = ("flow_id", "day", "source_pcap", "label", "match_status", "flow_start_epoch_utc",
          "canonical_flow_key", "first_packet_index", "last_packet_index", "packet_count")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group(row):
    return f"{row['source_pcap']}|{math.floor(float(row['flow_start_epoch_utc']) / 300)}"


def scan(columns=FIELDS):
    files = sorted(FLOW_ROOT.glob("*_flows.parquet"))
    if len(files) != 5:
        raise RuntimeError(f"expected five mapped weekday files, found{len(files)}")
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=list(columns)):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                yield {key: d[key][i] for key in columns}


def solve_groups(counts):
    """Metadata-only global assignment; shared groups have one common role."""
    names = sorted(counts)
    totals = Counter()
    for values in counts.values():
        totals.update(values)
    g, c = len(names), len(CLASSES)
    n_binary = g * 3
    n = n_binary + c * 3 * 2
    objective = np.zeros(n)
    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    upper[:n_binary] = 1
    integrality = np.zeros(n)
    integrality[:n_binary] = 1
    # Tiny fixed tie-break only; principal objective is equal-weight relative
    # per-class count deviation from8:1:1, never a model/feature measurement.
    for i, name in enumerate(names):
        for r, role in enumerate(ROLES):
            h = int(hashlib.sha256(f"{SEED}|{name}|{role}".encode()).hexdigest()[:8], 16)
            objective[i * 3 + r] = h / 2**32 * 1e-10
    a = lil_matrix((g + c * 6, n), dtype=np.float64)
    lo = np.zeros(g + c * 6)
    hi = np.zeros_like(lo)
    for i in range(g):
        a[i, i * 3:i * 3 + 3] = 1
        lo[i] = hi[i] = 1
    fractions = (.8, .1, .1)
    for j, label in enumerate(CLASSES):
        for r in range(3):
            row = g + j * 6 + r * 2
            positions = [i * 3 + r for i, name in enumerate(names) if counts[name][label]]
            values = [counts[names[pos // 3]][label] for pos in positions]
            a[row, positions] = values
            lo[row], hi[row] = 1, np.inf
            a[row + 1, positions] = values
            deviation = n_binary + (j * 3 + r) * 2
            a[row + 1, deviation] = 1
            a[row + 1, deviation + 1] = -1
            lo[row + 1] = hi[row + 1] = totals[label] * fractions[r]
            objective[deviation:deviation + 2] = 1 / totals[label]
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(a.tocsr(), lo, hi),
                  options={"time_limit": 120.0, "mip_rel_gap": .01})
    if result.x is None:
        raise RuntimeError(f"no feasible metadata-only grouped split: {result.message}")
    binary = result.x[:n_binary].reshape(g, 3)
    if np.max(np.abs(binary - np.rint(binary))) > 1e-6 or not np.all(np.rint(binary).sum(1) == 1):
        raise RuntimeError("solver returned a nonintegral group assignment")
    chosen = {name: ROLES[int(binary[i].argmax())] for i, name in enumerate(names)}
    solver = {"method": "scipy.optimize.milp / HiGHS", "status": int(result.status),
              "message": result.message, "objective": float(result.fun),
              "time_limit_seconds": 120, "relative_gap_target": .01,
              "mip_gap": float(result.mip_gap), "used_model_outcomes": False}
    return chosen, solver


def source_files():
    paths = [*FLOW_ROOT.glob("*_flows.parquet"), AUDIT_ROOT / "label_mismatches.csv",
             AUDIT_ROOT / "class_count_comparison.csv", ROOT / "PLAN.md",
             PROJECT / "stage34b_cic_balanced_retest/RESULTS.md",
             PROJECT / "stage34b_cic_balanced_retest/completion_verification.json"]
    for directory in ("stage31_four_dataset_three_view_equal",
                      "stage22_pretrained_trafficformer_e3_closed_set_comparison/scripts",
                      "stage34_ustc_cic_closed_set", "src/preprocessing"):
        paths.extend((PROJECT / directory).glob("*.py"))
    paths.extend(ROOT.glob("*.py"))
    return sorted(set(p.resolve() for p in paths))


@lru_cache(maxsize=1)
def verified_rows():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    for name, digest in lock["frozen_outputs"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"frozen protocol artifact changed: {name}")
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != protocol["total_samples"] or len({r["flow_id"] for r in rows}) != len(rows):
        raise RuntimeError("sample count/unique ID mismatch")
    groups = {}
    counts = Counter()
    for row in rows:
        gid, role = row["group_id"], row["split"]
        if groups.setdefault(gid, role) != role:
            raise RuntimeError(f"group overlap: {gid}")
        if row["label"] not in CLASSES or role not in ROLES or group(row) != gid:
            raise RuntimeError("label/role/group mismatch")
        counts[row["label"], role] += 1
    if min(counts[label, role] for label in CLASSES for role in ROLES) < 1:
        raise RuntimeError("class missing in a role")
    for role in ROLES:
        if counts["BENIGN", role] != sum(counts[label, role] for label in CLASSES if label != "BENIGN"):
            raise RuntimeError("benign/attack balance mismatch")
    return rows


def verify_sources():
    before = json.loads((ROOT / "source_hashes_before.json").read_text())
    after = {name: sha(name) for name in before}
    if before != after:
        raise RuntimeError("a registered source changed")
    return after


def main():
    if MANIFEST.exists() or (ROOT / "protocol.json").exists():
        raise FileExistsError("protocol already exists; do not regenerate frozen roles")
    with (AUDIT_ROOT / "label_mismatches.csv").open(encoding="utf-8-sig", newline="") as f:
        mismatches = {r["flow_id"]: r for r in csv.DictReader(f)}
    if len(mismatches) != 783:
        raise RuntimeError("independent-audit mismatch count changed")
    with (AUDIT_ROOT / "class_count_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        reference = {r["class_name"]: r for r in csv.DictReader(f)}
    raw = Counter()
    clean = Counter()
    statuses = Counter()
    counts = defaultdict(Counter)
    quarantined = []
    for row in scan():
        statuses[str(row["match_status"])] += 1
        if not str(row["match_status"]).startswith("MATCHED") or row["label"] is None:
            continue
        label = row["label"]
        raw[label] += 1
        if row["flow_id"] in mismatches:
            if mismatches[row["flow_id"]]["existing_label"] != label:
                raise RuntimeError("mismatch audit no longer aligns to mapping labels")
            quarantined.append({"flow_id": row["flow_id"], "label": label,
                                "reason": "historical independent LABEL_MISMATCH"})
            continue
        clean[label] += 1
        if label in CLASSES:
            counts[group(row)][label] += 1
    for label, ref in reference.items():
        if raw[label] != int(ref["existing_mapping_count"]) or clean[label] != int(ref["verified_label_match_count"]):
            raise RuntimeError(f"mapping/independent-audit count drift: {label}")
    if len(quarantined) != 783:
        raise RuntimeError("not every disagreement row was quarantined")
    print(json.dumps({"phase": "verified_metadata", "matched": sum(raw.values()),
                      "label_rechecked": sum(clean.values()), "groups": len(counts)}), flush=True)
    assignment, solver = solve_groups(counts)
    whole = Counter()
    for gid, values in counts.items():
        for label, value in values.items():
            whole[label, assignment[gid]] += value
    target = {}
    for label in CLASSES:
        if label == "BENIGN":
            continue
        fraction = .1 if clean[label] >= 10000 else 1.0
        for role in ROLES:
            target[label, role] = max(1, round(whole[label, role] * fraction))
    for role in ROLES:
        target["BENIGN", role] = sum(target[label, role] for label in CLASSES if label != "BENIGN")
        if target["BENIGN", role] > whole["BENIGN", role]:
            raise RuntimeError(f"not enough BENIGN flows for registered balancing in {role}")
    print(json.dumps({"phase": "grouped_split_frozen", "solver": solver,
                      "target_samples": sum(target.values())}), flush=True)
    heaps = defaultdict(list)
    for row in scan():
        label = row["label"]
        if label not in CLASSES or not str(row["match_status"]).startswith("MATCHED") or row["flow_id"] in mismatches:
            continue
        gid = group(row)
        role = assignment[gid]
        row.update(group_id=gid, split=role)
        rank = int.from_bytes(hashlib.sha256(f"{SEED}|{label}|{role}|{row['flow_id']}".encode()).digest(), "big")
        key = (label, role)
        entry = (-rank, row["flow_id"], row)
        if len(heaps[key]) < target[key]:
            heapq.heappush(heaps[key], entry)
        elif entry > heaps[key][0]:
            heapq.heapreplace(heaps[key], entry)
    selected = sorted((entry[2] for heap in heaps.values() for entry in heap), key=lambda r: r["flow_id"])
    if len(selected) != sum(target.values()):
        raise RuntimeError("selected sample count does not match targets")
    write_csv(MANIFEST, (*FIELDS, "group_id", "split"), selected)
    write_csv(ROOT / "quarantined_label_disagreements.csv", ("flow_id", "label", "reason"), quarantined)
    write_csv(ROOT / "group_assignments.csv", ("group_id", "split", *CLASSES),
              ({"group_id": gid, "split": assignment[gid], **{label: counts[gid][label] for label in CLASSES}}
               for gid in sorted(counts)))
    actual = Counter((r["label"], r["split"]) for r in selected)
    records = []
    for label in CLASSES:
        for role in ROLES:
            selected_groups = {r["group_id"] for r in selected if r["label"] == label and r["split"] == role}
            records.append({"class": label, "split": role, "whole_group_flows": whole[label, role],
                            "selected_flows": actual[label, role], "selected_groups": len(selected_groups)})
    write_csv(ROOT / "split_counts.csv", tuple(records[0]), records)
    inventory = []
    for label in sorted(reference):
        kept = label in CLASSES
        inventory.append({"class": label, "official_csv_rows": int(reference[label]["official_csv_count"]),
                          "matched_flows": raw[label], "label_disagreement_removed": raw[label] - clean[label],
                          "verified_flows": clean[label], "included": kept,
                          "selected_flows": sum(actual[label, role] for role in ROLES),
                          "reason": ("retained; endpoint/schedule warnings disclosed" if kept else
                                     "two five-minute groups cannot populate three roles" if label == "DoS GoldenEye" else
                                     "only11-34 reliable flows; insufficient sample support")})
    write_csv(ROOT / "class_audit.csv", tuple(inventory[0]), inventory)
    pcaps = [{"path": str(p), "size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
             for p in sorted((DATA / "PCAPs").glob("*.pcap"))]
    config = {"status": "FROZEN", "dataset": "CIC-IDS-2017", "task": "eleven-class closed-set",
              "seed": SEED, "classes": CLASSES, "group_rule": "source_pcap|floor(flow_start_epoch_utc/300)",
              "solver": solver, "large_attack_threshold": 10000, "large_attack_sample_rate": .1,
              "small_attack_sample_rate": 1, "benign_rule": "equal to total selected attacks per split",
              "sample_rank": "SHA256(seed|class|split|flow_id)", "total_samples": len(selected),
              "split_counts": {role: sum(actual[label, role] for label in CLASSES) for role in ROLES},
              "matched_labelled": sum(raw.values()), "independently_verified": sum(clean.values()),
              "quarantined": len(quarantined), "unmatched": statuses["UNMATCHED"],
              "unknown_samples_used": 0, "test_values_used_for_protocol_selection": 0,
              "sample_manifest_sha256": sha(MANIFEST), "pcaps_read_only": pcaps,
              "limitation": "previously exposed development dataset; class-day and endpoint confounding remains"}
    write_json(ROOT / "protocol.json", config)
    write_json(ROOT / "cicids2017_protocol_v2_audit.json", {"status": "PASS",
               "v2_manifest_sha256": sha(MANIFEST), "actual_protocol": "Stage49-eleven-class-group-disjoint",
               "classes": CLASSES, "meaning": "Stage34 builder API compatibility only"})
    sources = {str(p): sha(p) for p in source_files()}
    write_json(ROOT / "source_hashes_before.json", sources)
    frozen = {name: sha(ROOT / name) for name in ("split_manifest.csv", "group_assignments.csv",
              "protocol.json", "cicids2017_protocol_v2_audit.json", "class_audit.csv", "split_counts.csv")}
    write_json(ROOT / "protocol_lock.json", {"frozen_outputs": frozen, "source_count": len(sources)})
    verified_rows()
    write_json(ROOT / "protocol_verification.json", {"status": "PASS", "groups_disjoint": True,
               "classes_in_every_role": True, "label_audit_counts_match": True,
               "selected_samples": len(selected), "split_counts": config["split_counts"],
               "selected_quarantined_samples": 0, "test_feature_values_read": 0})
    print(json.dumps({"status": "PASS", "classes": len(CLASSES), "splits": config["split_counts"],
                      "protocol_sha256": sha(ROOT / "protocol.json")}), flush=True)


if __name__ == "__main__":
    main()
