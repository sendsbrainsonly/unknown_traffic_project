#!/usr/bin/env python3
"""Freeze a three-class, five-minute-group-disjoint CIC-IDS-2017 sample.

Uses the existing mapped-flow Parquet and labels as-is.  No model, packet
features, or validation outcomes are consulted.  Group assignment precedes
the deterministic 10% within-class-and-role subsample.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)")
PARQUETS = DATA / "outputs/cicids2017_pcap_label_mapping/flows"
CLASSES = ("BENIGN", "DoS Slowhttptest", "PortScan")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
SEED = 2022
FRACTION = 0.10
FIELDS = ("flow_id", "day", "source_pcap", "label", "match_status", "flow_start_epoch_utc",
          "canonical_flow_key", "first_packet_index", "last_packet_index", "packet_count")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def group(row):
    return f"{row['source_pcap']}|{int(float(row['flow_start_epoch_utc']) // 300)}"


def rows():
    for day in DAYS:
        path = PARQUETS / f"{day}_flows.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        for batch in pq.ParquetFile(path).iter_batches(columns=FIELDS, batch_size=65536):
            data = batch.to_pydict()
            for i, label in enumerate(data["label"]):
                if label in CLASSES and str(data["match_status"][i] or "").startswith("MATCHED"):
                    yield {name: data[name][i] for name in FIELDS}


def attack_assignment(groups, label, counts):
    own = sorted(g for g in groups if counts[g][label] > 0)
    total = sum(counts[g][label] for g in own)
    best = None
    for choices in itertools.product(("train", "validation", "test"), repeat=len(own)):
        sizes = Counter()
        for g, role in zip(own, choices, strict=True):
            sizes[role] += counts[g][label]
        if min(sizes[r] for r in ("train", "validation", "test")) == 0:
            continue
        # Fixed pre-outcome criterion: approximate 8:1:1 in whole group units.
        objective = (abs(sizes["validation"] - total * .1) +
                     abs(sizes["test"] - total * .1)) / total
        tie = hashlib.sha256(f"{SEED}|{label}|{choices}".encode()).hexdigest()
        candidate = (objective, tie, choices, sizes)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"no three-way group split for {label}")
    return dict(zip(own, best[2], strict=True)), dict(best[3]), best[0]


def assign_groups(counts):
    roles = {}
    objective = {}
    for label in CLASSES[1:]:
        chosen, sizes, loss = attack_assignment(counts, label, counts)
        if set(roles) & set(chosen):
            raise RuntimeError("attack labels share a capture-window group")
        roles.update(chosen)
        objective[label] = {"whole_group_role_flows": sizes, "normalized_deviation": loss}
    benign_total = sum(c["BENIGN"] for c in counts.values())
    known_benign = Counter()
    for g, role in roles.items():
        known_benign[role] += counts[g]["BENIGN"]
    remaining = [g for g in counts if g not in roles]
    remaining.sort(key=lambda g: hashlib.sha256(f"{SEED}|BENIGN|{g}".encode()).digest())
    for role in ("validation", "test"):
        target = benign_total * .1
        for g in list(remaining):
            value = counts[g]["BENIGN"]
            if known_benign[role] + value <= target or (known_benign[role] < target and
                abs(known_benign[role] + value - target) < abs(known_benign[role] - target)):
                roles[g] = role
                known_benign[role] += value
                remaining.remove(g)
            if known_benign[role] >= target:
                break
    for g in remaining:
        roles[g] = "train"
        known_benign["train"] += counts[g]["BENIGN"]
    if len(roles) != len(counts):
        raise RuntimeError("not all capture-window groups assigned")
    for label in CLASSES:
        sizes = Counter()
        for g, c in counts.items():
            sizes[roles[g]] += c[label]
        if min(sizes[part] for part in ("train", "validation", "test")) <= 0:
            raise RuntimeError(f"empty {label} role in grouped split")
        objective[label] = {"whole_group_role_flows": dict(sizes), **objective.get(label, {})}
    return roles, objective


def main():
    manifest = ROOT / "cicids2017_three_class_10pct_manifest.csv"
    assignments = ROOT / "cicids2017_group_assignments.csv"
    if manifest.exists() or assignments.exists():
        raise FileExistsError("CIC protocol already frozen; refusing to regenerate")
    counts = defaultdict(Counter)
    total = Counter()
    for row in rows():
        g = group(row)
        counts[g][row["label"]] += 1
        total[row["label"]] += 1
    expected = {"BENIGN": 1668300, "DoS Slowhttptest": 5096, "PortScan": 158863}
    if dict(total) != expected:
        raise RuntimeError(f"readiness-audit source count drift: {dict(total)} != {expected}")
    roles, objective = assign_groups(counts)
    split_counts = Counter()
    for g, values in counts.items():
        for label, value in values.items():
            split_counts[(label, roles[g])] += value
    target = {key: max(1, round(value * FRACTION)) for key, value in split_counts.items()}
    heaps = defaultdict(list)
    seen = Counter()
    for row in rows():
        label = row["label"]
        role = roles[group(row)]
        key = (label, role)
        seen[key] += 1
        rank = int.from_bytes(hashlib.sha256(f"{SEED}|{label}|{role}|{row['flow_id']}".encode()).digest(), "big")
        entry = (-rank, str(row["flow_id"]), row)
        if len(heaps[key]) < target[key]:
            heapq.heappush(heaps[key], entry)
        elif entry > heaps[key][0]:
            heapq.heapreplace(heaps[key], entry)
    if seen != split_counts:
        raise RuntimeError("CIC source changed between grouped and sample scans")
    sample = []
    for (label, role), heap in heaps.items():
        for _, _, row in heap:
            row["group_id"] = group(row)
            row["split"] = role
            sample.append(row)
    sample.sort(key=lambda row: row["flow_id"])
    if len(sample) != sum(target.values()) or len({r["flow_id"] for r in sample}) != len(sample):
        raise RuntimeError("invalid CIC sample cardinality/uniqueness")
    with assignments.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("group_id", "split", *CLASSES))
        for g in sorted(roles):
            writer.writerow((g, roles[g], *(counts[g][label] for label in CLASSES)))
    with manifest.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(*FIELDS, "group_id", "split"))
        writer.writeheader()
        writer.writerows(sample)
    audit = {"status": "PASS", "seed": SEED, "sample_fraction_per_class_split": FRACTION,
             "classes": CLASSES, "group_definition": "source_pcap|floor(flow_start_epoch_utc/300)",
             "source_class_counts": dict(total), "group_count": len(counts),
             "whole_group_split_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(split_counts.items())},
             "sample_split_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(target.items())},
             "group_objective": objective, "sample_manifest_sha256": digest(manifest),
             "group_assignments_sha256": digest(assignments),
             "source_parquet_sha256": {day: digest(PARQUETS / f"{day}_flows.parquet") for day in DAYS},
             "unknown_samples_used": 0, "test_features_read": 0,
             "limitation": "Slowhttptest and PortScan are single-day, single-capture labels; group-disjoint does not remove class-day/capture confounding"}
    (ROOT / "cicids2017_protocol_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "counts": audit["sample_split_counts"],
                      "groups": len(counts)}), flush=True)


if __name__ == "__main__":
    main()
