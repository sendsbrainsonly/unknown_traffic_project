#!/usr/bin/env python3
"""Freeze Stage44 group-disjoint VNAT service-level LOSO protocols."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SERVICE_BY_APP = {
    app: service
    for service, applications in CONFIG["service_taxonomy"].items()
    for app in applications
}
SPLITS = ("train", "validation", "test")
TARGET = {name: float(CONFIG["known_split_target"][name]) for name in SPLITS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_assignment(assignment: dict[str, str], sizes: dict[str, int]) -> float:
    total = sum(sizes.values())
    counts = Counter()
    groups = Counter()
    for group, split in assignment.items():
        counts[split] += sizes[group]
        groups[split] += 1
    if any(groups[name] == 0 for name in SPLITS):
        return 1e9
    # Flow-ratio mismatch is primary; group-ratio mismatch breaks close ties.
    flow_error = sum(((counts[name] / total) - TARGET[name]) ** 2 for name in SPLITS)
    group_total = len(sizes)
    group_error = sum(((groups[name] / group_total) - TARGET[name]) ** 2 for name in SPLITS)
    return flow_error + 0.01 * group_error


def build_assignment(service: str, sizes: dict[str, int], seed: int) -> dict[str, str]:
    """Deterministic multi-start greedy + local search over indivisible groups."""
    if len(sizes) < 3:
        raise RuntimeError(f"{service} has fewer than three groups")
    groups = list(sizes)
    rng = random.Random(f"{seed}:{service}")
    best: dict[str, str] | None = None
    best_score = float("inf")
    base = sorted(groups, key=lambda group: (-sizes[group], group))
    # Twenty-four deterministic restarts are enough for this audit-sized problem;
    # exhaustive search would be wasteful for Communication's 59 groups.
    for restart in range(24):
        order = base[:] if restart == 0 else rng.sample(groups, len(groups))
        assignment: dict[str, str] = {}
        for group in order:
            choices = []
            for split in SPLITS:
                trial = dict(assignment)
                trial[group] = split
                # Partial target error; absent roles are allowed until the final steps.
                counts = Counter()
                for item, role in trial.items():
                    counts[role] += sizes[item]
                total = sum(sizes.values())
                partial = sum(((counts[name] / total) - TARGET[name]) ** 2 for name in SPLITS)
                choices.append((partial, SPLITS.index(split), split))
            assignment[group] = min(choices)[2]
        # Repair empty roles with the move that minimizes the final objective.
        for missing in (name for name in SPLITS if name not in assignment.values()):
            candidates = []
            for group in groups:
                trial = dict(assignment)
                trial[group] = missing
                candidates.append((score_assignment(trial, sizes), group, trial))
            assignment = min(candidates, key=lambda item: (item[0], item[1]))[2]
        # Deterministic hill-climb over one-group moves and pairwise swaps.
        improved = True
        while improved:
            improved = False
            current = score_assignment(assignment, sizes)
            candidate = (current, "", "", assignment)
            for group in groups:
                for split in SPLITS:
                    if split == assignment[group]:
                        continue
                    trial = dict(assignment)
                    trial[group] = split
                    value = score_assignment(trial, sizes)
                    key = (value, group, split, trial)
                    if key[:3] < candidate[:3]:
                        candidate = key
            for index, left in enumerate(groups):
                for right in groups[index + 1:]:
                    if assignment[left] == assignment[right]:
                        continue
                    trial = dict(assignment)
                    trial[left], trial[right] = trial[right], trial[left]
                    value = score_assignment(trial, sizes)
                    key = (value, left, right, trial)
                    if key[:3] < candidate[:3]:
                        candidate = key
            if candidate[0] + 1e-15 < current:
                assignment = candidate[3]
                improved = True
        value = score_assignment(assignment, sizes)
        signature = tuple(assignment[group] for group in sorted(groups))
        incumbent = tuple(best[group] for group in sorted(groups)) if best else ()
        if value < best_score - 1e-15 or (abs(value - best_score) <= 1e-15 and signature < incumbent):
            best, best_score = dict(assignment), value
    assert best is not None
    return best


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = [
            row for row in csv.DictReader(handle)
            if row["protocol_id"] == CONFIG["source_protocol_id_for_unique_pool"]
        ]
    if len(source_rows) != 23449 or len({row["flow_uid"] for row in source_rows}) != 23449:
        raise RuntimeError("Stage14B unique pool is not exactly 23,449 flows")
    applications = {row["application"].lower() for row in source_rows}
    if applications != set(SERVICE_BY_APP):
        raise RuntimeError(f"taxonomy mismatch: {sorted(applications ^ set(SERVICE_BY_APP))}")

    by_service_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in source_rows:
        service = SERVICE_BY_APP[row["application"].lower()]
        by_service_group[service][row["group_id"]] += 1
    assignments = {
        service: build_assignment(service, dict(sizes), int(CONFIG["seed"]))
        for service, sizes in sorted(by_service_group.items())
    }

    canonical_rows: list[dict[str, object]] = []
    for row in sorted(source_rows, key=lambda item: item["flow_uid"]):
        service = SERVICE_BY_APP[row["application"].lower()]
        canonical_rows.append({
            "flow_uid": row["flow_uid"],
            "application": row["application"].lower(),
            "service": service,
            "vpn_status": row["vpn_status"],
            "capture_id": row["capture_id"],
            "group_id": row["group_id"],
            "source_flow_id": row["source_flow_id"],
            "protocol": row["protocol"],
            "packet_count": row["packet_count"],
            "byte_count": row["byte_count"],
            "source_pcap_path": row["source_pcap_path"],
            "global_known_split": assignments[service][row["group_id"]],
        })
    write_csv(ROOT / "vnat_service_manifest.csv", canonical_rows)

    split_audit: list[dict[str, object]] = []
    for service in sorted(assignments):
        rows = [row for row in canonical_rows if row["service"] == service]
        total = len(rows)
        for split in SPLITS:
            selected = [row for row in rows if row["global_known_split"] == split]
            split_audit.append({
                "service": service,
                "split": split,
                "flows": len(selected),
                "flow_ratio": len(selected) / total,
                "groups": len({row["group_id"] for row in selected}),
                "captures": len({row["capture_id"] for row in selected}),
                "applications": ";".join(sorted({str(row["application"]) for row in selected})),
                "vpn_flows": sum(row["vpn_status"] == "vpn" for row in selected),
                "nonvpn_flows": sum(row["vpn_status"] == "nonvpn" for row in selected),
            })
    write_csv(ROOT / "split_audit.csv", split_audit)

    protocols: dict[str, object] = {
        "status": "FROZEN_PRETRAIN",
        "dataset": "VNAT",
        "seed": CONFIG["seed"],
        "source_manifest": str(SOURCE),
        "source_manifest_sha256": sha256(SOURCE),
        "service_manifest_sha256": sha256(ROOT / "vnat_service_manifest.csv"),
        "taxonomy": CONFIG["service_taxonomy"],
        "split_unit": "group_id",
        "folds": {},
    }
    all_role_rows: list[dict[str, object]] = []
    for heldout in CONFIG["folds"]:
        fold = ROOT / "protocols" / heldout
        role_rows: list[dict[str, object]] = []
        for row in canonical_rows:
            role = "unknown_test" if row["service"] == heldout else {
                "train": "known_train",
                "validation": "known_validation",
                "test": "known_test",
            }[str(row["global_known_split"])]
            item = {
                "fold": heldout,
                "flow_uid": row["flow_uid"],
                "role": role,
                "service": row["service"],
                "application": row["application"],
                "vpn_status": row["vpn_status"],
                "capture_id": row["capture_id"],
                "group_id": row["group_id"],
            }
            role_rows.append(item)
            all_role_rows.append(item)
        write_csv(fold / "role_manifest.csv", role_rows)
        counts = Counter(str(row["role"]) for row in role_rows)
        known_services = sorted({str(row["service"]) for row in role_rows if row["role"] == "known_train"})
        group_roles: dict[str, set[str]] = defaultdict(set)
        capture_roles: dict[str, set[str]] = defaultdict(set)
        for row in role_rows:
            if str(row["role"]).startswith("known_"):
                group_roles[str(row["group_id"])].add(str(row["role"]))
                capture_roles[str(row["capture_id"])].add(str(row["role"]))
        group_leaks = {key: value for key, value in group_roles.items() if len(value) > 1}
        capture_leaks = {key: value for key, value in capture_roles.items() if len(value) > 1}
        if group_leaks or capture_leaks:
            raise RuntimeError(f"group/capture leakage in {heldout}: {len(group_leaks)}/{len(capture_leaks)}")
        if set(known_services) != set(CONFIG["folds"]) - {heldout}:
            raise RuntimeError(f"known service mismatch in {heldout}")
        if sum(counts.values()) != 23449 or any(counts[name] == 0 for name in (
            "known_train", "known_validation", "known_test", "unknown_test"
        )):
            raise RuntimeError(f"invalid fold counts in {heldout}: {counts}")
        role_hash = sha256(fold / "role_manifest.csv")
        fold_protocol = {
            "status": "FROZEN_PRETRAIN",
            "fold": heldout,
            "unknown_service": heldout,
            "known_services": known_services,
            "counts": dict(counts),
            "role_manifest_sha256": role_hash,
            "group_overlap_count": 0,
            "capture_overlap_count": 0,
            "unknown_training_samples": 0,
            "unknown_validation_samples": 0,
        }
        (fold / "protocol.json").write_text(json.dumps(fold_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        protocols["folds"][heldout] = fold_protocol
    write_csv(ROOT / "vnat_loso_role_manifest.csv", all_role_rows)
    (ROOT / "vnat_loso_protocol.json").write_text(json.dumps(protocols, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = {
        "status": "PASS",
        "unique_flows": len(canonical_rows),
        "services": Counter(str(row["service"]) for row in canonical_rows),
        "folds": len(CONFIG["folds"]),
        "group_overlap_count": 0,
        "capture_overlap_count": 0,
        "unknown_train_usage": 0,
        "unknown_validation_usage": 0,
        "source_manifest_sha256": sha256(SOURCE),
        "protocol_sha256": sha256(ROOT / "vnat_loso_protocol.json"),
        "role_manifest_sha256": sha256(ROOT / "vnat_loso_role_manifest.csv"),
    }
    (ROOT / "protocol_freeze_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True, default=dict) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, sort_keys=True, default=dict))


if __name__ == "__main__":
    main()
