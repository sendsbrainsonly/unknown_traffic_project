from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def percentile(values: Sequence[int | float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _penalty(value: int, target: float) -> float:
    scale = max(target, 10.0)
    return ((value - target) / scale) ** 2


def assign_groups(
    rows: list[dict[str, str]],
    ratios: Mapping[str, float],
    seed: int,
    objective_weights: Mapping[str, float],
    max_refinement_passes: int,
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign each indivisible split_group_id using a fixed metadata-only objective."""
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    class_totals = Counter()
    stratum_totals = Counter()
    for row in rows:
        group_rows[row["split_group_id"]].append(row)
        class_totals[row["class_name"]] += 1
        stratum_totals[(row["class_name"], row["cipher_source"])] += 1
    group_stats = {}
    for group_id, members in group_rows.items():
        group_stats[group_id] = {
            "total": len(members),
            "class": Counter(row["class_name"] for row in members),
            "stratum": Counter((row["class_name"], row["cipher_source"]) for row in members),
        }
    split_names = list(ratios)
    total_target = {split: len(rows) * ratios[split] for split in split_names}
    class_target = {split: {name: count * ratios[split] for name, count in class_totals.items()} for split in split_names}
    stratum_target = {split: {key: count * ratios[split] for key, count in stratum_totals.items()} for split in split_names}
    current_total = Counter()
    current_class = {split: Counter() for split in split_names}
    current_stratum = {split: Counter() for split in split_names}
    weights = {
        "total": float(objective_weights["total"]),
        "class": float(objective_weights["per_class"]),
        "stratum": float(objective_weights["per_class_cipher_source"]),
    }

    def delta(group_id: str, destination: str, source: str | None = None) -> float:
        stats = group_stats[group_id]
        change = 0.0
        affected = [destination] if source is None else [source, destination]
        for split in affected:
            sign = 1 if split == destination else -1
            old_total = current_total[split]
            new_total = old_total + sign * stats["total"]
            change += weights["total"] * (
                _penalty(new_total, total_target[split]) - _penalty(old_total, total_target[split])
            )
            for class_name, count in stats["class"].items():
                old = current_class[split][class_name]
                new = old + sign * count
                target = class_target[split][class_name]
                change += weights["class"] * (_penalty(new, target) - _penalty(old, target))
            for key, count in stats["stratum"].items():
                old = current_stratum[split][key]
                new = old + sign * count
                target = stratum_target[split][key]
                change += weights["stratum"] * (_penalty(new, target) - _penalty(old, target))
        return change

    def apply(group_id: str, destination: str, source: str | None = None) -> None:
        stats = group_stats[group_id]
        if source is not None:
            current_total[source] -= stats["total"]
            current_class[source].subtract(stats["class"])
            current_stratum[source].subtract(stats["stratum"])
        current_total[destination] += stats["total"]
        current_class[destination].update(stats["class"])
        current_stratum[destination].update(stats["stratum"])

    order = sorted(group_rows, key=lambda group_id: (-group_stats[group_id]["total"], stable_int(f"{seed}:{group_id}")))
    assignment: dict[str, str] = {}
    for group_id in order:
        destination = min(split_names, key=lambda split: (delta(group_id, split), stable_int(f"{seed}:{group_id}:{split}")))
        assignment[group_id] = destination
        apply(group_id, destination)

    moves = 0
    for _ in range(max_refinement_passes):
        changed = False
        for group_id in sorted(group_rows, key=lambda value: stable_int(f"refine:{seed}:{value}")):
            source = assignment[group_id]
            candidates = [split for split in split_names if split != source]
            destination = min(candidates, key=lambda split: (delta(group_id, split, source), stable_int(f"move:{seed}:{group_id}:{split}")))
            improvement = delta(group_id, destination, source)
            if improvement < -1e-12:
                apply(group_id, destination, source)
                assignment[group_id] = destination
                moves += 1
                changed = True
        if not changed:
            break
    metadata = {
        "group_count": len(group_rows),
        "sample_counts": dict(current_total),
        "moves": moves,
        "objective": "2*total + 5*per-class + 2*per-class-by-source normalized squared ratio error",
    }
    return assignment, metadata
