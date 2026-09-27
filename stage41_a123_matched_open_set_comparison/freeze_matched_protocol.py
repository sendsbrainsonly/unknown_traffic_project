#!/usr/bin/env python3
"""Freeze one seed-2022 USTC flow pool for paired Scenario A-1/A-2/A-3.

This reuses Stage34's 10%-per-class/split SHA256 rule. It does not open packet
values, image arrays, checkpoints, or any Test score.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = PROJECT / "opendetect_ustc_encoder_audit/outputs/input_alignment_manifest.csv"
STAGE34 = PROJECT / "stage34_ustc_cic_closed_set/ustc_a2_10pct_manifest.csv"
STAGE40 = PROJECT / "stage40_ustc_cic_open_set/ustc_a2_roles.csv"
STAGE3 = PROJECT / "stage3_unknown_utility/outputs"
UNKNOWN = {
    "A-1": {"Tinba"},
    "A-2": {"Geodo", "Htbot", "Tinba"},
    "A-3": {"Geodo", "Htbot", "Tinba", "Miuref", "Neris"},
}
SEED = 2022


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (ROOT / "matched_protocol.json").exists():
        raise FileExistsError("matched protocol is already frozen")
    original = read_csv(SOURCE)
    if len(original) != len({r["flow_id"] for r in original}):
        raise RuntimeError("source flow IDs are not unique")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in original:
        if row["input_valid"] != "True":
            raise RuntimeError(f"invalid aligned flow {row['flow_id']}")
        if row["original_split"] not in {"train", "val", "test"}:
            raise RuntimeError("unexpected source split")
        grouped[(row["class_name"], row["original_split"])].append(row)
    classes = {name for name, _ in grouped}
    if len(classes) != 20 or len(grouped) != 60:
        raise RuntimeError("expected 20 classes each with Train/Val/Test")
    sampled = []
    for (name, split), rows in sorted(grouped.items()):
        n = max(1, round(.1 * len(rows)))
        sampled.extend(sorted(rows, key=lambda r: (
            hashlib.sha256(f"{SEED}|{split}|{name}|{r['flow_id']}".encode()).digest(),
            r["flow_id"],
        ))[:n])
    sampled.sort(key=lambda r: r["flow_id"])
    if len(sampled) != len({r["flow_id"] for r in sampled}):
        raise RuntimeError("sampled flow duplication")
    stage34 = read_csv(STAGE34)
    stage34_ids = {r["flow_id"] for r in stage34}
    selected_a2 = {r["flow_id"] for r in sampled if r["class_name"] not in UNKNOWN["A-2"]}
    if stage34_ids != selected_a2 or any(r["class_name"] in UNKNOWN["A-2"] for r in stage34):
        raise RuntimeError("A-2 Known sample is not identical to Stage34")
    path = ROOT / "common_flow_pool.csv"
    write_csv(path, sampled)
    stage40 = read_csv(STAGE40)
    audit = {}
    for setting, unknown in UNKNOWN.items():
        config = json.loads((STAGE3 / setting / "training_config.json").read_text())
        if set(config["known_classes"]) != classes - unknown or set(config["unknown_classes_excluded"]) != unknown:
            raise RuntimeError(f"{setting}: paper-scenario class list mismatch")
        roles = []
        for row in sampled:
            cls, split = row["class_name"], row["original_split"]
            if cls in unknown:
                if split != "test":
                    continue
                role = "unknown_test"
            else:
                role = {"train": "known_train", "val": "known_validation", "test": "known_test"}[split]
            roles.append({"flow_id": row["flow_id"], "class_name": cls,
                          "role": role, "source_split": split,
                          "opendetect_input_id": row["opendetect_input_id"],
                          "source_file": row["source_file"]})
        roles.sort(key=lambda r: r["flow_id"])
        if len(roles) != len({r["flow_id"] for r in roles}):
            raise RuntimeError(f"{setting}: overlapping roles")
        if setting == "A-2":
            left = {(r["flow_id"], r["class_name"], r["role"]) for r in roles}
            right = {(r["flow_id"], r["class_name"], r["role"]) for r in stage40}
            if left != right:
                raise RuntimeError("A-2 roles differ from Stage40")
        role_path = ROOT / f"{setting.replace('-', '').lower()}_roles.csv"
        write_csv(role_path, roles)
        count = Counter(r["role"] for r in roles)
        if set(count) != {"known_train", "known_validation", "known_test", "unknown_test"}:
            raise RuntimeError(f"{setting}: missing role")
        audit[setting] = {"known_classes": sorted(classes - unknown),
                          "unknown_classes": sorted(unknown), "role_counts": dict(count),
                          "role_manifest": str(role_path), "role_manifest_sha256": digest(role_path)}
    protocol = {"status": "PASS", "seed": SEED, "fraction": .1,
                "sampling": "SHA256(seed|source_split|class_name|flow_id), first round(10%) per class/split",
                "source_manifest": str(SOURCE), "source_sha256": digest(SOURCE),
                "stage34_a2_sha256": digest(STAGE34), "stage40_a2_sha256": digest(STAGE40),
                "common_flow_pool": str(path), "common_flow_pool_sha256": digest(path),
                "a2_stage34_known_ids_exact": True, "a2_stage40_roles_exact": True,
                "unknown_train_or_val_flows": 0, "test_feature_values_read": 0,
                "units": audit}
    with (ROOT / "matched_protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({name: item["role_counts"] for name, item in audit.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
