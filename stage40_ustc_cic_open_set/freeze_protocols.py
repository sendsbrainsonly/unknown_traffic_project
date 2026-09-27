#!/usr/bin/env python3
"""Freeze USTC A-2 and two CIC leave-one-attack-out roles before scoring."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
USTC = PROJECT / "stage34_ustc_cic_closed_set"
CIC = PROJECT / "stage34b_cic_balanced_retest"
USTC_SOURCE = PROJECT / "opendetect_ustc_encoder_audit/outputs/input_alignment_manifest.csv"
USTC_KNOWN = USTC / "ustc_a2_10pct_manifest.csv"
USTC_CONFIG = PROJECT / "stage3_unknown_utility/outputs/A-2/training_config.json"
CIC_SOURCE = CIC / "balanced_manifest.csv"
UNKNOWN_USTC = ("Geodo", "Htbot", "Tinba")
ATTACKS = ("DoS Slowhttptest", "PortScan")
SEED = 2022


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"empty role manifest: {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def lock_ustc() -> dict:
    if digest(USTC_KNOWN) != json.loads((USTC / "ustc_sample_audit.json").read_text())["sample_manifest_sha256"]:
        raise RuntimeError("USTC Known 10% manifest changed")
    config = json.loads(USTC_CONFIG.read_text())
    known = set(config["known_classes"])
    if set(config["unknown_classes_excluded"]) != set(UNKNOWN_USTC) or known & set(UNKNOWN_USTC):
        raise RuntimeError("USTC frozen A-2 class role mismatch")
    trainval = csv_rows(USTC_KNOWN)
    if any(r["class_name"] not in known for r in trainval):
        raise RuntimeError("USTC Known manifest includes Unknown class")
    source = csv_rows(USTC_SOURCE)
    unknown_test: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for name in UNKNOWN_USTC:
        candidate = [r for r in source if r["class_name"] == name and
                     r["original_split"] == "test" and r["input_valid"] == "True"]
        if len(candidate) != len({r["flow_id"] for r in candidate}) or not candidate:
            raise RuntimeError(f"invalid USTC Unknown Test {name}")
        source_counts[name] = len(candidate)
        n = max(1, round(0.1 * len(candidate)))
        unknown_test += sorted(candidate, key=lambda r: (
            hashlib.sha256(f"{SEED}|test|{name}|{r['flow_id']}".encode()).digest(),
            r["flow_id"]))[:n]
    rows: list[dict[str, str]] = []
    for r in trainval:
        rows.append({"flow_id": r["flow_id"], "class_name": r["class_name"],
                     "role": {"train": "known_train", "val": "known_validation",
                              "test": "known_test"}[r["original_split"]],
                     "source_split": r["original_split"], "source_file": r["source_file"]})
    for r in unknown_test:
        rows.append({"flow_id": r["flow_id"], "class_name": r["class_name"],
                     "role": "unknown_test", "source_split": "test",
                     "source_file": r["source_file"]})
    rows.sort(key=lambda r: r["flow_id"])
    if len(rows) != len({r["flow_id"] for r in rows}):
        raise RuntimeError("USTC overlapping flow role")
    path = ROOT / "ustc_a2_roles.csv"
    write_csv(path, rows)
    return {"dataset": "USTC-TFC2016", "known_classes": sorted(known),
            "unknown_classes": list(UNKNOWN_USTC), "role_counts": dict(Counter(r["role"] for r in rows)),
            "unknown_source_test_counts": source_counts,
            "selection": "existing Stage34 Known 10% rows; Unknown valid Test SHA256 10% per class, seed2022",
            "role_manifest": str(path), "role_manifest_sha256": digest(path),
            "source_hashes": {str(p): digest(p) for p in (USTC_KNOWN, USTC_SOURCE, USTC_CONFIG)}}


def lock_cic(heldout: str, source: list[dict[str, str]]) -> dict:
    known_attack = next(x for x in ATTACKS if x != heldout)
    selected: list[dict[str, str]] = []
    for role in ("train", "validation", "test"):
        known_attack_rows = [r for r in source if r["split"] == role and r["label"] == known_attack]
        benign = [r for r in source if r["split"] == role and r["label"] == "BENIGN"]
        if not known_attack_rows or len(benign) < len(known_attack_rows):
            raise RuntimeError(f"CIC Known balance impossible: {heldout}/{role}")
        benign = sorted(benign, key=lambda r: (
            hashlib.sha256(f"{SEED}|cic40|{known_attack}|{role}|{r['flow_id']}".encode()).digest(),
            r["flow_id"]))[:len(known_attack_rows)]
        for r in known_attack_rows + benign:
            selected.append({"flow_id": r["flow_id"], "class_name": r["label"],
                             "role": {"train": "known_train", "validation": "known_validation",
                                      "test": "known_test"}[role], "source_split": role,
                             "source_file": r["source_pcap"], "group_id": r["group_id"]})
    for r in source:
        if r["label"] == heldout and r["split"] == "test":
            selected.append({"flow_id": r["flow_id"], "class_name": heldout,
                             "role": "unknown_test", "source_split": "test",
                             "source_file": r["source_pcap"], "group_id": r["group_id"]})
    selected.sort(key=lambda r: r["flow_id"])
    if len(selected) != len({r["flow_id"] for r in selected}):
        raise RuntimeError("CIC flow role overlap")
    group_splits: dict[str, set[str]] = {}
    for r in selected:
        group_splits.setdefault(r["group_id"], set()).add(r["source_split"])
    if any(len(x) != 1 for x in group_splits.values()):
        raise RuntimeError("CIC group crosses source splits")
    key = "unknown_slowhttptest" if heldout == ATTACKS[0] else "unknown_portscan"
    path = ROOT / f"{key}_roles.csv"
    write_csv(path, selected)
    return {"dataset": "CIC-IDS-2017", "key": key,
            "known_classes": ["BENIGN", known_attack], "unknown_classes": [heldout],
            "role_counts": dict(Counter(r["role"] for r in selected)),
            "per_class_role": {f"{cls}|{role}": n for (cls, role), n in
                               Counter((r["class_name"], r["role"]) for r in selected).items()},
            "selection": "subset frozen Stage34B; all Known attack rows; SHA256 1:1 Known BENIGN per split; held-out Test attack only",
            "role_manifest": str(path), "role_manifest_sha256": digest(path),
            "group_disjoint_across_train_validation_test": True,
            "source_hashes": {str(CIC_SOURCE): digest(CIC_SOURCE)}}


def main() -> None:
    if (ROOT / "protocols.json").exists():
        raise FileExistsError("Stage40 roles already frozen")
    cic_audit = json.loads((CIC / "balanced_manifest_audit.json").read_text())
    if digest(CIC_SOURCE) != cic_audit["balanced_manifest_sha256"]:
        raise RuntimeError("Stage34B balanced pool hash changed")
    source = csv_rows(CIC_SOURCE)
    if len(source) != len({r["flow_id"] for r in source}):
        raise RuntimeError("CIC balanced source duplicate flow")
    units = [lock_ustc(), *(lock_cic(attack, source) for attack in ATTACKS)]
    payload = {"status": "PASS", "seed": SEED, "created_before_test_feature_access": True,
               "score_methods": ["msp", "energy", "centroid", "des_v1"],
               "threshold": "Known Validation P95, method=higher, reject score>threshold",
               "des_v1": {"k": 10, "global_weight": 0.5, "local_weight": 0.5,
                          "normalization": "Known-Validation median/MAD"},
               "units": units}
    write_json(ROOT / "protocols.json", payload)
    print(json.dumps({"status": "PASS", "units": [
        {"name": x.get("key", "ustc_a2"), "role_counts": x["role_counts"]} for x in units]}))


if __name__ == "__main__":
    main()
