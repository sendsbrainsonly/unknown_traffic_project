#!/usr/bin/env python3
"""Freeze exact Stage15R pilot membership for RoNeTC without opening features."""
from __future__ import annotations

from collections import Counter

import numpy as np

from common import PROJECT, PROTOCOL_MANIFEST, STAGE12, STAGE14B, STAGE14C, config, read_csv, read_json, sha256_file, write_csv, write_json


FIELDS = [
    "dataset", "protocol_id", "setting", "protocol_seed", "role", "flow_uid",
    "class_name", "local_label", "is_unknown", "source_file", "source_flow_id",
]


def iscx_rows(spec: dict) -> list[dict]:
    dataset, setting, protocol_id = spec["dataset"], spec["setting"], spec["protocol_id"]
    protocol = read_json(STAGE12 / "artifacts" / dataset / "protocol" / setting / "protocol.json")
    known = list(protocol["known_classes"])
    local = {name: i for i, name in enumerate(known)}
    rows = []
    for row in read_csv(STAGE12 / "protocol" / dataset / "split_manifest.csv"):
        if row["setting"] != setting:
            continue
        role = row["role"]
        unknown = role == "unknown_test"
        rows.append({
            "dataset": dataset, "protocol_id": protocol_id, "setting": setting,
            "protocol_seed": spec["seed"], "role": role, "flow_uid": row["flow_id_sha256"],
            "class_name": row["canonical_class"], "local_label": len(known) if unknown else local[row["canonical_class"]],
            "is_unknown": int(unknown), "source_file": row["source_file"], "source_flow_id": "",
        })
    return rows


def vnat_rows(spec: dict) -> list[dict]:
    protocol_id = spec["protocol_id"]
    label_map = read_json(STAGE14C / "runs" / protocol_id / "inputs" / "label_map.json")
    known = [label_map["local_to_class"][str(i)] for i in range(len(label_map["local_to_class"]))]
    local = {name: i for i, name in enumerate(known)}
    rows = []
    for row in read_csv(STAGE14B / "vnat_split_manifest.csv"):
        if row["protocol_id"] != protocol_id:
            continue
        unknown = row["class_role"].lower() == "unknown"
        split = row["split"].lower()
        role = "unknown_test" if unknown else {"train": "known_train", "validation": "known_validation", "test": "known_test"}[split]
        rows.append({
            "dataset": "vnat", "protocol_id": protocol_id, "setting": spec["setting"],
            "protocol_seed": spec["seed"], "role": role, "flow_uid": row["flow_uid"],
            "class_name": row["application"], "local_label": len(known) if unknown else local[row["application"]],
            "is_unknown": int(unknown), "source_file": row["source_pcap_path"], "source_flow_id": row["source_flow_id"],
        })
    return rows


def ustc_rows(spec: dict) -> list[dict]:
    training = read_json(PROJECT / "stage3_unknown_utility" / "outputs" / "A-2" / "training_config.json")
    known = list(training["known_classes"])
    local = {name: i for i, name in enumerate(known)}
    class_to_id = read_json(PROJECT / "data" / "trafficformer_input" / "compatible_min1" / "label_map.json")["class_to_id"]
    id_to_class = {int(value): name for name, value in class_to_id.items()}
    source_by_uid = {row["flow_id"]: row["source_file"] for row in read_csv(PROJECT / "opendetect_ustc_encoder_audit" / "outputs" / "input_alignment_manifest.csv")}
    rows = []
    for split, role_known in (("train", "known_train"), ("val", "known_validation"), ("test", "known_test")):
        uids = np.load(PROJECT / "outputs" / "stage1" / "modelA" / "embeddings" / f"flow_ids_{split}.npy", allow_pickle=False)
        labels = np.load(PROJECT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{split}.npy", allow_pickle=False)
        for uid_value, label_value in zip(uids, labels):
            uid, class_name = str(uid_value), id_to_class[int(label_value)]
            unknown = class_name not in local
            if split != "test" and unknown:
                continue
            role = "unknown_test" if unknown else role_known
            rows.append({
                "dataset": "ustc", "protocol_id": "A-2", "setting": "A-2", "protocol_seed": 2022,
                "role": role, "flow_uid": uid, "class_name": class_name,
                "local_label": len(known) if unknown else local[class_name], "is_unknown": int(unknown),
                "source_file": source_by_uid[uid], "source_flow_id": "",
            })
    return rows


def main() -> None:
    if PROTOCOL_MANIFEST.exists():
        raise RuntimeError(f"refusing to overwrite {PROTOCOL_MANIFEST}")
    rows = []
    for spec in config()["protocols"]:
        rows.extend(iscx_rows(spec) if spec["dataset"].startswith("iscx_") else vnat_rows(spec) if spec["dataset"] == "vnat" else ustc_rows(spec))
    identities = [(row["dataset"], row["protocol_id"], row["role"], row["flow_uid"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate protocol-role-flow identity")
    write_csv(PROTOCOL_MANIFEST, rows, FIELDS)
    counts = Counter((row["dataset"], row["protocol_id"], row["role"]) for row in rows)
    expected = {
        ("iscx_vpn", "medium_seed2022"): (12910, 1611),
        ("iscx_tor", "medium_seed2022"): (8876, 1107),
        ("vnat", "medium_seed2025"): (15704, 1960),
        ("vnat", "medium_seed2026"): (15328, 1911),
        ("ustc", "A-2"): (346651, 43331),
    }
    for key, pair in expected.items():
        got = (counts[key + ("known_train",)], counts[key + ("known_validation",)])
        if got != pair:
            raise RuntimeError(f"frozen count mismatch {key}: {got} != {pair}")
    write_json(PROTOCOL_MANIFEST.with_suffix(".audit.json"), {
        "status": "PASS", "rows": len(rows), "sha256": sha256_file(PROTOCOL_MANIFEST),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "unknown_train_rows": sum(row["is_unknown"] == 1 and row["role"] == "known_train" for row in rows),
        "unknown_validation_rows": sum(row["is_unknown"] == 1 and row["role"] == "known_validation" for row in rows),
    })
    print(PROTOCOL_MANIFEST.with_suffix(".audit.json").read_text())


if __name__ == "__main__":
    main()
