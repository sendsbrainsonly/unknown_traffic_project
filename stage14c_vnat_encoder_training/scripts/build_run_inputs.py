#!/usr/bin/env python3
"""Materialize model-visible Known Train/Validation arrays for all frozen runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

from common import (
    CACHE_ROOT, EXPECTED_FREEZE_HASH, IMAGE_SIDE, RUNS_ROOT, iter_split_rows,
    label_maps, protocols_by_id, sha256_file, string_set_sha256, verify_freeze,
)


def main() -> None:
    freeze = verify_freeze()
    cache_audit = json.loads((CACHE_ROOT / "cache_audit.json").read_text())
    if cache_audit["status"] != "PASS" or cache_audit["freeze_hash"] != EXPECTED_FREEZE_HASH:
        raise RuntimeError("flow cache did not pass the frozen-input gate")
    flow_uids = np.load(CACHE_ROOT / "flow_uids.npy", allow_pickle=False)
    images = np.load(CACHE_ROOT / "images.npy", mmap_mode="r", allow_pickle=False)
    uid_to_index = {str(uid): i for i, uid in enumerate(flow_uids)}
    if len(uid_to_index) != 23_449 or images.shape != (23_449, IMAGE_SIDE * IMAGE_SIDE):
        raise RuntimeError("unexpected clean cache shape")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in iter_split_rows():
        grouped[row["protocol_id"]].append(row)
    protocols = protocols_by_id()
    if set(grouped) != set(protocols):
        raise RuntimeError("split manifest/protocol identities differ")

    RUNS_ROOT.mkdir(exist_ok=True)
    for protocol_id, protocol in sorted(protocols.items()):
        input_dir = RUNS_ROOT / protocol_id / "inputs"
        if input_dir.exists():
            raise RuntimeError(f"refusing to overwrite {input_dir}")
        input_dir.mkdir(parents=True)
        known = set(map(str, protocol["known_applications"]))
        unknown = set(map(str, protocol["unknown_applications"]))
        class_to_local, local_to_class = label_maps(protocol)
        source_rows = grouped[protocol_id]
        role_counts = Counter((r["class_role"], r["split"]) for r in source_rows)
        selected = {
            split: sorted(
                [r for r in source_rows if r["class_role"] == "known" and r["split"] == split],
                key=lambda r: (r["application"], r["flow_uid"]),
            )
            for split in ("train", "validation")
        }
        selected_uids = {r["flow_uid"] for rows in selected.values() for r in rows}
        unknown_uids = {r["flow_uid"] for r in source_rows if r["class_role"] == "unknown"}
        train_uids = {r["flow_uid"] for r in selected["train"]}
        val_uids = {r["flow_uid"] for r in selected["validation"]}
        if known & unknown or selected_uids & unknown_uids or train_uids & val_uids:
            raise RuntimeError(f"{protocol_id}: split leakage")
        if any(r["application"] not in known for rows in selected.values() for r in rows):
            raise RuntimeError(f"{protocol_id}: non-Known row selected")
        expected_train = int(protocol["split_statistics"]["train"]["flows"])
        expected_val = int(protocol["split_statistics"]["validation"]["flows"])
        if [len(selected["train"]), len(selected["validation"])] != [expected_train, expected_val]:
            raise RuntimeError(f"{protocol_id}: frozen counts changed")

        manifest_rows: list[dict[str, object]] = []
        for split, rows in selected.items():
            indices = np.asarray([uid_to_index[r["flow_uid"]] for r in rows], dtype=np.int64)
            split_images = np.asarray(images[indices]).reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
            labels = np.asarray([class_to_local[r["application"]] for r in rows], dtype=np.int64)
            np.save(input_dir / f"{split}_images.npy", split_images, allow_pickle=False)
            np.save(input_dir / f"{split}_labels.npy", labels, allow_pickle=False)
            np.save(input_dir / f"{split}_flow_uids.npy", np.asarray([r["flow_uid"] for r in rows], dtype="U64"), allow_pickle=False)
            for local_index, (row, cache_index) in enumerate(zip(rows, indices)):
                manifest_rows.append({
                    "protocol_id": protocol_id, "setting": protocol["setting"],
                    "seed": protocol["seed"], "split": split,
                    "local_index": local_index, "cache_index": int(cache_index),
                    "flow_uid": row["flow_uid"], "application": row["application"],
                    "local_label": int(labels[local_index]), "class_role": row["class_role"],
                })
        manifest_path = input_dir / "input_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader(); writer.writerows(manifest_rows)
        (input_dir / "label_map.json").write_text(json.dumps({
            "class_to_local": class_to_local,
            "local_to_class": {str(k): v for k, v in local_to_class.items()},
        }, indent=2, sort_keys=True) + "\n")
        audit = {
            "status": "PASS", "protocol_id": protocol_id,
            "setting": protocol["setting"], "seed": int(protocol["seed"]),
            "freeze_hash": freeze["freeze_hash"],
            "protocol_sha256": freeze["protocol_sha256"],
            "split_manifest_sha256": freeze["split_manifest_sha256"],
            "cache_audit_sha256": sha256_file(CACHE_ROOT / "cache_audit.json"),
            "input_manifest_sha256": sha256_file(manifest_path),
            "known_classes": list(protocol["known_applications"]),
            "unknown_classes": list(protocol["unknown_applications"]),
            "known_train_class_intersection_unknown": sorted({r["application"] for r in selected["train"]} & unknown),
            "known_validation_class_intersection_unknown": sorted({r["application"] for r in selected["validation"]} & unknown),
            "unknown_samples_used_in_training": 0,
            "unknown_samples_used_in_validation": 0,
            "known_test_samples_used": 0,
            "train_validation_flow_overlap": len(train_uids & val_uids),
            "train_samples": len(selected["train"]), "validation_samples": len(selected["validation"]),
            "selected_flow_uid_set_sha256": string_set_sha256(selected_uids),
            "source_role_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(role_counts.items())},
            "model_visible_roles": ["known:train", "known:validation"],
        }
        (input_dir / "input_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"protocol_id": protocol_id, "train": expected_train, "validation": expected_val, "status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
