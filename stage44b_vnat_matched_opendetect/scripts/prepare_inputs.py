#!/usr/bin/env python3
"""Expose only frozen Known Train/Validation images to native training."""
from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np

from matched_common import FOLDS, IMAGE_CACHE, ROOT, STAGE44, image_index, load_protocols, role_rows, sha256, verify_freeze


def main() -> None:
    frozen = verify_freeze()
    uid_to_image = image_index()
    images = np.load(IMAGE_CACHE / "images.npy", mmap_mode="r", allow_pickle=False)
    protocols = load_protocols()
    output_root = ROOT / "inputs"
    output_root.mkdir(exist_ok=True)

    for fold in FOLDS:
        roles = role_rows(fold)
        if {row["flow_uid"] for rows in roles.values() for row in rows} != set(uid_to_image):
            raise RuntimeError(f"{fold}: native image cache and frozen roles have different flow IDs")
        selected = roles["known_train"] + roles["known_validation"]
        if len(selected) != len({row["flow_uid"] for row in selected}):
            raise RuntimeError(f"{fold}: duplicate model-visible flow")
        known_classes = list(protocols[fold]["known_applications"])
        class_to_local = {name: index for index, name in enumerate(known_classes)}
        output = output_root / fold
        output.mkdir(exist_ok=False)
        manifest: list[dict[str, object]] = []
        for role, name in (("known_train", "train"), ("known_validation", "validation")):
            rows = roles[role]
            if {row["service"] for row in rows} != set(known_classes):
                raise RuntimeError(f"{fold}: {role} lacks a Known service")
            positions = np.asarray([uid_to_image[row["flow_uid"]] for row in rows], dtype=np.int64)
            selected_images = np.asarray(images[positions]).reshape(-1, 32, 32)
            labels = np.asarray([class_to_local[row["service"]] for row in rows], dtype=np.int64)
            np.save(output / f"{name}_images.npy", selected_images, allow_pickle=False)
            np.save(output / f"{name}_labels.npy", labels, allow_pickle=False)
            np.save(output / f"{name}_flow_uids.npy", np.asarray([row["flow_uid"] for row in rows], dtype="U64"), allow_pickle=False)
            for local_index, (row, position) in enumerate(zip(rows, positions)):
                manifest.append({
                    "fold": fold, "role": role, "local_index": local_index,
                    "flow_uid": row["flow_uid"], "service": row["service"],
                    "application": row["application"], "group_id": row["group_id"],
                    "capture_id": row["capture_id"], "cache_index": int(position),
                    "local_label": int(labels[local_index]),
                })
        with (output / "input_manifest.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        (output / "label_map.json").write_text(json.dumps({
            "class_to_local": class_to_local,
            "local_to_class": {str(index): name for name, index in class_to_local.items()},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audit = {
            "status": "PASS", "fold": fold, "seed": 2022,
            "frozen_inputs": frozen,
            "role_manifest_sha256": sha256(STAGE44 / "protocols" / fold / "role_manifest.csv"),
            "input_manifest_sha256": sha256(output / "input_manifest.csv"),
            "train_samples": len(roles["known_train"]),
            "validation_samples": len(roles["known_validation"]),
            "train_class_counts": dict(Counter(row["service"] for row in roles["known_train"])),
            "validation_class_counts": dict(Counter(row["service"] for row in roles["known_validation"])),
            "known_classes": known_classes,
            "unknown_classes": [fold],
            "unknown_samples_used_in_training": 0,
            "unknown_samples_used_in_validation": 0,
            "known_test_samples_used": 0,
            "train_validation_flow_overlap": 0,
            "model_visible_roles": ["known_train", "known_validation"],
        }
        (output / "input_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"fold": fold, "train": len(roles["known_train"]), "validation": len(roles["known_validation"]), "status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
