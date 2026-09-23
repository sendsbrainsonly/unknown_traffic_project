#!/usr/bin/env python3
"""Build only frozen Known Train/Validation Open-Detect inputs for one setting."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from common import (
    ROLE_TO_PREFIX,
    SPLIT_MANIFEST,
    STAGE6_OUTPUTS,
    TRAINING_ROLES,
    VALID_SETTINGS,
    encode_pcap_first_eight,
    label_maps,
    load_fold,
    select_training_rows,
    sha256_file,
    verify_stage6_protocol_hashes,
)


def encode_worker(row: dict[str, str]) -> dict[str, object]:
    path = Path(row["pcap_path"])
    try:
        image, packets_used = encode_pcap_first_eight(path)
        return {
            "sample_id": row["sample_id"],
            "image": image,
            "packets_used": packets_used,
            "input_valid": True,
            "failure_reason": "",
        }
    except Exception as exc:
        return {
            "sample_id": row["sample_id"],
            "image": None,
            "packets_used": 0,
            "input_valid": False,
            "failure_reason": f"{type(exc).__name__}:{exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--smoke-train-per-class", type=int, default=32)
    parser.add_argument("--smoke-validation-per-class", type=int, default=16)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if not (args.run_dir / "manifest.json").is_file():
        raise RuntimeError("run-dir must be initialized as an experiment bundle first")
    input_dir = args.run_dir / "inputs"
    if input_dir.exists():
        raise RuntimeError("refusing to overwrite an existing input directory")
    input_dir.mkdir(parents=True)

    protocol_hashes = verify_stage6_protocol_hashes()
    fold = load_fold(args.setting)
    class_to_local, local_to_class = label_maps(fold)
    selected, selection_metadata = select_training_rows(
        args.setting,
        args.mode,
        args.smoke_train_per_class,
        args.smoke_validation_per_class,
        args.seed,
    )
    (input_dir / "label_map.json").write_text(
        json.dumps(
            {
                "class_to_local_id": class_to_local,
                "local_id_to_class": {
                    str(key): value for key, value in local_to_class.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for role in TRAINING_ROLES:
        prefix = ROLE_TO_PREFIX[role]
        rows = selected[role]
        images = open_memmap(
            input_dir / f"{prefix}_images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(rows), 32, 32),
        )
        labels = np.asarray(
            [class_to_local[row["class_name"]] for row in rows], dtype=np.int64
        )
        sample_ids = np.asarray([row["sample_id"] for row in rows], dtype="U35")
        np.save(input_dir / f"{prefix}_labels.npy", labels, allow_pickle=False)
        np.save(input_dir / f"{prefix}_sample_ids.npy", sample_ids, allow_pickle=False)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(encode_worker, rows, chunksize=32)
            for index, (row, result) in enumerate(zip(rows, results)):
                if result["input_valid"]:
                    images[index] = result["image"]
                else:
                    failures.append(
                        {
                            "setting": args.setting,
                            "role": role,
                            "sample_id": row["sample_id"],
                            "pcap_path": row["pcap_path"],
                            "failure_reason": result["failure_reason"],
                        }
                    )
                manifest_rows.append(
                    {
                        "setting": args.setting,
                        "mode": args.mode,
                        "role": role,
                        "array_index": index,
                        "sample_id": row["sample_id"],
                        "class_name": row["class_name"],
                        "local_class_id": class_to_local[row["class_name"]],
                        "cipher_source": row["cipher_source"],
                        "split_group_id": row["split_group_id"],
                        "pcap_path": row["pcap_path"],
                        "packets_used": result["packets_used"],
                        "input_valid": result["input_valid"],
                        "failure_reason": result["failure_reason"],
                    }
                )
                if (index + 1) % 5000 == 0:
                    print(
                        json.dumps(
                            {
                                "setting": args.setting,
                                "role": role,
                                "processed": index + 1,
                                "total": len(rows),
                                "failures": len(failures),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        images.flush()

    manifest_path = input_dir / "input_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    failure_path = input_dir / "input_failures.csv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["setting", "role", "sample_id", "pcap_path", "failure_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)

    role_counts = selection_metadata["selected_role_counts"]
    audit = {
        **selection_metadata,
        "status": "PASS" if not failures else "FAIL",
        "stage6_protocol_hashes_verified": len(protocol_hashes),
        "stage6_split_manifest": str(SPLIT_MANIFEST.resolve()),
        "stage6_split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "stage6_fold_sha256": sha256_file(
            STAGE6_OUTPUTS / f"splits/{args.setting}_fold.json"
        ),
        "label_map_sha256": sha256_file(input_dir / "label_map.json"),
        "input_manifest_sha256": sha256_file(manifest_path),
        "input_failures": len(failures),
        "pcap_files_opened": int(sum(int(value) for value in role_counts.values())),
        "known_train_pcap_files_opened": int(role_counts["KNOWN_TRAIN"]),
        "known_validation_pcap_files_opened": int(role_counts["KNOWN_VALIDATION"]),
        "known_test_pcap_files_opened": 0,
        "unknown_pcap_files_opened": 0,
        "unknown_inference_executed": False,
        "pid": os.getpid(),
    }
    (input_dir / "input_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(f"input gate failed: {len(failures)} PCAPs could not be encoded")


if __name__ == "__main__":
    main()
