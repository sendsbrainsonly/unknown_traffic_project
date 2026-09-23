#!/usr/bin/env python3
"""Open the frozen Test split once and encode PCAPs with the Stage 6 preprocessing."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from stage9_common import (
    SPLIT_MANIFEST,
    STAGE9_ROOT,
    VALID_SETTINGS,
    STAGE7,
    assert_test_open_record,
    load_fold,
    read_csv,
    require,
    sha256_file,
    verify_all_provenance,
    write_csv,
    write_json,
)


ROLE_SPECS = (("KNOWN_TEST", "known_test"), ("UNKNOWN_TEST", "unknown_test"))


def encode_worker(row: dict[str, str]) -> dict[str, object]:
    path = Path(row["pcap_path"])
    try:
        image, packets_used = STAGE7.encode_pcap_first_eight(path)
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
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    require(args.workers > 0, "workers must be positive")
    verify_all_provenance()
    test_open = assert_test_open_record()
    setting = args.setting
    fold = load_fold(setting)
    known_classes = [str(value) for value in fold["known_classes"]]
    unknown_classes = [str(value) for value in fold["unknown_classes"]]
    class_to_local = {name: index for index, name in enumerate(known_classes)}
    artifact_dir = STAGE9_ROOT / "artifacts" / setting
    artifact_dir.mkdir(parents=True, exist_ok=True)
    guard = [artifact_dir / f"{prefix}_manifest.csv" for _, prefix in ROLE_SPECS]
    guard += [artifact_dir / f"{prefix}_images.npy" for _, prefix in ROLE_SPECS]
    require(not any(path.exists() for path in guard), f"{setting}: refusing to overwrite Test inputs")

    # This is the first role-level Test metadata read. The durable Test-open record exists above.
    rows = [row for row in read_csv(SPLIT_MANIFEST) if row["setting"] == setting]
    role_rows = {
        role: sorted(
            [row for row in rows if row["role"] == role],
            key=lambda row: (row["class_name"], row["sample_id"]),
        )
        for role, _ in ROLE_SPECS
    }
    require(role_rows["KNOWN_TEST"], f"{setting}: no Known Test rows")
    require(role_rows["UNKNOWN_TEST"], f"{setting}: no Unknown Test rows")
    require(set(row["class_name"] for row in role_rows["KNOWN_TEST"]) == set(known_classes), f"{setting}: Known Test class set mismatch")
    require(set(row["class_name"] for row in role_rows["UNKNOWN_TEST"]) == set(unknown_classes), f"{setting}: Unknown Test class set mismatch")
    selected_ids = [row["sample_id"] for role in role_rows.values() for row in role]
    require(len(selected_ids) == len(set(selected_ids)), f"{setting}: duplicate Test sample IDs")

    failures: list[dict[str, object]] = []
    role_audits: dict[str, object] = {}
    for role, prefix in ROLE_SPECS:
        selected = role_rows[role]
        images = open_memmap(
            artifact_dir / f"{prefix}_images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(selected), 32, 32),
        )
        packet_counts = np.zeros(len(selected), dtype=np.int16)
        manifests: list[dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(encode_worker, selected, chunksize=32)
            for index, (row, result) in enumerate(zip(selected, results)):
                if bool(result["input_valid"]):
                    images[index] = result["image"]
                    packet_counts[index] = int(result["packets_used"])
                else:
                    failures.append({
                        "setting": setting,
                        "role": role,
                        "sample_id": row["sample_id"],
                        "pcap_path": row["pcap_path"],
                        "failure_reason": result["failure_reason"],
                    })
                manifests.append({
                    "sample_id": row["sample_id"],
                    "pcap_path": row["pcap_path"],
                    "true_class": row["class_name"],
                    "role": role,
                    "cipher_source": row["cipher_source"],
                    "split_group_id": row["split_group_id"],
                    "row_index": index,
                })
                if (index + 1) % 5000 == 0:
                    print(json.dumps({"setting": setting, "role": role, "processed": index + 1, "total": len(selected), "failures": len(failures)}, sort_keys=True), flush=True)
        images.flush()
        np.save(artifact_dir / f"{prefix}_packet_counts.npy", packet_counts, allow_pickle=False)
        np.save(artifact_dir / f"{prefix}_sample_ids.npy", np.asarray([row["sample_id"] for row in selected], dtype="U64"), allow_pickle=False)
        if role == "KNOWN_TEST":
            labels = np.asarray([class_to_local[row["class_name"]] for row in selected], dtype=np.int64)
            np.save(artifact_dir / "known_test_labels.npy", labels, allow_pickle=False)
        manifest_path = artifact_dir / f"{prefix}_manifest.csv"
        write_csv(manifest_path, manifests)
        role_audits[role] = {
            "samples": len(selected),
            "classes": len(set(row["class_name"] for row in selected)),
            "minimum_packets_used": int(packet_counts.min()),
            "maximum_packets_used": int(packet_counts.max()),
            "zero_packet_rows": int(np.sum(packet_counts == 0)),
            "manifest_sha256": sha256_file(manifest_path),
            "images_sha256": sha256_file(artifact_dir / f"{prefix}_images.npy"),
        }
    write_csv(
        artifact_dir / "test_input_failures.csv",
        failures,
        ["setting", "role", "sample_id", "pcap_path", "failure_reason"],
    )
    audit = {
        "setting": setting,
        "status": "PASS" if not failures else "FAIL",
        "test_open_started_at": test_open["test_open_started_at"],
        "preprocessing": "frozen Stage 6 first-eight-packet Open-Detect encoding",
        "image_shape": [32, 32],
        "input_failures": len(failures),
        "roles": role_audits,
        "pcap_files_opened": sum(int(value["samples"]) for value in role_audits.values()),
        "different_preprocessing_retry_performed": False,
    }
    write_json(artifact_dir / "test_input_integrity.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    require(not failures, f"{setting}: strict Test input gate failed for {len(failures)} PCAPs")


if __name__ == "__main__":
    main()
