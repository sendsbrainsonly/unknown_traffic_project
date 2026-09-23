#!/usr/bin/env python3
"""Build F1/F2/F3 arrays from F0 using Known-Train-only scalers."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from common import NEW_FEATURES, RAW_CACHE, ROOT, RUNS_ROOT, source_input_dir, transformed_statistics, robust_fit, robust_quantize, json_dump, load_known_input_rows, protocol_ids, sha256_file, verify_frozen_inputs


SLOT_INDEX = np.asarray([[packet * 128 + offset for offset in range(12, 20)] for packet in range(8)], dtype=np.int64)


def fit_scalers(train_indices: np.ndarray, iats: np.ndarray, lengths: np.ndarray, mask: np.ndarray, stats: np.ndarray) -> dict[str, object]:
    train_mask = mask[train_indices].astype(bool)
    iat_values = np.log1p(np.maximum(iats[train_indices][train_mask], 0.0) * 1e6)
    length_values = np.log1p(np.maximum(lengths[train_indices][train_mask], 0.0))
    stat_values = transformed_statistics(stats[train_indices])
    iat_median, iat_iqr = robust_fit(iat_values)
    len_median, len_iqr = robust_fit(length_values)
    stat_median, stat_iqr = robust_fit(stat_values)
    return {
        "iat_median": float(iat_median), "iat_iqr": float(iat_iqr),
        "length_median": float(len_median), "length_iqr": float(len_iqr),
        "statistics_median": np.asarray(stat_median).tolist(),
        "statistics_iqr": np.asarray(stat_iqr).tolist(),
        "fit_packet_values": int(train_mask.sum()),
        "fit_flow_values": int(len(train_indices)),
    }


def encode(images: np.ndarray, raw_indices: np.ndarray, feature: str, scalers: dict[str, object], iats: np.ndarray, lengths: np.ndarray, directions: np.ndarray, mask: np.ndarray, stats: np.ndarray) -> np.ndarray:
    flat = np.asarray(images).reshape(len(images), 1024).copy()
    if np.any(flat[:, SLOT_INDEX.reshape(-1)] != 0):
        raise RuntimeError("F0 address-byte injection slots are not all zero")
    raw_iats = iats[raw_indices]
    raw_lengths = lengths[raw_indices]
    raw_directions = directions[raw_indices]
    raw_mask = mask[raw_indices]
    raw_stats = stats[raw_indices]
    iat_q = robust_quantize(np.log1p(np.maximum(raw_iats, 0.0) * 1e6), np.asarray(scalers["iat_median"]), np.asarray(scalers["iat_iqr"]))
    len_q = robust_quantize(np.log1p(np.maximum(raw_lengths, 0.0)), np.asarray(scalers["length_median"]), np.asarray(scalers["length_iqr"]))
    iat_q = np.where(raw_mask.astype(bool), iat_q, 0).astype(np.uint8)
    len_q = np.where(raw_mask.astype(bool), len_q, 0).astype(np.uint8)
    stat_q = robust_quantize(transformed_statistics(raw_stats), np.asarray(scalers["statistics_median"]), np.asarray(scalers["statistics_iqr"]))
    for packet in range(8):
        slot = SLOT_INDEX[packet]
        if feature == "f1":
            flat[:, slot] = iat_q[:, packet, None]
        elif feature == "f2":
            flat[:, slot] = stat_q[:, packet, None]
        elif feature == "f3":
            direction_q = np.where(
                raw_mask[:, packet] == 0,
                0,
                np.where(raw_directions[:, packet] > 0, 255, np.where(raw_directions[:, packet] < 0, 0, 128)),
            ).astype(np.uint8)
            valid_q = (raw_mask[:, packet] * 255).astype(np.uint8)
            flat[:, slot[0]] = len_q[:, packet]
            flat[:, slot[1]] = len_q[:, packet]
            flat[:, slot[2]] = iat_q[:, packet]
            flat[:, slot[3]] = iat_q[:, packet]
            flat[:, slot[4]] = direction_q
            flat[:, slot[5]] = valid_q
            flat[:, slot[6]] = stat_q[:, packet]
            flat[:, slot[7]] = stat_q[:, packet]
        else:
            raise KeyError(feature)
    return flat.reshape(-1, 32, 32)


def build_protocol(protocol_id: str) -> None:
    verify_frozen_inputs()
    audit = json.loads((RAW_CACHE / "cache_audit.json").read_text())
    if audit["status"] != "PASS" or audit["unknown_test_rows_loaded"] != 0 or audit["known_test_rows_loaded"] != 0:
        raise RuntimeError("raw feature cache boundary audit failed")
    source = source_input_dir(protocol_id)
    known_rows = load_known_input_rows(protocol_id)
    expected_uids = {row["flow_uid"] for row in known_rows}
    raw_uids = np.load(RAW_CACHE / "flow_uids.npy", allow_pickle=False)
    uid_to_raw = {str(uid): i for i, uid in enumerate(raw_uids)}
    iats = np.load(RAW_CACHE / "packet_iat_seconds.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(RAW_CACHE / "packet_lengths.npy", mmap_mode="r", allow_pickle=False)
    directions = np.load(RAW_CACHE / "packet_directions.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(RAW_CACHE / "packet_mask.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(RAW_CACHE / "flow_statistics.npy", mmap_mode="r", allow_pickle=False)
    uids_by_split = {split: np.load(source / f"{split}_flow_uids.npy", allow_pickle=False) for split in ("train", "validation")}
    if set(map(str, np.concatenate(list(uids_by_split.values())))) != expected_uids:
        raise RuntimeError(f"{protocol_id}: Stage 14C input UID arrays/manifest differ")
    raw_indices = {split: np.asarray([uid_to_raw[str(uid)] for uid in values], dtype=np.int64) for split, values in uids_by_split.items()}
    scalers = fit_scalers(raw_indices["train"], iats, lengths, mask, stats)
    for feature in NEW_FEATURES:
        target = RUNS_ROOT / feature / protocol_id / "inputs"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite {target}")
        target.mkdir(parents=True)
        for split in ("train", "validation"):
            source_images = np.load(source / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
            encoded = encode(source_images, raw_indices[split], feature, scalers, iats, lengths, directions, mask, stats)
            np.save(target / f"{split}_images.npy", encoded, allow_pickle=False)
            np.save(target / f"{split}_labels.npy", np.load(source / f"{split}_labels.npy", allow_pickle=False), allow_pickle=False)
            np.save(target / f"{split}_flow_uids.npy", uids_by_split[split], allow_pickle=False)
        shutil.copy2(source / "input_manifest.csv", target / "input_manifest.csv")
        shutil.copy2(source / "label_map.json", target / "label_map.json")
        scaler_payload = {
            "feature": feature,
            "protocol_id": protocol_id,
            "fit_split": "Known Train only",
            "unknown_test_rows_loaded": 0,
            "known_test_rows_loaded": 0,
            "transform": "log1p/log-ratio -> median/IQR -> clip[-4,4] -> uint8[0,255]",
            **scalers,
        }
        json_dump(target / "scaler.json", scaler_payload)
        input_audit = {
            "status": "PASS", "feature": feature, "protocol_id": protocol_id,
            "model_visible_roles": ["known:train", "known:validation"],
            "train_samples": len(uids_by_split["train"]), "validation_samples": len(uids_by_split["validation"]),
            "unknown_samples_loaded": 0, "known_test_samples_loaded": 0,
            "source_f0_manifest_sha256": sha256_file(source / "input_manifest.csv"),
            "derived_manifest_sha256": sha256_file(target / "input_manifest.csv"),
            "scaler_sha256": sha256_file(target / "scaler.json"),
            "injection_slots_verified_zero": True,
            "image_shape": [32, 32], "channels": 1,
        }
        json_dump(target / "input_audit.json", input_audit)
        print(json.dumps({"event": "inputs_complete", "feature": feature, "protocol_id": protocol_id, "train": len(uids_by_split["train"]), "validation": len(uids_by_split["validation"])}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol-id", choices=protocol_ids()); args = parser.parse_args()
    if args.protocol_id:
        build_protocol(args.protocol_id)
    else:
        for protocol_id in protocol_ids():
            build_protocol(protocol_id)


if __name__ == "__main__":
    main()
