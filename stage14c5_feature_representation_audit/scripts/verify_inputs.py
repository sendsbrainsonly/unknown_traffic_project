#!/usr/bin/env python3
"""Fail-closed audit of all Stage 14C.5 model-visible inputs."""

from __future__ import annotations

import json

import numpy as np

from common import NEW_FEATURES, RAW_CACHE, ROOT, RUNS_ROOT, json_dump, protocol_ids, verify_frozen_inputs


def main() -> None:
    frozen = verify_frozen_inputs()
    cache = json.loads((RAW_CACHE / "cache_audit.json").read_text())
    failures = []
    checked = 0
    for feature in NEW_FEATURES:
        for protocol_id in protocol_ids():
            input_dir = RUNS_ROOT / feature / protocol_id / "inputs"
            audit = json.loads((input_dir / "input_audit.json").read_text())
            scaler = json.loads((input_dir / "scaler.json").read_text())
            train_images = np.load(input_dir / "train_images.npy", mmap_mode="r", allow_pickle=False)
            val_images = np.load(input_dir / "validation_images.npy", mmap_mode="r", allow_pickle=False)
            checks = {
                "status": audit["status"] == "PASS",
                "unknown_zero": audit["unknown_samples_loaded"] == 0 and scaler["unknown_test_rows_loaded"] == 0,
                "known_test_zero": audit["known_test_samples_loaded"] == 0 and scaler["known_test_rows_loaded"] == 0,
                "roles": audit["model_visible_roles"] == ["known:train", "known:validation"],
                "shape": train_images.shape == (audit["train_samples"], 32, 32) and val_images.shape == (audit["validation_samples"], 32, 32),
                "dtype": train_images.dtype == np.uint8 and val_images.dtype == np.uint8,
                "scaler_fit_train": scaler["fit_flow_values"] == audit["train_samples"] and scaler["fit_split"] == "Known Train only",
                "slots": audit["injection_slots_verified_zero"] is True,
            }
            checked += 1
            if not all(checks.values()):
                failures.append({"feature": feature, "protocol_id": protocol_id, "checks": checks})
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked_feature_protocol_inputs": checked,
        "expected": 45,
        "raw_cache_status": cache["status"],
        "raw_cache_unique_flows": cache["unique_target_flows"],
        "packet_count_mismatches": cache["packet_count_mismatches"],
        "unknown_test_rows_loaded": cache["unknown_test_rows_loaded"],
        "known_test_rows_loaded": cache["known_test_rows_loaded"],
        "stage14b_freeze_hash": frozen["freeze_hash"],
        "failures": failures,
    }
    json_dump(ROOT / "input_verification.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS" or checked != 45:
        raise RuntimeError("Stage 14C.5 input verification failed")


if __name__ == "__main__":
    main()
