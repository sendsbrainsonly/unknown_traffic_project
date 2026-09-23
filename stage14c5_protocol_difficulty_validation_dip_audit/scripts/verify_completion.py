#!/usr/bin/env python3
"""Verify Stage 14C-5 requested outputs and frozen-data invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_HASHES = {
    PROJECT / "stage14b_vnat_protocol_freeze/vnat_open_set_protocol.json": "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced",
    PROJECT / "stage14b_vnat_protocol_freeze/vnat_split_manifest.csv": "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e",
    PROJECT / "stage14c4_our_method_training_cleanup/scripts/train_cleaned.py": "a72b0e52be8c10917c057b49fea58a228253a867007e7390097c27e98cab8655",
    PROJECT / "stage14c4_our_method_training_cleanup/training_dynamics.csv": "0405d1638f848e109567fdeb25065f1196a8557c8c80ee8d06baf48ed0f94a8c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        "RESULTS.md",
        "manifest.json",
        "stage14c5_report.md",
        "protocol_summary.csv",
        "protocol_class_composition.csv",
        "per_class_comparison.csv",
        "nonzero_confusion_pairs.csv",
        "validation_drop_events.csv",
        "replay_drop_telemetry.csv",
        "replay_reset_window.csv",
        "per_class_drop_recall.csv",
        "validation_dip_summary.csv",
        "audit_summary.json",
    ]
    assert all((OUT / path).is_file() and (OUT / path).stat().st_size > 0 for path in required)
    assert all(sha256(path) == digest for path, digest in EXPECTED_HASHES.items())

    audit = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["conclusion"] == "PIPELINE_READY_WITH_KNOWN_TRANSIENT"
    assert audit["freeze_hash"] == EXPECTED_FREEZE
    assert audit["known_test_samples_used"] == 0
    assert audit["unknown_test_samples_used"] == 0
    assert audit["des_executed"] is False
    assert audit["training_code_modified"] is False
    assert audit["observational_replays_parity_pass"] is True
    assert audit["maximum_replay_absolute_difference"] <= 1e-10

    historical = pd.read_csv(OUT / "validation_drop_events.csv")
    assert len(historical) == 24
    assert int(historical.prototype_reset.sum()) == 0
    assert int((historical.epoch >= 51).sum()) == 0

    for protocol in ("medium_seed2025", "medium_seed2026"):
        base = OUT / "telemetry_replay" / protocol
        status = json.loads((base / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "PASS" and status["parity_pass"] is True
        assert status["known_test_samples_used"] == 0
        assert status["unknown_test_samples_used"] == 0
        assert status["des_executed"] is False
        assert len(pd.read_csv(base / "epoch_telemetry.csv")) == 100
        assert len(pd.read_csv(base / "per_class_validation.csv")) == 700

    report = (OUT / "stage14c5_report.md").read_text(encoding="utf-8")
    for phrase in (
        "Medium-2025 与 Medium-2026 差距的主要原因",
        "validation dip 是否由 scheduler/prototype reset 引起",
        "PIPELINE_READY_WITH_KNOWN_TRANSIENT",
    ):
        assert phrase in report

    result = {
        "status": "PASS",
        "conclusion": audit["conclusion"],
        "freeze_hash": EXPECTED_FREEZE,
        "external_input_hashes_unchanged": True,
        "required_artifacts_nonempty": True,
        "replay_parity_pass": True,
        "maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"],
        "historical_drop_events": len(historical),
        "drop_events_at_reset": int(historical.prototype_reset.sum()),
        "drop_events_after_epoch_50": int((historical.epoch >= 51).sum()),
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
    }
    (OUT / "completion_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
