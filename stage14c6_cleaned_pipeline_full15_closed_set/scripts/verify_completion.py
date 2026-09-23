#!/usr/bin/env python3
"""Verify Stage 14C-6 completion, integrity, and reporting constraints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED_INPUT_HASHES = {
    PROJECT / "stage14b_vnat_protocol_freeze/vnat_open_set_protocol.json": "5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced",
    PROJECT / "stage14b_vnat_protocol_freeze/vnat_split_manifest.csv": "66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e",
    PROJECT / "stage14c4_our_method_training_cleanup/scripts/train_cleaned.py": "a72b0e52be8c10917c057b49fea58a228253a867007e7390097c27e98cab8655",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        "preflight.json", "frozen_training_config.json", "input_audit.csv",
        "run_queue_status.json", "run_level_results.csv", "setting_summary.csv",
        "per_class_results.csv", "class_summary.csv", "training_dynamics.csv",
        "paired_vs_native.csv", "paired_vs_native_summary.csv",
        "per_class_vs_native.csv", "hard_class_vs_native_summary.csv",
        "known_class_pair_effects.csv", "focus_class_pair_effects.csv",
        "stage14c4_parity.csv", "aggregate_summary.json", "stage14c6_report.md",
    ]
    assert all((ROOT / path).is_file() and (ROOT / path).stat().st_size > 0 for path in required)
    assert all(sha256(path) == expected for path, expected in EXPECTED_INPUT_HASHES.items())
    preflight = json.loads((ROOT / "preflight.json").read_text(encoding="utf-8"))
    queue = json.loads((ROOT / "run_queue_status.json").read_text(encoding="utf-8"))
    aggregate = json.loads((ROOT / "aggregate_summary.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(ROOT / "run_level_results.csv")
    per_class = pd.read_csv(ROOT / "per_class_results.csv")
    dynamics = pd.read_csv(ROOT / "training_dynamics.csv")
    parity = pd.read_csv(ROOT / "stage14c4_parity.csv")
    assert preflight["status"] == "PASS" and preflight["freeze_hash"] == EXPECTED_FREEZE
    assert queue["status"] == "success" and len(queue["completed"]) == 15 and not queue["failures"]
    assert aggregate["status"] == "PASS" and aggregate["runs_successful"] == 15
    assert len(runs) == 15 and len(dynamics) == 1500
    assert set(runs.epochs_completed) == {100}
    assert runs.status.eq("success").all()
    assert not runs.weak_run.any()
    assert runs.known_test_samples_used.sum() == 0
    assert runs.unknown_test_samples_used.sum() == 0
    assert not runs.des_executed.any()
    assert aggregate["checkpoints_verified"] == 15
    assert parity.exact_metric_parity_1e_12.all()
    assert per_class.groupby("protocol_id").size().shape[0] == 15
    for row in runs.itertuples(index=False):
        checkpoint = Path(row.checkpoint_path)
        assert checkpoint.is_file() and sha256(checkpoint) == row.checkpoint_sha256
    report = (ROOT / "stage14c6_report.md").read_text(encoding="utf-8")
    for phrase in (
        "CLOSED_SET_PASS_WITH_HARD_PROTOCOLS",
        "Known Test samples used: `0`",
        "Unknown Test samples used: `0`",
        "Can the cleaned pipeline be frozen?",
    ):
        assert phrase in report
    result = {
        "status": "PASS",
        "conclusion": "CLOSED_SET_PASS_WITH_HARD_PROTOCOLS",
        "runs_completed": 15,
        "epochs_verified": 1500,
        "checkpoints_sha256_verified": 15,
        "stage14c4_metric_parity": True,
        "freeze_hash": EXPECTED_FREEZE,
        "frozen_input_hashes_unchanged": True,
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
    }
    (ROOT / "completion_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
