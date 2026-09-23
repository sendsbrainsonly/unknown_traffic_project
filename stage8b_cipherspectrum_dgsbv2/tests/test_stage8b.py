from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage8b_common import AccessLedger, empirical_threshold, gate_status  # noqa: E402


def test_empirical_threshold_tie_uses_higher_threshold() -> None:
    threshold, acceptance = empirical_threshold(np.array([0.0, 1.0, 2.0, 3.0]), target_acceptance=0.625)
    assert threshold == 2.0
    assert acceptance == 0.5


def test_empirical_threshold_hits_preregistered_acceptance() -> None:
    scores = np.arange(40, dtype=np.float64)
    threshold, acceptance = empirical_threshold(scores, target_acceptance=0.975)
    assert threshold == 1.0
    assert acceptance == 0.975


@pytest.mark.parametrize(
    ("global_count", "local_count", "expected"),
    [
        (1, 1, "DUAL_GATE_ACTIVE"),
        (0, 1, "GLOBAL_GATE_REDUNDANT"),
        (1, 0, "LOCAL_GATE_REDUNDANT"),
        (0, 0, "BOTH_GATES_REDUNDANT"),
    ],
)
def test_gate_status(global_count: int, local_count: int, expected: str) -> None:
    assert gate_status(global_count, local_count) == expected


def test_access_ledger_blocks_test_before_open() -> None:
    ledger = AccessLedger("low")
    with pytest.raises(RuntimeError, match="blocked before open"):
        ledger.record(Path("known_test.npy"), "npy", "KNOWN_TEST")
    assert ledger.reads == []


def test_access_ledger_records_known_validation_only() -> None:
    ledger = AccessLedger("medium")
    ledger.record(Path("z_val.npy"), "npy", "KNOWN_VALIDATION", records=12)
    payload = ledger.payload()
    assert payload["known_test_opened"] == 0
    assert payload["unknown_test_opened"] == 0
    assert payload["unknown_inference_executed"] is False
    assert payload["reads"][0]["role"] == "KNOWN_VALIDATION"
