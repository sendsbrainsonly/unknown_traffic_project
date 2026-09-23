from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_setting import evaluate_role, primary_conclusion  # noqa: E402
from finalize_stage9 import cross_gate  # noqa: E402


class FakeK1:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def score_samples(self, _: np.ndarray) -> np.ndarray:
        return self.values


class FakeK2:
    def __init__(self, local: np.ndarray) -> None:
        self.local = local

    def _estimate_weighted_log_prob(self, _: np.ndarray) -> np.ndarray:
        return self.local


def test_six_frozen_methods_and_dgsb_and_gate() -> None:
    z = np.zeros((3, 2))
    models = {
        "k1": {"a": FakeK1(np.array([2.0, 0.0, 2.0])), "b": FakeK1(np.array([0.0, 2.0, 1.0]))},
        "k2": {
            "a": FakeK2(np.array([[1.0, 0.0], [-2.0, -3.0], [0.9, 0.8]])),
            "b": FakeK2(np.array([[-2.0, -3.0], [1.0, 0.0], [0.7, 0.6]])),
        },
    }
    thresholds = {
        "native": 0.5,
        "single": 0.5,
        "multi": 0.5,
        "class": np.array([0.5, 0.5]),
        "component": np.array([0.5, 0.5, 0.5, 0.5]),
        "dgsb_global": 0.5,
        "dgsb_local": np.array([1.1, 0.5, 0.5, 0.5]),
    }
    result = evaluate_role(z, np.array([1.0, 0.0, 1.0]), np.array([0, 1, 0]), ["a", "b"], models, thresholds)
    assert list(result) == ["Native", "Single-Full-K1", "Multi-Global-K2", "Class-P05-K2", "Component-P05-K2", "DGSB-v2"]
    dgsb = result["DGSB-v2"]
    assert np.array_equal(dgsb["accept"], dgsb["global_pass"] & dgsb["local_pass"])
    assert np.allclose(dgsb["detector_score"], np.minimum(dgsb["global_margin"], dgsb["local_margin"]))
    assert dgsb["prediction"].tolist() == ["a", "b", "a"]


def test_primary_conclusion_is_frozen_and_ci_based() -> None:
    win = [
        {"metric": "DeltaUFAR", "delta": -0.1, "ci_low": -0.2, "ci_high": -0.01},
        {"metric": "Delta Known FRR", "delta": 0.001, "ci_low": -0.001, "ci_high": 0.003},
    ]
    tradeoff = [
        {"metric": "DeltaUFAR", "delta": -0.1, "ci_low": -0.2, "ci_high": -0.01},
        {"metric": "Delta Known FRR", "delta": 0.02, "ci_low": 0.01, "ci_high": 0.03},
    ]
    no_gain = [
        {"metric": "DeltaUFAR", "delta": -0.01, "ci_low": -0.03, "ci_high": 0.01},
        {"metric": "Delta Known FRR", "delta": 0.0, "ci_low": -0.01, "ci_high": 0.01},
    ]
    harm = [
        {"metric": "DeltaUFAR", "delta": 0.1, "ci_low": 0.01, "ci_high": 0.2},
        {"metric": "Delta Known FRR", "delta": 0.0, "ci_low": -0.01, "ci_high": 0.01},
    ]
    assert primary_conclusion(win) == "WIN"
    assert primary_conclusion(tradeoff) == "TRADEOFF"
    assert primary_conclusion(no_gain) == "NO GAIN"
    assert primary_conclusion(harm) == "HARM"


def test_cross_setting_gate_rules() -> None:
    rows = [
        {"primary_conclusion": "WIN", "DeltaUFAR": -0.1},
        {"primary_conclusion": "WIN", "DeltaUFAR": -0.05},
        {"primary_conclusion": "NO GAIN", "DeltaUFAR": -0.01},
    ]
    assert cross_gate(rows) == ("A", "CONSISTENT EXTERNAL UTILITY")
    mixed = [
        {"primary_conclusion": "WIN", "DeltaUFAR": -0.1},
        {"primary_conclusion": "HARM", "DeltaUFAR": 0.05},
        {"primary_conclusion": "NO GAIN", "DeltaUFAR": 0.0},
    ]
    assert cross_gate(mixed) == ("E", "MIXED EXTERNAL RESULT")
