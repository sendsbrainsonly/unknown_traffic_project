from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (  # noqa: E402
    AccessLedger,
    conservative_empirical_threshold,
    linear_p05,
    true_class_component_assignments,
)


def test_conservative_threshold_uses_higher_value_on_exact_acceptance_tie() -> None:
    threshold, acceptance = conservative_empirical_threshold(np.array([0.0, 1.0, 2.0, 3.0]), 0.625)
    assert threshold == 2.0
    assert acceptance == 0.5


def test_linear_p05_matches_frozen_quantile_definition() -> None:
    values = np.arange(21, dtype=np.float64)
    assert linear_p05(values) == 1.0


def test_true_class_component_assignment_cannot_use_other_class_components() -> None:
    labels = np.array(["A", "B"], dtype=object)
    local_scores = np.array([[1.0, 2.0, 100.0, 99.0], [100.0, 99.0, 4.0, 5.0]])
    components, scores, posteriors = true_class_component_assignments(labels, ["A", "B"], local_scores)
    assert components.tolist() == [1, 1]
    assert scores.tolist() == [2.0, 5.0]
    assert np.all((posteriors > 0.5) & (posteriors < 1.0))


def test_access_ledger_blocks_forbidden_role_before_record(tmp_path: Path) -> None:
    ledger = AccessLedger("low", tmp_path / "audit.json")
    with pytest.raises(RuntimeError, match="blocked before open"):
        ledger.record(tmp_path / "forbidden.npy", "npy", "KNOWN_TEST")
    assert ledger.payload["reads"] == []


def test_frozen_config_has_no_search_or_test_access() -> None:
    config = json.loads((ROOT / "configs/stage8a_config.json").read_text(encoding="utf-8"))
    assert config["encoder"]["representation"] == "deterministic mu_x"
    assert config["pca"]["n_components"] == 64
    assert config["single_k1"]["n_components"] == 1
    assert config["multi_k2"]["n_components"] == 2
    assert config["thresholds"]["class_quantile"] == 0.05
    assert config["thresholds"]["component_fallback_min_validation_n"] == 30
    assert config["test_access"] is False
    assert config["unknown_inference"] is False
