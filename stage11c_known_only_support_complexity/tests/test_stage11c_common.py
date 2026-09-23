from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage11c_common import mle_covariance, spherical_nll, spectrum_statistics  # noqa: E402


def test_covariance_and_spherical_nll_are_finite() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 8))
    mean, covariance = mle_covariance(x)
    assert covariance.shape == (8, 8)
    stats = spectrum_statistics(x, "test")
    assert stats["condition_number"] >= 1
    nll = spherical_nll(x, mean, np.trace(covariance) / 8)
    assert nll.shape == (80,)
    assert np.all(np.isfinite(nll))


def test_config_is_frozen_known_only() -> None:
    config = json.loads((ROOT / "configs" / "stage11c_config.json").read_text())
    assert config["bootstrap"] == {"iterations": 100, "seed": 0, "subspace_k": [5, 10]}
    assert config["new_encoder_training"] is False
    assert config["new_detector_fitting"] is False
    assert "unknown_test arrays" in config["forbidden_inputs"]


def test_analyzer_has_no_test_array_access_or_model_fit() -> None:
    common = (ROOT / "scripts" / "stage11c_common.py").read_text()
    analyzer = (ROOT / "scripts" / "analyze_run.py").read_text()
    executable = common + analyzer
    assert 'archive["known_test' not in executable
    assert 'archive["unknown_test' not in executable
    assert ".fit(" not in executable
    assert ".fit_predict(" not in executable
    assert "backward(" not in executable
    assert "optimizer" not in analyzer.lower()
