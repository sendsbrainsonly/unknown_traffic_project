from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage14d_common import (  # noqa: E402
    detection_metrics,
    local_knn10_scores,
    paired_bootstrap_ci,
    robust_normalize,
    robust_parameters,
    threshold_p95,
)


def test_threshold_is_higher_p95() -> None:
    values = np.arange(20, dtype=np.float64)
    assert threshold_p95(values) == 19.0


def test_unknown_is_positive_and_ufar_frr_are_full_sample_rates() -> None:
    metrics = detection_metrics(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.0, 4.0]),
        np.asarray([1.0, 5.0]),
    )
    assert metrics["threshold"] == 3.0
    assert metrics["known_frr"] == 0.5
    assert metrics["ufar"] == 0.5


def test_robust_normalization_uses_median_mad() -> None:
    params = robust_parameters(np.asarray([0.0, 1.0, 2.0]), 1e-12)
    normalized = robust_normalize(np.asarray([0.0, 1.0, 2.0]), params)
    np.testing.assert_allclose(normalized, [-1.0, 0.0, 1.0], atol=2e-12)


def test_class_conditional_knn10() -> None:
    train0 = np.arange(12, dtype=np.float64)[:, None]
    train1 = (100 + np.arange(12, dtype=np.float64))[:, None]
    train = np.vstack([train0, train1])
    labels = np.asarray([0] * 12 + [1] * 12)
    queries = np.asarray([[0.0], [100.0]])
    scores = local_knn10_scores(queries, np.asarray([0, 1]), train, labels)
    np.testing.assert_allclose(scores, [4.5, 4.5])


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_ci([1.0, 2.0, 3.0], 1000, 0)
    second = paired_bootstrap_ci([1.0, 2.0, 3.0], 1000, 0)
    assert first == second

