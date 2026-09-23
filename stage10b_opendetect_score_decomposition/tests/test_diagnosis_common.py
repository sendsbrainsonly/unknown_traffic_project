from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnosis_common import anomaly_metrics, squared_distance_matrix
from run_score_decomposition import empirical_centers, nearest


def test_squared_distance_matrix_and_clipping() -> None:
    values = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(squared_distance_matrix(values, centers), [[0.0, 1.0], [4.0, 1.0]])


def test_unknown_positive_metrics_have_expected_orientation() -> None:
    result = anomaly_metrics(np.asarray([0.0, 0.1]), np.asarray([0.9, 1.0]))
    assert result["AUROC"] == 1.0
    assert result["AUPRC"] == 1.0
    assert result["cliffs_delta_unknown_minus_known"] == 1.0


def test_empirical_centers_use_only_matching_train_labels() -> None:
    values = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [14.0, 0.0]])
    labels = np.asarray(["a", "a", "b", "b"])
    centers = empirical_centers(values, labels, ["a", "b"])
    assert np.allclose(centers, [[1.0, 0.0], [12.0, 0.0]])
    scores, prediction = nearest(np.asarray([[0.0, 0.0], [13.0, 0.0]]), centers)
    assert prediction.tolist() == [0, 1]
    assert np.allclose(scores, [1.0, 1.0])
