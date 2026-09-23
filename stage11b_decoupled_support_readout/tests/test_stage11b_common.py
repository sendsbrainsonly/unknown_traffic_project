from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage11b_common import CONFIG_PATH, centroid_scores, empirical_centroids


class DummyLatents:
    def __init__(self) -> None:
        self.mu = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        self.labels = np.asarray([0, 0, 1, 1])


def test_empirical_centroid_and_squared_distance() -> None:
    centers = empirical_centroids(DummyLatents())
    np.testing.assert_allclose(centers, [[1.0, 0.0], [11.0, 0.0]])
    scores, predictions = centroid_scores(np.asarray([[2.0, 0.0], [9.0, 0.0]]), centers)
    np.testing.assert_allclose(scores, [1.0, 4.0])
    np.testing.assert_array_equal(predictions, [0, 1])


def test_only_four_fixed_detectors_and_no_training_config() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert set(config["detectors"]) == {"R0", "R1", "R2", "R3"}
    assert config["density_protocol"]["allowed_components"] == [1, 2]
    assert config["frozen_inference"]["optimizer"] is None
    assert config["frozen_inference"]["backward"] is False
    assert config["frozen_inference"]["checkpoint_update"] is False


def test_evaluator_contains_no_training_operations() -> None:
    source = (SCRIPTS / "evaluate_frozen.py").read_text()
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "model.train(" not in source
    assert "model.prototypes.copy_" not in source
