from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage16s_common import SERVICES, load_role, protocol_id  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_partition_and_unknown_free() -> None:
    for service in SERVICES:
        pid = protocol_id(service)
        roles = {role: load_role(pid, role) for role in ("known_train", "known_validation", "known_test", "unknown_test")}
        assert sum(len(value["data"]) for value in roles.values()) == 3065
        flow_sets = {role: {row["flow_id"] for row in value["metadata"]} for role, value in roles.items()}
        assert sum(map(len, flow_sets.values())) == len(set().union(*flow_sets.values()))
        assert {row["service_label"] for row in roles["unknown_test"]["metadata"]} == {service}
        assert all(service not in {row["service_label"] for row in roles[role]["metadata"]} for role in ("known_train", "known_validation", "known_test"))
        assert np.bincount(roles["known_train"]["labels"]).min() >= 10


def test_frozen_des_and_h1_definitions() -> None:
    stage14d = load_module(
        "stage16s_test_stage14d",
        ROOT.parent / "stage14d_vnat_frozen_open_set_evaluation" / "scripts" / "stage14d_common.py",
    )
    stage15b = load_module(
        "stage16s_test_stage15b",
        ROOT.parent / "stage15b_known_only_hybrid_detector" / "scripts" / "run_stage15b.py",
    )
    train = np.asarray([[0.0], [2.0], [10.0], [12.0]])
    labels = np.asarray([0, 0, 1, 1])
    centroids = stage14d.empirical_centroids(train, labels)
    np.testing.assert_allclose(centroids[:, 0], [1.0, 11.0])
    scores, predictions = stage14d.centroid_scores(np.asarray([[0.0], [12.0]]), centroids)
    np.testing.assert_allclose(scores, [1.0, 1.0])
    np.testing.assert_array_equal(predictions, [0, 1])
    percentile = stage15b.empirical_percentile(np.asarray([1.0, 2.0, 2.0, 4.0]), np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(percentile, [0.75, 0.75])
    np.testing.assert_allclose(np.maximum(percentile, np.asarray([0.5, 1.0])), [0.75, 1.0])


def test_p95_uses_higher_and_large_score_is_unknown() -> None:
    stage14d = load_module(
        "stage16s_test_stage14d_p95",
        ROOT.parent / "stage14d_vnat_frozen_open_set_evaluation" / "scripts" / "stage14d_common.py",
    )
    validation = np.arange(20, dtype=float)
    assert stage14d.threshold_p95(validation) == 19.0
    result = stage14d.detection_metrics(validation, np.asarray([0.0, 1.0]), np.asarray([20.0, 21.0]))
    assert result["ufar"] == 0.0
    assert result["known_frr"] == 0.0
