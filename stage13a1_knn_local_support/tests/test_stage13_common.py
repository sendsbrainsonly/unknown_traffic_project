from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage13_common import apply_gate, centroid_scores, empirical_centroids, local_knn_scores


def test_empirical_centroids_and_predictions() -> None:
    train = np.asarray([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    centroids = empirical_centroids(train, labels)
    np.testing.assert_allclose(centroids[:, 0], [1.0, 11.0])
    scores, predictions = centroid_scores(np.asarray([[0.5], [11.5]]), centroids)
    np.testing.assert_allclose(scores, [0.25, 0.25])
    np.testing.assert_array_equal(predictions, [0, 1])


def test_knn_is_class_conditional_and_excludes_exact_self() -> None:
    class0 = np.arange(0.0, 12.0)[:, None]
    class1 = np.arange(100.0, 112.0)[:, None]
    train = np.vstack([class0, class1])
    labels = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)
    predictions = labels.copy()
    scores = local_knn_scores(
        train,
        predictions,
        train,
        labels,
        (5, 10),
        query_train_indices=np.arange(len(train)),
        chunk_size=4,
    )
    assert np.all(scores[5] > 0.0)
    assert np.all(scores[10] >= scores[5])
    # The first point uses neighbours at distances 1..5 after masking itself.
    assert scores[5][0] == 3.0


def _rows(delta_by_scenario: dict[str, list[float]], delta_ufar: float = -0.01):
    rows = []
    for scenario, values in delta_by_scenario.items():
        for seed, value in enumerate(values):
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "delta_auroc": value,
                "delta_ufar": delta_ufar,
                "delta_known_frr": 0.0,
            })
    return rows


def _config():
    return {
        "gate": {
            "material_auroc_delta": 0.02,
            "clear_harm_auroc_delta": -0.02,
            "majority_positive_seed_count_overall": 8,
            "max_mean_delta_known_frr": 0.02,
            "material_scenarios_required_for_go": 2,
            "material_scenarios_required_for_conditional_go": 1,
            "require_lower_ufar_in_each_material_scenario": True,
        },
        "stability_definition": {"used_for_primary_gate": False},
    }


def test_gate_go_and_conditional_and_no_go() -> None:
    k5 = _rows({"a1": [0.01] * 5, "a2": [0.01] * 5, "a3": [0.01] * 5})
    go = _rows({"a1": [0.03] * 5, "a2": [0.02] * 5, "a3": [0.001] * 5})
    assert apply_gate(go, k5, _config())["final_gate"] == "GO"
    conditional = _rows({"a1": [0.03] * 5, "a2": [0.0] * 5, "a3": [-0.01] * 5})
    assert apply_gate(conditional, k5, _config())["final_gate"] == "CONDITIONAL_GO"
    no_go = _rows({"a1": [0.03] * 5, "a2": [-0.03] * 5, "a3": [0.0] * 5})
    assert apply_gate(no_go, k5, _config())["final_gate"] == "NO_GO"

