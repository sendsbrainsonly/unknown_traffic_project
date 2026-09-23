from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage13a2_common import apply_gate, fixed_gl_fusion, robust_parameters


def test_robust_parameters_and_fixed_fusion_are_validation_only() -> None:
    validation_c = np.array([1.0, 2.0, 3.0, 7.0])
    validation_k = np.array([10.0, 11.0, 14.0, 30.0])
    pc = robust_parameters(validation_c, 1e-12)
    pk = robust_parameters(validation_k, 1e-12)
    assert pc["median"] == 2.5
    assert pc["mad"] == 1.0
    assert pk["median"] == 12.5
    assert pk["mad"] == 2.0
    _, _, first = fixed_gl_fusion(np.array([4.0]), np.array([20.0]), pc, pk)
    _, _, second = fixed_gl_fusion(np.array([4000.0]), np.array([-2000.0]), pc, pk)
    assert np.isclose(first[0], 0.5 * ((4.0 - 2.5) / (1.0 + 1e-12)) + 0.5 * ((20.0 - 12.5) / (2.0 + 1e-12)))
    assert pc == robust_parameters(validation_c, 1e-12)
    assert pk == robust_parameters(validation_k, 1e-12)
    assert not np.isclose(first[0], second[0])


def _rows(a1: float, a2: float, a3: float, positive_counts=(5, 5, 5)):
    rows = []
    for scenario, delta, count in zip(("a1", "a2", "a3"), (a1, a2, a3), positive_counts):
        for index in range(5):
            signed = delta if index < count else -abs(delta or 0.001)
            rows.append({
                "scenario": scenario,
                "delta_auroc": signed,
                "delta_auprc": 0.0,
                "delta_ufar": -0.01 if scenario == "a2" else 0.0,
                "delta_known_frr": 0.0,
            })
    return rows


def _config():
    return {
        "gate": {
            "a1_min_delta_auroc_exclusive": -0.005,
            "a2_min_delta_auroc_inclusive": 0.02,
            "a3_min_delta_auroc_exclusive": -0.005,
            "min_positive_seed_count": 10,
            "max_mean_delta_known_frr": 0.02,
        },
        "conditional_gate": {
            "a1_min_delta_auroc_exclusive": -0.02,
            "a2_min_delta_auroc_exclusive": 0.0,
            "a3_min_delta_auroc_exclusive": -0.02,
            "min_positive_seed_count": 8,
            "max_mean_delta_known_frr": 0.02,
        },
    }


def test_gate_go_conditional_and_no_go() -> None:
    assert apply_gate(_rows(0.001, 0.03, 0.001), _config())["final_gate"] == "GO"
    conditional_rows = _rows(-0.01, 0.015, 0.001, positive_counts=(0, 5, 5))
    assert apply_gate(conditional_rows, _config())["final_gate"] == "CONDITIONAL_GO"
    assert apply_gate(_rows(-0.03, 0.03, 0.001), _config())["final_gate"] == "NO_GO"

