from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_boundary_diagnosis as boundary  # noqa: E402


def test_rule_decisions_use_fixed_margin_definitions() -> None:
    names = ["A", "B"]
    class_scores = np.array([[5.0, 2.0], [1.0, 4.0]])
    local_scores = np.array([[4.0, 3.0, 1.0, 0.0], [0.0, -1.0, 3.0, 2.0]])
    decisions = boundary.rule_decisions(
        names, class_scores, local_scores, global_threshold=4.5,
        class_thresholds=np.array([4.0, 3.0]),
        local_thresholds=np.array([3.5, 3.5, 2.5, 2.5]),
    )
    assert decisions["Global"]["accepted"].tolist() == [True, False]
    assert decisions["Class-P05"]["accepted"].tolist() == [True, True]
    assert decisions["Component-P05"]["accepted"].tolist() == [True, True]


def test_percentile_is_empirical_right_cdf() -> None:
    actual = boundary.percentile(np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(actual, [0.0, 2.0 / 3.0, 1.0])


def test_fixed_protocol_constants() -> None:
    assert boundary.QUANTILE == 0.05
    assert boundary.MIN_LOCAL_VALIDATION == 30
    assert boundary.RULES == ("Global", "Class-P05", "Component-P05")


def test_source_forbids_refit_and_quantile_search() -> None:
    source = (SCRIPTS / "run_boundary_diagnosis.py").read_text(encoding="utf-8")
    assert ".fit" + "(" not in source
    assert "Adaptive" + "K" not in source
    assert "0.01, 0.02" not in source
