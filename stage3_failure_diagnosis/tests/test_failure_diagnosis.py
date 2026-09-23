from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_failure as af  # noqa: E402


def test_transition_code_covers_four_states() -> None:
    single = pd.Series([False, False, True, True])
    multi = pd.Series([False, True, False, True])
    assert af.transition_code(single, multi).tolist() == ["SS", "SM", "MS", "MM"]


def test_epsilon_squared_matches_stage26_definition() -> None:
    assert np.isclose(af.epsilon_squared_kruskal(10.0, 100, 2), 9.0 / 98.0)
    assert np.isnan(af.epsilon_squared_kruskal(1.0, 2, 2))


def test_diagnosis_source_has_no_refit_or_test_embedding_path() -> None:
    source = (SCRIPTS / "analyze_failure.py").read_text(encoding="utf-8")
    forbidden = (".fit" + "(", "test_" + "known_mu", "evaluate_" + "final.py")
    for token in forbidden:
        assert token not in source


def test_snapshot_constants_are_frozen() -> None:
    assert af.SNAPSHOT_COMMIT == "ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc"
    assert set(af.SNAPSHOT_PACKAGE_SHA256) == {"A-1", "A-2", "A-3"}
