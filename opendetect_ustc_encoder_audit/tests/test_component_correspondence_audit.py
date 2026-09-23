from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_component_correspondence_audit import (  # noqa: E402
    CLASSES,
    assert_flow_identity,
    contingency_and_matching,
    correspondence_label,
)


def test_hungarian_correspondence_is_label_permutation_invariant() -> None:
    tf = np.asarray([0, 0, 0, 1, 1, 1])
    od = 1 - tf
    table, mapping, accuracy = contingency_and_matching(tf, od)
    assert table.tolist() == [[0, 3], [3, 0]]
    assert mapping == {0: 1, 1: 0}
    assert accuracy == 1.0


def test_correspondence_label_combines_metrics() -> None:
    assert correspondence_label(0.9, 0.9, 0.95) == "HIGH_CORRESPONDENCE"
    assert correspondence_label(0.1, 0.1, 0.60) == "LOW_CORRESPONDENCE"
    assert correspondence_label(0.9, 0.1, 0.95) == "MODERATE_CORRESPONDENCE"


def test_hungarian_supports_k3_permutations() -> None:
    tf = np.asarray([0, 0, 1, 1, 2, 2])
    od = np.asarray([2, 2, 0, 0, 1, 1])
    table, mapping, accuracy = contingency_and_matching(tf, od, n_components=3)
    assert table.tolist() == [[0, 0, 2], [2, 0, 0], [0, 2, 0]]
    assert mapping == {0: 2, 1: 0, 2: 1}
    assert accuracy == 1.0


def test_flow_identity_is_independent_of_component_count() -> None:
    rows = [
        {"flow_id": f"{name}-{split}", "class_name": name, "split": split}
        for name in CLASSES
        for split in ("train", "val")
    ]
    assertions = assert_flow_identity(pd.DataFrame(rows), pd.DataFrame(rows))
    assert len(assertions) == 8
    assert all(item["sets_equal"] for item in assertions)
