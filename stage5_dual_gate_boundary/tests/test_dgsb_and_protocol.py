from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_cstnet_protocol as protocol  # noqa: E402
import run_known_only_sanity as dgsb  # noqa: E402


def test_dual_global_threshold_hits_target_and_uses_conservative_tie_break() -> None:
    scores = np.array([1.0, 2.0, 3.0, 100.0])
    local_pass = np.array([True, True, True, False])
    threshold, rate, count = dgsb.calibrate_dual_global_threshold(scores, local_pass, 0.5)
    assert threshold == 2.0
    assert rate == 0.5
    assert count == 2

    threshold, rate, count = dgsb.calibrate_dual_global_threshold(scores, local_pass, 0.375)
    assert threshold == 3.0
    assert rate == 0.25
    assert count == 1


def test_dgsb_uses_raw_winning_component_and_strict_and_gate() -> None:
    names = ["KnownA"]
    class_scores = np.array([[10.0], [4.0], [10.0]])
    local_scores = np.array([[10.0, 9.0], [4.0, 3.0], [10.0, 9.0]])
    class_thresholds = np.array([0.0])
    local_thresholds = np.array([5.0, 0.0])
    result, winner, local_pass = dgsb.decisions(
        names,
        class_scores,
        local_scores,
        old_global_threshold=0.0,
        class_thresholds=class_thresholds,
        local_thresholds=local_thresholds,
        dual_global_threshold=9.0,
    )
    assert winner.tolist() == [0, 0, 0]
    assert local_pass.tolist() == [True, False, True]
    assert result["Component-P05"].tolist() == [True, True, True]
    assert result["DGSB"].tolist() == [True, False, True]


def test_local_threshold_fallback_is_class_p05() -> None:
    labels = np.array(["A"] * 4, dtype=object)
    names = ["A"]
    class_scores = np.array([[1.0], [2.0], [3.0], [4.0]])
    local_scores = np.array([[10.0, 0.0], [9.0, 0.0], [0.0, 8.0], [0.0, 7.0]])
    class_tau, local_tau, _, _, rows = dgsb.build_known_calibration(
        "toy", labels, names, class_scores, local_scores, 0.05, minimum_local_samples=3
    )
    assert all(row["fallback_to_class_p05"] for row in rows)
    assert np.allclose(local_tau, np.repeat(class_tau, 2))


def test_group_split_is_disjoint_and_deterministic() -> None:
    rows = []
    for class_name in ("a", "b"):
        for group in range(20):
            rows.append({"class_name": class_name, "group_id": f"g{group:02d}"})
    frame = pd.DataFrame(rows)
    split1, val1, test1, details1 = protocol.assign_known_splits(frame, seed=42, n_splits=10)
    split2, val2, test2, details2 = protocol.assign_known_splits(frame, seed=42, n_splits=10)
    assert split1.tolist() == split2.tolist()
    assert (val1, test1, details1) == (val2, test2, details2)
    check = frame.assign(split=split1.to_numpy())
    assert (check.groupby("group_id")["split"].nunique() == 1).all()


def test_runnable_sources_do_not_call_fit_or_name_unknown_result_files() -> None:
    sources = [
        ROOT / "scripts" / "run_known_only_sanity.py",
        ROOT / "scripts" / "freeze_cstnet_protocol.py",
    ]
    forbidden_names = {
        "test_unknown_mu.parquet",
        "frozen_final_predictions.parquet",
        "posthoc_unknown_boundary_results.csv",
    }
    for path in sources:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "fit" not in called_attributes
        assert "fit_transform" not in called_attributes
        assert not (forbidden_names & set(ast.literal_eval(node) for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)))
