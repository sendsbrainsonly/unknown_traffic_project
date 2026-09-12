from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    AUDIT_ROOT,
    VALID_SETTINGS,
    KnownImageDataset,
    known_acceptance_threshold,
    load_fold,
    verify_frozen_inputs,
)
from evaluate_final import detector_metrics, paired_class_stratified_bootstrap  # noqa: E402
from prepare_setting import FORBIDDEN_UNKNOWN_FLAGS, usage_flags  # noqa: E402

sys.path.insert(0, str(AUDIT_ROOT))
from stage3_model import Stage3OpenDetectNet  # noqa: E402


def test_frozen_protocol_and_execution_plan_hashes_match() -> None:
    verify_frozen_inputs()


def test_all_primary_fold_views_exclude_unknown_source_labels() -> None:
    for setting in VALID_SETTINGS:
        fold = load_fold(setting)
        dataset = KnownImageDataset("train", fold, augment=False)
        selected_source_labels = np.asarray(dataset.source_labels[dataset.source_indices])
        assert len(dataset) == fold["formal_counts"]["known_train"]
        assert set(selected_source_labels).isdisjoint(fold["unknown_current_project_ids"])
        assert dataset.local_labels.min() == 0
        assert dataset.local_labels.max() == fold["known_count"] - 1


def test_unknown_rows_never_receive_fit_or_calibration_flags() -> None:
    for split in ("train", "val", "test"):
        flags = usage_flags(split, "Unknown")
        assert all(not flags[name] for name in FORBIDDEN_UNKNOWN_FLAGS)
        assert flags["used_final_test"] is (split == "test")


def test_known_usage_is_fixed_by_original_split() -> None:
    train = usage_flags("train", "Known")
    val = usage_flags("val", "Known")
    test = usage_flags("test", "Known")
    assert train["used_encoder_train"] and train["used_scaler_fit"] and train["used_pca_fit"]
    assert train["used_density_fit"] and not train["used_threshold_calibration"]
    assert val["used_encoder_val"] and val["used_threshold_calibration"]
    assert not any(val[name] for name in ("used_scaler_fit", "used_pca_fit", "used_density_fit"))
    assert test["used_final_test"] and not test["used_encoder_train"] and not test["used_encoder_val"]


def test_threshold_uses_higher_is_more_known_semantics() -> None:
    scores = np.arange(1000, dtype=np.float64)
    threshold = known_acceptance_threshold(scores, 0.95)
    acceptance = np.mean(scores >= threshold)
    assert abs(acceptance - 0.95) <= 1.0 / len(scores)


def test_logvar_guard_bounds_only_overflow_prone_upper_tail() -> None:
    raw = torch.tensor([-45.0, -30.0, 19.0, 20.0, 21.0, 87.5])
    bounded = Stage3OpenDetectNet.bound_logvar(raw)
    assert torch.equal(bounded, torch.tensor([-45.0, -30.0, 19.0, 20.0, 20.0, 20.0]))


def test_unknown_positive_metrics_and_far_are_consistent() -> None:
    known_scores = np.asarray([4.0, 3.0, 2.0, 1.0])
    unknown_scores = np.asarray([-4.0, -3.0, 2.5, -2.0])
    result = detector_metrics("test", known_scores, unknown_scores, threshold=0.0)
    assert result["known_false_rejection_rate"] == 0.0
    assert result["unknown_false_acceptance_rate"] == 0.25
    assert result["unknown_rejection_rate"] == 0.75
    assert 0.0 <= result["auroc_unknown_positive"] <= 1.0
    assert 0.0 <= result["auprc_unknown_positive"] <= 1.0


def test_paired_class_stratified_bootstrap_is_deterministic() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "known_or_unknown": ["Known"] * 4 + ["Unknown"] * 4,
            "class_name": ["K1", "K1", "K2", "K2", "U1", "U1", "U2", "U2"],
            "single_known_score": [4.0, 3.0, 2.0, 1.0, 0.5, -1.0, -2.0, -3.0],
            "multi_known_score": [4.0, 3.0, 2.0, 1.0, -0.5, -1.0, -2.0, -3.0],
        }
    )
    thresholds = {"Single-Full": 0.0, "Multi-Full-K2": 0.0}
    first, first_ci = paired_class_stratified_bootstrap(frame, thresholds, 20, 0)
    second, second_ci = paired_class_stratified_bootstrap(frame, thresholds, 20, 0)
    assert first == second
    assert first_ci == second_ci
    assert len(first) == 20
