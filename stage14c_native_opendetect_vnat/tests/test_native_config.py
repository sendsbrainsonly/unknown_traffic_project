#!/usr/bin/env python3
"""Static guards for the frozen released-code training contract."""

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from common import NATIVE_CONFIG, TRAINING_SEED  # noqa: E402


def test_native_configuration() -> None:
    assert NATIVE_CONFIG["epochs"] == 100
    assert NATIVE_CONFIG["train_batch_size"] == 128
    assert NATIVE_CONFIG["validation_batch_size"] == 64
    assert NATIVE_CONFIG["learning_rate"] == 0.001
    assert NATIVE_CONFIG["lambda"] == 0.005
    assert NATIVE_CONFIG["latent_dim"] == 128
    assert NATIVE_CONFIG["lr_milestones"] == [50, 80]
    assert NATIVE_CONFIG["prototype_reset_zero_based_epochs"] == [50, 80]
    assert NATIVE_CONFIG["checkpoint_selection"] == "maximum Known Validation Accuracy"
    assert NATIVE_CONFIG["early_stopping"] is False
    assert NATIVE_CONFIG["dataloader_tensor_sharing_strategy"] == "default_file_descriptor"
    assert TRAINING_SEED == 2022


def test_runner_has_no_test_or_unknown_array_loads() -> None:
    source = (SCRIPTS / "train_one.py").read_text(encoding="utf-8")
    forbidden = (
        "test_images.npy", "test_labels.npy", "unknown_images.npy",
        "unknown_labels.npy",
    )
    for token in forbidden:
        assert token not in source.lower()
