from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage11a_common import (  # noqa: E402
    anchor_prototypes,
    assert_prototype_is_buffer,
    detection_metrics,
)


class ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, values: torch.Tensor):
        mean = values * self.scale
        return mean, torch.zeros_like(mean), {}


class ToyAnchoredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ToyEncoder()
        self.register_buffer("prototypes", torch.tensor([[9.0, 9.0], [-9.0, -9.0]]))


def test_prototype_is_optimizer_free_registered_buffer() -> None:
    model = ToyAnchoredModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert_prototype_is_buffer(model, optimizer)
    assert "prototypes" not in dict(model.named_parameters())
    assert "prototypes" in dict(model.named_buffers())


def test_anchor_uses_deterministic_class_centroids_and_zeroes_after_gap() -> None:
    values = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [14.0, 14.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(values, labels), batch_size=2, shuffle=False)
    model = ToyAnchoredModel()
    rows = anchor_prototypes(model, loader, torch.device("cpu"), [100, 200], 1, "TEST")
    assert torch.equal(model.prototypes, torch.tensor([[1.0, 1.0], [12.0, 12.0]]))
    assert len(rows) == 2
    assert all(row["after_anchor_gap"] == pytest.approx(0.0) for row in rows)
    assert all(row["prototype_source"] == "KNOWN_TRAIN_DETERMINISTIC_MU_ONLY" for row in rows)


def test_detection_threshold_is_validation_p95_higher_only() -> None:
    validation = np.arange(20, dtype=np.float64)
    known = np.arange(10, dtype=np.float64)
    unknown = np.arange(10, 20, dtype=np.float64)
    metrics = detection_metrics(validation, known, unknown, eval_seed=2021)
    assert metrics["threshold"] == 19.0
    assert metrics["threshold_source"].startswith("Known Validation")
    assert "youden" not in metrics["threshold_source"].lower()

