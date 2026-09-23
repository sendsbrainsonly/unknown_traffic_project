from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model import MaskSafeSequenceCNN  # noqa: E402


def padded(window: int) -> torch.Tensor:
    torch.manual_seed(7)
    x = torch.zeros(3, 3, window)
    x[:, :2, :5] = torch.randn(3, 2, 5)
    x[:, 2, :5] = 1.0
    if window > 5:
        x[:, :2, 5:] = 1000.0 * torch.randn(3, 2, window - 5)
    return x


def test_invalid_padding_does_not_change_eval_logits():
    torch.manual_seed(11)
    model = MaskSafeSequenceCNN(4).eval()
    with torch.no_grad():
        out8 = model(padded(8))
        out16 = model(padded(16))
        out32 = model(padded(32))
    torch.testing.assert_close(out8, out16, rtol=0, atol=1e-6)
    torch.testing.assert_close(out8, out32, rtol=0, atol=1e-6)


def test_invalid_padding_does_not_change_training_batch_statistics():
    torch.manual_seed(13)
    base = MaskSafeSequenceCNN(4)
    models = [copy.deepcopy(base).train() for _ in range(3)]
    outputs = [model(x) for model, x in zip(models, (padded(8), padded(16), padded(32)))]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=2e-6)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=2e-6)
    for left, right in ((models[0], models[1]), (models[0], models[2])):
        torch.testing.assert_close(left.bn1.running_mean, right.bn1.running_mean, rtol=0, atol=2e-6)
        torch.testing.assert_close(left.bn1.running_var, right.bn1.running_var, rtol=0, atol=2e-6)
        torch.testing.assert_close(left.bn2.running_mean, right.bn2.running_mean, rtol=0, atol=2e-6)
        torch.testing.assert_close(left.bn2.running_var, right.bn2.running_var, rtol=0, atol=2e-6)


def test_preregistered_windows_and_budget_are_frozen():
    config = json.loads((ROOT / "config.json").read_text())
    assert config["windows"] == [8, 16, 32]
    assert config["training"]["epochs"] == 100
    assert config["training"]["batch_size"] == 128
    assert config["training"]["seed"] == 2022
    assert config["features"]["length"] == "sign(direction) * log1p(captured_frame_length)"
    assert config["features"]["iat"] == "log1p(max(0, delta_timestamp_seconds) * 1000000)"
    assert config["features"]["mask"] == "1 for observed packet, 0 for structural padding"
