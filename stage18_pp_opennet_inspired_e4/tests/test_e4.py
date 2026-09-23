from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from e4_model import E4Encoder
from stage18_common import CACHE, CACHE_SHA256, VARIANT_ORDER, apply_robust_scaler, fit_robust_scaler, load_frozen_protocol, select_variant_channels, sha256_file, variant_spec


def test_frozen_cache_and_packet_features() -> None:
    frozen = load_frozen_protocol("Email")
    assert sha256_file(CACHE) == CACHE_SHA256
    assert frozen["raw_x"].shape == (3065, 30, 3)
    assert frozen["mask"].shape == (3065, 30)
    assert np.all(frozen["raw_x"][:, :, 1] >= 0)
    assert np.all(frozen["raw_x"][~frozen["mask"]] == 0)
    assert set(np.unique(frozen["raw_x"][:, :, 2][frozen["mask"]])).issubset({-1.0, 1.0})


def test_train_only_scaler_and_variants() -> None:
    frozen = load_frozen_protocol("Streaming")
    train_idx = frozen["indices"]["known_train"]
    scaler = fit_robust_scaler(frozen["raw_x"][train_idx], frozen["mask"][train_idx])
    scaled = apply_robust_scaler(frozen["raw_x"], frozen["mask"], scaler)
    expected_dims = {"time": 1, "length": 1, "time_length": 2, "time_length_direction": 3}
    for name in VARIANT_ORDER:
        mode = variant_spec(name)["features"]
        assert select_variant_channels(scaled, mode).shape[-1] == expected_dims[mode]
    assert np.all(scaled[~frozen["mask"]] == 0)


def test_shapes_gradients_and_padding_invariance() -> None:
    torch.manual_seed(7)
    mask = torch.zeros(4, 30, dtype=torch.bool)
    mask[:, :7] = True
    base = torch.randn(4, 30, 3)
    changed = base.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 1000
    for multi_scale in (False, True):
        for recurrent in (False, True):
            model = E4Encoder(3, 5, multi_scale=multi_scale, recurrent=recurrent, dropout=0.0)
            model.eval()
            logits_a, z_a = model(base, mask)
            logits_b, z_b = model(changed, mask)
            assert logits_a.shape == (4, 5)
            assert z_a.shape == (4, 128)
            assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-6)
            assert torch.allclose(z_a, z_b, atol=1e-6, rtol=1e-6)
            model.train()
            loss = model(base, mask)[0].square().mean()
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
