#!/usr/bin/env python3
"""Analytic and numerical tests of the preregistered two-view entropy rule."""
from __future__ import annotations

import json

import numpy as np
import torch

from preflight import OUT
from run_one import Fusion


def main() -> None:
    torch.manual_seed(5)
    model = Fusion(3, True, np.array([-1.0, 2.0], np.float32),
                   np.array([0.2, 0.7], np.float32))
    model.eval()
    mc = torch.randn(4, 64)
    mb = torch.randn(4, 64)
    h = torch.tensor([[-3., 2.], [-1., 3.], [0., -4.], [12., 11.]])
    with torch.no_grad():
        y, w = model(mc, mb, h)
        if not torch.allclose(w, torch.full_like(w, 0.5), atol=1e-7):
            raise RuntimeError("lambda=0.5 must algebraically reduce to equal weights")
        h_changed = h.clone()
        h_changed[3] = torch.tensor([-20., 20.])
        y2, _ = model(mc, mb, h_changed)
        batch_effect = float((y[:3] - y2[:3]).abs().max())
        if batch_effect != 0:
            raise RuntimeError("other sample's entropy affected prediction")
        # Symmetric logits [a,-a] also cancel exactly because lambda_b+lambda_s=1.
        model.mix_logits.copy_(torch.tensor([1.0, 0.0]))
        _, w3 = model(mc, mb, h)
        if not torch.isfinite(w3).all() or not torch.allclose(w3.sum(1), torch.ones(4), atol=1e-7):
            raise RuntimeError("weights invalid under mixed positive/negative entropy")
        if float(w3[:, 0].std()) <= 1e-4:
            raise RuntimeError("nontrivial lambda should permit per-sample weights")
    result = {"status": "PASS", "negative_entropy_supported": True,
              "lambda_half_equals_equal": True, "batch_effect_max_abs": batch_effect,
              "nontrivial_lambda_weight_std": float(w3[:, 0].std()),
              "weight_sum_max_abs_error": float((w3.sum(1) - 1).abs().max())}
    path = OUT / "mechanism_test.json"
    if path.exists():
        raise RuntimeError("refusing overwrite mechanism test evidence")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
