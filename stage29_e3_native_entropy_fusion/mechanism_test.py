#!/usr/bin/env python3
"""Check shared-lambda two-view rule is trainable and batch invariant."""
from __future__ import annotations

import json

import numpy as np
import torch

from preflight import OUT
from run_one import Fusion


def main():
    torch.manual_seed(7)
    model = Fusion(3, True, np.array([0., 0.], np.float32), np.array([1., 1.], np.float32))
    model.eval()
    t, g = torch.randn(3, 64), torch.randn(3, 64)
    h = torch.tensor([[0., 5.], [4., 0.], [-3., 1.]])
    logits, weights = model(t, g, h)
    if not torch.allclose(weights, torch.full_like(weights, .5), atol=1e-7):
        raise RuntimeError("initial shared lambda does not yield equal weights")
    weights[0, 0].backward()
    gradient = float(model.shared_lambda_logit.grad)
    if abs(gradient) <= 1e-6:
        raise RuntimeError("shared lambda has zero gradient at 0.5 initialization")
    with torch.no_grad():
        model.shared_lambda_logit.copy_(torch.tensor(1.0))
        logits2, weights2 = model(t, g, h)
        if float(weights2[:, 0].std()) < .01:
            raise RuntimeError("nontrivial lambda did not create sample-dependent weights")
        h_changed = h.clone()
        h_changed[2] = torch.tensor([100., -100.])
        logits_changed, _ = model(t, g, h_changed)
        if not torch.equal(logits2[:2], logits_changed[:2]):
            raise RuntimeError("other sample affected score")
    path = OUT / "mechanism_test.json"
    if path.exists():
        raise RuntimeError("mechanism test evidence exists")
    result = {"status": "PASS", "initial_weights_equal": True,
              "shared_lambda_initial_gradient": gradient,
              "sample_dependent_weight_std": float(weights2[:, 0].std()),
              "batch_cross_sample_effect": 0.0}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
