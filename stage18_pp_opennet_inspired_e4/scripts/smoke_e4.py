#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import torch

from e4_model import E4Encoder
from stage18_common import OUT, apply_robust_scaler, fit_robust_scaler, load_frozen_protocol, write_json


def main() -> None:
    frozen = load_frozen_protocol("Email")
    train_idx = frozen["indices"]["known_train"]
    scaler = fit_robust_scaler(frozen["raw_x"][train_idx], frozen["mask"][train_idx])
    x = apply_robust_scaler(frozen["raw_x"][train_idx[:64]], frozen["mask"][train_idx[:64]], scaler)
    mask = frozen["mask"][train_idx[:64]]
    labels = frozen["labels"]["known_train"][:64]
    device = torch.device("cuda:0")
    model = E4Encoder(3, 5, multi_scale=True, recurrent=True, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tx = torch.as_tensor(x, device=device)
    tm = torch.as_tensor(mask, device=device)
    ty = torch.as_tensor(labels, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, embedding = model(tx, tm)
    loss = torch.nn.functional.cross_entropy(logits, ty)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        logits1, z1 = model(tx, tm)
        logits2, z2 = model(tx, tm)
    result = {
        "status": "PASS",
        "device": str(device),
        "physical_gpu_visible": str(torch.cuda.get_device_name(0)),
        "input_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "embedding_shape": list(embedding.shape),
        "loss": float(loss.item()),
        "finite": bool(np.isfinite(logits.detach().cpu().numpy()).all() and np.isfinite(embedding.detach().cpu().numpy()).all()),
        "deterministic_eval": bool(torch.equal(logits1, logits2) and torch.equal(z1, z2)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "unknown_samples_used": 0,
    }
    write_json(OUT / "smoke_verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
