#!/usr/bin/env python3
"""Deterministic A-2 first-epoch NaN reproducer with targeted diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


STAGE3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE3_ROOT / "scripts"))
from common import AUDIT_ROOT, KnownImageDataset, load_fold, seed_everything  # noqa: E402

sys.path.insert(0, str(AUDIT_ROOT))
from adapters.opendetect_model import AuditedOpenDetectNet, released_weight_init  # noqa: E402
from stage3_model import Stage3OpenDetectNet  # noqa: E402


UPSTREAM_CODE = AUDIT_ROOT.parent.parent / "Open-Detect/code"


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    output: dict[str, object] = {
        "shape": list(detached.shape),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
    }
    if finite.any():
        output["finite_min"] = float(detached[finite].min())
        output["finite_max"] = float(detached[finite].max())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("released", "guarded"), default="released")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("debug reproducer requires CUDA")
    seed_everything(2022)
    fold = load_fold("A-2")
    dataset = KnownImageDataset("train", fold, augment=True)
    generator = torch.Generator().manual_seed(2022)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    model_class = AuditedOpenDetectNet if args.model == "released" else Stage3OpenDetectNet
    model = model_class(
        upstream_code=UPSTREAM_CODE,
        channels=1,
        latent_dim=128,
        num_classes=17,
        temp_inter=1.0,
        temp_intra=1.0,
    ).to(device)
    model.apply(released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    result: dict[str, object] = {
        "status": "STABLE" if args.model == "guarded" else "NO_FAILURE",
        "model": args.model,
        "batches_checked": 0,
    }
    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        details, _, losses = model.loss(images, labels)
        total = 0.005 * (losses["rec"] + losses["kld"] + losses["ent"]) + 0.995 * losses["dis"]
        component_stats = {name: tensor_stats(value) for name, value in losses.items()}
        if not torch.isfinite(total):
            result = {
                "status": "REPRODUCED_NONFINITE_TOTAL",
                "batch_index_zero_based": batch_index,
                "total": tensor_stats(total),
                "losses": component_stats,
                "mu": tensor_stats(details["mu"]),
                "logvar": tensor_stats(details["logvar"]),
                "exp_logvar": tensor_stats(details["logvar"].exp()),
                "dist": tensor_stats(details["dist"]),
                "kl_div": tensor_stats(details["kl_div"]),
                "prototypes": tensor_stats(model.prototypes),
                "labels": tensor_stats(labels),
            }
            break
        total.backward()
        bad_gradients = {
            name: tensor_stats(parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        }
        if bad_gradients:
            result = {
                "status": "REPRODUCED_NONFINITE_GRADIENT",
                "batch_index_zero_based": batch_index,
                "losses": component_stats,
                "mu": tensor_stats(details["mu"]),
                "logvar": tensor_stats(details["logvar"]),
                "bad_gradients": bad_gradients,
            }
            break
        optimizer.step()
        bad_parameters = {
            name: tensor_stats(parameter)
            for name, parameter in model.named_parameters()
            if not torch.isfinite(parameter).all()
        }
        if bad_parameters:
            result = {
                "status": "REPRODUCED_NONFINITE_PARAMETER",
                "batch_index_zero_based": batch_index,
                "losses": component_stats,
                "bad_parameters": bad_parameters,
            }
            break
        result["batches_checked"] = batch_index + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "NO_FAILURE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
