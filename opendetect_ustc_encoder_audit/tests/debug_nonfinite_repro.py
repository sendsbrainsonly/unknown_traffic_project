#!/usr/bin/env python3
"""Deterministic first-epoch replay for the formal non-finite loss failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(AUDIT_ROOT / "scripts"))

from adapters.opendetect_model import AuditedOpenDetectNet, released_weight_init  # noqa: E402
from train_opendetect import AlignedImageDataset, seed_everything  # noqa: E402


def finite_summary(name: str, tensor: torch.Tensor) -> dict[str, object]:
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    return {
        "name": name,
        "shape": list(tensor.shape),
        "all_finite": bool(finite.all()),
        "finite_ratio": float(finite.float().mean()),
        "finite_min": float(values.min()) if len(values) else None,
        "finite_max": float(values.max()) if len(values) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    dataset = AlignedImageDataset(
        AUDIT_ROOT / "artifacts/train_images.npy",
        PROJECT_ROOT / "outputs/stage1/modelA/embeddings/labels_train.npy",
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=True,
    )
    model = AuditedOpenDetectNet(
        UPSTREAM_CODE, "resnet18", 1, 128, 20, 1.0, 1.0
    ).to(device)
    model.apply(released_weight_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        details, _, losses = model.loss(images, labels)
        total = 0.005 * (losses["rec"] + losses["kld"] + losses["ent"]) + 0.995 * losses["dis"]
        tensors = {
            **details,
            **{f"loss_{name}": value for name, value in losses.items()},
            "loss_total": total,
        }
        bad_before = [name for name, value in tensors.items() if not torch.isfinite(value).all()]
        if bad_before:
            print(
                "[DEBUG-NAN-OD] "
                + json.dumps(
                    {
                        "verdict": "RED",
                        "batch_index": batch_index,
                        "batch_size": len(images),
                        "bad_before_backward": bad_before,
                        "summaries": [finite_summary(name, value) for name, value in tensors.items()],
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(2)
        total.backward()
        bad_grad = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if bad_grad:
            print(
                "[DEBUG-NAN-OD] "
                + json.dumps(
                    {
                        "verdict": "RED_GRAD",
                        "batch_index": batch_index,
                        "batch_size": len(images),
                        "bad_gradients": bad_grad,
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(3)
        optimizer.step()
        bad_param = [name for name, parameter in model.named_parameters() if not torch.isfinite(parameter).all()]
        if bad_param:
            print(
                "[DEBUG-NAN-OD] "
                + json.dumps(
                    {
                        "verdict": "RED_PARAM",
                        "batch_index": batch_index,
                        "batch_size": len(images),
                        "bad_parameters": bad_param,
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(4)
        if batch_index % 25 == 0:
            print(
                "[DEBUG-NAN-OD] "
                + json.dumps(
                    {
                        "verdict": "finite_so_far",
                        "batch_index": batch_index,
                        "total": float(total.detach()),
                        "logvar_min": float(details["logvar"].min()),
                        "logvar_max": float(details["logvar"].max()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(
        "[DEBUG-NAN-OD] "
        + json.dumps(
            {"verdict": "GREEN", "batches_checked": min(len(loader), args.max_batches)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

