#!/usr/bin/env python3
"""Deterministically extract frozen Stage 7 mu_x and transform with frozen scaler/PCA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset

from stage9_common import (
    PROJECT_ROOT,
    STAGE8A,
    STAGE9_ROOT,
    VALID_SETTINGS,
    assert_test_open_record,
    load_fold,
    load_frozen_models,
    load_json,
    require,
    sha256_file,
    stage7_run,
    verify_all_provenance,
    write_json,
)


AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
STAGE3_SCRIPTS = PROJECT_ROOT / "stage3_unknown_utility/scripts"
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(STAGE3_SCRIPTS))
from stage3_model import Stage3OpenDetectNet  # noqa: E402


class FrozenImageDataset(Dataset):
    def __init__(self, images: np.ndarray) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.images[index]).copy()).unsqueeze(0).float().div_(255.0)


@torch.inference_mode()
def extract(
    model: Stage3OpenDetectNet,
    images: np.ndarray,
    mu_path: Path,
    score_path: Path,
    pred_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> None:
    loader = DataLoader(
        FrozenImageDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    mu = open_memmap(mu_path, mode="w+", dtype=np.float32, shape=(len(images), 128))
    scores = open_memmap(score_path, mode="w+", dtype=np.float32, shape=(len(images),))
    predictions = open_memmap(pred_path, mode="w+", dtype=np.int64, shape=(len(images),))
    model.eval()
    offset = 0
    for batch_index, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        batch_mu, logvar, _ = model.stable_encode(batch)
        require(batch_mu.shape[1] == 128, "checkpoint latent dimension changed")
        kld = model.kl_div_to_prototypes(batch_mu, logvar)
        minimum, prediction = torch.min(kld, dim=1)
        count = len(batch)
        mu[offset : offset + count] = batch_mu.cpu().numpy().astype(np.float32, copy=False)
        scores[offset : offset + count] = -minimum.cpu().numpy().astype(np.float32, copy=False)
        predictions[offset : offset + count] = prediction.cpu().numpy().astype(np.int64, copy=False)
        offset += count
        if (batch_index + 1) % 25 == 0:
            print(json.dumps({"event": "test_mu_progress", "rows": offset, "total": len(images)}), flush=True)
    require(offset == len(images), "mu extraction row count mismatch")
    mu.flush()
    scores.flush()
    predictions.flush()


def array_audit(values: np.ndarray, expected_rows: int, expected_dim: int) -> dict[str, object]:
    require(values.shape == (expected_rows, expected_dim), f"shape mismatch: {values.shape}")
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    all_zero_rows = int(np.sum(np.all(values == 0, axis=1)))
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_zero_embedding_rows": all_zero_rows,
        "status": "PASS" if nan_count == 0 and inf_count == 0 and all_zero_rows == 0 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=VALID_SETTINGS, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.workers >= 0, "invalid loader configuration")
    verify_all_provenance()
    assert_test_open_record()
    require(torch.cuda.is_available(), "Stage 9 deterministic Test inference requires CUDA")
    setting = args.setting
    artifact = STAGE9_ROOT / "artifacts" / setting
    integrity = load_json(artifact / "test_input_integrity.json")
    require(integrity["status"] == "PASS", f"{setting}: input integrity gate failed")
    outputs = [
        artifact / "mu_known_test.npy",
        artifact / "mu_unknown_test.npy",
        artifact / "z_known_test_pca64.npy",
        artifact / "z_unknown_test_pca64.npy",
    ]
    require(not any(path.exists() for path in outputs), f"{setting}: refusing to overwrite Test representations")
    run_dir = stage7_run(setting)
    checkpoint_path = run_dir / "artifacts/training_best_checkpoint.pt"
    expected = STAGE8A.EXPECTED_STAGE7[setting]
    require(sha256_file(checkpoint_path) == expected["checkpoint_sha256"], f"{setting}: checkpoint hash changed")
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    fold = load_fold(setting)
    class_names = [str(value) for value in fold["known_classes"]]
    require(config["setting"] == setting, "checkpoint setting mismatch")
    require(int(config["latent_dimension"]) == 128, "checkpoint latent dimension mismatch")
    require(int(config["num_classes"]) == len(class_names), "checkpoint class count mismatch")
    model = Stage3OpenDetectNet(
        upstream_code=UPSTREAM_CODE,
        channels=1,
        latent_dim=128,
        num_classes=len(class_names),
        temp_inter=1.0,
        temp_intra=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    role_audits: dict[str, object] = {}
    for role in ("known_test", "unknown_test"):
        images = np.load(artifact / f"{role}_images.npy", mmap_mode="r", allow_pickle=False)
        extract(
            model,
            images,
            artifact / f"mu_{role}.npy",
            artifact / f"native_{role}_scores.npy",
            artifact / f"native_{role}_predictions.npy",
            device,
            args.batch_size,
            args.workers,
        )
        mu = np.load(artifact / f"mu_{role}.npy", mmap_mode="r", allow_pickle=False)
        role_audits[role] = array_audit(mu, len(images), 128)
        native_scores = np.load(artifact / f"native_{role}_scores.npy", mmap_mode="r", allow_pickle=False)
        native_pred = np.load(artifact / f"native_{role}_predictions.npy", mmap_mode="r", allow_pickle=False)
        require(native_scores.shape == (len(images),) and np.isfinite(native_scores).all(), f"{setting}/{role}: invalid Native scores")
        require(native_pred.shape == (len(images),) and np.all((native_pred >= 0) & (native_pred < len(class_names))), f"{setting}/{role}: invalid Native predictions")

    frozen = load_frozen_models(setting)
    for role in ("known_test", "unknown_test"):
        mu = np.load(artifact / f"mu_{role}.npy", mmap_mode="r", allow_pickle=False)
        # Frozen objects expose transform only here. No fit or fit_transform is called in Stage 9.
        z = frozen["pca"].transform(frozen["scaler"].transform(np.asarray(mu, dtype=np.float64)))
        require(z.shape == (len(mu), 64), f"{setting}/{role}: PCA64 shape mismatch")
        require(np.isfinite(z).all(), f"{setting}/{role}: non-finite PCA64 representation")
        np.save(artifact / f"z_{role}_pca64.npy", z.astype(np.float32), allow_pickle=False)
        role_audits[role]["pca64_shape"] = list(z.shape)
        role_audits[role]["pca64_nan_count"] = int(np.isnan(z).sum())
        role_audits[role]["pca64_inf_count"] = int(np.isinf(z).sum())
    require(all(value["status"] == "PASS" for value in role_audits.values()), f"{setting}: representation integrity failed")
    audit = {
        "setting": setting,
        "status": "PASS",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "representation": "deterministic mu_x; sampled z forbidden",
        "latent_dimension": 128,
        "scaler_operation": "transform_only",
        "pca_operation": "transform_only",
        "pca_dimension": 64,
        "roles": role_audits,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "stability_guard_activations": model.stability_guard_activations,
        "input_failures": integrity["input_failures"],
    }
    write_json(artifact / "test_representation_integrity.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
