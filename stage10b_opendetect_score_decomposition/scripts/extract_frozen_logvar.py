#!/usr/bin/env python3
"""Diagnostic-only frozen inference for Stage 9 Test mu/logvar.

No optimizer, loss, backward pass, fit operation, or threshold is available in
this script. Stage 9 mu remains canonical and is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from diagnosis_common import CONFIG_PATH, PROJECT_ROOT, ROOT, SETTINGS, SCOPE, load_json, require, write_json


STAGE9_SCRIPTS = PROJECT_ROOT / "stage9_cipherspectrum_final_test/scripts"
STAGE3_SCRIPTS = PROJECT_ROOT / "stage3_unknown_utility/scripts"
AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
sys.path.insert(0, str(STAGE9_SCRIPTS))
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(STAGE3_SCRIPTS))
from extract_test_representations import FrozenImageDataset  # noqa: E402
from stage3_model import Stage3OpenDetectNet  # noqa: E402
from stage9_common import (  # noqa: E402
    STAGE9_ROOT,
    assert_test_open_record,
    load_fold,
    sha256_file,
    stage7_run,
    verify_all_provenance,
)


@torch.inference_mode()
def extract(model: Stage3OpenDetectNet, images: np.ndarray, mu_path: Path, logvar_path: Path,
            device: torch.device, batch_size: int, workers: int) -> None:
    loader = DataLoader(
        FrozenImageDataset(images), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    mu = open_memmap(mu_path, mode="w+", dtype=np.float32, shape=(len(images), 128))
    logvar = open_memmap(logvar_path, mode="w+", dtype=np.float32, shape=(len(images), 128))
    model.eval()
    offset = 0
    for batch_index, batch in enumerate(loader):
        batch_mu, batch_logvar, _ = model.stable_encode(batch.to(device, non_blocking=True))
        count = len(batch)
        require(batch_mu.shape == batch_logvar.shape == (count, 128), "latent output shape changed")
        mu[offset:offset + count] = batch_mu.cpu().numpy().astype(np.float32, copy=False)
        logvar[offset:offset + count] = batch_logvar.cpu().numpy().astype(np.float32, copy=False)
        offset += count
        if (batch_index + 1) % 25 == 0:
            print(json.dumps({"event": "frozen_inference_progress", "rows": offset, "total": len(images)}), flush=True)
    require(offset == len(images), "frozen inference row count mismatch")
    mu.flush()
    logvar.flush()


def alignment_stats(recomputed: np.ndarray, frozen: np.ndarray) -> dict[str, object]:
    require(recomputed.shape == frozen.shape, "recomputed/frozen mu shape mismatch")
    maximum = 0.0
    total = 0.0
    count = 0
    for start in range(0, len(frozen), 2048):
        diff = np.abs(np.asarray(recomputed[start:start + 2048], dtype=np.float64) -
                      np.asarray(frozen[start:start + 2048], dtype=np.float64))
        maximum = max(maximum, float(np.max(diff)))
        total += float(np.sum(diff))
        count += diff.size
    return {"shape": list(frozen.shape), "max_abs_error": maximum, "mean_abs_error": total / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.workers >= 0, "invalid loader configuration")
    require(torch.cuda.is_available(), "frozen diagnostic inference requires CUDA")
    verify_all_provenance()
    assert_test_open_record()
    config = load_json(CONFIG_PATH)
    tolerance = config["numerical_gates"]
    setting = args.setting
    fold = load_fold(setting)
    class_names = [str(value) for value in fold["known_classes"]]
    source = STAGE9_ROOT / "artifacts" / setting
    target = ROOT / "artifacts" / setting
    output = ROOT / "outputs" / setting
    target.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "frozen_inference_audit.json"
    require(not audit_path.exists(), f"refusing to overwrite completed inference audit: {audit_path}")

    run_dir = stage7_run(setting)
    checkpoint_path = run_dir / "artifacts/training_best_checkpoint.pt"
    source_integrity = load_json(source / "test_representation_integrity.json")
    require(source_integrity["status"] == "PASS", "Stage 9 representation integrity is not PASS")
    require(sha256_file(checkpoint_path) == source_integrity["checkpoint_sha256"], "checkpoint hash changed")
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    require(checkpoint_config["setting"] == setting, "checkpoint setting mismatch")
    require(int(checkpoint_config["latent_dimension"]) == 128, "checkpoint latent dimension mismatch")
    require(int(checkpoint_config["num_classes"]) == len(class_names), "checkpoint class count mismatch")
    model = Stage3OpenDetectNet(
        upstream_code=UPSTREAM_CODE, channels=1, latent_dim=128,
        num_classes=len(class_names), temp_inter=1.0, temp_intra=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    roles: dict[str, object] = {}
    for role in ("known_test", "unknown_test"):
        images = np.load(source / f"{role}_images.npy", mmap_mode="r", allow_pickle=False)
        source_ids = np.load(source / f"{role}_sample_ids.npy", mmap_mode="r", allow_pickle=False).astype(str)
        manifest_rows = []
        import csv
        with (source / f"{role}_manifest.csv").open(encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        manifest_ids = np.asarray([row["sample_id"] for row in manifest_rows], dtype=str)
        require(len(images) == len(source_ids) == len(manifest_ids), f"{role}: sample count mismatch")
        require(np.array_equal(source_ids, manifest_ids), f"{role}: sample ID/order mismatch")
        mu_path = target / f"mu_{role}_recomputed.npy"
        logvar_path = target / f"logvar_{role}.npy"
        require(not mu_path.exists() and not logvar_path.exists(), f"{role}: refusing to overwrite inference arrays")
        extract(model, images, mu_path, logvar_path, device, args.batch_size, args.workers)
        recomputed = np.load(mu_path, mmap_mode="r", allow_pickle=False)
        frozen = np.load(source / f"mu_{role}.npy", mmap_mode="r", allow_pickle=False)
        logvar = np.load(logvar_path, mmap_mode="r", allow_pickle=False)
        require(np.isfinite(recomputed).all() and np.isfinite(logvar).all(), f"{role}: non-finite inference output")
        stats = alignment_stats(recomputed, frozen)
        aligned = (
            float(stats["max_abs_error"]) <= float(tolerance["mu_alignment_atol"])
            and float(stats["mean_abs_error"]) <= float(tolerance["mu_alignment_mean_abs_max"])
        )
        stats.update({
            "sample_count": len(images), "sample_id_order": "PASS",
            "finite_mu": True, "finite_logvar": True,
            "logvar_shape": list(logvar.shape), "alignment_status": "PASS" if aligned else "FAIL",
        })
        roles[role] = stats
        require(aligned, f"{setting}/{role}: recomputed mu differs materially from frozen Stage 9 mu")

    audit = {
        "analysis_scope": SCOPE,
        "setting": setting,
        "status": "PASS",
        "operation": "DIAGNOSTIC-ONLY FROZEN INFERENCE; eval mode; mu/logvar only",
        "training_performed": False,
        "threshold_tuning_performed": False,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "preprocessing": "exact Stage9 FrozenImageDataset: uint8 [32,32] -> [1,32,32] float32 / 255",
        "sample_order_source": "Stage9 frozen manifests and sample_id arrays",
        "stability_guard_activations": model.stability_guard_activations,
        "mu_alignment_tolerances": {
            "max_abs_error": tolerance["mu_alignment_atol"],
            "mean_abs_error": tolerance["mu_alignment_mean_abs_max"],
        },
        "roles": roles,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
