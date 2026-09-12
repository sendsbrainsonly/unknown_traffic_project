#!/usr/bin/env python3
"""Export deterministic setting-specific Open-Detect mu_x without test tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    AUDIT_ROOT,
    IMAGE_ROOT,
    STAGE3_ROOT,
    load_fold,
    load_label_maps,
    setting_indices,
    sha256_file,
    split_arrays,
    string_set_sha256,
    verify_frozen_inputs,
)


sys.path.insert(0, str(AUDIT_ROOT))
from stage3_model import Stage3OpenDetectNet  # noqa: E402


UPSTREAM_CODE = AUDIT_ROOT.parent.parent / "Open-Detect/code"


class SelectedImageDataset(Dataset):
    def __init__(self, split: str, indices: np.ndarray) -> None:
        self.images = np.load(IMAGE_ROOT / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.asarray(self.images[int(self.indices[index])]).copy()
        return torch.from_numpy(image).unsqueeze(0).float().div_(255.0)


def verify_frozen_model_hashes(path: Path) -> None:
    if not path.exists():
        raise RuntimeError("final-test latent extraction requires frozen_model_hashes.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("created_before_final_test"):
        raise RuntimeError("model freeze does not predate final test")
    for item in payload["files"]:
        artifact = Path(item["path"])
        if sha256_file(artifact) != item["sha256"]:
            raise RuntimeError(f"frozen artifact hash mismatch: {artifact}")


@torch.no_grad()
def extract_one(
    model: Stage3OpenDetectNet,
    device: torch.device,
    split: str,
    role: str,
    fold: dict[str, object],
    output_path: Path,
    batch_size: int,
    workers: int,
) -> dict[str, object]:
    flow_ids, labels = split_arrays(split)
    indices = setting_indices(labels, fold, role)
    dataset = SelectedImageDataset(split, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    mu = np.empty((len(indices), model.latent_dim), dtype=np.float32)
    native_unknown_score = np.empty(len(indices), dtype=np.float32)
    native_pred = np.empty(len(indices), dtype=np.int64)
    offset = 0
    model.eval()
    for images in loader:
        images = images.to(device, non_blocking=True)
        batch_mu, logvar, _ = model.stable_encode(images)
        kld = model.kl_div_to_prototypes(batch_mu, logvar)
        scores, predictions = torch.min(kld, dim=1)
        count = len(images)
        mu[offset : offset + count] = batch_mu.cpu().numpy().astype(np.float32, copy=False)
        native_unknown_score[offset : offset + count] = scores.cpu().numpy().astype(np.float32, copy=False)
        native_pred[offset : offset + count] = predictions.cpu().numpy()
        offset += count
    if offset != len(indices):
        raise RuntimeError("latent extraction row count mismatch")
    if not np.isfinite(mu).all() or not np.isfinite(native_unknown_score).all():
        raise RuntimeError("non-finite latent or native score")

    _, id_to_class = load_label_maps()
    selected_labels = np.asarray(labels[indices], dtype=np.int64)
    selected_ids = np.asarray(flow_ids[indices])
    class_names = np.asarray([id_to_class[int(value)] for value in selected_labels], dtype=object)
    local_names = np.asarray([str(value) for value in fold["known_classes"]], dtype=object)
    predicted_names = local_names[native_pred]
    frame = pd.DataFrame(
        {
            "flow_id": selected_ids,
            "class_name": class_names,
            "known_or_unknown": "Known" if role == "known" else "Unknown",
            "original_split": split,
            "original_label": selected_labels,
            "native_unknown_score": native_unknown_score,
            "native_predicted_known_class": predicted_names,
        }
    )
    latent_frame = pd.DataFrame(mu, columns=[f"mu_{index:03d}" for index in range(model.latent_dim)])
    frame = pd.concat([frame, latent_frame], axis=1)
    if frame["flow_id"].duplicated().any():
        raise RuntimeError(f"duplicate flow_id in {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    verified = pd.read_parquet(output_path, columns=["flow_id"])
    if len(verified) != len(frame) or verified["flow_id"].duplicated().any():
        raise RuntimeError("written parquet verification failed")
    return {
        "file": output_path.name,
        "path": str(output_path.resolve()),
        "rows": len(frame),
        "latent_dim": model.latent_dim,
        "flow_id_unique": True,
        "flow_id_set_sha256": string_set_sha256(selected_ids),
        "sha256": sha256_file(output_path),
        "nan_count": 0,
        "inf_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--phase", choices=("fit", "final"), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    verify_frozen_inputs()
    fold = load_fold(args.setting)
    output_dir = STAGE3_ROOT / "outputs" / args.setting
    artifact_dir = STAGE3_ROOT / "artifacts" / args.setting
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    selection = json.loads((output_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
    if sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("best checkpoint does not match frozen checkpoint selection")
    if args.phase == "final":
        verify_frozen_model_hashes(output_dir / "frozen_model_hashes.json")
    if not torch.cuda.is_available():
        raise RuntimeError("latent extraction requires CUDA")
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config["setting"] != args.setting or config["unknown_samples_loaded"] != 0:
        raise RuntimeError("checkpoint setting/leakage metadata mismatch")
    model = Stage3OpenDetectNet(
        upstream_code=UPSTREAM_CODE,
        channels=1,
        latent_dim=int(config["latent_dimension"]),
        num_classes=int(config["num_classes"]),
        temp_inter=1.0,
        temp_intra=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    phase_specs = (
        (("train", "known", "train_known_mu.parquet"), ("val", "known", "val_known_mu.parquet"))
        if args.phase == "fit"
        else (("test", "known", "test_known_mu.parquet"), ("test", "unknown", "test_unknown_mu.parquet"))
    )
    manifest_path = output_dir / "latent_export_manifest.json"
    manifest = {"setting": args.setting, "checkpoint_sha256": selection["checkpoint_sha256"], "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split, role, filename in phase_specs:
        path = artifact_dir / filename
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing latent export: {path}")
        result = extract_one(model, device, split, role, fold, path, args.batch_size, args.workers)
        manifest["files"][filename] = result
        print(json.dumps(result, sort_keys=True), flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.phase == "final":
        files = [
            artifact_dir / "train_known_mu.parquet",
            artifact_dir / "val_known_mu.parquet",
            artifact_dir / "test_known_mu.parquet",
            artifact_dir / "test_unknown_mu.parquet",
        ]
        id_sets = []
        for path in files:
            values = pd.read_parquet(path, columns=["flow_id"])["flow_id"]
            if values.duplicated().any():
                raise RuntimeError(f"duplicate flow IDs in {path.name}")
            id_sets.append(set(values.astype(str)))
        for left in range(len(id_sets)):
            for right in range(left + 1, len(id_sets)):
                if id_sets[left] & id_sets[right]:
                    raise RuntimeError(f"flow-ID overlap: {files[left].name} vs {files[right].name}")
        checks = {
            "setting": args.setting,
            "status": "PASS",
            "latent_dim": 128,
            "flow_ids_unique": True,
            "all_pairwise_split_overlaps": 0,
            "nan_count": 0,
            "inf_count": 0,
        }
        (output_dir / "latent_quality_gate.json").write_text(
            json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
