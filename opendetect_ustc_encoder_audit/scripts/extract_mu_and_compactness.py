#!/usr/bin/env python3
"""Export deterministic Open-Detect mu_x and enforce the latent quality gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
UPSTREAM_CODE = PROJECT_ROOT.parent / "Open-Detect/code"
CLASSES = ("FTP", "Cridex", "Miuref", "Outlook")
EXPECTED_ROWS = {"train": 391_280, "val": 48_910}
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.opendetect_model import AuditedOpenDetectNet  # noqa: E402
from train_opendetect import AlignedImageDataset  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def string_set_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values.tolist()):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def label_maps(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    forward = {str(key): int(value) for key, value in payload["class_to_id"].items()}
    return forward, {value: key for key, value in forward.items()}


@torch.no_grad()
def extract(model: AuditedOpenDetectNet, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    output = np.empty((len(loader.dataset), model.latent_dim), dtype=np.float32)
    offset = 0
    for images, _ in loader:
        mu, _, _ = model.encoder(images.to(device, non_blocking=True))
        batch = mu.detach().cpu().numpy().astype(np.float32, copy=False)
        output[offset : offset + len(batch)] = batch
        offset += len(batch)
    if offset != len(output):
        raise AssertionError("latent extraction row count mismatch")
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=AUDIT_ROOT / "artifacts/latent_gaussian_audit/best_checkpoint.pt",
    )
    parser.add_argument("--images-dir", type=Path, default=AUDIT_ROOT / "artifacts")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage1/modelA/embeddings",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json",
    )
    parser.add_argument(
        "--stage25-metadata",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage2_5_covariance_diagnosis/run_metadata.json",
    )
    parser.add_argument(
        "--alignment-summary",
        type=Path,
        default=AUDIT_ROOT / "outputs/input_alignment_summary.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=AUDIT_ROOT / "artifacts/latent_gaussian_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_ROOT / "outputs/latent_gaussian_audit",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    selection = json.loads((args.output_dir / "checkpoint_selection.json").read_text())
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != selection["checkpoint_sha256"]:
        raise ValueError("latent checkpoint SHA-256 differs from frozen selection")
    stage25 = json.loads(args.stage25_metadata.read_text(encoding="utf-8"))
    if Path(stage25["zt_dir"]).resolve() != args.embedding_dir.resolve():
        raise ValueError("Stage 2.5 TrafficFormer flow-ID source differs from extraction source")
    if stage25["loaded_splits"] != ["train", "val"] or stage25.get("test_loaded"):
        raise ValueError("Stage 2.5 split metadata is incompatible")
    if {item["class"] for item in stage25["classes"]} != set(CLASSES):
        raise ValueError("Stage 2.5 representative classes differ")
    alignment = json.loads(args.alignment_summary.read_text(encoding="utf-8"))
    if not alignment["all_inputs_valid"] or alignment["success"] != 489_101:
        raise ValueError("Open-Detect input alignment gate is not satisfied")

    if not torch.cuda.is_available():
        raise RuntimeError("mu extraction requires CUDA")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = AuditedOpenDetectNet(
        upstream_code=UPSTREAM_CODE,
        channels=int(config["channels"]),
        latent_dim=int(config["latent_dimension"]),
        num_classes=int(config["num_classes"]),
        temp_inter=float(config["temp_inter_code"]),
        temp_intra=float(config["temp_intra_code"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    class_to_id, id_to_class = label_maps(args.label_map)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_by_split: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, np.ndarray]] = {}
    manifest_rows: list[dict[str, object]] = []
    dimension_rows: list[dict[str, object]] = []
    class_quality_rows: list[dict[str, object]] = []
    class_flow_assertions: list[dict[str, object]] = []

    for split in ("train", "val"):
        flow_ids = np.load(args.embedding_dir / f"flow_ids_{split}.npy", allow_pickle=False)
        labels = np.load(args.embedding_dir / f"labels_{split}.npy", allow_pickle=False)
        if len(flow_ids) != EXPECTED_ROWS[split] or len(labels) != EXPECTED_ROWS[split]:
            raise ValueError(f"{split} source row count differs from frozen protocol")
        dataset = AlignedImageDataset(
            args.images_dir / f"{split}_images.npy",
            args.embedding_dir / f"labels_{split}.npy",
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        mu = extract(model, loader, device)
        if mu.shape != (EXPECTED_ROWS[split], model.latent_dim):
            raise ValueError(f"{split} latent shape mismatch: {mu.shape}")
        if len(np.unique(flow_ids)) != len(flow_ids):
            raise ValueError(f"duplicate {split} flow_id")
        names = np.asarray([id_to_class[int(label)] for label in labels])
        frame = pd.DataFrame(
            {"flow_id": flow_ids, "class_name": names, "split": np.full(len(mu), split)}
        )
        latent_columns = pd.DataFrame(
            mu, columns=[f"mu_{index}" for index in range(model.latent_dim)]
        )
        frame = pd.concat([frame, latent_columns], axis=1)
        parquet_path = args.artifact_dir / f"{split}_mu.parquet"
        frame.to_parquet(parquet_path, index=False, engine="pyarrow", compression="zstd")
        if len(pd.read_parquet(parquet_path, columns=["flow_id"])) != EXPECTED_ROWS[split]:
            raise ValueError(f"{split} parquet row-count verification failed")
        variances = mu.astype(np.float64).var(axis=0)
        for dimension, variance in enumerate(variances):
            dimension_rows.append(
                {"split": split, "dimension": dimension, "variance": float(variance)}
            )
        manifest_rows.append(
            {
                "split": split,
                "rows": len(mu),
                "latent_dim": model.latent_dim,
                "flow_id_unique": True,
                "flow_id_set_sha256": string_set_sha256(flow_ids),
                "parquet_path": str(parquet_path.resolve()),
                "parquet_sha256": sha256_file(parquet_path),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
        for class_name in CLASSES:
            class_id = class_to_id[class_name]
            mask = labels == class_id
            expected_ids = flow_ids[mask]
            extracted_ids = frame.loc[frame["class_name"] == class_name, "flow_id"].to_numpy()
            equal = np.array_equal(np.sort(expected_ids), np.sort(extracted_ids))
            class_flow_assertions.append(
                {
                    "class_name": class_name,
                    "split": split,
                    "trafficformer_flow_count": len(expected_ids),
                    "opendetect_flow_count": len(extracted_ids),
                    "trafficformer_flow_set_sha256": string_set_sha256(expected_ids),
                    "opendetect_flow_set_sha256": string_set_sha256(extracted_ids),
                    "flow_id_sets_equal": equal,
                }
            )
            if not equal:
                raise ValueError(f"{class_name} {split} flow-ID equality assertion failed")
            values = mu[mask].astype(np.float64)
            center = values.mean(axis=0)
            within = np.linalg.norm(values - center, axis=1)
            norms = np.linalg.norm(values, axis=1)
            class_quality_rows.append(
                {
                    "class_name": class_name,
                    "split": split,
                    "sample_count": len(values),
                    "mu_norm_mean": float(norms.mean()),
                    "mu_norm_median": float(np.median(norms)),
                    "within_class_distance_mean": float(within.mean()),
                    "within_class_distance_median": float(np.median(within)),
                    "within_class_distance_p95": float(np.quantile(within, 0.95)),
                    "within_class_variance": float(values.var(axis=0).sum()),
                }
            )
        latent_by_split[split] = mu
        metadata[split] = {"flow_id": flow_ids, "label": labels, "class_name": names}
        print(f"{split}: exported {mu.shape} to {parquet_path}", flush=True)

    overlap = np.intersect1d(metadata["train"]["flow_id"], metadata["val"]["flow_id"])
    if len(overlap):
        raise ValueError(f"train/validation flow-ID overlap: {len(overlap)}")
    if not all(np.isfinite(values).all() for values in latent_by_split.values()):
        raise ValueError("NaN or Inf found in extracted mu_x")
    train_variances = latent_by_split["train"].astype(np.float64).var(axis=0)
    collapsed = int((train_variances <= 1e-12).sum())
    near_collapsed = int((train_variances <= 1e-8).sum())
    if collapsed > 0 or near_collapsed > math.floor(0.10 * model.latent_dim):
        raise ValueError(
            f"latent collapse gate failed: collapsed={collapsed}, near_collapsed={near_collapsed}"
        )

    prototypes = model.prototypes.detach().cpu().numpy().astype(np.float64)
    train_mu = latent_by_split["train"].astype(np.float64)
    train_labels = metadata["train"]["label"]
    compact_rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        class_id = class_to_id[class_name]
        values = train_mu[train_labels == class_id]
        assigned = np.linalg.norm(values - prototypes[class_id], axis=1)
        wrong_indices = [index for index in range(len(class_to_id)) if index != class_id]
        wrong_prototypes = prototypes[wrong_indices]
        wrong_squared = (
            np.sum(values * values, axis=1, keepdims=True)
            + np.sum(wrong_prototypes * wrong_prototypes, axis=1)[None, :]
            - 2.0 * values @ wrong_prototypes.T
        )
        wrong = np.sqrt(np.maximum(wrong_squared, 0.0)).min(axis=1)
        compact_rows.append(
            {
                "class_name": class_name,
                "sample_count": len(values),
                "assigned_prototype_distance_mean": float(assigned.mean()),
                "assigned_prototype_distance_median": float(np.median(assigned)),
                "assigned_prototype_distance_p95": float(np.quantile(assigned, 0.95)),
                "within_class_variance": float(values.var(axis=0).sum()),
                "nearest_wrong_prototype_distance_mean": float(wrong.mean()),
                "nearest_wrong_prototype_distance_median": float(np.median(wrong)),
                "nearest_wrong_prototype_distance_p05": float(np.quantile(wrong, 0.05)),
                "mean_wrong_minus_assigned_margin": float((wrong - assigned).mean()),
            }
        )

    write_csv(args.output_dir / "latent_extraction_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "latent_dimension_statistics.csv", dimension_rows)
    write_csv(args.output_dir / "latent_class_quality.csv", class_quality_rows)
    write_csv(args.output_dir / "flow_id_equality_assertions.csv", class_flow_assertions)
    write_csv(args.output_dir / "class_compactness.csv", compact_rows)

    lines = [
        "# Latent Quality Report",
        "",
        "## Gate",
        "",
        "- Status: PASS",
        f"- Frozen checkpoint SHA-256 verified: `{checkpoint_hash}`",
        f"- Latent dimension: {model.latent_dim}",
        f"- Train rows: {len(latent_by_split['train'])} (expected {EXPECTED_ROWS['train']})",
        f"- Validation rows: {len(latent_by_split['val'])} (expected {EXPECTED_ROWS['val']})",
        "- Train flow IDs unique: yes",
        "- Validation flow IDs unique: yes",
        "- Train/validation overlap: 0",
        "- NaN: 0",
        "- Inf: 0",
        f"- Collapsed train dimensions (variance <= 1e-12): {collapsed}/{model.latent_dim}",
        f"- Near-collapsed train dimensions (variance <= 1e-8): {near_collapsed}/{model.latent_dim}",
        f"- Train per-dimension variance min/median/max: {train_variances.min():.9g} / {np.median(train_variances):.9g} / {train_variances.max():.9g}",
        "- Four-class Open-Detect/TrafficFormer train and validation flow-ID sets: 8/8 exact",
        "",
        "## Representative-class latent quality",
        "",
        "| class | split | n | mean mu norm | mean within-class distance | p95 within-class distance | total within-class variance |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in class_quality_rows:
        lines.append(
            f"| {row['class_name']} | {row['split']} | {row['sample_count']} | {row['mu_norm_mean']:.6f} | {row['within_class_distance_mean']:.6f} | {row['within_class_distance_p95']:.6f} | {row['within_class_variance']:.6f} |"
        )
    lines += [
        "",
        "## Open-Detect prototype compactness (train)",
        "",
        "| class | n | assigned mean | assigned median | assigned p95 | within-class variance | nearest-wrong mean | mean margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compact_rows:
        lines.append(
            f"| {row['class_name']} | {row['sample_count']} | {row['assigned_prototype_distance_mean']:.6f} | {row['assigned_prototype_distance_median']:.6f} | {row['assigned_prototype_distance_p95']:.6f} | {row['within_class_variance']:.6f} | {row['nearest_wrong_prototype_distance_mean']:.6f} | {row['mean_wrong_minus_assigned_margin']:.6f} |"
        )
    lines += [
        "",
        "Distances are descriptive checks of the learned compactness constraint. They do not establish that a single Gaussian is adequate and are not semantic-mode evidence.",
    ]
    (args.output_dir / "latent_quality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
