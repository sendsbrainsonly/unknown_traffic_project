#!/usr/bin/env python3
"""Known Train/Validation-only frozen-input audit for Stage 28."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
S20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
S22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
S23 = PROJECT / "stage23_closed_set_method_table"
S27 = PROJECT / "stage27_yatc_feature_level_fusion"
DATASETS = ("iscx_vpn", "iscx_tor")
ENCODER_SEEDS = (2022, 2023)
ROLES = ("known_train", "known_validation")
EXPECTED_S20 = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources(dataset: str, seed: int) -> dict[str, Path]:
    return {
        "stage20_manifest": S20,
        "stage22_config": S22 / "config.json",
        "stage22_representations": S22 / "runs" / dataset / f"seed{seed}" / "representations.npz",
        "stage22_e3_checkpoint": S22 / "runs" / dataset / f"seed{seed}" / "E3_model_best.pt",
        "stage23_yatc_checkpoint": S23 / "runs" / "yatc_stage20" / dataset / f"seed{seed}_formal" / "model_best.pt",
        "stage27_yatc_features": S27 / "runs" / dataset / f"seed{seed}" / "features.npz",
        "stage27_source_verification": S27 / "runs" / dataset / f"seed{seed}" / "verification.json",
    }


def manifest_labels() -> dict[str, dict[str, tuple[str, str]]]:
    result = {dataset: {} for dataset in DATASETS}
    with S20.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset = row["dataset"]
            if dataset not in result:
                continue
            flow_id = row["flow_id"]
            if flow_id in result[dataset]:
                raise RuntimeError(f"duplicate Stage20 flow ID: {dataset}/{flow_id}")
            result[dataset][flow_id] = (row["service_label"], row["closed_role"])
    return result


def load_pair(dataset: str, seed: int, labels: dict[str, tuple[str, str]]) -> dict:
    """Load only the Known Train/Validation NPZ keys; never materialize Test values."""
    paths = sources(dataset, seed)
    config = json.loads(paths["stage22_config"].read_text())
    services = list(config["datasets"][dataset]["services"])
    stage27_check = json.loads(paths["stage27_source_verification"].read_text())
    if stage27_check["status"] != "PASS" or stage27_check["dataset"] != dataset or stage27_check["seed"] != seed:
        raise RuntimeError("Stage27 frozen pair verification failed")
    roles: dict[str, dict] = {}
    with np.load(paths["stage22_representations"], allow_pickle=False) as e3, np.load(
        paths["stage27_yatc_features"], allow_pickle=False
    ) as yatc:
        mean = yatc["y_mean"].astype(np.float32)
        std = yatc["y_std"].astype(np.float32)
        if mean.shape != (192,) or std.shape != (192,) or not (std > 0).all():
            raise RuntimeError("invalid frozen YaTC normalization")
        train_yatc = yatc["known_train_yatc"].astype(np.float32)
        check_mean = train_yatc.mean(0)
        check_std = train_yatc.std(0)
        check_std[check_std < 1e-8] = 1.0
        if not np.allclose(mean, check_mean, atol=2e-5) or not np.allclose(std, check_std, atol=2e-5):
            raise RuntimeError("YaTC scaler was not fitted to this Known Train")
        for role in ROLES:
            ids = yatc[f"{role}_flow_ids"].astype(str)
            eids = e3[f"{role}_flow_ids"].astype(str)
            if not np.array_equal(ids, eids) or len(set(ids)) != len(ids):
                raise RuntimeError(f"{dataset}/{seed}/{role}: flow alignment or uniqueness failed")
            representation = e3[f"{role}_e3_z"].astype(np.float32)
            yfeature = train_yatc if role == "known_train" else yatc[f"{role}_yatc"].astype(np.float32)
            if representation.shape != (len(ids), 896) or yfeature.shape != (len(ids), 192):
                raise RuntimeError(f"{dataset}/{seed}/{role}: feature shape mismatch")
            if not np.isfinite(representation).all() or not np.isfinite(yfeature).all():
                raise RuntimeError(f"{dataset}/{seed}/{role}: nonfinite features")
            truths = []
            for flow_id in ids:
                if flow_id not in labels or labels[flow_id][1] != role:
                    raise RuntimeError(f"{dataset}/{seed}/{role}: Stage20 role/flow mismatch {flow_id}")
                truths.append(labels[flow_id][0])
            if any(name not in services for name in truths):
                raise RuntimeError("Stage20 service not in frozen Stage22 class order")
            semantic = np.concatenate([representation[:, :768], (yfeature - mean) / std], axis=1)
            behavioral = representation[:, 768:896]
            roles[role] = {
                "flow_ids": ids, "labels": np.asarray([services.index(name) for name in truths], dtype=np.int64),
                "semantic": semantic.astype(np.float32), "behavioral": behavioral.astype(np.float32),
            }
    if set(roles["known_train"]["flow_ids"]) & set(roles["known_validation"]["flow_ids"]):
        raise RuntimeError("Known Train/Validation flow overlap")
    if set(roles["known_train"]["labels"]) != set(range(len(services))) or set(
        roles["known_validation"]["labels"]
    ) != set(range(len(services))):
        raise RuntimeError("missing class in Known Train or Validation")
    return {"services": services, "roles": roles}


def main() -> None:
    if (OUT / "input_audit.json").exists() or (OUT / "frozen_input_hashes_before.json").exists():
        raise RuntimeError("refusing to overwrite Stage28 preflight evidence")
    all_labels = manifest_labels()
    shared = {"stage20_manifest": sha256(S20)}
    if shared["stage20_manifest"] != EXPECTED_S20:
        raise RuntimeError("Stage20 frozen manifest hash mismatch")
    hashes = {}
    audit = []
    for dataset in DATASETS:
        for seed in ENCODER_SEEDS:
            paths = sources(dataset, seed)
            hashes[f"{dataset}/{seed}"] = {name: sha256(path) for name, path in paths.items()}
            pair = load_pair(dataset, seed, all_labels[dataset])
            for role in ROLES:
                item = pair["roles"][role]
                audit.append({
                    "dataset": dataset, "encoder_seed": seed, "role": role,
                    "samples": len(item["flow_ids"]), "classes": len(pair["services"]),
                    "semantic_shape": list(item["semantic"].shape),
                    "behavioral_shape": list(item["behavioral"].shape),
                    "class_counts": {name: int(np.sum(item["labels"] == i))
                                     for i, name in enumerate(pair["services"])},
                    "flow_id_first": str(item["flow_ids"][0]),
                    "flow_id_last": str(item["flow_ids"][-1]),
                    "finite": True,
                })
    output = {
        "status": "PASS", "stage20_sha256": shared["stage20_manifest"],
        "matched_pairs": len(hashes), "known_roles_checked": len(audit),
        "test_feature_values_loaded": 0, "unknown_usage": 0, "rows": audit,
    }
    (OUT / "frozen_input_hashes_before.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
