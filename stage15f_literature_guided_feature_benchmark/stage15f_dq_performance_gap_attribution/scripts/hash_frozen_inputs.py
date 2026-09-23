#!/usr/bin/env python3
"""Hash the frozen inputs protected during Stage 15F-DQ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]


EXTRA_INPUTS = (
    PROJECT / "stage12_dual_external_validation/protocol/iscx_vpn/split_manifest.csv",
    PROJECT / "stage12_dual_external_validation/protocol/iscx_vpn/preprocessing_manifest.json",
    PROJECT / "stage12_dual_external_validation/protocol/iscx_vpn/protocol_freeze.json",
    PROJECT / "stage12_dual_external_validation/artifacts/iscx_vpn/protocol/medium/known_train.npz",
    PROJECT / "stage12_dual_external_validation/artifacts/iscx_vpn/protocol/medium/known_validation.npz",
    PROJECT / "stage12_dual_external_validation/runs/iscx_vpn/medium/seed2022/train_manifest.csv",
    PROJECT / "stage12_dual_external_validation/runs/iscx_vpn/medium/seed2022/val_manifest.csv",
    PROJECT / "stage12_dual_external_validation/runs/iscx_vpn/medium/seed2022/known_train_validation_features.npz",
    PROJECT / "stage12_dual_external_validation/runs/iscx_vpn/medium/seed2022/model_best.pt",
    PROJECT / "stage15r_representation_bottleneck_audit/feature_cache/iscx_vpn/flow_statistics.csv",
    PROJECT / "stage15r_representation_bottleneck_audit/feature_cache/iscx_vpn/capture_audit.csv",
    PROJECT / "stage15r_representation_bottleneck_audit/pilot_runs/e0/iscx_vpn/medium_seed2022/result.json",
    WORKSPACE / "Projects/Open-Detect/reproduction/prepare_pcap_dataset.py",
    WORKSPACE / "Projects/TrafficFormer/reproduction/scripts/prepare_iscxvpn.py",
    WORKSPACE / "Projects/TrafficFormer/reproduction/datasets/iscxvpn_trafficformer/dataset_stats.json",
    WORKSPACE / "Projects/TrafficFormer/reproduction/datasets/iscxvpn_trafficformer/processing_audit.tsv",
    WORKSPACE / "Projects/TFE-GNN/artifacts/paper-reproduction-v2/iscx-vpn/flows/prepare_manifest.json",
    WORKSPACE / "Projects/TFE-GNN/artifacts/paper-reproduction-v2/iscx-vpn/paper_text_split_seed32/dataset_manifest.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()

    prior = PROJECT / "stage15f_literature_guided_feature_benchmark/frozen_asset_hashes_after.json"
    payload = json.loads(prior.read_text(encoding="utf-8"))
    paths = {PROJECT / relative for relative in payload["files"]}
    paths.update(EXTRA_INPUTS)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(missing)

    rows = {}
    for path in sorted(paths):
        try:
            key = str(path.relative_to(PROJECT))
        except ValueError:
            key = str(path)
        rows[key] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    result = {
        "phase": args.phase,
        "scope": "Stage15F-0 protected inputs plus DQ-specific Native, TFE-GNN, TrafficFormer code/manifests/checkpoint and Known Train/Validation evidence",
        "file_count": len(rows),
        "files": rows,
    }
    destination = OUT / f"frozen_asset_hashes_{args.phase}.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "file_count": len(rows), "output": str(destination)}))


if __name__ == "__main__":
    main()
