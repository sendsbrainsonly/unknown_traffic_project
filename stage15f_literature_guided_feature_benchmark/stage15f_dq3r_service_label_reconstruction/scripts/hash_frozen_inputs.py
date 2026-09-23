#!/usr/bin/env python3
"""Hash the bounded frozen inputs used by Stage 15F-DQ-3R."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
WORKSPACE = ROOT.parents[1]


def digest(path: Path) -> dict[str, object]:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    dq = ROOT / "stage15f_literature_guided_feature_benchmark" / "stage15f_dq_performance_gap_attribution"
    stage12 = ROOT / "stage12_dual_external_validation"
    tfe = WORKSPACE / "Projects" / "TFE-GNN"
    tf = WORKSPACE / "Projects" / "TrafficFormer"
    files = [
        dq / "cross_project_pcap_manifest.csv",
        dq / "cross_project_flow_manifest.csv",
        dq / "frozen_asset_hashes_after.json",
        dq / "label_mapping.json",
        stage12 / "protocol" / "iscx_vpn" / "split_manifest.csv",
        stage12 / "protocol" / "iscx_vpn" / "DATASET_AUDIT.md",
        stage12 / "scripts" / "stage12_common.py",
        stage12 / "artifacts" / "iscx_vpn" / "protocol" / "low" / "protocol.json",
        stage12 / "artifacts" / "iscx_vpn" / "protocol" / "medium" / "protocol.json",
        stage12 / "artifacts" / "iscx_vpn" / "protocol" / "high" / "protocol.json",
        stage12 / "runs" / "iscx_vpn" / "medium" / "seed2022" / "config.json",
        tf / "reproduction" / "scripts" / "prepare_iscxvpn.py",
        tf / "reproduction" / "datasets" / "iscxvpn_trafficformer" / "processing_audit.tsv",
        tf / "reproduction" / "datasets" / "iscxvpn_trafficformer" / "dataset_stats.json",
        tfe / "reproduction" / "protocols.py",
        tfe / "artifacts" / "paper-reproduction-v2" / "iscx-vpn" / "flows" / "prepare_manifest.json",
        tfe / "artifacts" / "paper-reproduction-v2" / "iscx-vpn" / "paper_text_split_seed32" / "dataset_manifest.json",
    ]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + json.dumps(missing))
    payload = {
        "phase": args.phase,
        "scope": "Stage15F-DQ-3R bounded inputs; Known Train/Validation metadata only; no Test feature arrays",
        "file_count": len(files),
        "files": {str(p): digest(p) for p in sorted(files)},
    }
    target = OUT / f"frozen_asset_hashes_{args.phase}.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "file_count": len(files), "output": str(target)}))


if __name__ == "__main__":
    main()
