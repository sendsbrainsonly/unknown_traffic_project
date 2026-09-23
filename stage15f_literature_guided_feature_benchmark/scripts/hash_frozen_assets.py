#!/usr/bin/env python3
"""Hash the protected Stage 12--15R evidence and formal Stage 15F-0 inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, ROOT, sha256_file, write_json


PROTECTED_ROOTS = (
    "stage12_dual_external_validation",
    "stage13a1_knn_local_support",
    "stage13a2_global_local_fusion",
    "stage14a_vnat_data_audit",
    "stage14a5_vnat_split_feasibility",
    "stage14b_vnat_protocol_freeze",
    "stage14b5_vnat_flow_retention_audit",
    "stage14c_vnat_encoder_training",
    "stage14c_native_opendetect_vnat",
    "stage14c3_vnat_closed_set_failure_diagnosis",
    "stage14c4_our_method_training_cleanup",
    "stage14c5_feature_representation_audit",
    "stage14c5_protocol_difficulty_validation_dip_audit",
    "stage14c6_cleaned_pipeline_full15_closed_set",
    "stage14d_vnat_frozen_open_set_evaluation",
    "stage15a_failure_regime_complementarity_diagnosis",
    "stage15b_known_only_hybrid_detector",
    "stage15r_representation_bottleneck_audit",
)

FORMAL_INPUTS = (
    "stage3_unknown_utility/outputs/A-1/data_manifest.csv",
    "stage3_unknown_utility/outputs/A-2/data_manifest.csv",
    "stage3_unknown_utility/outputs/A-3/data_manifest.csv",
    "stage12_dual_external_validation/configs/stage12_config.json",
    "stage12_dual_external_validation/protocol/iscx_vpn/preprocessing_manifest.json",
    "stage12_dual_external_validation/protocol/iscx_vpn/split_manifest.csv",
    "stage12_dual_external_validation/protocol/iscx_tor/preprocessing_manifest.json",
    "stage12_dual_external_validation/protocol/iscx_tor/split_manifest.csv",
    "stage14b_vnat_protocol_freeze/vnat_open_set_protocol.json",
    "stage14b_vnat_protocol_freeze/vnat_split_manifest.csv",
    "stage14a_vnat_data_audit/vnat_manifest.csv",
    "stage14c5_feature_representation_audit/raw_feature_cache/cache_audit.json",
    "stage14c5_feature_representation_audit/raw_feature_cache/raw_cache_manifest.csv",
    "opendetect_ustc_encoder_audit/outputs/input_alignment_manifest.csv",
    "stage15r_representation_bottleneck_audit/feature_cache/ustc/cache_audit.json",
    "stage15r_representation_bottleneck_audit/feature_cache/ustc/pkl_audit.csv",
    "stage15r_representation_bottleneck_audit/feature_cache/iscx_vpn/cache_audit.json",
    "stage15r_representation_bottleneck_audit/feature_cache/iscx_vpn/capture_audit.csv",
    "stage15r_representation_bottleneck_audit/feature_cache/iscx_tor/cache_audit.json",
    "stage15r_representation_bottleneck_audit/feature_cache/iscx_tor/capture_audit.csv",
)


def protected_metadata() -> set[Path]:
    files: set[Path] = set()
    names = {"RESULTS.md", "manifest.json", "completion_verification.json", "final_verification.json"}
    for root_name in PROTECTED_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for name in names:
            files.update(path for path in root.rglob(name) if path.is_file())
    files.update(PROJECT_ROOT / relative for relative in FORMAL_INPUTS)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    records = {}
    for path in sorted(protected_metadata()):
        relative = str(path.relative_to(PROJECT_ROOT))
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    payload = {
        "phase": args.phase,
        "scope": "protected stage-level result manifests/verifications plus every Stage15F-0 formal frozen manifest/lineage input; raw PCAP SHA256 values are inherited from the hashed Stage15R capture audits",
        "protected_roots": list(PROTECTED_ROOTS),
        "formal_inputs": list(FORMAL_INPUTS),
        "file_count": len(records),
        "files": records,
    }
    write_json(ROOT / f"frozen_asset_hashes_{args.phase}.json", payload)
    print(f"phase={args.phase} files={len(records)}")


if __name__ == "__main__":
    main()
