#!/usr/bin/env python3
"""Hash protected experiment metadata plus E0 checkpoints before/after Stage 15R."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, ROOT, sha256_file, write_json


PROTECTED_ROOTS = (
    "stage12_dual_external_validation",
    "stage14b_vnat_protocol_freeze",
    "stage14c_native_opendetect_vnat",
    "stage14c_vnat_encoder_training",
    "stage14c3_vnat_closed_set_failure_diagnosis",
    "stage14c4_our_method_training_cleanup",
    "stage14c5_feature_representation_audit",
    "stage14c5_protocol_difficulty_validation_dip_audit",
    "stage14c6_cleaned_pipeline_full15_closed_set",
    "stage14d_vnat_frozen_open_set_evaluation",
    "stage15a_failure_regime_complementarity_diagnosis",
    "stage15b_known_only_hybrid_detector",
)

E0_CHECKPOINTS = (
    "stage12_dual_external_validation/runs/iscx_vpn/medium/seed2022/model_best.pt",
    "stage12_dual_external_validation/runs/iscx_tor/medium/seed2022/model_best.pt",
    "stage14c_native_opendetect_vnat/runs/medium_seed2025/best_checkpoint.pt",
    "stage14c_native_opendetect_vnat/runs/medium_seed2026/best_checkpoint.pt",
    "stage3_unknown_utility/artifacts/A-2/best_checkpoint.pt",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    files: set[Path] = set()
    for name in PROTECTED_ROOTS:
        base = PROJECT_ROOT / name
        if not base.is_dir():
            raise FileNotFoundError(base)
        for suffix in ("*.json", "*.csv", "*.md", "*.sha256"):
            files.update(path for path in base.rglob(suffix) if path.is_file())
    files.update(PROJECT_ROOT / name for name in E0_CHECKPOINTS)
    records = {}
    for path in sorted(files):
        relative = str(path.relative_to(PROJECT_ROOT))
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(ROOT / f"frozen_asset_hashes_{args.phase}.json", {"phase": args.phase, "protected_roots": list(PROTECTED_ROOTS), "file_count": len(records), "files": records})
    print(f"phase={args.phase} files={len(records)}")


if __name__ == "__main__":
    main()
