#!/usr/bin/env python3
"""Create the immutable Test-open record immediately before any Test read."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from stage9_common import (
    CONFIG_PATH,
    EVALUATION_CONFIG_PATH,
    PROJECT_ROOT,
    STAGE8B_ROOT,
    STAGE9_ROOT,
    VALID_SETTINGS,
    load_json,
    require,
    sha256_file,
    write_json,
)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.rstrip()


def main() -> None:
    provenance_path = STAGE9_ROOT / "outputs/provenance_verification.json"
    require(provenance_path.is_file(), "upstream provenance verification must run first")
    provenance = load_json(provenance_path)
    require(provenance["status"] == "PASS", "upstream provenance did not pass")
    output = STAGE9_ROOT / "outputs/test_open_provenance.json"
    require(not output.exists(), "Test-open provenance already exists; one-shot record cannot be overwritten")
    prohibited = []
    for setting in VALID_SETTINGS:
        artifact = STAGE9_ROOT / "artifacts" / setting
        for name in (
            "known_test_manifest.csv",
            "unknown_test_manifest.csv",
            "mu_known_test.npy",
            "mu_unknown_test.npy",
            "z_known_test_pca64.npy",
            "z_unknown_test_pca64.npy",
        ):
            if (artifact / name).exists():
                prohibited.append(str(artifact / name))
    require(not prohibited, f"Test artifacts already exist before first-open record: {prohibited}")
    frozen_files = [
        CONFIG_PATH,
        EVALUATION_CONFIG_PATH,
        STAGE9_ROOT / "scripts/stage9_common.py",
        STAGE9_ROOT / "scripts/prepare_test_inputs.py",
        STAGE9_ROOT / "scripts/extract_test_representations.py",
        STAGE9_ROOT / "scripts/evaluate_setting.py",
        STAGE9_ROOT / "scripts/finalize_stage9.py",
    ]
    require(all(path.is_file() for path in frozen_files), "Stage 9 implementation is incomplete")
    gate = load_json(STAGE8B_ROOT / "outputs/summary/final_gate.json")
    payload = {
        "test_open_started_at": datetime.now(timezone.utc).isoformat(),
        "first_test_open": True,
        "methods_frozen_before_test": True,
        "created_before_any_test_result": True,
        "git_status": git_output("status", "--short"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "evaluation_config_v2_path": str(EVALUATION_CONFIG_PATH.resolve()),
        "evaluation_config_v2_sha256": sha256_file(EVALUATION_CONFIG_PATH),
        "stage8b_unified_rule_sha256": gate["unified_rule_sha256"],
        "stage8b_setting_rule_sha256": gate["setting_rule_sha256"],
        "stage9_frozen_implementation": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in frozen_files
        ],
        "post_test_tuning_performed": False,
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
