#!/usr/bin/env python3
"""Verify frozen Stage6--10A evidence without writing upstream directories."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
STAGE10B = PROJECT / "stage10b_opendetect_score_decomposition"
OUTPUT = STAGE10B / "outputs/summary/provenance_verification.json"
STAGE10A = PROJECT / "stage10a_opendetect_protocol_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_hash_file(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        expected, raw_path = line.split(maxsplit=1)
        candidate = Path(raw_path)
        target = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
        require(target.is_file(), f"missing hash target: {target}")
        require(sha256(target) == expected, f"hash mismatch: {target}")
        count += 1
    return count


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def main() -> int:
    require(os.environ.get("CONDA_PREFIX") == sys.prefix, "CONDA_PREFIX and sys.prefix differ")
    require(
        sys.prefix == "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
        f"wrong Python environment: {sys.prefix}",
    )

    sentinel_paths = [
        PROJECT / "stage6_cipherspectrum_protocol/outputs/protocol/protocol_hashes.sha256",
        PROJECT / "stage6_cipherspectrum_protocol/outputs/corpus/canonical120k.sha256",
        PROJECT / "stage8a_cipherspectrum_known_density/manifest.json",
        PROJECT / "stage8b_cipherspectrum_dgsbv2/manifest.json",
        PROJECT / "stage9_cipherspectrum_final_test/outputs/summary/stage9_final_hashes.sha256",
        PROJECT / "stage9_cipherspectrum_final_test/outputs/summary/stage9_final_manifest.json",
    ]
    for setting in ("low", "medium", "high"):
        run = next((PROJECT / "stage7_cipherspectrum_known_training/runs").glob(f"formal-{setting}-*"))
        sentinel_paths.extend(
            [
                run / "training_checkpoint_selection.json",
                run / "manifest.json",
                PROJECT / f"stage8a_cipherspectrum_known_density/artifacts/{setting}/bundle_manifest.json",
                PROJECT / f"stage8a_cipherspectrum_known_density/artifacts/{setting}/native_threshold.json",
                PROJECT / f"stage9_cipherspectrum_final_test/artifacts/{setting}/test_bundle_manifest.json",
            ]
        )
    before = {str(path.relative_to(PROJECT)): sha256(path) for path in sentinel_paths}
    stage10a_before = tree_hashes(STAGE10A)

    stage6 = {
        "protocol_hashes_verified": verify_hash_file(
            PROJECT / "stage6_cipherspectrum_protocol/outputs/protocol/protocol_hashes.sha256"
        ),
        "corpus_hashes_verified": verify_hash_file(
            PROJECT / "stage6_cipherspectrum_protocol/outputs/corpus/canonical120k.sha256"
        ),
    }

    verifier_results = []
    for name, script in (
        ("stage7", PROJECT / "stage7_cipherspectrum_known_training/scripts/verify_stage7.py"),
        ("stage8a", PROJECT / "stage8a_cipherspectrum_known_density/scripts/verify_stage8a.py"),
        ("stage8b", PROJECT / "stage8b_cipherspectrum_dgsbv2/scripts/verify_stage8b.py"),
        ("stage9", PROJECT / "stage9_cipherspectrum_final_test/scripts/verify_stage9.py"),
    ):
        result = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=PROJECT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
        )
        verifier_results.append(
            {
                "stage": name,
                "returncode": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
        require(result.returncode == 0, f"{name} verifier failed: {result.stderr or result.stdout}")

    stage10a_validation = json.loads(
        (STAGE10A / "outputs/summary/audit_validation.json").read_text(encoding="utf-8")
    )
    stage10a_integrity = json.loads(
        (STAGE10A / "outputs/summary/upstream_integrity.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (STAGE10A / "sources/source_manifest.json").read_text(encoding="utf-8")
    )
    require(stage10a_validation["status"] == "PASS", "Stage10A validation is not PASS")
    require(stage10a_integrity["status"] == "PASS", "Stage10A upstream integrity is not PASS")
    require(
        stage10a_validation["opendetect_sibling_integrity"] == "PASS",
        "Stage10A sibling Open-Detect integrity is not PASS",
    )
    source_items = source_manifest["files"] + source_manifest["v6_checkpoints"]
    for item in source_items:
        path = Path(item["path"])
        require(path.is_file(), f"missing Stage10A source: {path}")
        require(path.stat().st_size == item["size_bytes"], f"Stage10A source size mismatch: {path}")
        require(sha256(path) == item["sha256"], f"Stage10A source hash mismatch: {path}")

    after = {str(path.relative_to(PROJECT)): sha256(path) for path in sentinel_paths}
    stage10a_after = tree_hashes(STAGE10A)
    require(before == after, "Stage6--9 frozen sentinel changed during provenance verification")
    require(stage10a_before == stage10a_after, "Stage10A changed during provenance verification")

    payload = {
        "status": "PASS",
        "audit_mode": "READ_ONLY_PROVENANCE_GATE",
        "environment": {"CONDA_PREFIX": os.environ.get("CONDA_PREFIX"), "sys_prefix": sys.prefix},
        "stage6": {"status": "PASS", **stage6},
        "stage7_to_stage9": verifier_results,
        "stage10a": {
            "status": "PASS",
            "files_hashed": len(stage10a_before),
            "source_manifest_items_verified": len(source_items),
            "sibling_v6_checkpoints_verified": len(source_manifest["v6_checkpoints"]),
            "final_gate": stage10a_validation["final_gate"],
        },
        "frozen_sentinel_hashes_before": before,
        "frozen_sentinel_hashes_after": after,
        "frozen_sentinels_unchanged": True,
        "stage10a_unchanged": True,
        "frozen_experiment_modified": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "stage10a_files": len(stage10a_before)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
