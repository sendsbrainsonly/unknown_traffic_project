#!/usr/bin/env python3
"""Shared frozen paths, provenance checks, access audit, and DGSB-v2 math."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


STAGE8B_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE8B_ROOT.parent
STAGE8A_ROOT = PROJECT_ROOT / "stage8a_cipherspectrum_known_density"
CONFIG_PATH = STAGE8B_ROOT / "configs/stage8b_config.json"
VALID_SETTINGS = ("low", "medium", "high")
FORBIDDEN_ROLES = {"KNOWN_TEST", "UNKNOWN_TEST"}


def _load_stage8a_common():
    path = STAGE8A_ROOT / "scripts/common.py"
    spec = importlib.util.spec_from_file_location("frozen_stage8a_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Stage 8A helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE8A = _load_stage8a_common()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        require(bool(rows), f"cannot infer fields for empty CSV: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return STAGE8A.sha256_file(path)


def empirical_threshold(scores: np.ndarray, target_acceptance: float = 0.975) -> tuple[float, float]:
    """Empirical threshold whose acceptance is closest to target; ties use the higher threshold."""

    return STAGE8A.conservative_empirical_threshold(scores, target=target_acceptance)


def score_k2_matrices(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    return STAGE8A.score_k2_matrices(models, values)


def true_class_component_assignments(
    labels: np.ndarray, names: list[str], local_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return STAGE8A.true_class_component_assignments(labels, names, local_scores)


class AccessLedger:
    """Formal Stage 8B input ledger; forbidden roles fail before file open."""

    def __init__(self, setting: str) -> None:
        require(setting in VALID_SETTINGS, f"invalid setting: {setting}")
        self.setting = setting
        self.reads: list[dict[str, object]] = []

    def record(self, path: Path, file_type: str, role: str, records: int | None = None) -> None:
        if role in FORBIDDEN_ROLES:
            raise RuntimeError(f"forbidden Stage 8B data access blocked before open: {role}: {path}")
        item: dict[str, object] = {
            "path": str(path.resolve()),
            "file_type": file_type,
            "role": role,
        }
        if records is not None:
            item["records"] = int(records)
        self.reads.append(item)

    def payload(self) -> dict[str, object]:
        counts = Counter((str(item["role"]), str(item["file_type"])) for item in self.reads)
        return {
            "setting": self.setting,
            "reads": self.reads,
            "read_events": len(self.reads),
            "counts_by_role_and_type": {
                f"{role}:{file_type}": count for (role, file_type), count in sorted(counts.items())
            },
            "pcap_files_opened": 0,
            "known_test_opened": 0,
            "known_test_mu_generated": False,
            "unknown_test_opened": 0,
            "unknown_mu_generated": False,
            "unknown_inference_executed": False,
            "status": "PASS",
        }

    def save(self, path: Path) -> None:
        write_json(path, self.payload())


def verify_hash_lines(hash_file: Path, root: Path) -> int:
    count = 0
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = (root / relative).resolve()
        require(path.is_file(), f"missing frozen hash target: {path}")
        require(sha256_file(path) == expected, f"frozen hash mismatch: {path}")
        count += 1
    return count


def verify_stage8a_setting(setting: str, config: dict[str, object]) -> dict[str, object]:
    artifact_root = STAGE8A_ROOT / "artifacts" / setting
    hash_file = artifact_root / "bundle_hashes.sha256"
    manifest_path = artifact_root / "bundle_manifest.json"
    expected = config["expected_stage8a"]
    require(
        sha256_file(hash_file) == expected["bundle_hash_file_sha256"][setting],
        f"{setting}: Stage 8A bundle hash file changed",
    )
    require(
        sha256_file(manifest_path) == expected["bundle_manifest_sha256"][setting],
        f"{setting}: Stage 8A bundle manifest changed",
    )
    manifest = load_json(manifest_path)
    require(manifest["status"] == "FROZEN", f"{setting}: Stage 8A bundle not frozen")
    lines = hash_file.read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(manifest["files"]), f"{setting}: Stage 8A hash ledger length mismatch")
    verified = 0
    for line, item in zip(lines, manifest["files"]):
        expected_sha, relative = line.split(maxsplit=1)
        require(expected_sha == item["sha256"], f"{setting}: Stage 8A ledger SHA mismatch")
        require(relative == item["path"], f"{setting}: Stage 8A ledger path mismatch")
        path = Path(str(item["absolute_path"]))
        require(path.is_file(), f"{setting}: Stage 8A frozen file missing: {path}")
        require(sha256_file(path) == expected_sha, f"{setting}: Stage 8A frozen file changed: {path}")
        verified += 1
    return {
        "setting": setting,
        "verification_status": "PASS",
        "files_verified": verified,
        "bundle_hash_file": str(hash_file.resolve()),
        "bundle_hash_file_sha256": sha256_file(hash_file),
        "bundle_manifest": str(manifest_path.resolve()),
        "bundle_manifest_sha256": sha256_file(manifest_path),
    }


def verify_all_provenance() -> dict[str, object]:
    config = load_json(CONFIG_PATH)
    upstream = STAGE8A.verify_all_provenance()
    require(upstream["status"] == "PASS", "Stage 6/7 provenance failed")
    stage8a_config_path = STAGE8A_ROOT / "configs/evaluation_config.json"
    require(
        sha256_file(stage8a_config_path) == config["expected_stage8a"]["evaluation_config_sha256"],
        "Stage 8A evaluation_config changed",
    )
    stage8a_gate = load_json(STAGE8A_ROOT / "outputs/summary/final_gate.json")
    require(stage8a_gate["gate"] == "READY_FOR_FINAL_TEST", "Stage 8A final gate is not READY")
    require(all(stage8a_gate["checks"].values()), "Stage 8A final gate contains a failed check")
    settings = [verify_stage8a_setting(setting, config) for setting in VALID_SETTINGS]
    evaluation = load_json(stage8a_config_path)
    expected_methods = [
        "Native",
        "Single-Full-K1",
        "Multi-Global-K2",
        "Class-P05-K2",
        "Component-P05-K2",
    ]
    require(evaluation["method_order"] == expected_methods, "Stage 8A baseline method order changed")
    return {
        "status": "PASS",
        "stage6": upstream["stage6"],
        "stage7": upstream["settings"],
        "stage8a": {
            "verification_status": "PASS",
            "evaluation_config_path": str(stage8a_config_path.resolve()),
            "evaluation_config_sha256": sha256_file(stage8a_config_path),
            "final_gate": stage8a_gate["gate"],
            "settings": settings,
        },
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
    }


def gate_status(global_exclusive_count: int, local_exclusive_count: int) -> str:
    if global_exclusive_count > 0 and local_exclusive_count > 0:
        return "DUAL_GATE_ACTIVE"
    if global_exclusive_count == 0 and local_exclusive_count > 0:
        return "GLOBAL_GATE_REDUNDANT"
    if global_exclusive_count > 0 and local_exclusive_count == 0:
        return "LOCAL_GATE_REDUNDANT"
    return "BOTH_GATES_REDUNDANT"


def hash_manifest(paths: Iterable[Path], root: Path, output: Path) -> str:
    lines: list[str] = []
    for path in sorted((item.resolve() for item in paths), key=str):
        relative = path.relative_to(root.resolve())
        lines.append(f"{sha256_file(path)}  {relative.as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(output)
