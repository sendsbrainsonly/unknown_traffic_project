#!/usr/bin/env python3
"""Shared frozen paths, provenance checks, scoring, and Stage 9 I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np


STAGE9_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE9_ROOT.parent
STAGE6_ROOT = PROJECT_ROOT / "stage6_cipherspectrum_protocol"
STAGE7_ROOT = PROJECT_ROOT / "stage7_cipherspectrum_known_training"
STAGE8A_ROOT = PROJECT_ROOT / "stage8a_cipherspectrum_known_density"
STAGE8B_ROOT = PROJECT_ROOT / "stage8b_cipherspectrum_dgsbv2"
SPLIT_MANIFEST = STAGE6_ROOT / "outputs/splits/split_manifest.csv"
CONFIG_PATH = STAGE9_ROOT / "configs/stage9_config.json"
EVALUATION_CONFIG_PATH = STAGE8B_ROOT / "configs/evaluation_config_v2.json"
VALID_SETTINGS = ("low", "medium", "high")
METHOD_ORDER = (
    "Native",
    "Single-Full-K1",
    "Multi-Global-K2",
    "Class-P05-K2",
    "Component-P05-K2",
    "DGSB-v2",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE8B = _load_module("frozen_stage8b_common_for_stage9", STAGE8B_ROOT / "scripts/stage8b_common.py")
STAGE8A = STAGE8B.STAGE8A
STAGE7 = _load_module("frozen_stage7_common_for_stage9", STAGE7_ROOT / "scripts/common.py")


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


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold(setting: str) -> dict[str, object]:
    return STAGE8A.load_fold(setting)


def stage7_run(setting: str) -> Path:
    return STAGE8A.stage7_run(setting)


def verify_hash_lines(hash_file: Path, root: Path) -> int:
    count = 0
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = (root / relative).resolve()
        require(path.is_file(), f"missing frozen target: {path}")
        require(sha256_file(path) == expected, f"frozen hash mismatch: {path}")
        count += 1
    return count


def verify_all_provenance() -> dict[str, object]:
    """Verify Stage 6/7/8A/8B and the final evaluation config without Test access."""

    config = load_json(CONFIG_PATH)
    upstream = STAGE8B.verify_all_provenance()
    require(upstream["status"] == "PASS", "Stage 6/7/8A provenance failed")
    gate_path = STAGE8B_ROOT / "outputs/summary/final_gate.json"
    gate = load_json(gate_path)
    require(gate["gate"] == "READY_FOR_ONE_SHOT_FINAL_TEST", "Stage 8B final gate is not READY")
    require(all(bool(value) for value in gate["checks"].values()), "Stage 8B final gate has failed checks")
    expected = config["frozen_upstream"]
    evaluation_sha = sha256_file(EVALUATION_CONFIG_PATH)
    require(evaluation_sha == expected["evaluation_config_v2_sha256"], "evaluation_config_v2 hash mismatch")
    require(evaluation_sha == gate["evaluation_config_v2_sha256"], "evaluation_config_v2 differs from Stage 8B gate")
    unified_path = STAGE8B_ROOT / "artifacts/dgsbv2_global_spec.json"
    require(sha256_file(unified_path) == expected["dgsbv2_unified_rule_sha256"], "DGSB-v2 unified rule hash mismatch")
    require(sha256_file(unified_path) == gate["unified_rule_sha256"], "DGSB-v2 unified rule differs from Stage 8B gate")
    setting_rows: list[dict[str, object]] = []
    for setting in VALID_SETTINGS:
        artifact = STAGE8B_ROOT / "artifacts" / setting
        hash_path = artifact / "dgsbv2_hashes.sha256"
        rule_path = artifact / "dgsbv2_rule.json"
        require(sha256_file(hash_path) == expected["dgsbv2_setting_bundle_hash_file_sha256"][setting], f"{setting}: DGSB-v2 bundle hash-file mismatch")
        require(sha256_file(hash_path) == gate["setting_bundle_hashes"][setting], f"{setting}: DGSB-v2 bundle differs from Stage 8B gate")
        require(sha256_file(rule_path) == expected["dgsbv2_setting_rule_sha256"][setting], f"{setting}: DGSB-v2 rule hash mismatch")
        require(sha256_file(rule_path) == gate["setting_rule_sha256"][setting], f"{setting}: DGSB-v2 rule differs from Stage 8B gate")
        verified = verify_hash_lines(hash_path, artifact)
        require(verified == 5, f"{setting}: expected five DGSB-v2 rule assets")
        setting_rows.append({"setting": setting, "status": "PASS", "dgsbv2_files_verified": verified})
    evaluation = load_json(EVALUATION_CONFIG_PATH)
    require(tuple(evaluation["method_order"]) == METHOD_ORDER, "frozen method order changed")
    require(evaluation["final_test_metrics"]["paired_bootstrap_iterations"] == 1000, "bootstrap count changed")
    return {
        "status": "PASS",
        "stage6": upstream["stage6"],
        "stage7": upstream["stage7"],
        "stage8a": upstream["stage8a"],
        "stage8b": {
            "status": "PASS",
            "final_gate": gate["gate"],
            "settings": setting_rows,
            "unified_rule_sha256": sha256_file(unified_path),
        },
        "evaluation_config_v2_sha256": evaluation_sha,
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "created_before_any_test_result": True,
    }


def assert_test_open_record() -> dict[str, object]:
    path = STAGE9_ROOT / "outputs/test_open_provenance.json"
    require(path.is_file(), "Test-open provenance must exist before Test access")
    payload = load_json(path)
    require(payload["first_test_open"] is True, "Test-open authorization flag missing")
    require(payload["methods_frozen_before_test"] is True, "methods were not frozen before Test")
    require(payload["created_before_any_test_result"] is True, "Test-open record was not created before results")
    for item in payload["stage9_frozen_implementation"]:
        target = Path(str(item["path"]))
        require(sha256_file(target) == item["sha256"], f"Stage 9 frozen implementation changed after Test open: {target}")
    return payload


def score_density_matrix(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray]:
    return STAGE8A.score_density_matrix(models, values)


def score_k2_matrices(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    return STAGE8A.score_k2_matrices(models, values)


def load_frozen_models(setting: str) -> dict[str, object]:
    artifact = STAGE8A_ROOT / "artifacts" / setting
    return {
        "scaler": joblib.load(artifact / "scaler.joblib"),
        "pca": joblib.load(artifact / "pca64.joblib"),
        "k1": joblib.load(artifact / "single_k1/models.joblib"),
        "k2": joblib.load(artifact / "multi_k2/models.joblib"),
    }


def file_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((item.resolve() for item in paths), key=str):
        rows.append({
            "path": path.relative_to(root.resolve()).as_posix(),
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def write_hash_lines(paths: Iterable[Path], root: Path, output: Path) -> None:
    lines = []
    for path in sorted((item.resolve() for item in paths), key=str):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root.resolve()).as_posix()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
