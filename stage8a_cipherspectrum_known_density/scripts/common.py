#!/usr/bin/env python3
"""Frozen paths, provenance checks, access ledger, and Stage 8A math helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import logsumexp


STAGE8_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE8_ROOT.parent
STAGE6_ROOT = PROJECT_ROOT / "stage6_cipherspectrum_protocol"
STAGE7_ROOT = PROJECT_ROOT / "stage7_cipherspectrum_known_training"
STAGE6_OUTPUTS = STAGE6_ROOT / "outputs"
PROTOCOL_HASH_MANIFEST = STAGE6_OUTPUTS / "protocol/protocol_hashes.sha256"
SPLIT_MANIFEST = STAGE6_OUTPUTS / "splits/split_manifest.csv"
CONFIG_PATH = STAGE8_ROOT / "configs/stage8a_config.json"
VALID_SETTINGS = ("low", "medium", "high")
ALLOWED_ROLES = {"KNOWN_TRAIN", "KNOWN_VALIDATION"}
FORBIDDEN_ROLES = {"KNOWN_TEST", "UNKNOWN_TEST"}
EXPECTED_PROTOCOL_CONFIG_SHA256 = "adab7f9a251be7900f42eb3286d9f897154f06ebdc0ddb6f3dd4eff48730ad8e"
EXPECTED_PROTOCOL_HASH_MANIFEST_SHA256 = "c3783084f0f9b1242e1c05e263ecb9203f6c017f1b746cd885b99c637fccc290"
EXPECTED_SPLIT_MANIFEST_SHA256 = "021e46a5c9aa2d9e44798af450d493aa572601c575fa5ae5ac4eee94054b45b4"
EXPECTED_OPEN_SET_PROTOCOL_SHA256 = "96eba64c7fbc85323604c68c1ece03c4702548c6f1908dae4fd22d7c37f7f666"
EXPECTED_STAGE7 = {
    "low": {
        "best_epoch": 26,
        "checkpoint_sha256": "78e23b0d9795a7829c03d740c67c5a823ca9f6ea36bda4cd0e24338a168b4d92",
        "manifest_sha256": "9e163fbb61e190cf3f8b8c019b0194ffcb59c800e4879110b4340fbcf9a0c68b",
        "fold_sha256": "dc5fafcca97e55efe5d96aca0cf827db98cef66ba3fa449caf952073efb3cfd7",
    },
    "medium": {
        "best_epoch": 22,
        "checkpoint_sha256": "59858ee8d9dcc0d36e64a8a396fcf0f401b720dbed8d2aa2f783761be352ac48",
        "manifest_sha256": "d0afbc68db3be2a72d69f35c274f809614c30e2684edb64cbacea3a3be8f6556",
        "fold_sha256": "d378b933eed561b509c46a92f98f64941514219284f4ba6843b022fb40b85319",
    },
    "high": {
        "best_epoch": 22,
        "checkpoint_sha256": "588c9eeec5d352f28b5598163aed193e01e53b8a408cdbf6ddbbaf58dbaae213",
        "manifest_sha256": "6258abd8021355465b51a651e8ed945662644b329ade9fdb7b5484c9cfe5b250",
        "fold_sha256": "12bd599bcb43a4527b869341fdd5efdb7e8664745747516c61d72d335604278e",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        require(bool(rows), f"cannot infer CSV fields for empty rows: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stage7_run(setting: str) -> Path:
    require(setting in VALID_SETTINGS, f"invalid setting: {setting}")
    return STAGE7_ROOT / f"runs/formal-{setting}-20260913"


def fold_path(setting: str) -> Path:
    return STAGE6_OUTPUTS / f"splits/{setting}_fold.json"


def load_fold(setting: str) -> dict[str, object]:
    require(setting in VALID_SETTINGS, f"invalid setting: {setting}")
    path = fold_path(setting)
    require(sha256_file(path) == EXPECTED_STAGE7[setting]["fold_sha256"], f"{setting}: fold hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["setting"] == setting, f"{setting}: fold identity mismatch")
    require(not set(payload["known_classes"]) & set(payload["unknown_classes"]), f"{setting}: class overlap")
    require(int(payload["known_unknown_group_overlap"]) == 0, f"{setting}: Known/Unknown group overlap")
    require(all(int(value) == 0 for value in payload["known_group_overlap"].values()), f"{setting}: Known split group overlap")
    return payload


def verify_stage6_protocol() -> dict[str, object]:
    require(sha256_file(STAGE6_ROOT / "configs/protocol_config.json") == EXPECTED_PROTOCOL_CONFIG_SHA256, "Stage 6 protocol config hash mismatch")
    require(sha256_file(PROTOCOL_HASH_MANIFEST) == EXPECTED_PROTOCOL_HASH_MANIFEST_SHA256, "Stage 6 protocol hash-manifest mismatch")
    require(sha256_file(SPLIT_MANIFEST) == EXPECTED_SPLIT_MANIFEST_SHA256, "Stage 6 split manifest mismatch")
    verified: list[dict[str, object]] = []
    root = PROTOCOL_HASH_MANIFEST.parent.resolve()
    outputs = STAGE6_OUTPUTS.resolve()
    for line in PROTOCOL_HASH_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = (root / relative).resolve()
        require(path.is_relative_to(outputs), f"Stage 6 hash target escapes outputs: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"Stage 6 artifact changed: {relative}")
        verified.append({"path": str(path), "sha256": actual})
    require(len(verified) == 16, f"expected 16 Stage 6 hashes, got {len(verified)}")
    protocol_path = STAGE6_OUTPUTS / "protocol/cipherspectrum_open_set_protocol.json"
    require(sha256_file(protocol_path) == EXPECTED_OPEN_SET_PROTOCOL_SHA256, "Stage 6 open-set protocol hash mismatch")
    return {
        "verification_status": "PASS",
        "protocol_sha256": EXPECTED_OPEN_SET_PROTOCOL_SHA256,
        "protocol_hash_manifest_sha256": EXPECTED_PROTOCOL_HASH_MANIFEST_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "artifacts_verified": len(verified),
    }


def verify_run_bundle(run_dir: Path) -> int:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "success", f"{run_dir.name}: experiment package is not successful")
    artifacts = manifest.get("artifacts", [])
    require(bool(artifacts), f"{run_dir.name}: empty experiment artifact inventory")
    for item in artifacts:
        path = run_dir / str(item["path"])
        require(path.is_file(), f"{run_dir.name}: missing {item['path']}")
        require(sha256_file(path) == item["sha256"], f"{run_dir.name}: package hash mismatch {item['path']}")
    return len(artifacts)


def verify_stage7_setting(setting: str) -> dict[str, object]:
    expected = EXPECTED_STAGE7[setting]
    run_dir = stage7_run(setting)
    manifest_path = run_dir / "manifest.json"
    require(sha256_file(manifest_path) == expected["manifest_sha256"], f"{setting}: Stage 7 manifest changed")
    artifacts_verified = verify_run_bundle(run_dir)
    selection = json.loads((run_dir / "training_checkpoint_selection.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "artifacts/training_best_checkpoint.pt"
    require(int(selection["best_epoch"]) == expected["best_epoch"], f"{setting}: wrong best epoch")
    require(selection["checkpoint_sha256"] == expected["checkpoint_sha256"], f"{setting}: selection hash changed")
    require(sha256_file(checkpoint) == expected["checkpoint_sha256"], f"{setting}: checkpoint hash mismatch")
    require(selection["selection_data"] == "Known Validation only", f"{setting}: selection leakage")
    require(int(selection["known_test_samples_used"]) == 0, f"{setting}: Known Test used")
    require(int(selection["unknown_samples_used"]) == 0, f"{setting}: Unknown used")
    require(selection["unknown_inference_executed"] is False, f"{setting}: Unknown inference recorded")
    audit = json.loads((run_dir / "inputs/input_audit.json").read_text(encoding="utf-8"))
    require(int(audit["known_test_pcap_files_opened"]) == 0, f"{setting}: Known Test PCAP opened in Stage 7")
    require(int(audit["unknown_pcap_files_opened"]) == 0, f"{setting}: Unknown PCAP opened in Stage 7")
    return {
        "setting": setting,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "checkpoint_best_epoch": expected["best_epoch"],
        "stage7_manifest_path": str(manifest_path.resolve()),
        "stage7_manifest_sha256": expected["manifest_sha256"],
        "stage7_package_artifacts_verified": artifacts_verified,
        "protocol_sha256": EXPECTED_OPEN_SET_PROTOCOL_SHA256,
        "verification_status": "PASS",
    }


def verify_all_provenance() -> dict[str, object]:
    protocol = verify_stage6_protocol()
    settings = [verify_stage7_setting(setting) for setting in VALID_SETTINGS]
    require(len({item["checkpoint_sha256"] for item in settings}) == 3, "Stage 7 checkpoints are not distinct")
    return {
        "status": "PASS",
        "stage6": protocol,
        "settings": settings,
        "stage7_checkpoints_distinct": True,
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "unknown_inference_executed": False,
    }


class AccessLedger:
    """Explicit formal-data read ledger that rejects forbidden roles before access."""

    def __init__(self, setting: str, path: Path) -> None:
        self.setting = setting
        self.path = path
        if path.exists():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {"setting": setting, "reads": []}
        require(self.payload["setting"] == setting, "access ledger setting mismatch")

    def record(self, path: Path, file_type: str, role: str, records: int | None = None) -> None:
        if role in FORBIDDEN_ROLES:
            raise RuntimeError(f"forbidden Stage 8A data access blocked before open: {role}: {path}")
        entry: dict[str, object] = {
            "path": str(path.resolve()),
            "file_type": file_type,
            "role": role,
        }
        if records is not None:
            entry["records"] = int(records)
        self.payload["reads"].append(entry)

    def save(self) -> None:
        counts = Counter((item["role"], item["file_type"]) for item in self.payload["reads"])
        self.payload.update(
            {
                "read_events": len(self.payload["reads"]),
                "counts_by_role_and_type": {
                    f"{role}:{file_type}": count
                    for (role, file_type), count in sorted(counts.items())
                },
                "pcap_files_opened": 0,
                "known_test_opened": 0,
                "unknown_test_opened": 0,
                "known_test_mu_generated": False,
                "unknown_mu_generated": False,
                "unknown_inference_executed": False,
                "status": "PASS",
            }
        )
        write_json(self.path, self.payload)


def conservative_empirical_threshold(scores: np.ndarray, target: float = 0.95) -> tuple[float, float]:
    """Choose closest empirical acceptance; exact ties choose the higher threshold."""

    values = np.asarray(scores, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "threshold scores must be finite 1-D values")
    require(0.0 < target < 1.0, "target acceptance must lie in (0,1)")
    unique, first_indices = np.unique(np.sort(values), return_index=True)
    acceptances = (len(values) - first_indices) / len(values)
    distances = np.abs(acceptances - target)
    best_distance = distances.min()
    candidates = np.flatnonzero(np.isclose(distances, best_distance, rtol=0.0, atol=1e-15))
    best = int(candidates[-1])
    return float(unique[best]), float(acceptances[best])


def linear_p05(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "P05 scores must be finite and non-empty")
    return float(np.quantile(values, 0.05, method="linear"))


def score_density_matrix(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = list(models)
    return names, np.column_stack([models[name].score_samples(values) for name in names])


def score_k2_matrices(models: dict[str, object], values: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(models)
    local_blocks: list[np.ndarray] = []
    class_blocks: list[np.ndarray] = []
    for name in names:
        local = np.asarray(models[name]._estimate_weighted_log_prob(values), dtype=np.float64)
        require(local.shape == (len(values), 2), f"{name}: unexpected K2 local-score shape")
        local_blocks.append(local)
        class_blocks.append(logsumexp(local, axis=1))
    return names, np.column_stack(class_blocks), np.column_stack(local_blocks)


def true_class_component_assignments(
    labels: np.ndarray, names: list[str], local_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {name: index for index, name in enumerate(names)}
    components = np.empty(len(labels), dtype=np.int64)
    assigned_scores = np.empty(len(labels), dtype=np.float64)
    assigned_posteriors = np.empty(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        require(str(label) in lookup, f"true class absent from K2 models: {label}")
        positions = np.flatnonzero(labels == label)
        class_index = lookup[str(label)]
        block = local_scores[positions, 2 * class_index : 2 * class_index + 2]
        selected = np.argmax(block, axis=1)
        norm = logsumexp(block, axis=1)
        components[positions] = selected
        assigned_scores[positions] = block[np.arange(len(positions)), selected]
        assigned_posteriors[positions] = np.exp(assigned_scores[positions] - norm)
    return components, assigned_scores, assigned_posteriors


def file_set_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def project_revision() -> str:
    return os.environ.get("STAGE8A_PROJECT_REVISION", "ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc")
