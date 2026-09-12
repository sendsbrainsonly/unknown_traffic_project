#!/usr/bin/env python3
"""Shared frozen paths, split logic, metrics, and hashing for Stage 3."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


STAGE3_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE3_ROOT.parent
PROTOCOL_ROOT = PROJECT_ROOT / "stage3_protocol"
AUDIT_ROOT = PROJECT_ROOT / "opendetect_ustc_encoder_audit"
EMBEDDING_ROOT = PROJECT_ROOT / "outputs/stage1/modelA/embeddings"
IMAGE_ROOT = AUDIT_ROOT / "artifacts"
LABEL_MAP_PATH = PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json"
EXECUTION_PLAN_PATH = PROTOCOL_ROOT / "stage3_execution_plan.json"
EXPECTED_PROTOCOL_CANONICAL_SHA256 = (
    "1fff4ed33211de0b123cf4c0ec6b203cc33be683ac315b610ffd8f0194550ae1"
)
EXPECTED_PROTOCOL_RAW_SHA256 = (
    "86718b1930c71ef6a904f16f37a199b8ceeea6600d3ceba65e7592048f54677a"
)
EXPECTED_EXECUTION_PLAN_RAW_SHA256 = (
    "197e36be2e3559c20dddf4375d117855b24cab1c1ab0f89d76f08a7f164d13de"
)
VALID_SETTINGS = ("A-1", "A-2", "A-3")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(path: Path, excluded_key: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(excluded_key)
    # The frozen protocol used jq -Sc.  Its compact encoding matches these
    # separators for this ASCII-only object; booleans/numbers are unchanged.
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()


def verify_frozen_inputs() -> None:
    protocol_path = PROTOCOL_ROOT / "stage3_protocol.json"
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_RAW_SHA256:
        raise RuntimeError("frozen stage3_protocol.json raw SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_sha256"] != EXPECTED_PROTOCOL_CANONICAL_SHA256:
        raise RuntimeError("frozen protocol embedded canonical SHA-256 mismatch")
    if canonical_json_sha256(protocol_path, "protocol_sha256") != EXPECTED_PROTOCOL_CANONICAL_SHA256:
        raise RuntimeError("frozen protocol canonical SHA-256 mismatch")
    if sha256_file(EXECUTION_PLAN_PATH) != EXPECTED_EXECUTION_PLAN_RAW_SHA256:
        raise RuntimeError("frozen execution-plan SHA-256 mismatch")
    plan = json.loads(EXECUTION_PLAN_PATH.read_text(encoding="utf-8"))
    if tuple(plan["primary_settings"]) != VALID_SETTINGS:
        raise RuntimeError("execution plan primary settings changed")
    if tuple(plan["execution_order"]) != VALID_SETTINGS:
        raise RuntimeError("execution order changed")
    if plan["stop_after_good_result"] or not plan["report_all_primary_settings"]:
        raise RuntimeError("execution completion policy changed")


def load_fold(setting: str) -> dict[str, object]:
    if setting not in VALID_SETTINGS:
        raise ValueError(f"unknown setting: {setting}")
    payload = json.loads((PROTOCOL_ROOT / f"fold_{setting.replace('-', '')}.json").read_text(encoding="utf-8"))
    if payload["scenario"] != setting:
        raise RuntimeError("fold scenario mismatch")
    if set(payload["known_classes"]) & set(payload["unknown_classes"]):
        raise RuntimeError("Known and Unknown class lists overlap")
    if len(payload["known_classes"]) != payload["known_count"]:
        raise RuntimeError("Known class count mismatch")
    if len(payload["unknown_classes"]) != payload["unknown_count"]:
        raise RuntimeError("Unknown class count mismatch")
    return payload


def load_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    class_to_id = {str(key): int(value) for key, value in payload["class_to_id"].items()}
    id_to_class = {int(key): str(value) for key, value in payload["id_to_class"].items()}
    if len(class_to_id) != 20 or len(id_to_class) != 20:
        raise RuntimeError("expected the frozen bijective 20-class label map")
    return class_to_id, id_to_class


def split_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    flow_ids = np.load(EMBEDDING_ROOT / f"flow_ids_{split}.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(EMBEDDING_ROOT / f"labels_{split}.npy", mmap_mode="r", allow_pickle=False)
    if len(flow_ids) != len(labels):
        raise RuntimeError(f"{split}: flow IDs and labels differ in length")
    return flow_ids, labels


def setting_maps(fold: dict[str, object]) -> tuple[dict[int, int], dict[int, str]]:
    class_to_id, _ = load_label_maps()
    known_classes = [str(name) for name in fold["known_classes"]]
    source_to_local = {class_to_id[name]: index for index, name in enumerate(known_classes)}
    local_to_name = {index: name for index, name in enumerate(known_classes)}
    return source_to_local, local_to_name


def setting_indices(labels: np.ndarray, fold: dict[str, object], role: str) -> np.ndarray:
    if role == "known":
        allowed = np.asarray(fold["known_current_project_ids"], dtype=np.int64)
    elif role == "unknown":
        allowed = np.asarray(fold["unknown_current_project_ids"], dtype=np.int64)
    else:
        raise ValueError("role must be known or unknown")
    return np.flatnonzero(np.isin(labels, allowed))


class KnownImageDataset(Dataset):
    """View that fetches only frozen Known indices and remaps labels locally."""

    def __init__(self, split: str, fold: dict[str, object], augment: bool) -> None:
        self.images = np.load(IMAGE_ROOT / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        _, source_labels = split_arrays(split)
        self.source_labels = source_labels
        self.source_indices = setting_indices(source_labels, fold, "known")
        source_to_local, _ = setting_maps(fold)
        lookup = np.full(20, -1, dtype=np.int64)
        for source_id, local_id in source_to_local.items():
            lookup[source_id] = local_id
        self.local_labels = lookup[np.asarray(source_labels[self.source_indices], dtype=np.int64)]
        if np.any(self.local_labels < 0):
            raise RuntimeError("Known dataset contains an unmapped label")
        self.augment = augment
        self.crop = RandomCrop(32, padding=4)
        self.flip = RandomHorizontalFlip()

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        source_index = int(self.source_indices[index])
        image = torch.from_numpy(np.asarray(self.images[source_index]).copy()).unsqueeze(0)
        if self.augment:
            image = self.flip(self.crop(image))
        return image.float().div_(255.0), int(self.local_labels[index])


def balanced_positions(dataset: KnownImageDataset, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for local_id in range(int(dataset.local_labels.max()) + 1):
        positions = np.flatnonzero(dataset.local_labels == local_id)
        if len(positions) < per_class:
            raise RuntimeError(f"local class {local_id} has fewer than {per_class} samples")
        selected.extend(rng.choice(positions, size=per_class, replace=False).tolist())
    return sorted(selected)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def string_set_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def latent_columns(frame_columns: Iterable[str]) -> list[str]:
    columns = sorted(column for column in frame_columns if column.startswith("mu_"))
    if len(columns) != 128:
        raise RuntimeError(f"expected 128 mu columns, got {len(columns)}")
    return columns


def score_density_models(models: dict[str, object], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = list(models)
    matrix = np.column_stack([models[name].score_samples(values) for name in names])
    best = np.argmax(matrix, axis=1)
    return matrix[np.arange(len(values)), best], np.asarray(names, dtype=object)[best]


def known_acceptance_threshold(scores: np.ndarray, target: float = 0.95) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target must be strictly between 0 and 1")
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and non-empty")
    return float(np.quantile(scores, 1.0 - target, method="linear"))
