"""Frozen Stage32 inputs; no split generation or Unknown-feature access."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
S22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
S25 = PROJECT / "stage25_semantic_coarse_closed_set_development"
S27 = PROJECT / "stage27_yatc_feature_level_fusion"
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
VFREEZE = PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"
F2_INPUT = PROJECT / "stage14c5_feature_representation_audit" / "runs" / "f2" / "medium_seed2025" / "inputs"
F2_RUN = PROJECT / "stage14c6_cleaned_pipeline_full15_closed_set" / "runs" / "medium_seed2025"
DATASETS = ("iscx_vpn", "iscx_tor", "vnat")
VIEWS = ("trafficformer", "graph", "yatc")
DIMS = (768, 128, 192)
TRAINING_SEED = 2022
VNAT_PROTOCOL = "medium_seed2025"
VNAT_MAP = {
    "netflix": "Streaming", "vimeo": "Streaming", "youtube": "Streaming",
    "zoiper": "VoIP", "skype": "Chat", "rdp": "Command & Control",
    "ssh": "Command & Control", "rsync": "File Transfer",
    "scp": "File Transfer", "sftp": "File Transfer",
}
ISCOURSE = {"Chat": "Communication", "Email": "Communication", "VoIP": "Communication"}
COARSE_CLASSES = {
    "iscx_vpn": ("Communication", "File-Transfer", "P2P", "Streaming"),
    "iscx_tor": ("Browsing", "Communication", "File-Transfer", "P2P", "Streaming"),
    "vnat": ("Chat", "Command & Control", "File Transfer", "Streaming"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def coarse(dataset: str, fine: str) -> str:
    if dataset == "vnat":
        return VNAT_MAP[fine]
    return ISCOURSE.get(fine, fine)


@lru_cache(maxsize=1)
def stage30():
    path = PROJECT / "stage30_three_view_entropy_fusion" / "preflight.py"
    spec = importlib.util.spec_from_file_location("stage30_frozen_preflight", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def iscx_pair(dataset: str):
    if dataset not in ("iscx_vpn", "iscx_tor"):
        raise ValueError(dataset)
    source = stage30()
    if source.sha256(S20) != source.EXPECTED_S20:
        raise RuntimeError("Stage20 frozen manifest hash changed")
    return source.load_three(dataset, TRAINING_SEED, source.known_labels()[dataset])


@lru_cache(maxsize=1)
def vnat_rows():
    with VFREEZE.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["protocol_id"] == VNAT_PROTOCOL]
    if not rows:
        raise RuntimeError("missing frozen VNAT protocol")
    return rows


def expected_rows(dataset: str, role: str) -> list[tuple[str, str]]:
    if role not in ("known_train", "known_validation", "known_test"):
        raise ValueError(role)
    if dataset == "vnat":
        split = {"known_train": "train", "known_validation": "validation", "known_test": "test"}[role]
        rows = sorted((r["flow_uid"], r["application"]) for r in vnat_rows()
                      if r["class_role"] == "known" and r["split"] == split)
    else:
        with S20.open(newline="", encoding="utf-8") as f:
            rows = sorted((r["flow_id"], r["service_label"]) for r in csv.DictReader(f)
                          if r["dataset"] == dataset and r["closed_role"] == role)
    if not rows or len(rows) != len({uid for uid, _ in rows}):
        raise RuntimeError(f"missing/duplicate frozen {dataset}/{role} IDs")
    if any(coarse(dataset, fine) not in COARSE_CLASSES[dataset] for _, fine in rows):
        raise RuntimeError(f"unmapped coarse labels {dataset}/{role}")
    return rows


def _value_bundle(dataset: str, role: str, rows: list[tuple[str, str]], features: dict) -> dict:
    n = len(rows)
    result = {
        "flow_ids": np.asarray([uid for uid, _ in rows], dtype="U80"),
        "fine_labels": np.asarray([fine for _, fine in rows], dtype="U40"),
        "labels": np.asarray([COARSE_CLASSES[dataset].index(coarse(dataset, fine)) for _, fine in rows], dtype=np.int64),
    }
    for name, dim in zip(VIEWS, DIMS, strict=True):
        arr = np.asarray(features[name], dtype=np.float32)
        if arr.shape != (n, dim) or not np.isfinite(arr).all():
            raise RuntimeError(f"invalid {dataset}/{role}/{name} shape or finite values: {arr.shape}")
        result[name] = arr
    if set(result["labels"]) != set(range(len(COARSE_CLASSES[dataset]))):
        raise RuntimeError(f"empty coarse class in {dataset}/{role}")
    return result


def load_known(dataset: str, role: str) -> dict:
    """Only Known Train/Validation. Test feature values are a separate locked API."""
    if role not in ("known_train", "known_validation"):
        raise ValueError("load_known cannot read Test")
    rows = expected_rows(dataset, role)
    ids = [uid for uid, _ in rows]
    if dataset != "vnat":
        pair = iscx_pair(dataset)["roles"][role]
        if pair["flow_ids"].astype(str).tolist() != ids:
            raise RuntimeError(f"Stage30 {dataset}/{role} flow-ID mismatch")
        return _value_bundle(dataset, role, rows, {name: pair[name] for name in VIEWS})
    root = S31 / "runs" / "vnat" / VNAT_PROTOCOL
    features = {}
    for name in VIEWS:
        branch = root / name
        if not (branch / "SUCCESS").is_file():
            raise RuntimeError(f"incomplete VNAT branch: {branch}")
        meta = json.loads((branch / "metrics.json").read_text())
        if sha256(branch / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"VNAT branch checkpoint hash mismatch: {name}")
        got_ids = np.load(branch / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if got_ids != ids:
            raise RuntimeError(f"VNAT {name}/{role} flow-ID mismatch")
        features[name] = np.load(branch / f"{role}_features.npy", allow_pickle=False)
    return _value_bundle(dataset, role, rows, features)


def freeze_sources() -> dict[str, str]:
    paths = {"stage20_manifest": S20, "vnat_split_manifest": VFREEZE,
             "stage25_remap_predictions": S25 / "remap_predictions.csv",
             "stage25_coarse_head_predictions": S25 / "coarse_head_predictions.csv",
             "vnat_f2_val_predictions": F2_RUN / "validation_predictions.npy",
             "vnat_f2_input_manifest": F2_INPUT / "input_manifest.csv"}
    for dataset in ("iscx_vpn", "iscx_tor"):
        paths[f"{dataset}_stage22_representations"] = S22 / "runs" / dataset / "seed2022" / "representations.npz"
        paths[f"{dataset}_stage27_yatc_features"] = S27 / "runs" / dataset / "seed2022" / "features.npz"
    root = S31 / "runs" / "vnat" / VNAT_PROTOCOL
    for name in VIEWS:
        for role in ("known_train", "known_validation"):
            paths[f"vnat_{name}_{role}_features"] = root / name / f"{role}_features.npy"
        paths[f"vnat_{name}_checkpoint"] = root / name / "model_best.pt"
    return {name: sha256(path) for name, path in paths.items()}
