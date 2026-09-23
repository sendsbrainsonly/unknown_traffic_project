from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE15F0 = ROOT.parent
PROJECT = STAGE15F0.parent
STAGE12 = PROJECT / "stage12_dual_external_validation"
STAGE14C_INPUT = PROJECT / "stage14c_vnat_encoder_training"
STAGE14D = PROJECT / "stage14d_vnat_frozen_open_set_evaluation"
STAGE15R = PROJECT / "stage15r_representation_bottleneck_audit"
STAGE15F1A = PROJECT / "stage15f1a_packet_window_benchmark"
STAGE3 = PROJECT / "stage3_unknown_utility"
CONFIG_PATH = ROOT / "config.json"
DEFINITIONS_PATH = ROOT / "feature_definitions.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"empty rows need explicit fields: {path}")
    if fields is None:
        # Reused Stage15R S12 records and newly trained Stage15F-1B records
        # intentionally retain slightly different audit metadata.  Build a
        # stable first-seen union instead of silently dropping later fields or
        # assuming that the first row defines the complete schema.
        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    columns.append(key)
                    seen.add(key)
    else:
        columns = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def pilot_specs() -> list[dict]:
    return list(read_json(CONFIG_PATH)["pilot_protocols"])


def protocol_rows(dataset: str, protocol_id: str) -> tuple[list[str], list[dict], list[dict]]:
    if dataset in {"iscx_vpn", "iscx_tor"}:
        protocol = read_json(STAGE12 / "artifacts" / dataset / "protocol" / "medium" / "protocol.json")
        classes = list(protocol["known_classes"])
        local = {name: index for index, name in enumerate(classes)}
        rows = [row for row in read_csv(STAGE12 / "protocol" / dataset / "split_manifest.csv") if row["setting"] == "medium"]

        def select(role: str) -> list[dict]:
            return [
                {"sample_id": row["flow_id_sha256"], "class_name": row["canonical_class"], "label": local[row["canonical_class"]]}
                for row in rows if row["role"] == role
            ]

        return classes, select("known_train"), select("known_validation")
    if dataset == "vnat":
        base = STAGE14C_INPUT / "runs" / protocol_id / "inputs"
        label_map = read_json(base / "label_map.json")
        classes = [label_map["local_to_class"][str(index)] for index in range(len(label_map["local_to_class"]))]
        rows = read_csv(base / "input_manifest.csv")

        def select(role: str) -> list[dict]:
            return [
                {"sample_id": row["flow_uid"], "class_name": row["application"], "label": int(row["local_label"])}
                for row in rows if row["split"] == role
            ]

        return classes, select("train"), select("validation")
    if dataset == "ustc":
        config = read_json(STAGE3 / "outputs" / "A-2" / "training_config.json")
        classes = list(config["known_classes"])
        class_to_id = read_json(PROJECT / "data" / "trafficformer_input" / "compatible_min1" / "label_map.json")["class_to_id"]
        original_to_local = {int(class_to_id[name]): index for index, name in enumerate(classes)}
        selected = []
        for role in ("train", "val"):
            ids = np.load(PROJECT / "outputs" / "stage1" / "modelA" / "embeddings" / f"flow_ids_{role}.npy", allow_pickle=False)
            labels = np.load(PROJECT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{role}.npy", allow_pickle=False)
            keep = np.isin(labels, list(original_to_local))
            selected.append([
                {"sample_id": str(uid), "class_name": classes[original_to_local[int(label)]], "label": original_to_local[int(label)]}
                for uid, label in zip(ids[keep], labels[keep])
            ])
        return classes, selected[0], selected[1]
    raise KeyError(dataset)


def feature_names() -> list[str]:
    return list(read_json(DEFINITIONS_PATH)["features"])


def group_by_feature() -> dict[str, str]:
    return {name: spec["group"] for name, spec in read_json(DEFINITIONS_PATH)["features"].items()}


def feature_set_names(feature_set: str) -> list[str]:
    definitions = read_json(DEFINITIONS_PATH)
    groups = definitions["feature_sets"][feature_set]
    return [name for name, spec in definitions["features"].items() if spec["group"] in groups]


def cache_dir(dataset: str) -> Path:
    return ROOT / "behavior_cache" / dataset


def run_dir(scenario: str, feature_set: str, dataset: str, protocol_id: str) -> Path:
    return ROOT / "runs" / scenario / feature_set / dataset / protocol_id
