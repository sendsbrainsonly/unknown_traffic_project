from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE12 = PROJECT / "stage12_dual_external_validation"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
STAGE14C_INPUT = PROJECT / "stage14c_vnat_encoder_training"
STAGE14D = PROJECT / "stage14d_vnat_frozen_open_set_evaluation"
STAGE15R = PROJECT / "stage15r_representation_bottleneck_audit"
STAGE15F0 = PROJECT / "stage15f_literature_guided_feature_benchmark"
STAGE3 = PROJECT / "stage3_unknown_utility"
USTC_AUDIT = PROJECT / "opendetect_ustc_encoder_audit"
CONFIG_PATH = ROOT / "config.json"


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"empty rows need explicit fields: {path}")
    columns = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def protocol_rows(dataset: str, protocol_id: str) -> tuple[list[str], list[dict], list[dict]]:
    if dataset in {"iscx_vpn", "iscx_tor"}:
        setting = "medium"
        protocol = read_json(STAGE12 / "artifacts" / dataset / "protocol" / setting / "protocol.json")
        classes = list(protocol["known_classes"])
        local = {name: i for i, name in enumerate(classes)}
        rows = [row for row in read_csv(STAGE12 / "protocol" / dataset / "split_manifest.csv") if row["setting"] == setting]
        def convert(role: str):
            return [{"sample_id": row["flow_id_sha256"], "class_name": row["canonical_class"], "label": local[row["canonical_class"]]} for row in rows if row["role"] == role]
        return classes, convert("known_train"), convert("known_validation")
    if dataset == "vnat":
        base = STAGE14C_INPUT / "runs" / protocol_id / "inputs"
        label_map = read_json(base / "label_map.json")
        classes = [label_map["local_to_class"][str(i)] for i in range(len(label_map["local_to_class"]))]
        rows = read_csv(base / "input_manifest.csv")
        def convert(role: str):
            return [{"sample_id": row["flow_uid"], "class_name": row["application"], "label": int(row["local_label"])} for row in rows if row["split"] == role]
        return classes, convert("train"), convert("validation")
    if dataset == "ustc":
        config = read_json(STAGE3 / "outputs" / "A-2" / "training_config.json")
        classes = list(config["known_classes"])
        class_to_id = read_json(PROJECT / "data" / "trafficformer_input" / "compatible_min1" / "label_map.json")["class_to_id"]
        original_to_local = {int(class_to_id[name]): i for i, name in enumerate(classes)}
        outputs = []
        for role in ("train", "val"):
            ids = np.load(PROJECT / "outputs" / "stage1" / "modelA" / "embeddings" / f"flow_ids_{role}.npy", allow_pickle=False)
            labels = np.load(PROJECT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{role}.npy", allow_pickle=False)
            keep = np.isin(labels, list(original_to_local))
            outputs.append([{"sample_id": str(uid), "class_name": classes[original_to_local[int(label)]], "label": original_to_local[int(label)]} for uid, label in zip(ids[keep], labels[keep])])
        return classes, outputs[0], outputs[1]
    raise KeyError(dataset)


def pilot_specs() -> list[dict]:
    return list(read_json(CONFIG_PATH)["pilot_protocols"])
