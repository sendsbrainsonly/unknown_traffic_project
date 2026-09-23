#!/usr/bin/env python3
"""Shared frozen definitions for Stage 15F-DQ-4."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
STAGE15F = PROJECT / "stage15f_literature_guided_feature_benchmark"
DQ = STAGE15F / "stage15f_dq_performance_gap_attribution"
DQ3R = STAGE15F / "stage15f_dq3r_service_label_reconstruction"
DQ3F = STAGE15F / "stage15f_dq3f_fine_service_matched_benchmark"
TF = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/TrafficFormer")
TF_DATA = TF / "reproduction/datasets/iscxvpn_trafficformer"

sys.path.insert(0, str(DQ3F / "scripts"))
from dq3f_common import (  # noqa: E402
    SERVICES, array_digest, encoded_targets, hash_protected_inputs as hash_dq3f_inputs,
    load_matched_split, read_json, sha256_file, write_csv, write_json,
)

SUBSETS = ("A", "B", "C_PARENT", "C_FINAL", "D")
TRAINED_MODELS = {"M-A": "A", "M-B": "B", "M-C": "C_FINAL", "M-D": "D"}
SEEDS = (2022, 2023, 2024)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_paths(paths: Iterable[Path]) -> dict:
    resolved = sorted({path.resolve() for path in paths})
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected input missing: {missing}")
    return {
        "file_count": len(resolved),
        "files": {str(path): sha256_file(path) for path in resolved},
    }


def protected_paths() -> list[Path]:
    paths = [
        DQ / "cross_project_flow_manifest.csv",
        DQ / "flow_reconstruction_audit.csv",
        DQ / "scripts/run_dq0_dq2_audit.py",
        DQ3R / "service_label_provenance.csv",
        DQ3R / "dq3f_preregistered_config.json",
        DQ3F / "dq3f_manifest_parity.csv",
        DQ3F / "dq3f_training_configs.json",
        DQ3F / "scripts/dq3f_common.py",
        DQ3F / "scripts/train_dq3f.py",
        TF / "reproduction/scripts/prepare_iscxvpn.py",
        TF_DATA / "processing_audit.tsv",
        TF_DATA / "service/source_manifest.tsv",
        TF_DATA / "dataset_stats.json",
    ]
    for seed in SEEDS:
        run = DQ3F / "runs/service" / f"seed{seed}"
        paths.extend([
            run / "model_best.pt", run / "checkpoint.sha256", run / "config.json",
            run / "result.json", run / "validation_predictions.csv",
        ])
    return paths


def hash_protected_inputs() -> dict:
    payload = sha256_paths(protected_paths())
    payload["dq3f_transitive_inputs"] = hash_dq3f_inputs()
    return payload


def trafficformer_final_parity() -> dict:
    audit = pd.read_csv(TF_DATA / "processing_audit.tsv", sep="\t")
    final = pd.read_csv(TF_DATA / "service/source_manifest.tsv", sep="\t")
    success = audit[audit["status"].eq("success")].copy()
    keys = ["source_file", "app_label", "service_label", "packets", "bytes"]
    success_counter = Counter(map(tuple, success[keys].itertuples(index=False, name=None)))
    final_counter = Counter(map(tuple, final[keys].itertuples(index=False, name=None)))
    status_counts = {str(key): int(value) for key, value in audit["status"].value_counts().items()}
    return {
        "processing_audit_rows": int(len(audit)),
        "success_parent_rows": int(len(success)),
        "final_service_manifest_rows": int(len(final)),
        "status_counts": status_counts,
        "success_to_final_service_multiset_parity": success_counter == final_counter,
        "service_label_counts": {str(key): int(value) for key, value in final["service_label"].value_counts().items()},
        "final_rule": "All successful parents enter the Service task because every Service has >=10 parents.",
        "row_identity_limit": "The final manifest omits five-tuples/parent IDs; membership is proven as a multiset, not as a one-to-one native-session row map.",
    }


def load_mother(role: str) -> dict:
    base = load_matched_split(role)
    flow = pd.read_csv(DQ / "cross_project_flow_manifest.csv")
    if flow["flow_id"].duplicated().any():
        raise RuntimeError("cross_project_flow_manifest flow_id is not unique")
    lookup = flow.set_index("flow_id", drop=False)
    rows = []
    for index, meta in enumerate(base["metadata"]):
        flow_id = meta["flow_id"]
        if flow_id not in lookup.index:
            raise RuntimeError(f"mother flow absent from DQ manifest: {flow_id}")
        raw = lookup.loc[flow_id].to_dict()
        if raw["application"] != meta["application_label"]:
            raise RuntimeError(f"application mismatch: {flow_id}")
        if raw["service"] != meta["service_label"]:
            raise RuntimeError(f"service mismatch: {flow_id}")
        duration = max(0.0, float(raw["flow_end_time"]) - float(raw["flow_start_time"]))
        cate = str(raw["cate_match_status"])
        c_parent = bool(raw["trafficformer_eligibility"])
        enriched = {
            **meta,
            "mother_row_index": index,
            "packet_count": int(raw["packet_count"]),
            "total_bytes": int(raw["total_bytes"]),
            "captured_bytes": int(raw["captured_bytes"]),
            "duration_seconds": duration,
            "cate_match_status": cate,
            "cate_match_reason": str(raw["cate_match_reason"]),
            "cate_candidate_count": int(raw["cate_candidate_count"]),
            "cate_paper_text_eligible": bool(raw["cate_paper_text_eligible"]),
            "trafficformer_parent_flow_id": str(raw["trafficformer_parent_flow_id"]),
            "trafficformer_parent_packet_count": int(raw["trafficformer_parent_packet_count"]),
            "trafficformer_parent_captured_bytes": int(raw["trafficformer_parent_captured_bytes"]),
            "trafficformer_audit_exact_parity": bool(raw["trafficformer_audit_exact_parity"]),
            "A": True,
            "B": cate == "MATCHED",
            "C_PARENT": c_parent,
            "C_FINAL": c_parent,
            "D": cate == "MATCHED" and c_parent,
        }
        rows.append(enriched)
    if len(rows) != len(base["data"]):
        raise RuntimeError("mother metadata/array length mismatch")
    if not all(row["trafficformer_audit_exact_parity"] for row in rows):
        raise RuntimeError("TrafficFormer capture parity failed")
    return {**base, "metadata": rows}


def service_targets(metadata: list[dict]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(SERVICES)}
    return np.asarray([mapping[row["service_label"]] for row in metadata], dtype=np.int64)


def subset_data(role: str, subset: str) -> dict:
    if subset not in SUBSETS:
        raise ValueError(subset)
    mother = load_mother(role)
    keep = np.asarray([bool(row[subset]) for row in mother["metadata"]], dtype=bool)
    metadata = [row for row, selected in zip(mother["metadata"], keep) if selected]
    data = mother["data"][keep]
    target = service_targets(metadata)
    return {"role": role, "subset": subset, "data": data, "target": target, "metadata": metadata}


def flow_id_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["flow_id"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_a_reuse() -> dict:
    result = {"status": "PASS", "seeds": {}}
    for role in ("known_train", "known_validation"):
        split = subset_data(role, "A")
        expected_count = 2730 if role == "known_train" else 335
        if len(split["target"]) != expected_count:
            raise RuntimeError(f"A count changed for {role}")
    train = subset_data("known_train", "A")
    validation = subset_data("known_validation", "A")
    for seed in SEEDS:
        run = DQ3F / "runs/service" / f"seed{seed}"
        config = read_json(run / "config.json")
        prediction = pd.read_csv(run / "validation_predictions.csv")
        expected_ids = [row["flow_id"] for row in validation["metadata"]]
        checks = {
            "classes": config["classes"] == SERVICES,
            "train_count": int(config["train_samples"]) == len(train["target"]),
            "validation_count": int(config["validation_samples"]) == len(validation["target"]),
            "train_array_digest": config["train_subset_array_digest"] == array_digest(train["data"], train["target"]),
            "validation_array_digest": config["validation_subset_array_digest"] == array_digest(validation["data"], validation["target"]),
            "prediction_flow_order": prediction["flow_id"].tolist() == expected_ids,
            "checkpoint_sha256": sha256_file(run / "model_best.pt") == (run / "checkpoint.sha256").read_text().split()[0],
        }
        if not all(checks.values()):
            result["status"] = "FAIL"
        result["seeds"][str(seed)] = checks
    if result["status"] != "PASS":
        raise RuntimeError(f"M-A reuse parity failed: {result}")
    return result

