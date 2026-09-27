#!/usr/bin/env python3
"""Frozen three-view Known Train/Validation preflight for Stage30."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
sys.path.insert(0, str(PROJECT))
from stage28_er_cmgi_entropy_fusion.preflight import (  # noqa: E402
    S20, S27, EXPECTED_S20, load_pair as load_aligned_pair,
    sha256, sources as upstream_sources,
)

DATASETS = ("iscx_vpn", "iscx_tor")
ENCODER_SEEDS = (2022, 2023)
ROLES = ("known_train", "known_validation")


def sources(dataset: str, seed: int) -> dict[str, Path]:
    result = upstream_sources(dataset, seed)
    result["stage28_alignment_code"] = PROJECT / "stage28_er_cmgi_entropy_fusion" / "preflight.py"
    result["stage27_run_metrics"] = S27 / "runs" / dataset / f"seed{seed}" / "run_metrics.csv"
    return result


def known_labels() -> dict[str, dict[str, tuple[str, str]]]:
    result = {dataset: {} for dataset in DATASETS}
    with S20.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] not in result or row["closed_role"] not in ROLES:
                continue
            fid = row["flow_id"]
            if fid in result[row["dataset"]]:
                raise RuntimeError("duplicate Known flow ID")
            result[row["dataset"]][fid] = (row["service_label"], row["closed_role"])
    return result


def load_three(dataset: str, seed: int, mapping: dict[str, tuple[str, str]]) -> dict:
    pair = load_aligned_pair(dataset, seed, mapping)
    roles = {}
    for role in ROLES:
        old = pair["roles"][role]
        semantic = old["semantic"]
        graph = old["behavioral"]
        if semantic.shape[1] != 960 or graph.shape[1] != 128:
            raise RuntimeError("unexpected frozen feature dimensions")
        roles[role] = {"flow_ids": old["flow_ids"], "labels": old["labels"],
                       "trafficformer": semantic[:, :768].copy(),
                       "graph": graph.copy(), "yatc": semantic[:, 768:].copy()}
        for name, dim in (("trafficformer", 768), ("graph", 128), ("yatc", 192)):
            values = roles[role][name]
            if values.shape != (len(old["flow_ids"]), dim) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid {name} view: {dataset}/{seed}/{role}")
    return {"services": pair["services"], "roles": roles}


def main() -> None:
    if (OUT / "frozen_input_hashes_before.json").exists() or (OUT / "input_audit.json").exists():
        raise RuntimeError("preflight evidence already exists")
    if sha256(S20) != EXPECTED_S20:
        raise RuntimeError("Stage20 manifest hash changed")
    mapping = known_labels()
    hashes, audit = {}, []
    for dataset in DATASETS:
        for seed in ENCODER_SEEDS:
            key = f"{dataset}/{seed}"
            hashes[key] = {name: sha256(path) for name, path in sources(dataset, seed).items()}
            pair = load_three(dataset, seed, mapping[dataset])
            for role in ROLES:
                values = pair["roles"][role]
                audit.append({"dataset": dataset, "encoder_seed": seed, "role": role,
                              "samples": len(values["flow_ids"]), "classes": len(pair["services"]),
                              "views": {name: int(values[name].shape[1]) for name in
                                        ("trafficformer", "graph", "yatc")},
                              "class_counts": {name: int(np.sum(values["labels"] == i))
                                               for i, name in enumerate(pair["services"])}})
    output = {"status": "PASS", "frozen_pairs": 4, "known_role_arrays": len(audit),
              "independent_views": 3, "test_feature_values_loaded": 0,
              "unknown_values_loaded": 0, "rows": audit}
    (OUT / "frozen_input_hashes_before.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
