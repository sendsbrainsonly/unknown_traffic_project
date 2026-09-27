#!/usr/bin/env python3
"""Stage29 frozen native-E3 two-branch Known-only preflight."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
S20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
S22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
DATASETS = ("iscx_vpn", "iscx_tor")
ENCODER_SEEDS = (2022, 2023)
ROLES = ("known_train", "known_validation")
EXPECTED_S20 = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def paths(dataset: str, seed: int) -> dict[str, Path]:
    run = S22 / "runs" / dataset / f"seed{seed}"
    return {"stage20_manifest": S20, "stage22_config": S22 / "config.json",
            "e3_representations": run / "representations.npz",
            "e3_checkpoint": run / "E3_model_best.pt",
            "stage22_run_manifest": run / "run_manifest.json"}


def labels() -> dict[str, dict[str, tuple[str, str]]]:
    out = {name: {} for name in DATASETS}
    with S20.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset = row["dataset"]
            if dataset in out:
                fid = row["flow_id"]
                if fid in out[dataset]:
                    raise RuntimeError(f"duplicate Stage20 flow ID: {dataset}/{fid}")
                out[dataset][fid] = (row["service_label"], row["closed_role"])
    return out


def load_pair(dataset: str, seed: int, mapping: dict[str, tuple[str, str]]) -> dict:
    config = json.loads((S22 / "config.json").read_text())
    services = list(config["datasets"][dataset]["services"])
    run_manifest = json.loads(paths(dataset, seed)["stage22_run_manifest"].read_text())
    if run_manifest["status"] != "SUCCESS" or run_manifest["dataset"] != dataset:
        raise RuntimeError("Stage22 source run is not complete")
    roles = {}
    with np.load(paths(dataset, seed)["e3_representations"], allow_pickle=False) as bundle:
        train_raw_t = bundle["known_train_e1_z"].astype(np.float32)
        train_raw_g = bundle["known_train_e2_z"].astype(np.float32)
        t_mean, g_mean = train_raw_t.mean(0), train_raw_g.mean(0)
        t_std, g_std = train_raw_t.std(0), train_raw_g.std(0)
        t_std[t_std < 1e-8] = 1.0
        g_std[g_std < 1e-8] = 1.0
        for role in ROLES:
            ids = bundle[f"{role}_flow_ids"].astype(str)
            raw_t = bundle[f"{role}_e1_z"].astype(np.float32)
            raw_g = bundle[f"{role}_e2_z"].astype(np.float32)
            fused = bundle[f"{role}_e3_z"].astype(np.float32)
            prediction = bundle[f"{role}_e3_pred"].astype(np.int64)
            if raw_t.shape != (len(ids), 768) or raw_g.shape != (len(ids), 128) or \
               fused.shape != (len(ids), 896) or prediction.shape != (len(ids),):
                raise RuntimeError(f"{dataset}/{seed}/{role}: shape mismatch")
            trafficformer, graph = fused[:, :768], fused[:, 768:]
            expected = np.concatenate(((raw_t - t_mean) / t_std, (raw_g - g_mean) / g_std), axis=1)
            if not np.allclose(expected, fused, atol=2e-5, rtol=2e-5):
                raise RuntimeError("E3 is not the Known-Train standardized 768+128 concat")
            if len(set(ids)) != len(ids) or not np.isfinite(fused).all():
                raise RuntimeError("duplicate flow ID or nonfinite feature")
            truth = []
            for fid in ids:
                if fid not in mapping or mapping[fid][1] != role:
                    raise RuntimeError(f"Stage20 role mismatch: {fid}")
                if mapping[fid][0] not in services:
                    raise RuntimeError(f"Stage20 service not in Stage22 class order: {fid}")
                truth.append(services.index(mapping[fid][0]))
            if prediction.min() < 0 or prediction.max() >= len(services):
                raise RuntimeError("E3 original prediction invalid")
            roles[role] = {"flow_ids": ids, "trafficformer": trafficformer,
                           "graph": graph, "labels": np.asarray(truth, np.int64),
                           "e3_original_pred": prediction}
    if set(roles["known_train"]["flow_ids"]) & set(roles["known_validation"]["flow_ids"]):
        raise RuntimeError("Known Train/Val overlap")
    for role in ROLES:
        if set(roles[role]["labels"]) != set(range(len(services))):
            raise RuntimeError(f"class missing in {role}")
    return {"services": services, "roles": roles}


def main() -> None:
    if (OUT / "input_audit.json").exists() or (OUT / "frozen_input_hashes_before.json").exists():
        raise RuntimeError("preflight evidence already exists")
    if sha(S20) != EXPECTED_S20:
        raise RuntimeError("Stage20 manifest hash changed")
    mapping = labels()
    checks, audit = {}, []
    for dataset in DATASETS:
        for seed in ENCODER_SEEDS:
            key = f"{dataset}/{seed}"
            checks[key] = {name: sha(path) for name, path in paths(dataset, seed).items()}
            pair = load_pair(dataset, seed, mapping[dataset])
            for role in ROLES:
                item = pair["roles"][role]
                audit.append({"dataset": dataset, "encoder_seed": seed, "role": role,
                              "flows": len(item["flow_ids"]), "classes": len(pair["services"]),
                              "z_t_dim": item["trafficformer"].shape[1], "z_g_dim": item["graph"].shape[1],
                              "class_counts": {name: int(np.sum(item["labels"] == i))
                                               for i, name in enumerate(pair["services"])}})
    result = {"status": "PASS", "pairs": len(checks), "known_role_arrays": len(audit),
              "native_e3_branches": 2, "yatc_usage": 0,
              "known_test_values_loaded": 0, "unknown_values_loaded": 0, "rows": audit}
    (OUT / "frozen_input_hashes_before.json").write_text(json.dumps(checks, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
