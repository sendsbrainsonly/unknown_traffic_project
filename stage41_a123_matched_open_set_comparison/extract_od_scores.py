#!/usr/bin/env python3
"""Extract per-flow Native Open-Detect KL scores from a frozen Stage41 checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from freeze_matched_protocol import ROOT, digest
from prepare_od_inputs import OFFICIAL

SIBLING = ROOT.parent.parent / "Open-Detect/reproduction"
sys.path.insert(0, str(SIBLING))
from run_ustc_reproduction import OpenDetectNet, TrafficImages, get_splits  # noqa: E402
from run_reproduction import collect_outputs  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("A-1", "A-2", "A-3"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen Open-Detect inference")
    unit = json.loads((ROOT / "matched_protocol.json").read_text())["units"][args.setting]
    with Path(unit["role_manifest"]).open(newline="", encoding="utf-8") as handle:
        roles = list(csv.DictReader(handle))
    if digest(Path(unit["role_manifest"])) != unit["role_manifest_sha256"]:
        raise RuntimeError("role manifest hash changed")
    base = ROOT / args.setting
    # The initial A-1 GPU0 attempt was interrupted when the user lowered the
    # concurrency cap. It is preserved but must never be used as a full run.
    od_run = "od_run_2gpu" if args.setting == "A-1" else "od_run"
    checkpoint_path = base / od_run / "model_best.pt"
    checkpoint_sha = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    official_known, _, _, _ = get_splits("USTC", ("A-1", "A-2", "A-3").index(args.setting))
    if checkpoint["n_classes"] != len(official_known) or checkpoint["latent_dim"] != 128:
        raise RuntimeError("checkpoint class/latent count mismatch")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model = OpenDetectNet("resnet18", 1, 128, len(official_known), 1, 1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inverse = {value: key for key, value in OFFICIAL.items()}
    audit = {"status": "PASS", "setting": args.setting, "checkpoint_sha256": checkpoint_sha,
             "role_manifest_sha256": unit["role_manifest_sha256"], "roles": {}}
    for part, source_roles in (("validation", {"known_validation"}),
                               ("test", {"known_test", "unknown_test"})):
        with np.load(base / "od_inputs" / f"ustc_{part}.npz", allow_pickle=False) as data:
            x, y = data["data"], data["target"]
        ids = np.load(base / "od_inputs" / f"ustc_{part}_flow_ids.npy", allow_pickle=False).astype(str)
        selected = sorted((r for r in roles if r["role"] in source_roles), key=lambda r: r["flow_id"])
        if ids.tolist() != [r["flow_id"] for r in selected]:
            raise RuntimeError(f"{part}: flow ID mismatch")
        if any(int(y[i]) != OFFICIAL[r["class_name"]] for i, r in enumerate(selected)):
            raise RuntimeError(f"{part}: label mismatch")
        ds = TrafficImages(x, y, range(20), train=False, reindex=False)
        labels, scores, predictions = collect_outputs(
            model, DataLoader(ds, batch_size=256, shuffle=False, num_workers=0), device)
        if len(scores) != len(selected) or not np.isfinite(scores).all() or not np.array_equal(labels, y):
            raise RuntimeError(f"{part}: Native score extraction failed")
        output = []
        for i, row in enumerate(selected):
            output.append({"flow_id": row["flow_id"], "class_name": row["class_name"],
                           "role": row["role"], "score_od_native": float(scores[i]),
                           "predicted_known": inverse[official_known[int(predictions[i])]]})
        filename = "od_validation_scores.csv" if part == "validation" else "od_sample_scores.csv"
        write_csv(base / filename, output)
        audit["roles"][part] = len(output)
    if digest(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("checkpoint hash changed during frozen inference")
    (base / "od_score_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
