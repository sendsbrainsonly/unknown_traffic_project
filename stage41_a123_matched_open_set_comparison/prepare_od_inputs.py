#!/usr/bin/env python3
"""Materialize exact matched-flow Open-Detect image arrays, without refitting data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from freeze_matched_protocol import PROJECT, ROOT, digest

IMAGES = PROJECT / "opendetect_ustc_encoder_audit/artifacts"
LABELS = PROJECT / "outputs/stage1/modelA/embeddings"
LOCAL_MAP = PROJECT / "data/fig_graph/all_flows/label_map.json"
OFFICIAL = {
    "Gmail": 0, "FTP": 1, "Nsis-ay": 2, "Facetime": 3, "Weibo": 4,
    "Cridex": 5, "Zeus": 6, "SMB": 7, "BitTorrent": 8,
    "WorldOfWarcraft": 9, "Shifu": 10, "Outlook": 11, "Virut": 12,
    "Geodo": 13, "MySQL": 14, "Htbot": 15, "Tinba": 16,
    "Skype": 17, "Miuref": 18, "Neris": 19,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("A-1", "A-2", "A-3"), required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "matched_protocol.json").read_text())
    unit = protocol["units"][args.setting]
    path = Path(unit["role_manifest"])
    if digest(path) != unit["role_manifest_sha256"]:
        raise RuntimeError("frozen role manifest hash drift")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    local_map = json.loads(LOCAL_MAP.read_text())["class_to_id"]
    out = ROOT / args.setting / "od_inputs"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    arrays = {role: np.load(IMAGES / f"{role}_images.npy", mmap_mode="r", allow_pickle=False)
              for role in ("train", "val", "test")}
    labels = {role: np.load(LABELS / f"labels_{role}.npy", mmap_mode="r", allow_pickle=False)
              for role in arrays}
    audit = {"status": "PASS", "setting": args.setting,
             "role_manifest_sha256": unit["role_manifest_sha256"],
             "unknown_train_count": 0, "unknown_validation_count": 0,
             "roles": {}}
    for suffix, source_roles in (("train", {"known_train"}),
                                 ("validation", {"known_validation"}),
                                 ("test", {"known_test", "unknown_test"})):
        selected = sorted((r for r in rows if r["role"] in source_roles), key=lambda r: r["flow_id"])
        data = np.empty((len(selected), 32, 32), dtype=np.uint8)
        targets = np.empty(len(selected), dtype=np.int64)
        for i, row in enumerate(selected):
            split, raw_index = row["opendetect_input_id"].split(":", 1)
            index = int(raw_index)
            if split != row["source_split"] or index >= len(arrays[split]):
                raise RuntimeError(f"misaligned Open-Detect image ID: {row['flow_id']}")
            if int(labels[split][index]) != int(local_map[row["class_name"]]):
                raise RuntimeError(f"image/flow label mismatch: {row['flow_id']}")
            data[i] = arrays[split][index]
            targets[i] = OFFICIAL[row["class_name"]]
        name = f"ustc_{suffix}.npz"
        np.savez(out / name, data=data, target=targets)
        np.save(out / f"ustc_{suffix}_flow_ids.npy",
                np.asarray([r["flow_id"] for r in selected], dtype="U80"), allow_pickle=False)
        audit["roles"][suffix] = {"samples": len(selected), "npz_sha256": digest(out / name),
                                  "flow_ids_sha256": digest(out / f"ustc_{suffix}_flow_ids.npy")}
    (out / "input_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "setting": args.setting,
                      "counts": {k: v["samples"] for k, v in audit["roles"].items()}}), flush=True)


if __name__ == "__main__":
    main()
