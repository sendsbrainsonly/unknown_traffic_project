#!/usr/bin/env python3
"""Recover YaTC MFR from frozen USTC A-2 Known Train/Val Stage-0 flow PKLs."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from preflight import OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen

sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table" / "scripts"))
from build_yatc_mfr import author_mfr_packet  # noqa: E402

ALIGN = PROJECT / "opendetect_ustc_encoder_audit" / "outputs" / "input_alignment_manifest.csv"
CONFIG = PROJECT / "stage3_unknown_utility" / "outputs" / "A-2" / "training_config.json"
FLOWS = PROJECT / "data" / "flows"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    if args.phase == "test":
        ensure_all_heads_frozen()
    result = OUT / "input_caches" / "ustc" / "A-2" / ("yatc_mfr" if args.phase == "trainval" else "yatc_mfr_test")
    if result.exists():
        raise RuntimeError(f"refusing to overwrite cache: {result}")
    result.mkdir(parents=True)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    known = set(json.loads(CONFIG.read_text())["known_classes"])
    with ALIGN.open(newline="", encoding="utf-8") as handle:
        splits = ("train", "val") if args.phase == "trainval" else ("test",)
        target = {r["flow_id"]:r for r in csv.DictReader(handle)
                  if r["class_name"] in known and r["original_split"] in splits}
    if not target or any(r["input_valid"] != "True" for r in target.values()):
        raise RuntimeError("missing or invalid frozen A-2 Known alignment")
    role_name = {"train":"known_train","val":"known_validation","test":"known_test"}
    arrays, positions = {}, {}
    for split in splits:
        ids = sorted(uid for uid,row in target.items() if row["original_split"] == split)
        positions[split] = {uid:i for i,uid in enumerate(ids)}
        np.save(result/f"{role_name[split]}_flow_ids.npy",np.asarray(ids,dtype="U80"),allow_pickle=False)
        arrays[split] = np.lib.format.open_memmap(result/f"{role_name[split]}_mfr.npy",mode="w+",
                                                 dtype=np.uint8,shape=(len(ids),40,40))
        arrays[split][:] = 0
    seen = set()
    pkl_audit = []
    for path in sorted(FLOWS.glob("*.pkl")):
        pieces = path.stem.split("__")
        if len(pieces) < 2 or pieces[1] not in known:
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        count = 0
        for uid,packets in payload["packets"].items():
            if uid not in target:
                continue
            if uid in seen or target[uid]["class_name"] != payload["class_name"]:
                raise RuntimeError(f"duplicate or mislabeled USTC flow: {uid}")
            split = target[uid]["original_split"]
            position = positions[split][uid]
            if not packets:
                raise RuntimeError(f"empty Stage-0 flow: {uid}")
            for slot,packet in enumerate(packets[:5]):
                view = np.frombuffer(author_mfr_packet(bytes(packet[4])),dtype=np.uint8)
                arrays[split][position].reshape(-1)[slot*320:(slot+1)*320] = view
            seen.add(uid)
            count += 1
        pkl_audit.append({"path":str(path),"class":str(payload["class_name"]),
                          "flows_in_pkl":len(payload["packets"]),"known_trainval_matched":count})
        print(json.dumps({"pkl":path.name,"matched":count,"total_matched":len(seen),
                          "target":len(target)}),flush=True)
        del payload
        gc.collect()
    missing = sorted(set(target)-seen)
    if missing:
        (result/"failure.json").write_text(json.dumps({"missing_count":len(missing),
                                                        "sample":missing[:80]},indent=2))
        raise RuntimeError(f"unable to recover {len(missing)} A-2 Known flow MFR values")
    for array in arrays.values():
        array.flush()
    audit = {"status":"PASS","dataset":"ustc","protocol":"A-2",
             "roles":{role_name[role]:len(pos) for role,pos in positions.items()},
             "alignment_sha256":sha(ALIGN),"config_sha256":sha(CONFIG),
             "pkl_audit":pkl_audit,"mfr_method":"YaTC first-five 40x40 raw-packet rule",
             "unknown_class_files_loaded":0,
             "known_test_features_transformed":len(target) if args.phase == "test" else 0,
             "caveat":"pickle.load materializes full Known-class PKL; only selected phase packets are transformed or stored"}
    (result/"cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":"PASS","known_trainval_flows":len(target),
                      "pkls_read":len(pkl_audit)}),flush=True)


if __name__ == "__main__":
    main()
