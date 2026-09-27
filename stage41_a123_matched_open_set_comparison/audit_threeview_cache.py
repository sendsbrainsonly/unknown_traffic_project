#!/usr/bin/env python3
"""Check A-1/A-3 Known Train/Val packet inputs equal Stage34 A-2 on shared flows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from freeze_matched_protocol import PROJECT, ROOT


def check_pair(left: Path, right: Path, id_name: str, array_names: list[str]) -> int:
    ids_l = np.load(left / id_name, allow_pickle=False).astype(str)
    ids_r = np.load(right / id_name, allow_pickle=False).astype(str)
    index_l = {uid: i for i, uid in enumerate(ids_l)}
    index_r = {uid: i for i, uid in enumerate(ids_r)}
    shared = sorted(set(index_l) & set(index_r))
    if not shared:
        raise RuntimeError(f"no shared flow IDs: {left} / {right}")
    pos_l = np.asarray([index_l[uid] for uid in shared], dtype=np.int64)
    pos_r = np.asarray([index_r[uid] for uid in shared], dtype=np.int64)
    for name in array_names:
        a = np.load(left / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        b = np.load(right / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if a.shape[1:] != b.shape[1:]:
            raise RuntimeError(f"feature shape drift: {name}")
        for offset in range(0, len(shared), 4096):
            sl = slice(offset, offset + 4096)
            if not np.array_equal(a[pos_l[sl]], b[pos_r[sl]]):
                raise RuntimeError(f"input parity failed: {name}, shared offset {offset}")
    return len(shared)


def main() -> None:
    old = PROJECT / "stage34_ustc_cic_closed_set/input_caches/ustc/A-2"
    results = {}
    for setting in ("A-1", "A-3"):
        new = ROOT / setting / "input_caches/ustc/A-2"
        tf = check_pair(new / "tf_fig", old / "tf_fig", "flow_ids.npy",
                        ["token_ids", "segments", "fig_x", "fig_adj", "fig_mask"])
        mfr = {}
        for role in ("known_train", "known_validation"):
            mfr[role] = check_pair(new / "yatc_mfr", old / "yatc_mfr",
                                    f"{role}_flow_ids.npy", [f"{role}_mfr"])
        if tf != sum(mfr.values()):
            raise RuntimeError(f"{setting}: TF/FIG and MFR parity counts differ")
        results[setting] = {"shared_flows": tf, "mfr_role_counts": mfr,
                            "arrays_bitwise_equal": True}
    payload = {"status": "PASS", "test_features_read": 0,
               "unknown_features_read": 0, "results": results}
    path = ROOT / "input_parity_audit.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
