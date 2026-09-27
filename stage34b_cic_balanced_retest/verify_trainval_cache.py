#!/usr/bin/env python3
"""Read-only full-array parity check for balanced Train/Val packet inputs."""
from __future__ import annotations

import json

import numpy as np

from prepare_balanced import ROOT, SOURCE, MANIFEST, sha, verified_rows


def main() -> None:
    selected = verified_rows()
    old = SOURCE / "cicids2017/input_caches/ustc/A-2"
    new = ROOT / "cicids2017/input_caches/ustc/A-2"
    old_tf, new_tf = old / "tf_fig", new / "tf_fig"
    old_ids = np.load(old_tf / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    new_ids = np.load(new_tf / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    expected = sorted(r["flow_id"] for r in selected if r["split"] in ("train", "validation"))
    if new_ids != expected or len(old_ids) != len(set(old_ids)):
        raise RuntimeError("TF/FIG flow IDs differ")
    old_pos = {uid: i for i, uid in enumerate(old_ids)}
    for name in ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask"):
        source = np.load(old_tf / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        target = np.load(new_tf / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, len(new_ids), 512):
            batch = new_ids[start:start + 512]
            if not np.array_equal(target[start:start + len(batch)],
                                  source[[old_pos[uid] for uid in batch]]):
                raise RuntimeError(f"TF/FIG packet array differs: {name}/{start}")
    old_mfr, new_mfr = old / "yatc_mfr", new / "yatc_mfr"
    for role in ("known_train", "known_validation"):
        src_ids = np.load(old_mfr / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        dst_ids = np.load(new_mfr / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        old_pos = {uid: i for i, uid in enumerate(src_ids)}
        source = np.load(old_mfr / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
        target = np.load(new_mfr / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, len(dst_ids), 512):
            batch = dst_ids[start:start + 512]
            if not np.array_equal(target[start:start + len(batch)],
                                  source[[old_pos[uid] for uid in batch]]):
                raise RuntimeError(f"YaTC packet array differs: {role}/{start}")
    payload = {"status": "PASS", "flow_ids": len(new_ids),
               "tf_fig_arrays_bitwise_equal": ["token_ids", "segments", "fig_x", "fig_adj", "fig_mask"],
               "yatc_arrays_bitwise_equal": ["known_train_mfr", "known_validation_mfr"],
               "balanced_manifest_sha256": sha(MANIFEST), "unknown_or_test_features_read": 0}
    (ROOT / "input_parity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
