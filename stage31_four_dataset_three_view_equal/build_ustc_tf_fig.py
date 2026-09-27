#!/usr/bin/env python3
"""Recover exact frozen USTC A-2 Known Train/Val TrafficFormer and FIG inputs."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preflight import OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
from src.preprocessing.fig_graph import FigFormat, build_flow_graph  # noqa: E402
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.tokenizers import BertTokenizer  # noqa: E402

ALIGN = PROJECT / "opendetect_ustc_encoder_audit" / "outputs" / "input_alignment_manifest.csv"
CONFIG = PROJECT / "stage3_unknown_utility" / "outputs" / "A-2" / "training_config.json"
FLOWS = PROJECT / "data" / "flows"
VOCAB = PROJECT / "tf_runtime" / "code" / "models" / "encryptd_vocab.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    if args.phase == "test":
        ensure_all_heads_frozen()
    result = OUT / "input_caches" / "ustc" / "A-2" / ("tf_fig" if args.phase == "trainval" else "tf_fig_test")
    if result.exists():
        raise RuntimeError(f"refusing to overwrite cache: {result}")
    result.mkdir(parents=True)
    known = set(json.loads(CONFIG.read_text())["known_classes"])
    with ALIGN.open(newline="", encoding="utf-8") as handle:
        splits = ("train", "val") if args.phase == "trainval" else ("test",)
        rows = [r for r in csv.DictReader(handle) if r["class_name"] in known and
                r["original_split"] in splits]
    targets = {row["flow_id"]: row for row in rows}
    if not targets or len(targets) != len(rows) or any(r["input_valid"] != "True" for r in rows):
        raise RuntimeError("invalid or duplicate frozen Known A-2 alignment")
    ids = sorted(targets)
    positions = {uid: i for i, uid in enumerate(ids)}
    np.save(result / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    n = len(ids)
    specs = {"token_ids": (np.int32, (n, 320)), "segments": (np.uint8, (n, 320)),
             "fig_x": (np.float32, (n, 30, 7)), "fig_adj": (np.uint8, (n, 30, 30)),
             "fig_mask": (np.bool_, (n, 30))}
    arrays = {}
    for name, (dtype, shape) in specs.items():
        arrays[name] = np.lib.format.open_memmap(result / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        arrays[name][:] = 0
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(VOCAB), spm_model_path=None))
    tf_fmt, fig_fmt = TrafficFormerFormat(), FigFormat()
    found, source_audit = set(), []
    for pkl in sorted(FLOWS.glob("*.pkl")):
        pieces = pkl.stem.split("__")
        if len(pieces) < 2 or pieces[1] not in known:
            continue
        with pkl.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in targets:
                continue
            if uid in found or targets[uid]["class_name"] != payload["class_name"]:
                raise RuntimeError(f"duplicate or class-mismatched frozen USTC flow {uid}")
            flow = packets[:30]
            i = positions[uid]
            encoded = encode_flow(flow, tf_fmt)
            if encoded is None:
                raise RuntimeError(f"empty TrafficFormer flow {uid}")
            tokens = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
            arrays["token_ids"][i, :len(tokens)] = tokens
            arrays["segments"][i, :len(tokens)] = 1
            graph = build_flow_graph(uid, flow, fig_fmt)
            count = graph.node_count
            if not 1 <= count <= 30:
                raise RuntimeError(f"invalid FIG node count {uid}: {count}")
            arrays["fig_x"][i, :count] = np.asarray(graph.features, dtype=np.float32)
            arrays["fig_mask"][i, :count] = True
            for a, b in graph.edges:
                arrays["fig_adj"][i, a, b] = arrays["fig_adj"][i, b, a] = 1
            found.add(uid)
            matched += 1
        source_audit.append({"pkl": str(pkl), "flow_count": len(payload["packets"]),
                             "known_trainval_matched": matched})
        print(json.dumps({"pkl": pkl.name, "found": len(found), "target": n}), flush=True)
        del payload
        gc.collect()
    if found != set(targets):
        missing = sorted(set(targets) - found)
        (result / "failure.json").write_text(json.dumps({"missing": len(missing), "sample": missing[:80]}))
        raise RuntimeError(f"missing {len(missing)} frozen USTC Known flows")
    for array in arrays.values():
        array.flush()
    if not np.isfinite(arrays["fig_x"]).all():
        raise RuntimeError("non-finite FIG features")
    audit = {"status": "PASS", "dataset": "ustc", "protocol": "A-2",
             "known_train": sum(r["original_split"] == "train" for r in rows),
             "known_validation": sum(r["original_split"] == "val" for r in rows),
             "known_test": sum(r["original_split"] == "test" for r in rows),
             "flows": n, "alignment_sha256": sha(ALIGN), "config_sha256": sha(CONFIG),
             "source_audit": source_audit, "max_tf_packets": 5, "max_fig_nodes": 30,
             "source_stage0_max_packets": 8,
             "known_test_features_transformed": len(rows) if args.phase == "test" else 0,
             "unknown_class_files_loaded": 0,
             "caveat": "pickle.load materializes whole Known-class PKLs; only selected phase flows are transformed/stored"}
    (result / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "dataset": "ustc", "flows": n}), flush=True)


if __name__ == "__main__":
    main()
