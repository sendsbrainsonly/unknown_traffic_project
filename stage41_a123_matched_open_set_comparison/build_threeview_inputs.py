#!/usr/bin/env python3
"""Build Stage31-equivalent inputs for matched A-1/A-3 Known-only training or frozen Test."""
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

from freeze_matched_protocol import PROJECT, ROOT, digest

sys.path[:0] = [str(PROJECT), str(PROJECT / "tf_runtime/code"),
                str(PROJECT / "stage23_closed_set_method_table/scripts")]
from src.preprocessing.fig_graph import FigFormat, build_flow_graph  # noqa: E402
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.tokenizers import BertTokenizer  # noqa: E402
from build_yatc_mfr import author_mfr_packet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("A-1", "A-3"), required=True)
    parser.add_argument("--phase", choices=("trainval", "eval"), required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "matched_protocol.json").read_text())
    unit = protocol["units"][args.setting]
    path = Path(unit["role_manifest"])
    if digest(path) != unit["role_manifest_sha256"]:
        raise RuntimeError("frozen role manifest changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.phase == "trainval":
        selected = [r for r in rows if r["role"] in ("known_train", "known_validation")]
    else:
        run = ROOT / args.setting / "runs/ustc/A-2"
        for branch in ("trafficformer", "graph", "yatc"):
            meta = json.loads((run / branch / "metrics.json").read_text())
            if digest(run / branch / "model_best.pt") != meta["checkpoint_sha256"]:
                raise RuntimeError(f"{branch} checkpoint hash changed")
        head = json.loads((run / "T0_equal/known_validation_metrics.json").read_text())
        for name, expected in head["checkpoint_hashes"].items():
            if digest(run / "T0_equal" / name) != expected:
                raise RuntimeError(f"fusion checkpoint hash changed: {name}")
        selected = [r for r in rows if r["role"] in ("known_test", "unknown_test")]
    targets = {r["flow_id"]: r for r in selected}
    if not targets or len(targets) != len(selected):
        raise RuntimeError("empty or duplicated target IDs")
    ids = sorted(targets)
    positions = {uid: i for i, uid in enumerate(ids)}
    base = ROOT / args.setting / "input_caches/ustc/A-2"
    tf_dir = base / ("tf_fig" if args.phase == "trainval" else "tf_fig_eval")
    mfr_dir = base / ("yatc_mfr" if args.phase == "trainval" else "yatc_mfr_eval")
    if tf_dir.exists() or mfr_dir.exists():
        raise FileExistsError("refusing to overwrite three-view input cache")
    tf_dir.mkdir(parents=True)
    mfr_dir.mkdir(parents=True)
    n = len(ids)
    np.save(tf_dir / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    specs = {"token_ids": (np.int32, (n, 320)), "segments": (np.uint8, (n, 320)),
             "fig_x": (np.float32, (n, 30, 7)), "fig_adj": (np.uint8, (n, 30, 30)),
             "fig_mask": (np.bool_, (n, 30))}
    arrays = {name: np.lib.format.open_memmap(tf_dir / f"{name}.npy", mode="w+",
                                               dtype=dtype, shape=shape)
              for name, (dtype, shape) in specs.items()}
    for array in arrays.values():
        array[:] = 0
    if args.phase == "trainval":
        mfr_arrays = {}
        mfr_pos = {}
        for split, role in (("train", "known_train"), ("val", "known_validation")):
            role_ids = sorted(r["flow_id"] for r in selected if r["role"] == role)
            mfr_pos[role] = {uid: i for i, uid in enumerate(role_ids)}
            np.save(mfr_dir / f"{role}_flow_ids.npy", np.asarray(role_ids, dtype="U80"), allow_pickle=False)
            mfr_arrays[role] = np.lib.format.open_memmap(mfr_dir / f"{role}_mfr.npy", mode="w+",
                                                             dtype=np.uint8, shape=(len(role_ids), 40, 40))
            mfr_arrays[role][:] = 0
    else:
        np.save(mfr_dir / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
        mfr_arrays = {"eval": np.lib.format.open_memmap(mfr_dir / "mfr.npy", mode="w+",
                                                          dtype=np.uint8, shape=(n, 40, 40))}
        mfr_arrays["eval"][:] = 0
        mfr_pos = {"eval": positions}
    tokenizer = BertTokenizer(SimpleNamespace(
        vocab_path=str(PROJECT / "tf_runtime/code/models/encryptd_vocab.txt"), spm_model_path=None))
    tf_fmt, fig_fmt = TrafficFormerFormat(), FigFormat()
    classes = {r["class_name"] for r in selected}
    seen, source = set(), []
    for pkl in sorted((PROJECT / "data/flows").glob("*.pkl")):
        pieces = pkl.stem.split("__")
        if len(pieces) < 2 or pieces[1] not in classes:
            continue
        with pkl.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in targets:
                continue
            if uid in seen or payload["class_name"] != targets[uid]["class_name"] or not packets:
                raise RuntimeError(f"duplicate, mislabelled or empty flow: {uid}")
            i = positions[uid]
            encoded = encode_flow(packets[:30], tf_fmt)
            if encoded is None:
                raise RuntimeError(f"TrafficFormer encoding failed: {uid}")
            tokens = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
            arrays["token_ids"][i, :len(tokens)] = tokens
            arrays["segments"][i, :len(tokens)] = 1
            graph = build_flow_graph(uid, packets[:30], fig_fmt)
            count = graph.node_count
            if not 1 <= count <= 30:
                raise RuntimeError(f"invalid FIG node count: {uid}")
            arrays["fig_x"][i, :count] = np.asarray(graph.features, dtype=np.float32)
            arrays["fig_mask"][i, :count] = True
            for a, b in graph.edges:
                arrays["fig_adj"][i, a, b] = arrays["fig_adj"][i, b, a] = 1
            role = targets[uid]["role"] if args.phase == "trainval" else "eval"
            image = mfr_arrays[role][mfr_pos[role][uid]]
            for slot, packet in enumerate(packets[:5]):
                image.reshape(-1)[slot*320:(slot+1)*320] = np.frombuffer(
                    author_mfr_packet(bytes(packet[4])), dtype=np.uint8)
            seen.add(uid)
            matched += 1
        source.append({"pkl": pkl.name, "class": payload["class_name"], "matched": matched})
        print(json.dumps({"pkl": pkl.name, "matched": matched, "found": len(seen), "target": n}), flush=True)
        del payload
        gc.collect()
    if seen != set(ids):
        raise RuntimeError(f"missing {len(set(ids) - seen)} matched flows")
    for array in (*arrays.values(), *mfr_arrays.values()):
        array.flush()
    if not np.isfinite(arrays["fig_x"]).all():
        raise RuntimeError("nonfinite FIG input")
    audit = {"status": "PASS", "setting": args.setting, "phase": args.phase,
             "flows": n, "role_manifest_sha256": unit["role_manifest_sha256"],
             "source": source, "tf_max_packets": 5, "fig_max_nodes": 30,
             "mfr_max_packets": 5,
             "unknown_train_or_validation_flows": 0,
             "test_feature_values_loaded": n if args.phase == "eval" else 0}
    for output in (tf_dir, mfr_dir):
        (output / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "setting": args.setting, "phase": args.phase,
                      "flows": n}), flush=True)


if __name__ == "__main__":
    main()
