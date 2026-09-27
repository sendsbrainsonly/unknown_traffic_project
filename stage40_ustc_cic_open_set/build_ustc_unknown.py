#!/usr/bin/env python3
"""Transform frozen USTC A-2 Unknown Test rows with Stage31 input rules."""
from __future__ import annotations

import csv
import gc
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S34 = PROJECT / "stage34_ustc_cic_closed_set"
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
sys.path.insert(0, str(S34))
sys.path.insert(0, str(S31))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime/code"))
sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table/scripts"))
from ustc_runner import gate  # noqa: E402
from src.preprocessing.fig_graph import FigFormat, build_flow_graph  # noqa: E402
from src.preprocessing.trafficformer_input import TrafficFormerFormat, encode_flow  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.tokenizers import BertTokenizer  # noqa: E402
from build_yatc_mfr import author_mfr_packet  # noqa: E402
from freeze_protocols import digest  # noqa: E402


def main() -> None:
    protocol = json.loads((ROOT / "protocols.json").read_text())
    if protocol["status"] != "PASS":
        raise RuntimeError("protocol not frozen")
    unit = protocol["units"][0]
    path = Path(unit["role_manifest"])
    if digest(path) != unit["role_manifest_sha256"]:
        raise RuntimeError("USTC role manifest hash changed")
    gate()  # validates all branch and fusion checkpoint hashes before Unknown I/O
    with path.open(newline="", encoding="utf-8") as f:
        targets = {r["flow_id"]: r for r in csv.DictReader(f) if r["role"] == "unknown_test"}
    ids = sorted(targets)
    if not ids or len(ids) != unit["role_counts"]["unknown_test"]:
        raise RuntimeError("Unknown Test role count mismatch")
    expected = set(unit["unknown_classes"])
    if {r["class_name"] for r in targets.values()} != expected:
        raise RuntimeError("USTC Unknown class mismatch")
    tf = ROOT / "ustc/input_caches/tf_fig_eval"
    mfr = ROOT / "ustc/input_caches/yatc_mfr_eval"
    if tf.exists() or mfr.exists():
        raise FileExistsError("refusing to overwrite Unknown cache")
    tf.mkdir(parents=True)
    mfr.mkdir(parents=True)
    n = len(ids)
    index = {uid: i for i, uid in enumerate(ids)}
    np.save(tf / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    np.save(mfr / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    specs = {"token_ids": (np.int32, (n, 320)), "segments": (np.uint8, (n, 320)),
             "fig_x": (np.float32, (n, 30, 7)), "fig_adj": (np.uint8, (n, 30, 30)),
             "fig_mask": (np.bool_, (n, 30))}
    arrays = {name: np.lib.format.open_memmap(tf / f"{name}.npy", mode="w+",
                                               dtype=dtype, shape=shape)
              for name, (dtype, shape) in specs.items()}
    for array in arrays.values():
        array[:] = 0
    images = np.lib.format.open_memmap(mfr / "mfr.npy", mode="w+",
                                       dtype=np.uint8, shape=(n, 40, 40))
    images[:] = 0
    vocab = PROJECT / "tf_runtime/code/models/encryptd_vocab.txt"
    tokenizer = BertTokenizer(SimpleNamespace(vocab_path=str(vocab), spm_model_path=None))
    tf_fmt, graph_fmt = TrafficFormerFormat(), FigFormat()
    found = set()
    source = []
    for pkl in sorted((PROJECT / "data/flows").glob("*.pkl")):
        pieces = pkl.stem.split("__")
        if len(pieces) < 2 or pieces[1] not in expected:
            continue
        with pkl.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            if uid not in index:
                continue
            if uid in found or payload["class_name"] != targets[uid]["class_name"]:
                raise RuntimeError(f"duplicate or mislabeled USTC Unknown flow {uid}")
            i = index[uid]
            flow = packets[:30]
            encoded = encode_flow(flow, tf_fmt)
            if encoded is None:
                raise RuntimeError(f"empty TrafficFormer Unknown flow {uid}")
            tokens = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
            arrays["token_ids"][i, :len(tokens)] = tokens
            arrays["segments"][i, :len(tokens)] = 1
            graph = build_flow_graph(uid, flow, graph_fmt)
            count = graph.node_count
            if not 1 <= count <= 30:
                raise RuntimeError(f"invalid FIG Unknown node count {uid}")
            arrays["fig_x"][i, :count] = np.asarray(graph.features, dtype=np.float32)
            arrays["fig_mask"][i, :count] = True
            for a, b in graph.edges:
                arrays["fig_adj"][i, a, b] = arrays["fig_adj"][i, b, a] = 1
            if not packets:
                raise RuntimeError(f"empty MFR Unknown flow {uid}")
            for slot, packet in enumerate(packets[:5]):
                view = np.frombuffer(author_mfr_packet(bytes(packet[4])), dtype=np.uint8)
                images[i].reshape(-1)[slot * 320:(slot + 1) * 320] = view
            found.add(uid)
            matched += 1
        source.append({"path": str(pkl), "class": payload["class_name"],
                       "flows_in_pkl": len(payload["packets"]), "matched": matched})
        print(json.dumps({"source": pkl.name, "matched": matched, "total": len(found)}), flush=True)
        del payload
        gc.collect()
    if found != set(ids):
        raise RuntimeError(f"missing {len(set(ids)-found)} frozen USTC Unknown Test flows")
    for array in (*arrays.values(), images):
        array.flush()
    if not np.isfinite(arrays["fig_x"]).all():
        raise RuntimeError("nonfinite Unknown FIG input")
    audit = {"status": "PASS", "role": "unknown_test", "flows": n,
             "role_manifest_sha256": digest(path), "source": source,
             "tf_max_packets": 5, "fig_max_nodes": 30, "mfr_max_packets": 5,
             "known_training_updates": 0, "calibration_updates": 0}
    for output in (tf, mfr):
        with (output / "cache_audit.json").open("x", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
            f.write("\n")
    print(json.dumps({"status": "PASS", "unknown_test": n}), flush=True)


if __name__ == "__main__":
    main()
