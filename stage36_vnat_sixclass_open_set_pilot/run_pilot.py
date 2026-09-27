#!/usr/bin/env python3
"""Frozen Stage33 six-class VNAT open-set pilot; no training or test tuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scapy.all import IP, IPv6, PcapReader
from scapy.utils import RawPcapReader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S14 = PROJECT / "stage14b_vnat_protocol_freeze"
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
S32 = PROJECT / "stage32_three_dataset_coarse_single_seed"
S33 = PROJECT / "stage33_vnat_six_class_single_seed"
PROTOCOL = "medium_seed2025"
FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
EXPECTED = {"known_train": 15704, "known_validation": 1960, "known_test": 1960,
            "unknown_test": 3825}
VIEWS = ("trafficformer", "graph", "yatc")
sys.path.insert(0, str(S31))
sys.path.insert(0, str(S32))
sys.path.insert(0, str(S33))
import build_vnat_tf_fig as tf_builder  # noqa: E402
import build_vnat_mfr as mfr_builder  # noqa: E402
import evaluate_known_test as stage31_eval  # noqa: E402
import evaluate_one as stage32_eval  # noqa: E402
import common as prior  # noqa: E402
import run_six as stage33  # noqa: E402
from train_equal_fusion import Adapters, EqualFusion, encode  # noqa: E402
from train_tf_fig_branch import TAGCN, source as tf_source  # noqa: E402
from train_yatc_branch import protocol_rows, source as yatc_source  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen_rows() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    with (S14 / "vnat_split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["protocol_id"] != PROTOCOL:
                continue
            role = "unknown_test" if row["class_role"] == "unknown" else "known_" + row["split"]
            result[role].append(row)
    for role, count in EXPECTED.items():
        result[role].sort(key=lambda r: r["flow_uid"])
        if len(result[role]) != count or len({r["flow_uid"] for r in result[role]}) != count:
            raise RuntimeError(f"frozen {role} count/ID mismatch: {len(result[role])} != {count}")
    ids = [r["flow_uid"] for rows in result.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frozen roles overlap")
    known = set(stage33.LABEL_MAP)
    unknown = {"sftp", "vimeo", "zoiper"}
    if any(r["application"] not in known for role in ("known_train", "known_validation", "known_test")
           for r in result[role]) or {r["application"] for r in result["unknown_test"]} != unknown:
        raise RuntimeError("known/unknown application membership changed")
    return result


def frozen_hashes() -> dict[str, str]:
    selected = json.loads((S33 / "selected_head_before_test.json").read_text())
    if prior.freeze_sources() != selected["source_hashes"]:
        raise RuntimeError("Stage31/32 source hashes changed")
    run = S33 / "runs/vnat"
    for name, digest in selected["checkpoint_hashes"].items():
        if sha(run / name) != digest:
            raise RuntimeError(f"Stage33 checkpoint changed: {name}")
    protocol = json.loads((S14 / "vnat_open_set_protocol.json").read_text())
    obj = dict(protocol)
    digest = obj.pop("freeze_hash")
    computed = hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != FREEZE or computed != FREEZE:
        raise RuntimeError("Stage14B freeze hash mismatch")
    if sha(S14 / "vnat_split_manifest.csv") != protocol["split_manifest"]["sha256"]:
        raise RuntimeError("Stage14B split manifest hash mismatch")
    return {"stage14b_protocol": sha(S14 / "vnat_open_set_protocol.json"),
            "stage14b_split_manifest": sha(S14 / "vnat_split_manifest.csv"),
            **{f"stage33_{name}": digest for name, digest in selected["checkpoint_hashes"].items()},
            **{f"stage31_{name}": selected["source_hashes"][f"vnat_{name}_checkpoint"]
               for name in VIEWS}}


def preflight() -> None:
    rows = frozen_rows()
    hashes = frozen_hashes()
    if not all((S31 / "runs/vnat" / PROTOCOL / name / "SUCCESS").is_file() for name in VIEWS):
        raise RuntimeError("frozen Stage31 branch incomplete")
    if not (S33 / "runs/vnat/SUCCESS").is_file():
        raise RuntimeError("frozen Stage33 head incomplete")
    write_json(ROOT / "preflight.json", {
        "status": "PASS", "protocol": PROTOCOL, "freeze_hash": FREEZE,
        "role_counts": {name: len(items) for name, items in rows.items()},
        "application_counts": {name: dict(sorted(Counter(r["application"] for r in items).items()))
                               for name, items in rows.items()},
        "source_hashes": hashes, "unknown_feature_values_loaded": 0,
        "known_test_feature_values_loaded": 0, "training_or_fitting_performed": False,
    })
    print(json.dumps({"phase": "preflight", "status": "PASS", "counts": EXPECTED}), flush=True)


def recover(rows: list[dict], output: Path) -> None:
    """Reuse Stage31 stream/packet encoders; materialize only selected IDs."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    targets = {r["flow_uid"]: r for r in rows}
    ids = sorted(targets)
    pos = {uid: i for i, uid in enumerate(ids)}
    specs = {"token_ids": (np.int32, (len(ids), 320)),
             "segments": (np.uint8, (len(ids), 320)),
             "fig_x": (np.float32, (len(ids), 30, 7)),
             "fig_adj": (np.uint8, (len(ids), 30, 30)),
             "fig_mask": (np.bool_, (len(ids), 30)),
             "mfr": (np.uint8, (len(ids), 40, 40))}
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (dtype, shape) in specs.items()}
    for arr in arrays.values():
        arr[:] = 0
    np.save(output / "flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    by_pcap: dict[Path, dict[str, str]] = defaultdict(dict)
    for uid, row in targets.items():
        pcap = Path(row["source_pcap_path"])
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        stream = row["source_flow_id"]
        if stream in by_pcap[pcap]:
            raise RuntimeError(f"duplicate source stream: {pcap}/{stream}")
        by_pcap[pcap][stream] = uid
    tokenizer = tf_builder.BertTokenizer(tf_builder.SimpleNamespace(
        vocab_path=str(tf_builder.VOCAB), spm_model_path=None))
    tf_fmt, graph_fmt = tf_builder.TrafficFormerFormat(), tf_builder.FigFormat()
    capture_audit = []
    found = set()
    for number, (pcap, wanted) in enumerate(sorted(by_pcap.items(), key=lambda p: str(p[0])), 1):
        refs = mfr_builder.stream_refs(pcap, set(wanted), limit=30)
        frame_to_uid: dict[int, list[str]] = defaultdict(list)
        mfr_frame_to_uid: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for stream, frames in refs.items():
            uid = wanted[stream]
            for slot, frame in enumerate(frames):
                frame_to_uid[frame].append(uid)
                if slot < 5:
                    mfr_frame_to_uid[frame].append((uid, slot))
        packets: dict[str, list[tuple]] = defaultdict(list)
        last_frame = max(frame_to_uid)
        with PcapReader(str(pcap)) as reader:
            raw_ip = reader.linktype == 101
            for frame, packet in enumerate(reader, 1):
                for uid in frame_to_uid.get(frame, ()):
                    network = packet.getlayer(IP) or packet.getlayer(IPv6)
                    if network is None:
                        raise RuntimeError(f"non-IP selected frame {pcap}:{frame}:{uid}")
                    raw = bytes(packet)
                    initiator = packets[uid][0][5] if packets[uid] else str(network.src)
                    caplen = len(raw)
                    wirelen = int(getattr(packet, "wirelen", 0) or caplen)
                    packets[uid].append((float(packet.time), caplen, wirelen,
                                         int(str(network.src) == initiator),
                                         tf_builder.ethernet_compatible_frame(raw, raw_ip), initiator))
                if frame >= last_frame:
                    break
        with RawPcapReader(str(pcap)) as reader:
            for frame, (raw, _) in enumerate(reader, 1):
                for uid, slot in mfr_frame_to_uid.get(frame, ()):
                    try:
                        encoded = mfr_builder.author_mfr_packet(raw)
                    except ValueError as exc:
                        if "requires an IPv4 IP layer" not in str(exc):
                            raise
                        encoded = mfr_builder.ipv6_compatible_mfr_packet(raw)
                    arrays["mfr"][pos[uid]].reshape(-1)[slot*320:(slot+1)*320] = np.frombuffer(
                        encoded, dtype=np.uint8)
                if frame >= last_frame:
                    break
        for stream, uid in wanted.items():
            selected = packets[uid]
            expected = min(30, int(targets[uid]["packet_count"]))
            if len(selected) != expected:
                raise RuntimeError(f"packet coverage {uid}: {len(selected)} != {expected}")
            flow = [item[:5] for item in selected]
            i = pos[uid]
            encoded = tf_builder.encode_flow(flow, tf_fmt)
            if encoded is None:
                raise RuntimeError(f"empty TF flow {uid}")
            tokens = tokenizer.convert_tokens_to_ids(
                [tf_builder.CLS_TOKEN] + tokenizer.tokenize(encoded.text))[:320]
            arrays["token_ids"][i, :len(tokens)] = tokens
            arrays["segments"][i, :len(tokens)] = 1
            graph = tf_builder.build_flow_graph(uid, flow, graph_fmt)
            count = graph.node_count
            if not 1 <= count <= 30:
                raise RuntimeError(f"invalid graph node count {uid}: {count}")
            arrays["fig_x"][i, :count] = np.asarray(graph.features, dtype=np.float32)
            arrays["fig_mask"][i, :count] = True
            for a, b in graph.edges:
                arrays["fig_adj"][i, a, b] = arrays["fig_adj"][i, b, a] = 1
            found.add(uid)
        capture_audit.append({"pcap": str(pcap), "selected_flows": len(wanted),
                              "last_required_frame": last_frame})
        print(json.dumps({"phase": "recover", "capture": number,
                          "total_captures": len(by_pcap), "flows": len(found)}), flush=True)
    if found != set(ids):
        raise RuntimeError(f"missing selected flows: {len(set(ids)-found)}")
    for arr in arrays.values():
        arr.flush()
    write_json(output / "cache_audit.json", {"status": "PASS", "protocol": PROTOCOL,
        "flows": len(ids), "role": "known_validation_parity" if len(ids) < 100 else "unknown_test",
        "capture_audit": capture_audit, "stage31_encoding_functions_reused": True})


def parity() -> None:
    if json.loads((ROOT / "preflight.json").read_text())["status"] != "PASS":
        raise RuntimeError("preflight not PASS")
    rows = frozen_rows()["known_validation"]
    selected = []
    for app in stage33.LABEL_MAP:
        for vpn in ("nonvpn", "vpn"):
            match = next((r for r in rows if r["application"] == app and r["vpn_status"] == vpn), None)
            if match:
                selected.append(match)
    if len(selected) < 7:
        raise RuntimeError("insufficient Known Val parity coverage")
    cache = ROOT / "input_caches" / "known_val_parity"
    recover(selected, cache)
    ids = np.load(cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    old_tf = S31 / "input_caches/vnat" / PROTOCOL / "tf_fig"
    old_mfr = S31 / "input_caches/vnat" / PROTOCOL / "yatc_mfr_attempt3"
    tf_ids = np.load(old_tf / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    mfr_ids = np.load(old_mfr / "validation_flow_ids.npy", allow_pickle=False).astype(str).tolist()
    tf_pos, mfr_pos = {uid: i for i, uid in enumerate(tf_ids)}, {uid: i for i, uid in enumerate(mfr_ids)}
    fields = {"token_ids": "token_ids", "segments": "segments", "fig_x": "fig_x",
              "fig_adj": "fig_adj", "fig_mask": "fig_mask", "mfr": "validation_mfr"}
    failures = []
    for fresh, old in fields.items():
        new_values = np.load(cache / f"{fresh}.npy", mmap_mode="r", allow_pickle=False)
        old_values = np.load((old_mfr if fresh == "mfr" else old_tf) / f"{old}.npy",
                             mmap_mode="r", allow_pickle=False)
        for i, uid in enumerate(ids):
            index = mfr_pos[uid] if fresh == "mfr" else tf_pos[uid]
            if not np.array_equal(new_values[i], old_values[index]):
                failures.append({"flow_id": uid, "field": fresh})
    result = {"status": "PASS" if not failures else "FAIL", "samples": len(ids),
              "fields": list(fields), "exact_array_comparisons": len(ids)*len(fields),
              "failures": failures, "unknown_feature_values_loaded": 0}
    write_json(ROOT / "parity.json", result)
    if failures:
        raise RuntimeError(f"Known Validation input parity failed: {failures[:8]}")
    print(json.dumps({"phase": "parity", **result}), flush=True)


def extract_unknown() -> None:
    if json.loads((ROOT / "parity.json").read_text())["status"] != "PASS":
        raise RuntimeError("Known Val parity not PASS")
    if frozen_hashes() != json.loads((ROOT / "preflight.json").read_text())["source_hashes"]:
        raise RuntimeError("frozen inputs changed before Unknown extraction")
    rows = frozen_rows()["unknown_test"]
    recover(rows, ROOT / "input_caches" / "unknown_test")
    if frozen_hashes() != json.loads((ROOT / "preflight.json").read_text())["source_hashes"]:
        raise RuntimeError("frozen inputs changed during Unknown extraction")


class UnknownMFR(Dataset):
    def __init__(self, array: np.ndarray):
        self.array = array

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = torch.from_numpy(np.array(self.array[index], copy=True)).float().unsqueeze(0)
        return x.div_(255).sub_(0.5).div_(0.5)


def unknown_branches(device: torch.device) -> dict[str, np.ndarray]:
    cache = ROOT / "input_caches" / "unknown_test"
    audit = json.loads((cache / "cache_audit.json").read_text())
    if audit["status"] != "PASS" or audit["flows"] != EXPECTED["unknown_test"]:
        raise RuntimeError("Unknown input cache invalid")
    ids = np.load(cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if ids != [r["flow_uid"] for r in frozen_rows()["unknown_test"]]:
        raise RuntimeError("Unknown cache ID order mismatch")
    _, classes, _ = protocol_rows("vnat", PROTOCOL)
    run = S31 / "runs/vnat" / PROTOCOL
    result = {}
    model = tf_source.load_tf_module().Classifier(tf_source.make_tf_args(len(classes), 1)).to(device)
    model.load_state_dict(torch.load(run / "trafficformer/model_best.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
    result["trafficformer"], _ = tf_source.infer_tf(model,
        np.load(cache / "token_ids.npy", mmap_mode="r", allow_pickle=False),
        np.load(cache / "segments.npy", mmap_mode="r", allow_pickle=False), device)
    del model
    torch.cuda.empty_cache()
    model = TAGCN(in_dim=7, hidden=128, labels_num=len(classes), k_hops=2, dropout=.5).to(device)
    model.load_state_dict(torch.load(run / "graph/model_best.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
    with np.load(run / "graph/known_train_node_scaler.npz", allow_pickle=False) as s:
        x = ((np.load(cache / "fig_x.npy", allow_pickle=False)-s["mean"])/s["std"]).astype(np.float32)
    mask = np.load(cache / "fig_mask.npy", allow_pickle=False)
    x[~mask] = 0
    adj = tf_source.normalize_adjacency(np.load(cache / "fig_adj.npy", allow_pickle=False), mask)
    result["graph"], _ = tf_source.infer_graph(model, x, adj, mask, device)
    del model, x, adj
    torch.cuda.empty_cache()
    model, _ = yatc_source.initialized_model(classes)
    model.load_state_dict(torch.load(run / "yatc/model_best.pt", map_location="cpu",
                                     weights_only=False)["model_state_dict"])
    model.to(device).eval()
    images = np.load(cache / "mfr.npy", mmap_mode="r", allow_pickle=False)
    chunks = []
    with torch.no_grad():
        for image in DataLoader(UnknownMFR(images), batch_size=64, shuffle=False, num_workers=0):
            with torch.cuda.amp.autocast():
                chunks.append(model.forward_features(image.to(device)).float().cpu().numpy())
    result["yatc"] = np.concatenate(chunks).astype(np.float32)
    for name, dim in zip(VIEWS, (768, 128, 192), strict=True):
        if result[name].shape != (len(ids), dim) or not np.isfinite(result[name]).all():
            raise RuntimeError(f"invalid Unknown branch feature {name}")
    return result


@torch.no_grad()
def fused(features: dict[str, np.ndarray], device: torch.device,
          adapter: Adapters, head: EqualFusion, scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(features["trafficformer"])
    bundle = {"labels": np.zeros(n, dtype=np.int64)}
    for view in VIEWS:
        bundle[view] = ((features[view]-scaler[f"{view}_mean"])/scaler[f"{view}_std"]).astype(np.float32)
        if not np.isfinite(bundle[view]).all():
            raise RuntimeError(f"nonfinite normalized view {view}")
    mu, _, _ = encode(adapter, bundle, device)
    out_h, out_logits = [], []
    head.eval()
    for start in range(0, n, 512):
        x = torch.from_numpy(mu[start:start+512]).to(device)
        h = head.fuser((x / 3).flatten(1))
        out_h.append(h.cpu().numpy())
        out_logits.append(head.classifier(h).cpu().numpy())
    h = np.concatenate(out_h).astype(np.float32)
    logits = np.concatenate(out_logits).astype(np.float32)
    if h.shape != (n, 128) or not np.isfinite(h).all() or not np.isfinite(logits).all():
        raise RuntimeError("invalid fused embedding/logits")
    return h, logits


def known_smoke() -> None:
    """Prove frozen six-class head parity before any Unknown input is opened."""
    if json.loads((ROOT / "parity.json").read_text())["status"] != "PASS":
        raise RuntimeError("Known-Val packet parity not PASS")
    if not torch.cuda.is_available():
        raise RuntimeError("selected GPU required for frozen head smoke")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    run = S33 / "runs/vnat"
    values = stage33.load_known("vnat", "known_validation")
    with np.load(run / "known_train_view_scalers.npz", allow_pickle=False) as loaded:
        scaler = {name: loaded[name].copy() for name in loaded.files}
    adapter = Adapters(len(stage33.CLASSES)).to(device)
    adapter.load_state_dict(torch.load(run / "adapters_best.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    head = EqualFusion(len(stage33.CLASSES), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(run / "T0_equal_best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"])
    h, logits = fused(values, device, adapter, head, scaler)
    historical = np.load(run / "known_validation_logits.npy", allow_pickle=False)
    max_abs = float(np.max(np.abs(historical-logits)))
    if max_abs > 1e-4 or not np.array_equal(historical.argmax(1), logits.argmax(1)):
        raise RuntimeError(f"Known-Val frozen head parity failed: max_abs={max_abs}")
    result = {"status": "PASS", "known_validation": len(h), "embedding_dim": h.shape[1],
              "max_abs_logit_difference": max_abs, "prediction_parity": True,
              "unknown_feature_values_loaded": 0, "known_test_feature_values_loaded": 0}
    write_json(ROOT / "known_smoke.json", result)
    print(json.dumps({"phase": "known_smoke", **result}), flush=True)


def raw_scores(h: np.ndarray, logits: np.ndarray, centroids: np.ndarray,
               support: list[NearestNeighbors]) -> dict[str, np.ndarray]:
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= softmax.sum(axis=1, keepdims=True)
    msp = 1 - softmax.max(axis=1)
    energy = -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1)
    distances = ((h[:, None, :].astype(np.float64)-centroids[None, :, :])**2).sum(2)
    nearest = distances.argmin(1)
    global_distance = distances[np.arange(len(h)), nearest]
    local = np.empty(len(h), dtype=np.float64)
    for label, model in enumerate(support):
        positions = np.flatnonzero(nearest == label)
        if len(positions):
            local[positions] = model.kneighbors(h[positions], n_neighbors=10,
                                                return_distance=True)[0].mean(1)
    return {"msp": msp, "energy": energy, "centroid": global_distance, "local": local}


def metric(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    decision = score > threshold
    known = truth == 0
    unknown = ~known
    return {"auroc": float(roc_auc_score(truth, score)),
            "auprc": float(average_precision_score(truth, score)),
            "ufar": float(np.mean(~decision[unknown])),
            "known_frr": float(np.mean(decision[known])),
            "known_acceptance": float(np.mean(~decision[known])),
            "threshold": float(threshold), "known_test": int(known.sum()),
            "unknown_test": int(unknown.sum())}


def evaluate() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen branch inference")
    before = json.loads((ROOT / "preflight.json").read_text())["source_hashes"]
    if json.loads((ROOT / "parity.json").read_text())["status"] != "PASS" or frozen_hashes() != before:
        raise RuntimeError("pre-evaluation frozen/parity guard failed")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    roles = frozen_rows()
    train = stage33.load_known("vnat", "known_train")
    val = stage33.load_known("vnat", "known_validation")
    test_rows = [(r["flow_uid"], r["application"]) for r in roles["known_test"]]
    test_features = stage32_eval.load_vnat_test(test_rows, device)
    unknown_features = unknown_branches(device)
    run = S33 / "runs/vnat"
    with np.load(run / "known_train_view_scalers.npz", allow_pickle=False) as values:
        scaler = {name: values[name].copy() for name in values.files}
    adapter = Adapters(len(stage33.CLASSES)).to(device)
    adapter.load_state_dict(torch.load(run / "adapters_best.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    adapter.eval()
    head = EqualFusion(len(stage33.CLASSES), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(run / "T0_equal_best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"])
    head.eval()
    by_role = {}
    for role, features in (("known_train", train), ("known_validation", val),
                           ("known_test", test_features), ("unknown_test", unknown_features)):
        h, logits = fused(features, device, adapter, head, scaler)
        by_role[role] = {"h": h, "logits": logits}
        output = ROOT / "representations"
        output.mkdir(exist_ok=True)
        np.savez(output / f"{role}.npz", h=h, logits=logits,
                 flow_ids=np.asarray([r["flow_uid"] for r in roles[role]], dtype="U80"))
    # Stage33 Known Test logit parity prevents an accidental feature/head mismatch.
    historical = np.load(run / "known_test_evaluation/logits.npy", allow_pickle=False)
    max_abs = float(np.max(np.abs(historical-by_role["known_test"]["logits"])))
    if max_abs > 1e-4 or not np.array_equal(historical.argmax(1), by_role["known_test"]["logits"].argmax(1)):
        raise RuntimeError(f"Known Test saved-logit parity failed: max_abs={max_abs}")
    train_y = np.asarray([stage33.CLASSES.index(stage33.LABEL_MAP[r["application"]])
                          for r in roles["known_train"]], dtype=np.int64)
    train_h = by_role["known_train"]["h"]
    centroids = np.stack([train_h[train_y == i].mean(0) for i in range(len(stage33.CLASSES))])
    support = [NearestNeighbors(n_neighbors=10, metric="euclidean", algorithm="auto").fit(train_h[train_y == i])
               for i in range(len(stage33.CLASSES))]
    if min(int((train_y == i).sum()) for i in range(len(stage33.CLASSES))) < 10:
        raise RuntimeError("class-conditional support below k=10")
    raw = {role: raw_scores(value["h"], value["logits"], centroids, support)
           for role, value in by_role.items() if role != "known_train"}
    median = {key: float(np.median(raw["known_validation"][key])) for key in ("centroid", "local")}
    mad = {key: float(np.median(np.abs(raw["known_validation"][key]-median[key])))
           for key in median}
    for role in raw:
        raw[role]["des_v1"] = .5 * ((raw[role]["centroid"]-median["centroid"])/(mad["centroid"]+1e-8)
                                    + (raw[role]["local"]-median["local"])/(mad["local"]+1e-8))
    methods = ("msp", "energy", "centroid", "des_v1")
    thresholds = {name: float(np.percentile(raw["known_validation"][name], 95, method="higher"))
                  for name in methods}
    write_json(ROOT / "calibration.json", {"status": "PASS", "source": "Known Validation only",
        "known_validation_samples": EXPECTED["known_validation"], "thresholds": thresholds,
        "median": median, "mad": mad, "quantile_method": "higher", "unknown_used": 0,
        "test_used": 0, "centroids_from": "Known Train only", "knn_k": 10})
    score_rows = []
    for role in ("known_validation", "known_test", "unknown_test"):
        for i, row in enumerate(roles[role]):
            record = {"flow_uid": row["flow_uid"], "role": role,
                      "application": row["application"], "is_unknown": int(role == "unknown_test")}
            for name in methods:
                record[f"score_{name}"] = float(raw[role][name][i])
                record[f"prediction_{name}"] = int(raw[role][name][i] > thresholds[name])
            score_rows.append(record)
    write_csv(ROOT / "sample_scores.csv", score_rows)
    truth = np.r_[np.zeros(EXPECTED["known_test"], dtype=np.int64),
                  np.ones(EXPECTED["unknown_test"], dtype=np.int64)]
    overall, per_class = [], []
    for name in methods:
        score = np.r_[raw["known_test"][name], raw["unknown_test"][name]]
        overall.append({"method": name, **metric(truth, score, thresholds[name])})
        for app in sorted({r["application"] for r in roles["unknown_test"]}):
            indices = [i for i, row in enumerate(roles["unknown_test"]) if row["application"] == app]
            local_score = np.r_[raw["known_test"][name], raw["unknown_test"][name][indices]]
            local_truth = np.r_[np.zeros(EXPECTED["known_test"], dtype=int), np.ones(len(indices), dtype=int)]
            per_class.append({"method": name, "unknown_application": app,
                              **metric(local_truth, local_score, thresholds[name])})
    write_csv(ROOT / "open_set_results.csv", overall)
    write_csv(ROOT / "per_unknown_application.csv", per_class)
    after = frozen_hashes()
    if after != before:
        raise RuntimeError("frozen input hash changed during evaluation")
    write_json(ROOT / "evaluation_audit.json", {"status": "PASS", "source_hashes_before": before,
        "source_hashes_after": after, "known_test_logit_max_abs_difference": max_abs,
        "train_support": len(train_h), "known_validation": EXPECTED["known_validation"],
        "known_test": EXPECTED["known_test"], "unknown_test": EXPECTED["unknown_test"],
        "unknown_fit_count": 0, "test_fit_count": 0, "checkpoint_updates": 0})
    print(json.dumps({"phase": "evaluation", "status": "PASS", "results": overall}), flush=True)


def verify() -> None:
    audit = json.loads((ROOT / "evaluation_audit.json").read_text())
    if audit["status"] != "PASS" or audit["source_hashes_before"] != frozen_hashes():
        raise RuntimeError("frozen hash verification failed")
    with (ROOT / "sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if Counter(row["role"] for row in rows) != Counter(
            {"known_validation": 1960, "known_test": 1960, "unknown_test": 3825}):
        raise RuntimeError("saved sample-score role count mismatch")
    with (ROOT / "open_set_results.csv").open(newline="", encoding="utf-8") as handle:
        expected = {r["method"]: r for r in csv.DictReader(handle)}
    calibration = json.loads((ROOT / "calibration.json").read_text())
    for name in ("msp", "energy", "centroid", "des_v1"):
        val = np.asarray([float(r[f"score_{name}"]) for r in rows if r["role"] == "known_validation"])
        threshold = float(np.percentile(val, 95, method="higher"))
        if threshold != calibration["thresholds"][name]:
            raise RuntimeError(f"Known-Val threshold replay failed: {name}")
        test = [r for r in rows if r["role"] in ("known_test", "unknown_test")]
        y = np.asarray([int(r["is_unknown"]) for r in test])
        s = np.asarray([float(r[f"score_{name}"]) for r in test])
        if any(int(r[f"prediction_{name}"]) != int(float(r[f"score_{name}"]) > threshold)
               for r in test):
            raise RuntimeError(f"decision replay failed: {name}")
        got = metric(y, s, threshold)
        for key, value in got.items():
            if not math.isclose(value, float(expected[name][key]), abs_tol=1e-10):
                raise RuntimeError(f"metric replay failed: {name}/{key}")
    write_json(ROOT / "completion_verification.json", {"status": "PASS", "methods": 4,
        "saved_sample_scores": len(rows), "frozen_hashes_unchanged": True,
        "threshold_source": "Known Validation only", "metric_replay": "PASS"})
    print(json.dumps({"phase": "verify", "status": "PASS", "samples": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "parity", "known_smoke", "extract", "evaluate", "verify"))
    phase = parser.parse_args().phase
    {"preflight": preflight, "parity": parity, "known_smoke": known_smoke,
     "extract": extract_unknown,
     "evaluate": evaluate, "verify": verify}[phase]()
