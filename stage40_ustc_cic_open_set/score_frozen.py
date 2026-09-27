#!/usr/bin/env python3
"""Known-only P95 calibration and frozen three-view open-set evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from freeze_protocols import ROOT, PROJECT, digest

S31 = PROJECT / "stage31_four_dataset_three_view_equal"
S36 = PROJECT / "stage36_vnat_sixclass_open_set_pilot"
S38 = PROJECT / "stage38_three_view_des_open_set_transfer"
sys.path[:0] = [str(S31), str(S36), str(S38)]
from train_equal_fusion import Adapters, EqualFusion, metrics  # noqa: E402
from train_tf_fig_branch import TAGCN, source as tf_source  # noqa: E402
from train_yatc_branch import source as yatc_source  # noqa: E402
from run_pilot import UnknownMFR, fused, metric, raw_scores  # noqa: E402
from run_iscx_detection import support_models, add_des_v1  # noqa: E402

METHODS = ("msp", "energy", "centroid", "des_v1")
VIEWS = ("trafficformer", "graph", "yatc")


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(key: str):
    protocol = json.loads((ROOT / "protocols.json").read_text())
    if protocol["status"] != "PASS" or protocol["score_methods"] != list(METHODS):
        raise RuntimeError("Stage40 protocol mismatch")
    unit = next(x for x in protocol["units"] if x.get("key", "ustc_a2") == key)
    role_path = Path(unit["role_manifest"])
    if digest(role_path) != unit["role_manifest_sha256"]:
        raise RuntimeError("frozen role manifest changed")
    if any(digest(Path(p)) != h for p, h in unit["source_hashes"].items()):
        raise RuntimeError("source protocol hash changed")
    with role_path.open(newline="", encoding="utf-8") as f:
        roles = list(csv.DictReader(f))
    if len(roles) != len({r["flow_id"] for r in roles}):
        raise RuntimeError("duplicate role flow ID")
    if set(x["role"] for x in roles) != {"known_train", "known_validation", "known_test", "unknown_test"}:
        raise RuntimeError("incomplete role set")
    if any(r["class_name"] in unit["unknown_classes"] for r in roles
           if r["role"] in ("known_train", "known_validation", "known_test")):
        raise RuntimeError("Unknown class leaked into Known role")
    run = (PROJECT / "stage34_ustc_cic_closed_set/runs/ustc/A-2") if key == "ustc_a2" else \
          ROOT / key / "runs/ustc/A-2"
    out = ROOT / key / "detection"
    return unit, roles, run, out


def frozen_model(run: Path, classes: list[str], device: torch.device):
    hashes = {}
    for name in VIEWS:
        sub = run / name
        meta = json.loads((sub / "metrics.json").read_text())
        if not (sub / "SUCCESS").exists() or digest(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"frozen branch hash failed: {name}")
        hashes[name] = meta["checkpoint_sha256"]
    head_dir = run / "T0_equal"
    meta = json.loads((head_dir / "known_validation_metrics.json").read_text())
    if not (head_dir / "SUCCESS").exists():
        raise RuntimeError("missing frozen fusion checkpoint")
    for name, expected in meta["checkpoint_hashes"].items():
        if digest(head_dir / name) != expected:
            raise RuntimeError(f"frozen fusion hash failed: {name}")
        hashes[name] = expected
    adapters = Adapters(len(classes)).to(device)
    adapters.load_state_dict(torch.load(head_dir / "adapters_best.pt", map_location="cpu",
                                        weights_only=False)["state_dict"])
    adapters.eval()
    head = EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(head_dir / "T0_equal_best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"])
    head.eval()
    with np.load(head_dir / "known_train_view_scalers.npz", allow_pickle=False) as data:
        scaler = {name: data[name].copy() for name in data.files}
    return adapters, head, scaler, hashes, meta


def known_role(run: Path, role: str, roles: list[dict[str, str]], classes: list[str]):
    selected = sorted((r["flow_id"], r["class_name"]) for r in roles if r["role"] == role)
    ids = [x for x, _ in selected]
    y = np.asarray([classes.index(name) for _, name in selected], dtype=np.int64)
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"bad Known role: {role}")
    features = {}
    for view, size in (("trafficformer", 768), ("graph", 128), ("yatc", 192)):
        sub = run / view
        cached = np.load(sub / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if cached != ids:
            raise RuntimeError(f"Known branch flow alignment failed: {view}/{role}")
        x = np.load(sub / f"{role}_features.npy", allow_pickle=False).astype(np.float32)
        if x.shape != (len(ids), size) or not np.isfinite(x).all():
            raise RuntimeError(f"Known branch feature failed: {view}/{role}")
        features[view] = x
    return ids, y, features


def calibrate(key: str) -> None:
    torch.set_num_threads(4)
    unit, roles, run, out = setup(key)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {out}")
    out.mkdir(parents=True)
    device = torch.device("cpu")
    classes = sorted(unit["known_classes"])
    adapter, head, scaler, hashes, head_meta = frozen_model(run, classes, device)
    representations = {}
    for role in ("known_train", "known_validation"):
        ids, y, features = known_role(run, role, roles, classes)
        h, logits = fused(features, device, adapter, head, scaler)
        np.savez(out / f"{role}.npz", h=h, logits=logits,
                 flow_ids=np.asarray(ids, dtype="U80"), labels=y)
        representations[role] = (ids, y, h, logits)
    _, y_val, h_val, logit_val = representations["known_validation"]
    replay = metrics(y_val, logit_val.argmax(1), len(classes))[0]
    reference = head_meta["metrics"]
    if any(not math.isclose(replay[k], reference[k], abs_tol=1e-6)
           for k in ("accuracy", "macro_f1", "weighted_f1")):
        raise RuntimeError(f"Known Validation score replay failed: {replay} != {reference}")
    _, y_train, h_train, _ = representations["known_train"]
    centers, neighbor = support_models(h_train, y_train, classes)
    val_raw = raw_scores(h_val, logit_val, centers, neighbor)
    center = {k: float(np.median(val_raw[k])) for k in ("centroid", "local")}
    mad = {k: float(np.median(np.abs(val_raw[k]-center[k]))) for k in center}
    add_des_v1(val_raw, center, mad)
    thresholds = {name: float(np.percentile(val_raw[name], 95, method="higher"))
                  for name in METHODS}
    payload = {"status": "PASS", "unit": key, "known_train": len(y_train),
               "known_validation": len(y_val), "classes": classes,
               "known_validation_metrics_replay": replay, "thresholds": thresholds,
               "median": center, "mad": mad, "k": 10, "global_local_weights": [0.5, 0.5],
               "threshold_rule": "Known-Val P95 method=higher; reject score>threshold",
               "checkpoint_hashes": hashes, "role_manifest_sha256": unit["role_manifest_sha256"],
               "unknown_fitting_count": 0, "test_fitting_count": 0,
               "calibration_runtime": "CPU; same frozen adapters and head"}
    write_json(out / "calibration.json", payload)
    print(json.dumps({"status": "PASS", "key": key, "known_val": len(y_val),
                      "macro_f1": replay["macro_f1"]}), flush=True)


def input_paths(key: str):
    if key == "ustc_a2":
        known = PROJECT / "stage34_ustc_cic_closed_set/input_caches/ustc/A-2"
        unknown = ROOT / "ustc/input_caches"
        return [(known / "tf_fig_test", known / "yatc_mfr_test", "known_test"),
                (unknown / "tf_fig_eval", unknown / "yatc_mfr_eval", "unknown_test")]
    base = ROOT / key / "input_caches/ustc/A-2"
    return [(base / "tf_fig_eval", base / "yatc_mfr_eval", "combined")]


@torch.no_grad()
def extract_eval(run: Path, classes: list[str], ids: list[str], tf_cache: Path,
                 mfr_cache: Path, mfr_name: str, device: torch.device):
    if np.load(tf_cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist() != ids:
        raise RuntimeError("Test TF/FIG flow ID order mismatch")
    if np.load(mfr_cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist() != ids:
        raise RuntimeError("Test MFR flow ID order mismatch")
    n = len(ids)
    tf_run = run / "trafficformer"
    model = tf_source.load_tf_module().Classifier(tf_source.make_tf_args(len(classes), 1)).to(device)
    model.load_state_dict(torch.load(tf_run / "model_best.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
    tokens = np.load(tf_cache / "token_ids.npy", mmap_mode="r", allow_pickle=False)
    segments = np.load(tf_cache / "segments.npy", mmap_mode="r", allow_pickle=False)
    features = {"trafficformer": tf_source.infer_tf(model, tokens, segments, device)[0]}
    del model
    torch.cuda.empty_cache()
    graph_run = run / "graph"
    graph = TAGCN(in_dim=7, hidden=128, labels_num=len(classes), k_hops=2, dropout=.5).to(device)
    graph.load_state_dict(torch.load(graph_run / "model_best.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
    with np.load(graph_run / "known_train_node_scaler.npz", allow_pickle=False) as data:
        x = ((np.load(tf_cache / "fig_x.npy", allow_pickle=False)-data["mean"]) /
             data["std"]).astype(np.float32)
    mask = np.load(tf_cache / "fig_mask.npy", allow_pickle=False)
    x[~mask] = 0
    adj = tf_source.normalize_adjacency(np.load(tf_cache / "fig_adj.npy", allow_pickle=False), mask)
    features["graph"] = tf_source.infer_graph(graph, x, adj, mask, device)[0]
    del graph, x, adj
    torch.cuda.empty_cache()
    yatc_run = run / "yatc"
    yatc, _ = yatc_source.initialized_model(classes)
    yatc.load_state_dict(torch.load(yatc_run / "model_best.pt", map_location="cpu",
                                    weights_only=False)["model_state_dict"])
    yatc.to(device).eval()
    images = np.load(mfr_cache / mfr_name, mmap_mode="r", allow_pickle=False)
    chunks = []
    for image in DataLoader(UnknownMFR(images), batch_size=64, shuffle=False, num_workers=0):
        with torch.cuda.amp.autocast():
            chunks.append(yatc.forward_features(image.to(device)).float().cpu().numpy())
    features["yatc"] = np.concatenate(chunks).astype(np.float32)
    if any(features[name].shape != (n, dim) or not np.isfinite(features[name]).all()
           for name, dim in (("trafficformer", 768), ("graph", 128), ("yatc", 192))):
        raise RuntimeError("bad eval branch shape or value")
    return features


def evaluate(key: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen three-view inference")
    torch.set_num_threads(4)
    unit, roles, run, out = setup(key)
    calibration = json.loads((out / "calibration.json").read_text())
    if calibration["status"] != "PASS" or calibration["role_manifest_sha256"] != unit["role_manifest_sha256"]:
        raise RuntimeError("Known-only calibration absent or changed")
    if (out / "open_set_results.csv").exists():
        raise FileExistsError("refusing to overwrite open-set evaluation")
    classes = sorted(unit["known_classes"])
    device = torch.device("cuda:0")
    adapter, head, scaler, hashes, _ = frozen_model(run, classes, device)
    if hashes != calibration["checkpoint_hashes"]:
        raise RuntimeError("frozen checkpoints changed after calibration")
    evaluation = sorted((r for r in roles if r["role"] in ("known_test", "unknown_test")),
                        key=lambda r: r["flow_id"])
    all_ids = [r["flow_id"] for r in evaluation]
    features_by_id = {}
    for tf_cache, mfr_cache, role in input_paths(key):
        selected = evaluation if role == "combined" else [r for r in evaluation if r["role"] == role]
        ids = [r["flow_id"] for r in selected]
        for cache in (tf_cache, mfr_cache):
            if json.loads((cache / "cache_audit.json").read_text())["status"] != "PASS":
                raise RuntimeError(f"Test cache audit failed: {cache}")
        mfr_file = "known_test_mfr.npy" if key == "ustc_a2" and role == "known_test" else "mfr.npy"
        if key == "ustc_a2" and role == "known_test":
            mfr_ids = np.load(mfr_cache / "known_test_flow_ids.npy", allow_pickle=False).astype(str).tolist()
            if mfr_ids != ids:
                raise RuntimeError("Known Test MFR role ID mismatch")
            # Stage34's known_test cache has a role-specific ID filename.
            mfr_cache_id_check = False
        else:
            mfr_cache_id_check = True
        if not mfr_cache_id_check:
            # The generic branch extractor expects flow_ids.npy.  Read through the
            # existing cache without modifying historical files.
            pass
        if key == "ustc_a2" and role == "known_test":
            # For this one historical layout, the role-specific MFR ID was just
            # checked above; feed an isolated link-free mirror of IDs to extractor.
            feature = extract_eval_legacy_mfr(run, classes, ids, tf_cache, mfr_cache,
                                              mfr_file, device)
        else:
            feature = extract_eval(run, classes, ids, tf_cache, mfr_cache, mfr_file, device)
        for i, uid in enumerate(ids):
            features_by_id[uid] = {name: feature[name][i] for name in VIEWS}
    if set(features_by_id) != set(all_ids):
        raise RuntimeError("Test feature ID coverage failed")
    combined = {name: np.stack([features_by_id[uid][name] for uid in all_ids]) for name in VIEWS}
    h, logits = fused(combined, device, adapter, head, scaler)
    with np.load(out / "known_train.npz", allow_pickle=False) as data:
        centers, neighbor = support_models(data["h"], data["labels"], classes)
    raw = raw_scores(h, logits, centers, neighbor)
    add_des_v1(raw, calibration["median"], calibration["mad"])
    truth = np.asarray([int(r["role"] == "unknown_test") for r in evaluation], dtype=np.int64)
    result_rows = []
    for name in METHODS:
        threshold = calibration["thresholds"][name]
        result_rows.append({"dataset": unit["dataset"], "unit": key, "method": name,
                            "known_test": int((truth == 0).sum()), "unknown_test": int(truth.sum()),
                            "unknown_prevalence": float(truth.mean()),
                            **metric(truth, raw[name], threshold),
                            "binary_f1": float(f1_score(truth, raw[name] > threshold))})
    write_csv(out / "open_set_results.csv", result_rows)
    sample = []
    for i, row in enumerate(evaluation):
        item = {"flow_id": row["flow_id"], "role": row["role"],
                "class_name": row["class_name"], "is_unknown": int(truth[i]),
                "predicted_known": classes[int(logits[i].argmax())]}
        for name in METHODS:
            item[f"score_{name}"] = float(raw[name][i])
            item[f"reject_{name}"] = int(raw[name][i] > calibration["thresholds"][name])
        sample.append(item)
    write_csv(out / "sample_scores.csv", sample)
    per = []
    for unknown_name in unit["unknown_classes"]:
        chosen = (truth == 0) | np.asarray([r["class_name"] == unknown_name and r["role"] == "unknown_test"
                                               for r in evaluation])
        for name in METHODS:
            per.append({"unit": key, "unknown_class": unknown_name, "method": name,
                        "unknown_prevalence": float(truth[chosen].mean()),
                        **metric(truth[chosen], raw[name][chosen], calibration["thresholds"][name])})
    write_csv(out / "per_unknown_class.csv", per)
    known_idx = np.flatnonzero(truth == 0)
    known_y = np.asarray([classes.index(evaluation[i]["class_name"]) for i in known_idx])
    known_metric = metrics(known_y, logits[known_idx].argmax(1), len(classes))[0]
    write_json(out / "known_test_closed_set.json", known_metric)
    if key == "ustc_a2":
        historical = json.loads((run / "known_test_evaluation/results.json").read_text())
        if any(not math.isclose(known_metric[x], historical[x], abs_tol=1e-6)
               for x in ("accuracy", "macro_f1", "weighted_f1")):
            raise RuntimeError("USTC known-test historical prediction replay failed")
    hashes_after = frozen_model(run, classes, torch.device("cpu"))[3]
    if hashes_after != hashes:
        raise RuntimeError("frozen checkpoints changed during Test")
    write_json(out / "evaluation_audit.json", {"status": "PASS", "unit": key,
        "known_test": len(known_idx), "unknown_test": int(truth.sum()),
        "known_train_support": calibration["known_train"],
        "known_validation_threshold": calibration["known_validation"],
        "unknown_fit_count": 0, "test_fit_count": 0,
        "checkpoint_hashes_unchanged": True,
        "role_manifest_sha256": unit["role_manifest_sha256"]})
    (out / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status": "PASS", "unit": key, "open_set": result_rows}), flush=True)


def extract_eval_legacy_mfr(run, classes, ids, tf_cache, mfr_cache, mfr_name, device):
    # Stage34 uses known_test_flow_ids.npy while newer eval caches use flow_ids.npy.
    # Check source IDs separately, then call the common extraction through a
    # minimal facade that resolves the expected filename, without editing source.
    class Facade:
        def __init__(self, root):
            self.root = root
        def __truediv__(self, name):
            return self.root / ("known_test_flow_ids.npy" if name == "flow_ids.npy" else name)
    return extract_eval(run, classes, ids, tf_cache, Facade(mfr_cache), mfr_name, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    parser.add_argument("--key", choices=("ustc_a2", "unknown_slowhttptest", "unknown_portscan"), required=True)
    args = parser.parse_args()
    if args.phase == "calibrate":
        calibrate(args.key)
    else:
        evaluate(args.key)


if __name__ == "__main__":
    main()
