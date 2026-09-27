#!/usr/bin/env python3
"""Stage38B frozen Known-only calibration and one-shot open-set evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from audit_iscx_known import CATEGORY, ROOT, S12, S31, sha
from build_iscx_eval_inputs import frozen_test_rows
from run_iscx_known import PROTOCOL, coarse_protocol_rows, lock

sys.path.insert(0, str(S31))
sys.path.insert(0, str(ROOT.parent / "stage36_vnat_sixclass_open_set_pilot"))
from train_equal_fusion import Adapters, EqualFusion, metrics  # noqa: E402
from train_tf_fig_branch import TAGCN, source as tf_source  # noqa: E402
from train_yatc_branch import source as yatc_source  # noqa: E402
from run_pilot import UnknownMFR, fused, metric, raw_scores  # noqa: E402

METHODS = ("msp", "energy", "centroid", "des_v1")
VIEWS = ("trafficformer", "graph", "yatc")


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paths(dataset: str) -> tuple[Path, Path]:
    run = ROOT / "runs" / dataset / PROTOCOL
    return run, run / "detection"


def frozen_model(dataset: str, device: torch.device):
    lock(dataset)
    run, _ = paths(dataset)
    _, classes, _ = coarse_protocol_rows(dataset, PROTOCOL)
    head_run = run / "T0_equal"
    if not (head_run / "SUCCESS").is_file():
        raise RuntimeError("Known-only fusion training incomplete")
    meta = json.loads((head_run / "known_validation_metrics.json").read_text())
    for name, digest in meta["checkpoint_hashes"].items():
        if sha(head_run / name) != digest:
            raise RuntimeError(f"frozen fusion checkpoint hash mismatch: {name}")
    adapter = Adapters(len(classes)).to(device)
    adapter.load_state_dict(torch.load(head_run / "adapters_best.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    adapter.eval()
    head = EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(head_run / "T0_equal_best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"])
    head.eval()
    with np.load(head_run / "known_train_view_scalers.npz", allow_pickle=False) as data:
        scaler = {name: data[name].copy() for name in data.files}
    hashes = {name: sha(head_run / name) for name in meta["checkpoint_hashes"]}
    return classes, adapter, head, scaler, hashes


def known_role(dataset: str, role: str, classes: list[str]):
    selected, _, _ = coarse_protocol_rows(dataset, PROTOCOL)
    rows = sorted((uid, cls) for uid, cls, row_role in selected if row_role == role)
    ids = [uid for uid, _ in rows]
    run, _ = paths(dataset)
    features = {}
    for view in VIEWS:
        sub = run / view
        if not (sub / "SUCCESS").is_file():
            raise RuntimeError(f"Known branch incomplete: {view}")
        meta = json.loads((sub / "metrics.json").read_text())
        if sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"Known branch checkpoint hash mismatch: {view}")
        branch_ids = np.load(sub / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if branch_ids != ids:
            raise RuntimeError(f"Known branch ID mismatch: {view}/{role}")
        features[view] = np.load(sub / f"{role}_features.npy", allow_pickle=False).astype(np.float32)
        if features[view].shape != (len(ids), {"trafficformer": 768, "graph": 128, "yatc": 192}[view]):
            raise RuntimeError(f"Known branch feature shape mismatch: {view}/{role}")
    labels = np.asarray([classes.index(cls) for _, cls in rows], dtype=np.int64)
    return ids, labels, features


def support_models(train_h: np.ndarray, labels: np.ndarray, classes: list[str]):
    if min(int((labels == i).sum()) for i in range(len(classes))) < 10:
        raise RuntimeError("Known Train class support below fixed k=10")
    centers = np.stack([train_h[labels == i].mean(0) for i in range(len(classes))])
    nearest = [NearestNeighbors(n_neighbors=10, metric="euclidean", algorithm="auto")
               .fit(train_h[labels == i]) for i in range(len(classes))]
    return centers, nearest


def add_des_v1(raw: dict, median: dict, mad: dict) -> None:
    raw["des_v1"] = .5 * ((raw["centroid"] - median["centroid"]) / (mad["centroid"] + 1e-8)
                           + (raw["local"] - median["local"]) / (mad["local"] + 1e-8))


def calibrate(dataset: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen calibration")
    torch.set_num_threads(4)
    lock(dataset)
    run, out = paths(dataset)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    device = torch.device("cuda:0")
    classes, adapter, head, scaler, hashes = frozen_model(dataset, device)
    results = {}
    for role in ("known_train", "known_validation"):
        ids, labels, features = known_role(dataset, role, classes)
        h, logits = fused(features, device, adapter, head, scaler)
        np.savez(out / f"{role}.npz", h=h, logits=logits,
                 flow_ids=np.asarray(ids, dtype="U80"), labels=labels)
        results[role] = (ids, labels, h, logits)
    val_ids, val_y, val_h, val_logits = results["known_validation"]
    with (run / "T0_equal/known_validation_predictions.csv").open(newline="", encoding="utf-8") as handle:
        old = list(csv.DictReader(handle))
    if [r["flow_id"] for r in old] != val_ids or [r["predicted_class"] for r in old] != [classes[i] for i in val_logits.argmax(1)]:
        raise RuntimeError("frozen Known Validation prediction replay mismatch")
    original = json.loads((run / "T0_equal/known_validation_metrics.json").read_text())["metrics"]
    replay = metrics(val_y, val_logits.argmax(1), len(classes))[0]
    if any(not math.isclose(replay[key], original[key], abs_tol=1e-6) for key in ("accuracy", "macro_f1", "weighted_f1")):
        raise RuntimeError("frozen Known Validation metric replay mismatch")
    _, train_y, train_h, _ = results["known_train"]
    centers, neighbors = support_models(train_h, train_y, classes)
    val_raw = raw_scores(val_h, val_logits, centers, neighbors)
    median = {key: float(np.median(val_raw[key])) for key in ("centroid", "local")}
    mad = {key: float(np.median(np.abs(val_raw[key] - median[key]))) for key in median}
    add_des_v1(val_raw, median, mad)
    thresholds = {name: float(np.percentile(val_raw[name], 95, method="higher")) for name in METHODS}
    write_json(out / "calibration.json", {"status": "PASS", "dataset": dataset,
        "protocol": PROTOCOL, "known_train": len(train_y), "known_validation": len(val_y),
        "classes": classes, "known_validation_metrics": replay,
        "thresholds": thresholds, "median": median, "mad": mad,
        "quantile_method": "higher", "decision_rule": "score > threshold",
        "knn_k": 10, "global_local_weights": [0.5, 0.5],
        "checkpoint_hashes": hashes, "train_centroid_fit": len(train_y),
        "train_knn_fit": len(train_y), "val_threshold_fit": len(val_y),
        "unknown_fit_count": 0, "test_fit_count": 0})
    write_json(out / "calibration_verification.json", {"status": "PASS",
        "known_validation_prediction_replay": True, "known_validation_metric_replay": True,
        "unknown_feature_values_loaded": 0, "known_test_feature_values_loaded": 0})
    print(json.dumps({"status": "PASS", "phase": "calibrate", "dataset": dataset,
                      "known_val_macro_f1": replay["macro_f1"]}), flush=True)


def test_branches(dataset: str, classes: list[str], ids: list[str], device: torch.device):
    run, _ = paths(dataset)
    root = ROOT / "input_caches" / dataset
    tf_cache = root / "tf_fig_eval"
    mfr_cache = root / "yatc_mfr_eval"
    for cache in (tf_cache, mfr_cache):
        audit = json.loads((cache / "cache_audit.json").read_text())
        if audit["status"] != "PASS" or audit["model_and_threshold_frozen_before_read"] is not True:
            raise RuntimeError(f"frozen Test cache audit failed: {cache}")
        if np.load(cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist() != ids:
            raise RuntimeError(f"frozen Test input ID mismatch: {cache}")
    n = len(ids)
    features = {}
    sub = run / "trafficformer"
    tf_args = tf_source.make_tf_args(len(classes), 1)
    tf_model = tf_source.load_tf_module().Classifier(tf_args).to(device)
    tf_model.load_state_dict(torch.load(sub / "model_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    tokens = np.load(tf_cache / "token_ids.npy", mmap_mode="r", allow_pickle=False)
    segments = np.load(tf_cache / "segments.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != (n, 320) or segments.shape != (n, 320):
        raise RuntimeError("TrafficFormer Test token shape mismatch")
    features["trafficformer"], _ = tf_source.infer_tf(tf_model, tokens, segments, device)
    del tf_model
    torch.cuda.empty_cache()
    sub = run / "graph"
    graph = TAGCN(in_dim=7, hidden=128, labels_num=len(classes), k_hops=2, dropout=.5).to(device)
    graph.load_state_dict(torch.load(sub / "model_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    with np.load(sub / "known_train_node_scaler.npz", allow_pickle=False) as scaler:
        x = ((np.load(tf_cache / "fig_x.npy", allow_pickle=False) - scaler["mean"]) / scaler["std"]).astype(np.float32)
    mask = np.load(tf_cache / "fig_mask.npy", allow_pickle=False)
    x[~mask] = 0
    adj = tf_source.normalize_adjacency(np.load(tf_cache / "fig_adj.npy", allow_pickle=False), mask)
    features["graph"], _ = tf_source.infer_graph(graph, x, adj, mask, device)
    del graph, x, adj
    torch.cuda.empty_cache()
    sub = run / "yatc"
    yatc, _ = yatc_source.initialized_model(classes)
    yatc.load_state_dict(torch.load(sub / "model_best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    yatc.to(device).eval()
    images = np.load(mfr_cache / "mfr.npy", mmap_mode="r", allow_pickle=False)
    chunks = []
    with torch.no_grad():
        for image in DataLoader(UnknownMFR(images), batch_size=64, shuffle=False, num_workers=0):
            with torch.cuda.amp.autocast():
                chunks.append(yatc.forward_features(image.to(device)).float().cpu().numpy())
    features["yatc"] = np.concatenate(chunks).astype(np.float32)
    for name, dim in (("trafficformer", 768), ("graph", 128), ("yatc", 192)):
        if features[name].shape != (n, dim) or not np.isfinite(features[name]).all():
            raise RuntimeError(f"nonfinite or misaligned Test feature {name}")
    return features


def evaluate(dataset: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen Test inference")
    torch.set_num_threads(4)
    unit = lock(dataset)
    run, out = paths(dataset)
    if not (out / "calibration_verification.json").is_file() or (out / "sample_scores.csv").exists():
        raise RuntimeError("calibration missing or Test already evaluated")
    calibration = json.loads((out / "calibration.json").read_text())
    if calibration["status"] != "PASS" or calibration["unknown_fit_count"] or calibration["test_fit_count"]:
        raise RuntimeError("Known-only calibration failed")
    rows, _ = frozen_test_rows(dataset)
    ids = [r["flow_id_sha256"] for r in rows]
    device = torch.device("cuda:0")
    classes, adapter, head, scaler, hashes = frozen_model(dataset, device)
    if hashes != calibration["checkpoint_hashes"]:
        raise RuntimeError("frozen fusion checkpoint changed after calibration")
    features = test_branches(dataset, classes, ids, device)
    h, logits = fused(features, device, adapter, head, scaler)
    roles = np.asarray([r["role"] for r in rows])
    for role in ("known_test", "unknown_test"):
        choose = roles == role
        np.savez(out / f"{role}.npz", h=h[choose], logits=logits[choose],
                 flow_ids=np.asarray(ids, dtype="U80")[choose])
    with np.load(out / "known_train.npz", allow_pickle=False) as train:
        centers, neighbors = support_models(train["h"], train["labels"], classes)
    with np.load(out / "known_validation.npz", allow_pickle=False) as val:
        val_h = val["h"].copy()
        val_logits = val["logits"].copy()
        val_ids = val["flow_ids"].astype(str).tolist()
    val_raw = raw_scores(val_h, val_logits, centers, neighbors)
    test_raw = raw_scores(h, logits, centers, neighbors)
    for raw in (val_raw, test_raw):
        add_des_v1(raw, calibration["median"], calibration["mad"])
    for name in METHODS:
        replay = float(np.percentile(val_raw[name], 95, method="higher"))
        if replay != calibration["thresholds"][name]:
            raise RuntimeError(f"Known-Val threshold changed: {name}")
    score_rows = []
    for i, row in enumerate(rows):
        record = {"flow_id": ids[i], "role": row["role"],
                  "application": row["canonical_class"],
                  "official_category": row["official_category"],
                  "is_unknown": int(row["role"] == "unknown_test"),
                  "predicted_known_class": classes[int(logits[i].argmax())]}
        for name in METHODS:
            record[f"score_{name}"] = float(test_raw[name][i])
            record[f"prediction_{name}"] = int(test_raw[name][i] > calibration["thresholds"][name])
        score_rows.append(record)
    for i, uid in enumerate(val_ids):
        record = {"flow_id": uid, "role": "known_validation", "application": "",
                  "official_category": "", "is_unknown": 0,
                  "predicted_known_class": classes[int(val_logits[i].argmax())]}
        for name in METHODS:
            record[f"score_{name}"] = float(val_raw[name][i])
            record[f"prediction_{name}"] = int(val_raw[name][i] > calibration["thresholds"][name])
        score_rows.append(record)
    write_csv(out / "sample_scores.csv", score_rows)
    truth = np.asarray([int(r["role"] == "unknown_test") for r in rows], dtype=np.int64)
    overall, per_app = [], []
    for name in METHODS:
        overall.append({"dataset": dataset, "method": name, "unknown_prevalence": float(truth.mean()),
                        **metric(truth, test_raw[name], calibration["thresholds"][name])})
        for app in sorted({r["canonical_class"] for r in rows if r["role"] == "unknown_test"}):
            keep = (roles == "known_test") | np.asarray([r["canonical_class"] == app and r["role"] == "unknown_test" for r in rows])
            per_app.append({"dataset": dataset, "method": name, "unknown_application": app,
                            "unknown_prevalence": float(truth[keep].mean()),
                            **metric(truth[keep], test_raw[name][keep], calibration["thresholds"][name])})
    write_csv(out / "open_set_results.csv", overall)
    write_csv(out / "per_unknown_application.csv", per_app)
    test_y = np.asarray([classes.index(CATEGORY[r["official_category"]])
                         for r in rows if r["role"] == "known_test"], dtype=np.int64)
    predicted = logits[roles == "known_test"].argmax(1)
    score, per = metrics(test_y, predicted, len(classes))
    write_json(out / "closed_set_results.json", {"status": "PASS", "dataset": dataset,
        "known_test_samples": len(test_y), "metrics": score, "per_class": [
            {"class": name, "precision": float(per[0][i]), "recall": float(per[1][i]),
             "f1": float(per[2][i]), "support": int(per[3][i])} for i, name in enumerate(classes)]})
    write_json(out / "evaluation_audit.json", {"status": "PASS", "dataset": dataset,
        "known_test": int((roles == "known_test").sum()),
        "unknown_test": int((roles == "unknown_test").sum()),
        "known_train_support": calibration["known_train"],
        "known_validation_calibration": calibration["known_validation"],
        "known_test_for_fitting": 0, "unknown_for_fitting": 0,
        "checkpoint_updates": 0, "checkpoint_hashes": hashes,
        "source_hashes_unchanged": lock(dataset) == unit})
    print(json.dumps({"status": "PASS", "phase": "evaluate", "dataset": dataset,
                      "closed_set": score, "open_set": overall}), flush=True)


def verify(dataset: str) -> None:
    unit = lock(dataset)
    _, out = paths(dataset)
    audit = json.loads((out / "evaluation_audit.json").read_text())
    if audit["status"] != "PASS" or not audit["source_hashes_unchanged"]:
        raise RuntimeError("frozen evaluation audit failed")
    with (out / "sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(r["role"] for r in rows)
    if any(counts[role] != unit["role_counts"][role] for role in ("known_validation", "known_test", "unknown_test")):
        raise RuntimeError(f"saved score role count mismatch: {counts}")
    if len({r["flow_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate saved score flow ID")
    with (out / "open_set_results.csv").open(newline="", encoding="utf-8") as handle:
        expected = {r["method"]: r for r in csv.DictReader(handle)}
    calibration = json.loads((out / "calibration.json").read_text())
    test = [r for r in rows if r["role"] in ("known_test", "unknown_test")]
    truth = np.asarray([int(r["is_unknown"]) for r in test], dtype=np.int64)
    for name in METHODS:
        val = np.asarray([float(r[f"score_{name}"]) for r in rows if r["role"] == "known_validation"])
        threshold = float(np.percentile(val, 95, method="higher"))
        if threshold != calibration["thresholds"][name]:
            raise RuntimeError(f"saved Known-Val threshold replay failed: {name}")
        score = np.asarray([float(r[f"score_{name}"]) for r in test])
        if any(int(r[f"prediction_{name}"]) != int(float(r[f"score_{name}"]) > threshold) for r in test):
            raise RuntimeError(f"saved decision replay failed: {name}")
        replay = metric(truth, score, threshold)
        if any(not math.isclose(value, float(expected[name][key]), abs_tol=1e-10)
               for key, value in replay.items()):
            raise RuntimeError(f"saved open-set metric replay failed: {name}")
    write_json(out / "completion_verification.json", {"status": "PASS", "dataset": dataset,
        "saved_score_rows": len(rows), "methods": len(METHODS),
        "threshold_and_decision_replay": True, "open_set_metric_replay": True,
        "unknown_for_fitting": 0, "known_test_for_fitting": 0,
        "protected_sources_unchanged": lock(dataset) == unit})
    print(json.dumps({"status": "PASS", "phase": "verify", "dataset": dataset}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "evaluate", "verify"))
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    {"calibrate": calibrate, "evaluate": evaluate, "verify": verify}[args.phase](args.dataset)
