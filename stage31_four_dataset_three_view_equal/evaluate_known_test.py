#!/usr/bin/env python3
"""One-shot frozen Stage31 Known Test evaluation after all five T0 selections."""
from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from preflight import OUT, PROJECT, SOURCES, sha
from test_unlock import ensure_all_heads_frozen
from train_equal_fusion import Adapters, EqualFusion, encode, infer, metrics
from train_tf_fig_branch import TAGCN, source as tf_source
from train_yatc_branch import KnownMFR, cache_root, protocol_rows, source as yatc_source


def test_rows(dataset, protocol, classes):
    if dataset == "ustc":
        path = PROJECT / "opendetect_ustc_encoder_audit" / "outputs" / "input_alignment_manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [(r["flow_id"], r["class_name"]) for r in csv.DictReader(handle)
                    if r["class_name"] in classes and r["original_split"] == "test"
                    and r["input_valid"] == "True"]
    else:
        path = SOURCES[dataset]
        with path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if dataset == "vnat":
            rows = [(r["flow_uid"], r["application"]) for r in source_rows
                    if r["protocol_id"] == protocol and r["class_role"] == "known"
                    and r["split"] == "test"]
        else:
            rows = [(r["flow_id_sha256"], r["canonical_class"]) for r in source_rows
                    if r["setting"] == "medium" and r["role"] == "known_test"]
    rows.sort()
    if not rows or len(rows) != len({uid for uid, _ in rows}) or set(cls for _, cls in rows) != set(classes):
        raise RuntimeError("invalid frozen Known Test membership")
    return rows, sha(path)


def test_cache(dataset, protocol):
    root = OUT / "input_caches" / dataset
    if dataset in ("vnat", "ustc"):
        root = root / protocol
    return root / "tf_fig_test", root / "yatc_mfr_test"


def verify_model(run, filename, expected):
    path = run / filename
    if sha(path) != expected:
        raise RuntimeError(f"frozen checkpoint SHA256 mismatch: {path}")
    return path


def extract_branches(dataset, protocol, classes, rows, device, run):
    tf_cache, mfr_cache = test_cache(dataset, protocol)
    ta = json.loads((tf_cache / "cache_audit.json").read_text())
    ma = json.loads((mfr_cache / "cache_audit.json").read_text())
    if ta["status"] != ma["status"] or ta["status"] != "PASS":
        raise RuntimeError("Known Test cache audit failed")
    ids = [uid for uid, _ in rows]
    cache_ids = np.load(tf_cache / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    if ids != cache_ids:
        raise RuntimeError("TrafficFormer/FIG Known Test ID alignment failed")
    mfr_ids = np.load(mfr_cache / ("test_flow_ids.npy" if dataset == "vnat" else "known_test_flow_ids.npy"),
                      allow_pickle=False).astype(str).tolist()
    if ids != mfr_ids:
        raise RuntimeError("YaTC Known Test ID alignment failed")
    n = len(ids)
    features, checkpoint_hashes = {}, {}
    tf_run = run / "trafficformer"
    tf_meta = json.loads((tf_run / "metrics.json").read_text())
    tf_ckpt = verify_model(tf_run, "model_best.pt", tf_meta["checkpoint_sha256"])
    tf_args = tf_source.make_tf_args(len(classes), 1)
    tf_model = tf_source.load_tf_module().Classifier(tf_args).to(device)
    tf_model.load_state_dict(torch.load(tf_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    tokens = np.load(tf_cache / "token_ids.npy", mmap_mode="r", allow_pickle=False)
    segments = np.load(tf_cache / "segments.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != (n, 320) or segments.shape != (n, 320):
        raise RuntimeError("Known Test TF token shape mismatch")
    features["trafficformer"], _ = tf_source.infer_tf(tf_model, tokens, segments, device)
    checkpoint_hashes["trafficformer"] = sha(tf_ckpt)
    del tf_model
    torch.cuda.empty_cache()
    graph_run = run / "graph"
    graph_meta = json.loads((graph_run / "metrics.json").read_text())
    graph_ckpt = verify_model(graph_run, "model_best.pt", graph_meta["checkpoint_sha256"])
    model = TAGCN(in_dim=7, hidden=128, labels_num=len(classes), k_hops=2, dropout=.5).to(device)
    model.load_state_dict(torch.load(graph_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    scaler = np.load(graph_run / "known_train_node_scaler.npz", allow_pickle=False)
    raw_x = np.load(tf_cache / "fig_x.npy", allow_pickle=False)
    mask = np.load(tf_cache / "fig_mask.npy", allow_pickle=False)
    adj = np.load(tf_cache / "fig_adj.npy", allow_pickle=False)
    if raw_x.shape != (n, 30, 7) or mask.shape != (n, 30) or adj.shape != (n, 30, 30):
        raise RuntimeError("Known Test FIG shape mismatch")
    x = ((raw_x - scaler["mean"])/scaler["std"]).astype(np.float32)
    x[~mask] = 0
    normalized_adj = tf_source.normalize_adjacency(adj, mask)
    features["graph"], _ = tf_source.infer_graph(model, x, normalized_adj, mask, device)
    checkpoint_hashes["graph"] = sha(graph_ckpt)
    del model, x, normalized_adj
    torch.cuda.empty_cache()
    yatc_run = run / "yatc"
    yatc_meta = json.loads((yatc_run / "metrics.json").read_text())
    yatc_ckpt = verify_model(yatc_run, "model_best.pt", yatc_meta["checkpoint_sha256"])
    yatc_model, _ = yatc_source.initialized_model(classes)
    yatc_model.load_state_dict(torch.load(yatc_ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    yatc_model.to(device).eval()
    selected = [(uid, cls, "known_test") for uid, cls in rows]
    mfr = KnownMFR(mfr_cache, "known_test", selected, classes,
                   "test" if dataset == "vnat" else None)
    extracted = []
    with torch.no_grad():
        for image, _ in DataLoader(mfr, batch_size=64, shuffle=False, num_workers=0):
            with torch.cuda.amp.autocast():
                z = yatc_model.forward_features(image.to(device))
            extracted.append(z.float().cpu().numpy())
    features["yatc"] = np.concatenate(extracted).astype(np.float32)
    checkpoint_hashes["yatc"] = sha(yatc_ckpt)
    for name, dim in (("trafficformer", 768), ("graph", 128), ("yatc", 192)):
        if features[name].shape != (n, dim) or not np.isfinite(features[name]).all():
            raise RuntimeError(f"invalid Known Test feature {name}")
    return features, checkpoint_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    ensure_all_heads_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    run = OUT / "runs" / args.dataset / args.protocol
    result = run / "known_test_evaluation"
    if result.exists() or (run / "known_test_results.json").exists():
        raise RuntimeError(f"refusing to overwrite Known Test evidence: {result}")
    result.mkdir(parents=True)
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        _, classes, manifest_sha = protocol_rows(args.dataset, args.protocol)
        rows, test_manifest_sha = test_rows(args.dataset, args.protocol, classes)
        features, branch_hashes = extract_branches(args.dataset, args.protocol, classes, rows, device, run)
        head_run = run / "T0_equal"
        frozen = json.loads((head_run / "known_validation_metrics.json").read_text())
        paths = {name: verify_model(head_run, name, expected) for name, expected in
                 frozen["checkpoint_hashes"].items()}
        scaler = np.load(head_run / "known_train_view_scalers.npz", allow_pickle=False)
        for view in features:
            values = ((features[view] - scaler[f"{view}_mean"])/scaler[f"{view}_std"]).astype(np.float32)
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite standardized Test view {view}")
            features[view] = values
        labels = np.asarray([classes.index(cls) for _, cls in rows], dtype=np.int64)
        bundle = {"labels": labels, **features}
        adapters = Adapters(len(classes)).to(device)
        adapters.load_state_dict(torch.load(paths["adapters_best.pt"], map_location="cpu", weights_only=False)["state_dict"])
        encoded = encode(adapters, bundle, device)
        head = EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
        head.load_state_dict(torch.load(paths["T0_equal_best.pt"], map_location="cpu", weights_only=False)["state_dict"])
        logits = infer(head, encoded, device)
        prediction = logits.argmax(1)
        score, per_class = metrics(labels, prediction, len(classes))
        with (result / "sample_predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("flow_id", "true_class", "predicted_class", "correct"))
            writer.writeheader()
            for (uid, cls), chosen in zip(rows, prediction, strict=True):
                writer.writerow({"flow_id": uid, "true_class": cls,
                                 "predicted_class": classes[int(chosen)], "correct": int(cls==classes[int(chosen)])})
        np.save(result / "logits.npy", logits.astype(np.float32), allow_pickle=False)
        report = {"status": "PASS", "dataset": args.dataset, "protocol": args.protocol,
                  "known_test_samples": len(rows), "known_classes": classes,
                  "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
                  "weighted_f1": score["weighted_f1"],
                  "per_class": [{"class": name, "precision": float(per_class[0][i]),
                                 "recall": float(per_class[1][i]), "f1": float(per_class[2][i]),
                                 "support": int(per_class[3][i])} for i, name in enumerate(classes)],
                  "trainval_manifest_sha256": manifest_sha, "test_manifest_sha256": test_manifest_sha,
                  "branch_checkpoint_hashes": branch_hashes,
                  "adapter_checkpoint_sha256": sha(paths["adapters_best.pt"]),
                  "head_checkpoint_sha256": sha(paths["T0_equal_best.pt"]),
                  "unknown_samples_loaded": 0, "test_parameter_selection": 0}
        (result / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        (run / "known_test_results.json").write_text(json.dumps(report, indent=2) + "\n")
        (result / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "dataset": args.dataset,
                          "protocol": args.protocol, "test": score}), flush=True)
    except BaseException as exc:
        (result / "FAILURE.json").write_text(json.dumps({"error": repr(exc),
                "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
