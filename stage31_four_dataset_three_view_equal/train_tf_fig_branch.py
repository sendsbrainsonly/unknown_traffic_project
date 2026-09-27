#!/usr/bin/env python3
"""Train a frozen-protocol TrafficFormer or TAGCN branch with Stage22 recipes."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from preflight import OUT, PROJECT, sha
from train_yatc_branch import protocol_rows

sys.path.insert(0, str(PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "scripts"))
import train_closed_run as source  # noqa: E402
from train_pilot_run import TAGCN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--branch", choices=("trafficformer", "graph"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    expected = {"iscx_vpn": "medium_seed2022", "iscx_tor": "medium_seed2022",
                "vnat": ("medium_seed2025", "medium_seed2026"), "ustc": "A-2"}[args.dataset]
    if args.protocol not in (expected if isinstance(expected, tuple) else (expected,)):
        raise ValueError("protocol outside fixed Stage31 pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    run = OUT / "runs" / args.dataset / args.protocol / (args.branch + ("_smoke" if args.smoke else ""))
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    started = time.time()
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        selected, classes, manifest_sha = protocol_rows(args.dataset, args.protocol)
        root = OUT / "input_caches" / args.dataset
        root = root / (args.protocol if args.dataset == "vnat" else "A-2") if args.dataset in ("vnat", "ustc") else root
        root = root / "tf_fig"
        audit = json.loads((root / "cache_audit.json").read_text())
        if audit["status"] != "PASS":
            raise RuntimeError("TF/FIG cache audit not passed")
        ids = np.load(root / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if set(ids) != {uid for uid, _, _ in selected} or len(ids) != len(selected):
            raise RuntimeError("TF/FIG IDs do not equal frozen Known Train/Val")
        index = {uid: i for i, uid in enumerate(ids)}
        class_index = {name: i for i, name in enumerate(classes)}
        role = {}
        for part in ("known_train", "known_validation"):
            rows = sorted((uid, cls) for uid, cls, item_role in selected if item_role == part)
            positions = np.asarray([index[uid] for uid, _ in rows], dtype=np.int64)
            labels = np.asarray([class_index[cls] for _, cls in rows], dtype=np.int64)
            role[part] = {"rows": rows, "positions": positions, "labels": labels}
        if any(set(role[p]["labels"]) != set(range(len(classes))) for p in role):
            raise RuntimeError("class absent from frozen Known Train/Validation")
        source.write_json(run / "config.json", {
            "dataset": args.dataset, "protocol": args.protocol, "branch": args.branch,
            "seed": 2022, "classes": classes, "known_train": len(role["known_train"]["rows"]),
            "known_validation": len(role["known_validation"]["rows"]),
            "frozen_split_sha256": manifest_sha, "cache_audit_sha256": sha(root / "cache_audit.json"),
            "recipe": "Stage22 pretrained TrafficFormer 20e batch64 LR6e-5" if args.branch == "trafficformer"
                      else "Stage22 TAGCN K2 50e batch64 LR1e-3",
            "selection": "Known Validation Macro-F1", "unknown_training_samples": 0,
            "unknown_validation_samples": 0, "test_selection_samples": 0})
        train, val = role["known_train"], role["known_validation"]
        if args.branch == "trafficformer":
            config = source.read_json(source.CONFIG)
            if sha(source.PRETRAINED_MODEL) != config["pretrained_model_sha256"]:
                raise RuntimeError("official TrafficFormer pretrained hash mismatch")
            tokens = np.load(root / "token_ids.npy", mmap_mode="r", allow_pickle=False)
            segments = np.load(root / "segments.npy", mmap_mode="r", allow_pickle=False)
            def tf_pair(part):
                return (np.asarray(tokens[part["positions"]]),
                        np.asarray(segments[part["positions"]]), part["labels"])
            xtrain, xval = tf_pair(train), tf_pair(val)
            if args.smoke:
                source.seed_everything(2022)
                module = source.load_tf_module()
                tf_args = source.make_tf_args(len(classes), math.ceil(len(train["labels"])/64)*20)
                tf_args.batch_size = 64
                model = module.Classifier(tf_args)
                state = torch.load(source.PRETRAINED_MODEL, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=False)
                model.to(device)
                batch = min(8, len(xtrain[0]))
                logits, z = source.tf_forward(model,
                    torch.as_tensor(xtrain[0][:batch], device=device).long(),
                    torch.as_tensor(xtrain[1][:batch], device=device).long())
                loss = nn.CrossEntropyLoss()(logits, torch.as_tensor(xtrain[2][:batch], device=device))
                loss.backward()
                source.write_json(run / "smoke_result.json", {"status": "PASS", "batch": batch,
                    "loss": float(loss.detach()), "embedding_shape": list(z.shape)})
                (run / "SUCCESS").write_text("SUCCESS\n")
                return
            model, history, best_epoch, load = source.train_tf_pretrained(
                xtrain, xval, len(classes), 2022, device, run / "model_best.pt", config)
            feature_train, _ = source.infer_tf(model, xtrain[0], xtrain[1], device)
            feature_val, _ = source.infer_tf(model, xval[0], xval[1], device)
            source.write_json(run / "pretrained_load_report.json", load)
        else:
            graph_x = np.load(root / "fig_x.npy", mmap_mode="r", allow_pickle=False)
            graph_adj = np.load(root / "fig_adj.npy", mmap_mode="r", allow_pickle=False)
            graph_mask = np.load(root / "fig_mask.npy", mmap_mode="r", allow_pickle=False)
            train_nodes = np.asarray(graph_x[train["positions"]])[np.asarray(graph_mask[train["positions"]])]
            center = train_nodes.mean(0)
            scale = train_nodes.std(0)
            scale[scale < 1e-8] = 1
            def graph_pair(part):
                pos = part["positions"]
                raw_x = np.asarray(graph_x[pos])
                mask = np.asarray(graph_mask[pos])
                x = ((raw_x-center)/scale).astype(np.float32)
                x[~mask] = 0
                adj = source.normalize_adjacency(np.asarray(graph_adj[pos]), mask)
                return x, adj, mask, part["labels"]
            xtrain, xval = graph_pair(train), graph_pair(val)
            np.savez(run / "known_train_node_scaler.npz", mean=center, std=scale)
            if args.smoke:
                source.seed_everything(2022)
                model = TAGCN(in_dim=7, hidden=128, labels_num=len(classes), k_hops=2, dropout=.5).to(device)
                batch = min(8, len(xtrain[0]))
                logits, z = model(*(torch.as_tensor(item[:batch], device=device)
                                    for item in xtrain[:3]))
                loss = nn.CrossEntropyLoss()(logits, torch.as_tensor(xtrain[3][:batch], device=device))
                loss.backward()
                source.write_json(run / "smoke_result.json", {"status": "PASS", "batch": batch,
                    "loss": float(loss.detach()), "embedding_shape": list(z.shape)})
                (run / "SUCCESS").write_text("SUCCESS\n")
                return
            model, history, best_epoch = source.train_graph(
                xtrain, xval, len(classes), 2022, device, run / "model_best.pt")
            feature_train, _ = source.infer_graph(model, *xtrain[:3], device)
            feature_val, _ = source.infer_graph(model, *xval[:3], device)
        for part, value, rows in (("known_train", feature_train, train["rows"]),
                                  ("known_validation", feature_val, val["rows"])):
            if not np.isfinite(value).all():
                raise RuntimeError(f"nonfinite {args.branch} features {part}")
            np.save(run / f"{part}_features.npy", value.astype(np.float32), allow_pickle=False)
            np.save(run / f"{part}_flow_ids.npy", np.asarray([uid for uid, _ in rows], dtype="U80"), allow_pickle=False)
        source.write_json(run / "training_history.json", history)
        source.write_json(run / "metrics.json", {"status": "PASS", "completed_epochs":
            20 if args.branch == "trafficformer" else 50, "best_epoch": best_epoch,
            "checkpoint_sha256": sha(run / "model_best.pt"), "elapsed_seconds": time.time()-started})
        (run / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "dataset": args.dataset,
                          "protocol": args.protocol, "branch": args.branch,
                          "best_epoch": best_epoch}), flush=True)
    except BaseException as exc:
        source.write_json(run / "FAILURE.json", {"status": "FAIL", "error": repr(exc),
                                                "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
