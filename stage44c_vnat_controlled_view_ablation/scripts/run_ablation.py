#!/usr/bin/env python3
"""Controlled head retraining and fixed evaluation on Stage44 frozen views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CFG = json.loads((ROOT / "config.json").read_text())
SOURCE = PROJECT / CFG["source"]
VIEWS = tuple(CFG["views"])
ROLES = ("known_train", "known_validation", "known_test", "unknown_test")
sys.path.insert(0, str(PROJECT / "stage31_four_dataset_three_view_equal"))
from train_equal_fusion import Adapters, EqualFusion, encode, metrics, seeded

spec = importlib.util.spec_from_file_location("stage44_frozen_evaluator", SOURCE / "scripts/evaluate_fold.py")
old = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(old)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def branch_root(fold):
    return SOURCE / "protocols" / fold / "stage44_native_runs_attempt3/runs/vnat/medium_seed2025"


def frozen_roles(fold):
    base = SOURCE / "protocols" / fold
    protocol = read_json(base / "protocol.json")
    assert sha(base / "role_manifest.csv") == protocol["role_manifest_sha256"]
    roles = {role: [] for role in ROLES}
    for row in csv_rows(base / "role_manifest.csv"):
        roles[row["role"]].append(row)
    for role in ROLES:
        roles[role].sort(key=lambda r: r["flow_uid"])
        assert len(roles[role]) == protocol["counts"][role]
    known = set(protocol["known_services"])
    assert set(r["service"] for r in roles["unknown_test"]) == {protocol["unknown_service"]}
    for role in ("known_train", "known_validation", "known_test"):
        assert set(r["service"] for r in roles[role]) == known
    for key in ("flow_uid", "capture_id", "group_id"):
        sets = [{r[key] for r in roles[role]} for role in ROLES]
        assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4)), key
    assert min(Counter(r["service"] for r in roles["known_train"]).values()) >= 10
    return protocol, roles


def source_snapshot():
    paths = [ROOT / "config.json", ROOT / "PLAN.md", ROOT / "scripts/run_ablation.py",
             ROOT / "scripts/replay_results.py", SOURCE / "vnat_loso_protocol.json",
             SOURCE / "vnat_service_manifest.csv", SOURCE / "input_caches/full/flow_ids.npy",
             SOURCE / "scripts/evaluate_fold.py", PROJECT / "stage31_four_dataset_three_view_equal/train_equal_fusion.py"]
    for fold in CFG["folds"]:
        base = SOURCE / "protocols" / fold
        paths += [base / "protocol.json", base / "role_manifest.csv"]
        run = branch_root(fold)
        for view in VIEWS:
            paths += [run / view / "model_best.pt"]
            for role in ROLES[:2]:
                paths += [run / view / f"{role}_features.npy", run / view / f"{role}_flow_ids.npy"]
        paths += [run / "graph/known_train_node_scaler.npz"]
        paths += [run / "T0_equal" / name for name in ("adapters_best.pt", "T0_equal_best.pt",
                  "known_train_view_scalers.npz", "known_validation_metrics.json")]
    return {str(p.relative_to(PROJECT)): sha(p) for p in paths}


def progress(phase, **extra):
    heads = list((ROOT / "runs").glob("*/*/TRAIN_SUCCESS.json"))
    evaluated = list((ROOT / "runs").glob("*/*/EVALUATION_SUCCESS.json"))
    save_json(ROOT / "progress.json", {"status": "running", "phase": phase,
        "training_completed": len(heads), "training_total": 16,
        "evaluation_completed": len(evaluated), "evaluation_total": 16,
        "elapsed_seconds": time.time() - STARTED, **extra})


def adapter_model(fold, nclasses, device):
    run = branch_root(fold) / "T0_equal"
    model = Adapters(nclasses).to(device)
    model.load_state_dict(torch.load(run / "adapters_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with np.load(run / "known_train_view_scalers.npz", allow_pickle=False) as values:
        scaler = {name: values[name].copy() for name in values.files}
    return model, scaler


def to_mu(features, model, scaler, device):
    n = len(features[VIEWS[0]])
    bundle = {"labels": np.zeros(n, dtype=np.int64)}
    for view in VIEWS:
        bundle[view] = ((features[view] - scaler[f"{view}_mean"]) / scaler[f"{view}_std"]).astype(np.float32)
        assert np.isfinite(bundle[view]).all()
    return encode(model, bundle, device)[0]


def cache_trainval(fold, protocol, roles, device):
    folder = ROOT / "latent_cache" / fold
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "TRAINVAL_SUCCESS.json").exists():
        return
    model, scaler = adapter_model(fold, len(protocol["known_services"]), device)
    for role in ROLES[:2]:
        ids = [r["flow_uid"] for r in roles[role]]
        features = {}
        for view in VIEWS:
            base = branch_root(fold) / view
            actual = np.load(base / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
            assert ids == actual, (fold, view, role, "ID mismatch")
            features[view] = np.load(base / f"{role}_features.npy", allow_pickle=False)
        mu = to_mu(features, model, scaler, device)
        assert mu.shape == (len(ids), 3, 64)
        np.save(folder / f"{role}_mu.npy", mu, allow_pickle=False)
        np.save(folder / f"{role}_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
    save_json(folder / "TRAINVAL_SUCCESS.json", {"status": "PASS", "test_used": 0, "unknown_used": 0})
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def infer_head(model, mu, device):
    model.eval()
    embeddings, logits = [], []
    for start in range(0, len(mu), 512):
        x = torch.from_numpy(np.asarray(mu[start:start + 512], dtype=np.float32)).to(device)
        h = model.fuser((x / 3).flatten(1))
        embeddings.append(h.cpu().numpy())
        logits.append(model.classifier(h).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def classification(y, logits, nclasses):
    value, _ = metrics(y, logits.argmax(1), nclasses)
    shifted = logits.astype(np.float64) - logits.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    value["loss"] = float(log_loss(y, probability, labels=np.arange(nclasses)))
    return value


def train_head(fold, variant, protocol, roles, device):
    output = ROOT / "runs" / fold / variant
    if (output / "TRAIN_SUCCESS.json").exists():
        return
    output.mkdir(parents=True, exist_ok=False)
    classes = protocol["known_services"]
    y = {role: np.asarray([classes.index(r["service"]) for r in roles[role]], dtype=np.int64)
         for role in ROLES[:2]}
    mask = np.asarray(CFG["variants"][variant], dtype=np.float32)[None, :, None]
    mu = {role: np.load(ROOT / "latent_cache" / fold / f"{role}_mu.npy", allow_pickle=False) * mask
          for role in ROLES[:2]}
    old_checkpoint = torch.load(branch_root(fold) / "T0_equal/T0_equal_best.pt", map_location="cpu", weights_only=False)
    seeded(CFG["seed"])
    model = EqualFusion(len(classes), old_checkpoint["state_dict"]["center"],
                        old_checkpoint["state_dict"]["mad"]).to(device)
    initial = {key: v.cpu().clone() for key, v in model.state_dict().items()}
    init_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    loader = DataLoader(TensorDataset(torch.from_numpy(mu["known_train"]), torch.from_numpy(y["known_train"])),
                        batch_size=CFG["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(CFG["seed"]))
    save_json(output / "config.json", {**CFG, "fold": fold, "variant": variant,
        "classes": classes, "active_mask": CFG["variants"][variant], "initial_state_sha256": init_hash,
        "train_samples": len(y["known_train"]), "val_samples": len(y["known_validation"]),
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "protocol_sha256": protocol["role_manifest_sha256"]})
    best, best_epoch, best_state, best_optimizer = -1., -1, None, None
    history = []
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        total = 0.
        for values, target in loader:
            values, target = values.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model.fuser((values / 3).flatten(1))
            loss = F.cross_entropy(model.classifier(hidden), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss {fold}/{variant}/{epoch}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(target)
        epoch_row = {"epoch": epoch, "train_loss": total / len(y["known_train"])}
        for role, prefix in (("known_train", "train_eval"), ("known_validation", "val")):
            _, logits = infer_head(model, mu[role], device)
            value = classification(y[role], logits, len(classes))
            epoch_row.update({f"{prefix}_{key}": val for key, val in value.items()})
        score = epoch_row["val_macro_f1"]
        if score > best:
            best, best_epoch = score, epoch
            best_state = {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}
            # Copy at the selected epoch, rather than retaining live optimizer tensors.
            import copy
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        history.append(epoch_row)
        write_csv(output / "training_dynamics.csv", history)
        progress("training", fold=fold, variant=variant, epoch=epoch, best_epoch=best_epoch,
                 val_macro_f1=score, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(json.dumps({"phase": "train", "fold": fold, "variant": variant, **epoch_row}), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "optimizer_state_dict": best_optimizer,
        "best_epoch": best_epoch, "classes": classes, "config": CFG, "variant": variant,
        "protocol_sha256": protocol["role_manifest_sha256"]}, output / "head_best.pt")
    _, logits = infer_head(model, mu["known_validation"], device)
    validation = classification(y["known_validation"], logits, len(classes))
    parity = None
    if variant == "full":
        original = read_json(branch_root(fold) / "T0_equal/known_validation_metrics.json")
        parity = {"original_best_epoch": old_checkpoint["best_epoch"], "new_best_epoch": best_epoch,
                  "metric_delta": {key: validation[key] - original["metrics"][key]
                                   for key in ("accuracy", "macro_f1", "weighted_f1")},
                  "head_max_abs_parameter_delta": max(float((best_state[k] - v).abs().max())
                                                       for k, v in old_checkpoint["state_dict"].items())}
        if max(abs(v) for v in parity["metric_delta"].values()) > 1e-6:
            save_json(output / "VALIDATION_PARITY_FAILURE.json", parity)
            raise RuntimeError(f"Full Known-Val parity failed before Test: {fold}: {parity}")
    save_json(output / "TRAIN_SUCCESS.json", {"status": "PASS", "best_epoch": best_epoch,
        "completed_epochs": CFG["epochs"], "validation": validation, "initial_state_sha256": init_hash,
        "checkpoint_sha256": sha(output / "head_best.pt"), "historical_validation_parity": parity,
        "unknown_training_samples": 0, "unknown_validation_samples": 0, "test_selection_samples": 0})
    del model, optimizer
    torch.cuda.empty_cache()


def cache_test(fold, protocol, roles, device):
    assert (ROOT / "HEADS_FROZEN.json").exists()
    folder = ROOT / "latent_cache" / fold
    if (folder / "TEST_SUCCESS.json").exists():
        return
    ids = np.load(SOURCE / "input_caches/full/flow_ids.npy", allow_pickle=False).astype(str).tolist()
    index = {uid: i for i, uid in enumerate(ids)}
    positions = {role: np.asarray([index[r["flow_uid"]] for r in roles[role]]) for role in ROLES[2:]}
    arrays = {name: np.load(SOURCE / "input_caches/full" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask", "mfr")}
    features = {role: {} for role in ROLES[2:]}
    run = branch_root(fold)
    progress("test_feature_inference", fold=fold, branch="trafficformer")
    model = old.tf_source.load_tf_module().Classifier(old.tf_source.make_tf_args(len(protocol["known_services"]), 1)).to(device)
    model.load_state_dict(torch.load(run / "trafficformer/model_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    for role, pos in positions.items():
        features[role]["trafficformer"], _ = old.tf_source.infer_tf(model, np.asarray(arrays["token_ids"][pos]),
                                                                 np.asarray(arrays["segments"][pos]), device)
    del model
    torch.cuda.empty_cache()
    progress("test_feature_inference", fold=fold, branch="graph")
    model = old.TAGCN(in_dim=7, hidden=128, labels_num=len(protocol["known_services"]), k_hops=2, dropout=.5).to(device)
    model.load_state_dict(torch.load(run / "graph/model_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    with np.load(run / "graph/known_train_node_scaler.npz", allow_pickle=False) as s:
        mean, std = s["mean"].copy(), s["std"].copy()
    for role, pos in positions.items():
        mask = np.asarray(arrays["fig_mask"][pos])
        x = ((np.asarray(arrays["fig_x"][pos]) - mean) / std).astype(np.float32)
        x[~mask] = 0
        adjacency = old.tf_source.normalize_adjacency(np.asarray(arrays["fig_adj"][pos]), mask)
        features[role]["graph"], _ = old.tf_source.infer_graph(model, x, adjacency, mask, device)
    del model
    torch.cuda.empty_cache()
    progress("test_feature_inference", fold=fold, branch="yatc")
    model, _ = old.yatc_source.initialized_model(protocol["known_services"])
    model.load_state_dict(torch.load(run / "yatc/model_best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    model.to(device)
    for role, pos in positions.items():
        features[role]["yatc"] = old.infer_yatc(model, arrays["mfr"][pos], device)
    del model
    torch.cuda.empty_cache()
    adapter, scaler = adapter_model(fold, len(protocol["known_services"]), device)
    for role in ROLES[2:]:
        mu = to_mu(features[role], adapter, scaler, device)
        assert np.isfinite(mu).all()
        np.save(folder / f"{role}_mu.npy", mu, allow_pickle=False)
        np.save(folder / f"{role}_ids.npy", np.asarray([r["flow_uid"] for r in roles[role]], dtype="U80"), allow_pickle=False)
    save_json(folder / "TEST_SUCCESS.json", {"status": "PASS", "heads_frozen_sha256": sha(ROOT / "HEADS_FROZEN.json")})
    del adapter
    torch.cuda.empty_cache()


def evaluate_head(fold, variant, protocol, roles, device):
    output = ROOT / "runs" / fold / variant
    if (output / "EVALUATION_SUCCESS.json").exists():
        return
    classes = protocol["known_services"]
    checkpoint = torch.load(output / "head_best.pt", map_location="cpu", weights_only=False)
    model = EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    mask = np.asarray(CFG["variants"][variant], dtype=np.float32)[None, :, None]
    reps = {}
    for role in ROLES:
        mu = np.load(ROOT / "latent_cache" / fold / f"{role}_mu.npy", allow_pickle=False) * mask
        ids = np.load(ROOT / "latent_cache" / fold / f"{role}_ids.npy", allow_pickle=False).astype(str).tolist()
        assert ids == [r["flow_uid"] for r in roles[role]]
        h, logits = infer_head(model, mu, device)
        reps[role] = (h, logits)
        np.savez_compressed(output / f"{role}_representation.npz", embedding=h, logits=logits,
                            flow_ids=np.asarray(ids, dtype="U80"))
    train_y = np.asarray([classes.index(r["service"]) for r in roles["known_train"]])
    train_h = reps["known_train"][0]
    centroids = np.stack([train_h[train_y == i].mean(0) for i in range(len(classes))])
    support = [NearestNeighbors(n_neighbors=10, metric="euclidean").fit(train_h[train_y == i])
               for i in range(len(classes))]
    scores = {role: old.raw_scores(*reps[role], centroids, support) for role in ROLES[1:]}
    med = {key: float(np.median(scores["known_validation"][key])) for key in ("centroid", "local")}
    mad = {key: float(np.median(np.abs(scores["known_validation"][key] - med[key]))) for key in med}
    for role in scores:
        scores[role]["des_v1"] = .5 * sum((scores[role][key] - med[key]) / (mad[key] + CFG["epsilon"])
                                         for key in med)
    thresholds = {m: float(np.percentile(scores["known_validation"][m], 95, method="higher"))
                  for m in CFG["detectors"]}
    save_json(output / "calibration.json", {"source": "Known Validation", "thresholds": thresholds,
        "median": med, "mad": mad, "knn_k": 10, "unknown_used": 0, "test_used": 0})
    y = np.asarray([classes.index(r["service"]) for r in roles["known_test"]])
    closed = classification(y, reps["known_test"][1], len(classes))
    pred = reps["known_test"][1].argmax(1)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=np.arange(len(classes)), zero_division=0)
    write_csv(output / "per_class.csv", [{"class": name, "precision": p[i], "recall": r[i], "f1": f[i], "support": n[i]}
                                        for i, name in enumerate(classes)])
    save_json(output / "confusion_matrix.json", {"classes": classes, "matrix": confusion_matrix(y, pred, labels=np.arange(len(classes))).tolist()})
    save_json(output / "closed_metrics.json", closed)
    known_ids = [r["flow_uid"] for r in roles["known_test"]]
    unknown_ids = [r["flow_uid"] for r in roles["unknown_test"]]
    count = min(len(known_ids), len(unknown_ids))
    kp = old.deterministic_subset(known_ids, count, fold + ":known")
    up = old.deterministic_subset(unknown_ids, count, fold + ":unknown")
    result = []
    for method in CFG["detectors"]:
        ks, us = scores["known_test"][method], scores["unknown_test"][method]
        for view in CFG["evaluation_views"]:
            a, b = (ks, us) if view == "natural" else (ks[kp], us[up])
            truth = np.r_[np.zeros(len(a), dtype=np.int64), np.ones(len(b), dtype=np.int64)]
            result.append({"fold": fold, "variant": variant, "method": method, "view": view,
                "known_samples": len(a), "unknown_samples": len(b), "threshold": thresholds[method],
                "closed_accuracy": closed["accuracy"], "closed_macro_f1": closed["macro_f1"],
                "closed_weighted_f1": closed["weighted_f1"],
                **old.binary_metrics(truth, np.r_[a, b], thresholds[method])})
    write_csv(output / "open_metrics.csv", result)
    sample_rows = []
    for role in ROLES[1:]:
        prediction = reps[role][1].argmax(1)
        for i, row in enumerate(roles[role]):
            item = {"fold": fold, "variant": variant, "flow_uid": row["flow_uid"], "role": role,
                "service": row["service"], "application": row["application"], "vpn_status": row["vpn_status"],
                "is_unknown": int(role == "unknown_test"), "predicted_service": classes[prediction[i]]}
            for method in CFG["detectors"]:
                item["score_" + method] = float(scores[role][method][i])
                item["prediction_" + method + "_p95"] = int(scores[role][method][i] > thresholds[method])
            item["score_local"] = float(scores[role]["local"][i])
            sample_rows.append(item)
    write_csv(output / "sample_scores.csv", sample_rows)
    save_json(output / "EVALUATION_SUCCESS.json", {"status": "PASS", "rows": len(result),
        "samples": len(sample_rows), "checkpoint_sha256": sha(output / "head_best.pt"), "unknown_fitting": 0})
    progress("evaluation", fold=fold, variant=variant, closed=closed)
    print(json.dumps({"phase": "evaluation", "fold": fold, "variant": variant, "closed": closed,
                      "des_v1": next(r for r in result if r["method"] == "des_v1" and r["view"] == "natural")}), flush=True)
    del model
    torch.cuda.empty_cache()


def summarize():
    results, closed_rows, pairs = [], [], []
    for fold in CFG["folds"]:
        for variant in CFG["variants"]:
            run = ROOT / "runs" / fold / variant
            for row in csv_rows(run / "open_metrics.csv"):
                results.append(row)
            done = read_json(run / "TRAIN_SUCCESS.json")
            closed_rows.append({"fold": fold, "variant": variant, "best_epoch": done["best_epoch"],
                "completed_epochs": done["completed_epochs"], **read_json(run / "closed_metrics.json"),
                "checkpoint_sha256": sha(run / "head_best.pt")})
    write_csv(ROOT / "run_results.csv", results)
    write_csv(ROOT / "closed_results.csv", closed_rows)
    for row in results:
        if row["variant"] == "full":
            continue
        base = next(r for r in results if r["fold"] == row["fold"] and r["variant"] == "full"
                    and r["method"] == row["method"] and r["view"] == row["view"])
        keys = ("auroc", "auprc", "ufar", "known_frr", "binary_f1", "binary_accuracy",
                "closed_accuracy", "closed_macro_f1", "closed_weighted_f1")
        pairs.append({"fold": row["fold"], "variant": row["variant"], "method": row["method"],
                      "view": row["view"], **{"delta_" + key: float(row[key]) - float(base[key]) for key in keys}})
    write_csv(ROOT / "paired_comparison.csv", pairs)
    summary = []
    for variant in CFG["variants"]:
        selected = [r for r in results if r["variant"] == variant and r["method"] == "des_v1" and r["view"] == "natural"]
        item = {"variant": variant, "folds": len(selected)}
        for key in ("closed_accuracy", "closed_macro_f1", "closed_weighted_f1", "auroc", "auprc", "ufar", "known_frr", "binary_f1"):
            values = np.asarray([float(r[key]) for r in selected])
            item[key + "_mean"] = float(values.mean())
            item[key + "_std_across_folds"] = float(values.std(ddof=1))
        if variant != "full":
            diffs = [r for r in pairs if r["variant"] == variant and r["method"] == "des_v1" and r["view"] == "natural"]
            item["auroc_wins_over_full"] = sum(r["delta_auroc"] > 0 for r in diffs)
        summary.append(item)
    write_csv(ROOT / "variant_summary.csv", summary)
    old_results = csv_rows(SOURCE / "stage44_run_results.csv")
    history = []
    for row in results:
        if row["variant"] != "full":
            continue
        base = next(r for r in old_results if r["fold"] == row["fold"] and r["method"] == row["method"]
                    and r["view"] == row["view"] and float(r["threshold_percentile"]) == 95)
        history.append({"fold": row["fold"], "method": row["method"], "view": row["view"],
            **{"delta_" + key: float(row[key]) - float(base[key]) for key in ("auroc", "auprc", "ufar", "known_frr", "binary_f1")}})
    write_csv(ROOT / "historical_full_comparison.csv", history)
    lines = ["# Stage44C — VNAT controlled frozen-view ablation", "", "- Status: `success / independent replay PASS`.",
        "- Owner: codex. Single seed 2022; four previously developed VNAT Service LOSO folds.",
        "", "## Data and split", "", "Frozen Stage44 flow IDs/labels/roles; no protocol regeneration. Four folds, each 23,449 source flows.",
        "", "## Configuration and execution", "", "16 freshly initialized fusion heads; identical 192→128→128→C architecture, seed, 30 epochs, batch256, Adam1e-3 and Known-Val Macro-F1 selection. Encoders/adapters/scalers frozen. Missing 64D slots zeroed; scaling remains1/3. All heads frozen before Test inference. One live-selected GPU, serial queue.",
        "", "## Core results", "", "DES-v1, natural Test prevalence, Known-Val P95. Values below are four-fold descriptive means, not independent-seed confidence estimates.", "",
        "| Variant | Closed Macro-F1 | Closed Weighted-F1 | AUROC | AUPRC | Unknown F1 | UFAR | Known FRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        keys = ("closed_macro_f1", "closed_weighted_f1", "auroc", "auprc", "binary_f1", "ufar", "known_frr")
        lines.append("| " + str(r["variant"]) + " | " + " | ".join(f"{r[k + '_mean']:.6f}" for k in keys) + " |")
    lines += ["", "## Preserved evidence", "", "PLAN.md/config.json; source_lock.json; HEADS_FROZEN.json; latent caches; 16 best heads; training dynamics, role-level embeddings/logits, sample scores, per-class/confusion/calibration files; run_results.csv, closed_results.csv, paired_comparison.csv, variant_summary.csv, historical_full_comparison.csv; independent replay and source hash verification.",
        "", "## Limitations", "", "Existing Known application coverage gaps and VPN/non-VPN imbalance remain. This isolates view information under frozen jointly selected independent adapters; it does not establish encoder-training or adapter-design causality. Effective informative input dimensions change despite equal nominal architecture. Previously exposed Test; single seed; four folds share source data. Full closed-set scores can have a ceiling effect. No Test-driven architecture or threshold choice.",
        "", "## Conclusion and next step", "", "Inspect each fold and both closed/open metrics before attributing a benefit to any branch. An ablation improving a metric is evidence against assuming that branch always helps in this protocol. All variants retained; no automatic branch selection, protocol repair, hyperparameter search or subsequent training is triggered."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU selector must provide a CUDA device")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    snapshot = source_snapshot()
    lock = ROOT / "source_lock.json"
    if lock.exists():
        if not args.resume:
            raise FileExistsError("Existing run; use --resume only after checking preserved state")
        assert read_json(lock) == snapshot
    else:
        save_json(lock, snapshot)
    manifest = read_json(ROOT / "manifest.json")
    manifest.update({"status": "running", "inputs": [{"path": name, "sha256": digest} for name, digest in snapshot.items()],
        "code": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                 "dirty": True, "changes": [str(ROOT.relative_to(PROJECT))]},
        "configuration": {"files": ["config.json", "PLAN.md"], "parameters": CFG, "seeds": [2022]},
        "execution": {"tmux_session": "stage44c_serial_20261023", "command": "python scripts/run_ablation.py",
                      "exit_code": None, "environment": sys.prefix,
                      "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")},
        "limitations": ["previously developed Test and frozen-adapter conditional ablation", "single seed", "existing application/tunnel coverage limits"],
        "next_step": "complete 16 heads, then frozen Test evaluation and independent replay"})
    save_json(ROOT / "manifest.json", manifest)
    for fold in CFG["folds"]:
        protocol, roles = frozen_roles(fold)
        cache_trainval(fold, protocol, roles, device)
        for variant in CFG["variants"]:
            train_head(fold, variant, protocol, roles, device)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "runs").glob("*/*/head_best.pt"))}
    assert len(hashes) == 16
    save_json(ROOT / "HEADS_FROZEN.json", {"checkpoint_sha256": hashes, "config_sha256": sha(ROOT / "config.json"),
        "known_only_selection": True, "new_test_feature_access_before_this_gate": 0})
    for fold in CFG["folds"]:
        protocol, roles = frozen_roles(fold)
        cache_test(fold, protocol, roles, device)
        for variant in CFG["variants"]:
            evaluate_head(fold, variant, protocol, roles, device)
    assert source_snapshot() == snapshot, "source file changed during ablation"
    assert all(sha(ROOT / p) == value for p, value in hashes.items())
    summary = summarize()
    subprocess.run([sys.executable, str(ROOT / "scripts/replay_results.py")], check=True)
    save_json(ROOT / "completion_verification.json", {"status": "PASS", "heads": 16, "evaluations": 16,
        "source_files_unchanged": len(snapshot), "protocol_unchanged": True, "checkpoint_hashes_unchanged": True,
        "unknown_training_samples": 0, "unknown_validation_samples": 0, "test_selection_samples": 0,
        "detector_unknown_fitting_samples": 0, "independent_replay": "PASS"})
    manifest = read_json(ROOT / "manifest.json")
    manifest.update({"status": "success", "core_results": summary, "next_step": "report controlled ablation without automatic method changes"})
    manifest["execution"]["exit_code"] = 0
    save_json(ROOT / "manifest.json", manifest)
    skill = PROJECT.parent.parent / "skills/experiment-data-preservation/scripts"
    subprocess.run([sys.executable, str(skill / "refresh_artifact_manifest.py"), str(ROOT)], check=True)
    subprocess.run([sys.executable, str(skill / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True)
    progress("complete")
    done = read_json(ROOT / "progress.json")
    done["status"] = "success"
    save_json(ROOT / "progress.json", done)
    # Refresh after the final atomic progress write so its recorded hash is current.
    subprocess.run([sys.executable, str(skill / "refresh_artifact_manifest.py"), str(ROOT)], check=True)
    subprocess.run([sys.executable, str(skill / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True)
    print(json.dumps({"status": "PASS", "summary": summary}), flush=True)


STARTED = time.time()
if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        save_json(ROOT / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        progress("failed", error=repr(exc))
        failure_progress = read_json(ROOT / "progress.json")
        failure_progress["status"] = "failed"
        save_json(ROOT / "progress.json", failure_progress)
        raise
