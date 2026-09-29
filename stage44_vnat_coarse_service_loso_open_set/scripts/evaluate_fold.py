#!/usr/bin/env python3
"""Evaluate one frozen Stage44 fold with Known-Val-only calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
FULL = ROOT / "input_caches" / "full"
PROTOCOL_ID = "service_seed2022"
VIEWS = ("trafficformer", "graph", "yatc")
METHODS = ("msp", "energy", "centroid", "des_v1")
PERCENTILES = (90, 95, 99)

sys.path.insert(0, str(PROJECT / "stage31_four_dataset_three_view_equal"))
from train_equal_fusion import Adapters, EqualFusion, encode
from train_tf_fig_branch import TAGCN, source as tf_source
from train_yatc_branch import source as yatc_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MFRDataset(Dataset):
    def __init__(self, values: np.ndarray):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        value = torch.from_numpy(np.array(self.values[index], copy=True)).float().unsqueeze(0)
        return value.div_(255).sub_(0.5).div_(0.5)


def infer_yatc(model: torch.nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(MFRDataset(values), batch_size=64, shuffle=False, num_workers=0):
            with torch.cuda.amp.autocast():
                chunks.append(model.forward_features(batch.to(device)).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def fused(features: dict[str, np.ndarray], device: torch.device, adapter: Adapters,
          head: EqualFusion, scaler: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(features["trafficformer"])
    bundle: dict[str, np.ndarray] = {"labels": np.zeros(n, dtype=np.int64)}
    for view in VIEWS:
        bundle[view] = ((features[view] - scaler[f"{view}_mean"]) /
                        scaler[f"{view}_std"]).astype(np.float32)
        if not np.isfinite(bundle[view]).all():
            raise RuntimeError(f"nonfinite normalized feature: {view}")
    mu, _, _ = encode(adapter, bundle, device)
    embeddings, logits = [], []
    head.eval()
    for start in range(0, n, 512):
        value = torch.from_numpy(mu[start:start + 512]).to(device)
        hidden = head.fuser((value / 3).flatten(1))
        embeddings.append(hidden.cpu().numpy())
        logits.append(head.classifier(hidden).cpu().numpy())
    return np.concatenate(embeddings).astype(np.float32), np.concatenate(logits).astype(np.float32)


def raw_scores(embedding: np.ndarray, logits: np.ndarray, centroids: np.ndarray,
               support: list[NearestNeighbors]) -> dict[str, np.ndarray]:
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    msp = 1 - probability.max(axis=1)
    energy = -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1)
    distances = ((embedding[:, None, :].astype(np.float64) - centroids[None, :, :]) ** 2).sum(2)
    nearest = distances.argmin(1)
    global_distance = distances[np.arange(len(embedding)), nearest]
    local = np.empty(len(embedding), dtype=np.float64)
    for label, model in enumerate(support):
        selected = np.flatnonzero(nearest == label)
        if len(selected):
            local[selected] = model.kneighbors(
                embedding[selected], n_neighbors=10, return_distance=True
            )[0].mean(1)
    return {"msp": msp, "energy": energy, "centroid": global_distance, "local": local}


def binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score > threshold
    known = truth == 0
    unknown = ~known
    return {
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "binary_accuracy": float(accuracy_score(truth, prediction)),
        "binary_precision": float(precision_score(truth, prediction, zero_division=0)),
        "binary_recall": float(recall_score(truth, prediction, zero_division=0)),
        "binary_f1": float(f1_score(truth, prediction, zero_division=0)),
        "ufar": float(np.mean(~prediction[unknown])),
        "known_frr": float(np.mean(prediction[known])),
        "known_acceptance": float(np.mean(~prediction[known])),
    }


def deterministic_subset(ids: list[str], count: int, namespace: str) -> np.ndarray:
    ordered = sorted(range(len(ids)), key=lambda index: hashlib.sha256(
        f"2022:{namespace}:{ids[index]}".encode()
    ).hexdigest())
    return np.asarray(sorted(ordered[:count]), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen Stage44 inference")
    fold = ROOT / "protocols" / args.fold
    output = fold / "evaluation"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
    role_path = fold / "role_manifest.csv"
    if sha256(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("role manifest changed after freeze")
    with role_path.open(newline="", encoding="utf-8") as handle:
        role_rows = list(csv.DictReader(handle))
    roles: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in role_rows:
        roles[row["role"]].append(row)
    for role in roles:
        roles[role].sort(key=lambda row: row["flow_uid"])
    if any(len(roles[role]) != protocol["counts"][role] for role in protocol["counts"]):
        raise RuntimeError("role counts changed")
    classes = protocol["known_services"]
    if min(Counter(row["service"] for row in roles["known_train"]).values()) < 10:
        raise RuntimeError("Known Train service support below DES k=10")

    all_ids = np.load(FULL / "flow_ids.npy", allow_pickle=False).astype(str).tolist()
    index = {uid: position for position, uid in enumerate(all_ids)}
    positions = {
        role: np.asarray([index[row["flow_uid"]] for row in rows], dtype=np.int64)
        for role, rows in roles.items()
    }
    arrays = {
        name: np.load(FULL / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("token_ids", "segments", "fig_x", "fig_adj", "fig_mask", "mfr")
    }
    run = fold / "stage44_native_runs_attempt3" / "runs" / "vnat" / "medium_seed2025"
    for component in (*VIEWS, "T0_equal"):
        if not (run / component / "SUCCESS").is_file():
            raise RuntimeError(f"frozen component incomplete: {component}")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    features = {role: {} for role in roles}

    model = tf_source.load_tf_module().Classifier(tf_source.make_tf_args(len(classes), 1)).to(device)
    model.load_state_dict(torch.load(run / "trafficformer/model_best.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
    for role, pos in positions.items():
        features[role]["trafficformer"], _ = tf_source.infer_tf(
            model, np.asarray(arrays["token_ids"][pos]), np.asarray(arrays["segments"][pos]), device
        )
    del model
    torch.cuda.empty_cache()

    model = TAGCN(in_dim=7, hidden=128, labels_num=len(classes), k_hops=2, dropout=.5).to(device)
    model.load_state_dict(torch.load(run / "graph/model_best.pt", map_location="cpu",
                                     weights_only=False)["state_dict"])
    with np.load(run / "graph/known_train_node_scaler.npz", allow_pickle=False) as scaler_values:
        graph_mean, graph_std = scaler_values["mean"], scaler_values["std"]
    for role, pos in positions.items():
        mask = np.asarray(arrays["fig_mask"][pos])
        x = ((np.asarray(arrays["fig_x"][pos]) - graph_mean) / graph_std).astype(np.float32)
        x[~mask] = 0
        adjacency = tf_source.normalize_adjacency(np.asarray(arrays["fig_adj"][pos]), mask)
        features[role]["graph"], _ = tf_source.infer_graph(model, x, adjacency, mask, device)
    del model
    torch.cuda.empty_cache()

    model, _ = yatc_source.initialized_model(classes)
    model.load_state_dict(torch.load(run / "yatc/model_best.pt", map_location="cpu",
                                     weights_only=False)["model_state_dict"])
    model.to(device)
    for role, pos in positions.items():
        features[role]["yatc"] = infer_yatc(model, arrays["mfr"][pos], device)
    del model
    torch.cuda.empty_cache()

    equal = run / "T0_equal"
    with np.load(equal / "known_train_view_scalers.npz", allow_pickle=False) as values:
        scaler = {name: values[name].copy() for name in values.files}
    adapter = Adapters(len(classes)).to(device)
    adapter.load_state_dict(torch.load(equal / "adapters_best.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    adapter.eval()
    head = EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(equal / "T0_equal_best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"])
    head.eval()
    representations = {}
    representation_dir = output / "representations"
    representation_dir.mkdir()
    for role in roles:
        embedding, logits = fused(features[role], device, adapter, head, scaler)
        representations[role] = {"embedding": embedding, "logits": logits}
        np.savez_compressed(
            representation_dir / f"{role}.npz", embedding=embedding, logits=logits,
            flow_ids=np.asarray([row["flow_uid"] for row in roles[role]], dtype="U80"),
        )

    test_truth = np.asarray([classes.index(row["service"]) for row in roles["known_test"]])
    test_prediction = representations["known_test"]["logits"].argmax(1)
    closed = {
        "accuracy": float(accuracy_score(test_truth, test_prediction)),
        "macro_f1": float(f1_score(test_truth, test_prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_truth, test_prediction, average="weighted", zero_division=0)),
    }
    per_service = []
    for index_value, service in enumerate(classes):
        truth = test_truth == index_value
        prediction = test_prediction == index_value
        per_service.append({
            "fold": args.fold,
            "service": service,
            "support": int(truth.sum()),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
        })
    write_csv(output / "closed_set_per_service.csv", per_service)
    write_json(output / "closed_set_metrics.json", closed)

    train_y = np.asarray([classes.index(row["service"]) for row in roles["known_train"]])
    train_h = representations["known_train"]["embedding"]
    centroids = np.stack([train_h[train_y == index_value].mean(0) for index_value in range(len(classes))])
    support = [
        NearestNeighbors(n_neighbors=10, metric="euclidean").fit(train_h[train_y == index_value])
        for index_value in range(len(classes))
    ]
    scores = {
        role: raw_scores(value["embedding"], value["logits"], centroids, support)
        for role, value in representations.items() if role != "known_train"
    }
    median = {key: float(np.median(scores["known_validation"][key])) for key in ("centroid", "local")}
    mad = {
        key: float(np.median(np.abs(scores["known_validation"][key] - median[key])))
        for key in median
    }
    for role in scores:
        scores[role]["des_v1"] = .5 * (
            (scores[role]["centroid"] - median["centroid"]) / (mad["centroid"] + 1e-8)
            + (scores[role]["local"] - median["local"]) / (mad["local"] + 1e-8)
        )
    thresholds = {
        method: {
            percentile: float(np.percentile(scores["known_validation"][method], percentile, method="higher"))
            for percentile in PERCENTILES
        }
        for method in METHODS
    }
    write_json(output / "calibration.json", {
        "status": "PASS", "source": "Known Validation only", "thresholds": thresholds,
        "median": median, "mad": mad, "knn_k": 10, "unknown_used": 0, "test_used": 0,
    })

    known_ids = [row["flow_uid"] for row in roles["known_test"]]
    unknown_ids = [row["flow_uid"] for row in roles["unknown_test"]]
    natural_truth = np.r_[np.zeros(len(known_ids), dtype=np.int64), np.ones(len(unknown_ids), dtype=np.int64)]
    balanced_count = min(len(known_ids), len(unknown_ids))
    known_pick = deterministic_subset(known_ids, balanced_count, f"{args.fold}:known")
    unknown_pick = deterministic_subset(unknown_ids, balanced_count, f"{args.fold}:unknown")
    result_rows = []
    for method in METHODS:
        known_score = scores["known_test"][method]
        unknown_score = scores["unknown_test"][method]
        views = {
            "natural": (natural_truth, np.r_[known_score, unknown_score]),
            "balanced_1to1": (
                np.r_[np.zeros(balanced_count, dtype=np.int64), np.ones(balanced_count, dtype=np.int64)],
                np.r_[known_score[known_pick], unknown_score[unknown_pick]],
            ),
        }
        for view, (truth, value) in views.items():
            for percentile in PERCENTILES:
                result_rows.append({
                    "fold": args.fold,
                    "unknown_service": args.fold,
                    "method": method,
                    "view": view,
                    "threshold_percentile": percentile,
                    "threshold": thresholds[method][percentile],
                    "known_samples": int((truth == 0).sum()),
                    "unknown_samples": int((truth == 1).sum()),
                    "known_closed_accuracy": closed["accuracy"],
                    "known_closed_macro_f1": closed["macro_f1"],
                    "known_closed_weighted_f1": closed["weighted_f1"],
                    **binary_metrics(truth, value, thresholds[method][percentile]),
                })
    write_csv(output / "open_set_metrics.csv", result_rows)

    score_rows = []
    for role in ("known_validation", "known_test", "unknown_test"):
        for position_value, row in enumerate(roles[role]):
            record: dict[str, object] = {
                "fold": args.fold, "flow_uid": row["flow_uid"], "role": role,
                "service": row["service"], "application": row["application"],
                "vpn_status": row["vpn_status"], "is_unknown": int(role == "unknown_test"),
            }
            for method in METHODS:
                record[f"score_{method}"] = float(scores[role][method][position_value])
                record[f"prediction_{method}_p95"] = int(
                    scores[role][method][position_value] > thresholds[method][95]
                )
            score_rows.append(record)
    write_csv(output / "sample_scores.csv", score_rows)

    per_application = []
    for application in sorted({row["application"] for row in roles["unknown_test"]}):
        selected = np.asarray([
            index_value for index_value, row in enumerate(roles["unknown_test"])
            if row["application"] == application
        ])
        truth = np.r_[np.zeros(len(known_ids), dtype=np.int64), np.ones(len(selected), dtype=np.int64)]
        for method in METHODS:
            value = np.r_[scores["known_test"][method], scores["unknown_test"][method][selected]]
            per_application.append({
                "fold": args.fold, "unknown_application": application, "method": method,
                "unknown_samples": len(selected),
                **binary_metrics(truth, value, thresholds[method][95]),
            })
    write_csv(output / "per_unknown_application.csv", per_application)
    checkpoint_hashes = {
        str(path.relative_to(PROJECT)): sha256(path)
        for path in [
            run / "trafficformer/model_best.pt", run / "graph/model_best.pt",
            run / "yatc/model_best.pt", equal / "adapters_best.pt", equal / "T0_equal_best.pt",
        ]
    }
    write_json(output / "evaluation_verification.json", {
        "status": "PASS", "fold": args.fold, "unknown_training_samples": 0,
        "unknown_validation_samples": 0, "unknown_normalization_samples": 0,
        "unknown_support_samples": 0, "unknown_threshold_samples": 0,
        "test_threshold_samples": 0, "role_manifest_sha256": sha256(role_path),
        "checkpoint_hashes": checkpoint_hashes, "metric_rows": len(result_rows),
        "score_rows": len(score_rows), "primary_threshold": "Known Validation P95",
    })
    print(json.dumps({"status": "PASS", "fold": args.fold, "closed": closed,
                      "metric_rows": len(result_rows)}), flush=True)


if __name__ == "__main__":
    main()
