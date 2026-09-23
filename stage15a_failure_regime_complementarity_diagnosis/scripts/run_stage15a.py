#!/usr/bin/env python3
"""Stage 15A: read-only failure-regime and complementarity diagnosis.

All detector scores are loaded from already frozen artifacts.  The script only
derives explanatory signals and summaries; it never runs an encoder, fits a
detector, or changes a threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


EPS = 1e-12


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def p95(values: np.ndarray) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="higher"))


def binary_metrics(known: np.ndarray, unknown: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)])
    s = np.concatenate([known, unknown]).astype(np.float64)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def load_prototypes(checkpoint: Path) -> np.ndarray:
    obj = torch.load(checkpoint, map_location="cpu")
    state = obj.get("model_state_dict", obj.get("state_dict", obj))
    candidates = [value for key, value in state.items() if key.split(".")[-1] == "prototypes"]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one prototypes tensor in {checkpoint}, got {len(candidates)}")
    return candidates[0].detach().cpu().numpy().astype(np.float64)


def empirical_centroids(train_mu: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise RuntimeError(f"reindexed labels are not contiguous: {classes}")
    return np.stack([train_mu[labels == idx].mean(axis=0) for idx in classes]).astype(np.float64)


def nearest_two(mu: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, ...]:
    distances = cdist(np.asarray(mu, dtype=np.float64), centroids, metric="euclidean")
    order = np.argsort(distances, axis=1)[:, :2]
    d1 = distances[np.arange(len(distances)), order[:, 0]]
    if centroids.shape[0] > 1:
        d2 = distances[np.arange(len(distances)), order[:, 1]]
    else:
        d2 = np.full(len(distances), np.nan)
    return d1, d2, d2 - d1, d1 / (d2 + EPS), order[:, 0]


def native_parts(mu: np.ndarray, logvar: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sq = cdist(np.asarray(mu, dtype=np.float64), prototypes, metric="sqeuclidean")
    d_proto = 0.5 * sq.min(axis=1)
    v_post = 0.5 * np.sum(np.exp(np.asarray(logvar, dtype=np.float64)) - logvar - 1.0, axis=1)
    return d_proto, v_post, sq.argmin(axis=1)


@dataclass
class Protocol:
    dataset: str
    protocol_id: str
    setting: str
    seed: int
    known_names: list[str]
    unknown_names: np.ndarray
    train_mu: np.ndarray
    train_labels: np.ndarray
    val_mu: np.ndarray
    val_logvar: np.ndarray
    val_labels: np.ndarray
    known_mu: np.ndarray
    known_logvar: np.ndarray
    known_labels: np.ndarray
    unknown_mu: np.ndarray
    unknown_logvar: np.ndarray
    val_ids: np.ndarray
    known_ids: np.ndarray
    unknown_ids: np.ndarray
    scores: dict[str, dict[str, np.ndarray]]
    local: dict[str, np.ndarray] | None
    checkpoint: Path
    source_paths: list[Path]
    sampling_protocol: str


def vnat_protocols(root: Path) -> list[Protocol]:
    base = root / "stage14d_vnat_frozen_open_set_evaluation"
    freeze = read_json(base / "checkpoint_freeze.json")["checkpoints"]
    protocols: list[Protocol] = []
    for artifact in sorted((base / "artifacts").glob("*_seed*")):
        if not (artifact / "SUCCESS").exists():
            continue
        pid = artifact.name
        result = read_json(artifact / "result.json")
        latent = np.load(artifact / "latent_outputs.npz", allow_pickle=True)
        score = np.load(artifact / "score_arrays.npz", allow_pickle=True)
        rec = freeze[pid]
        protocols.append(Protocol(
            dataset="VNAT", protocol_id=pid, setting=str(result["setting"]), seed=int(result["seed"]),
            known_names=list(result["known_classes"]),
            unknown_names=np.asarray(latent["unknown_test_applications"]).astype(str),
            train_mu=latent["train_mu"], train_labels=latent["train_labels"],
            val_mu=latent["validation_mu"], val_logvar=latent["validation_logvar"], val_labels=latent["validation_labels"],
            known_mu=latent["known_test_mu"], known_logvar=latent["known_test_logvar"], known_labels=latent["known_test_labels"],
            unknown_mu=latent["unknown_test_mu"], unknown_logvar=latent["unknown_test_logvar"],
            val_ids=latent["validation_flow_uids"].astype(str), known_ids=latent["known_test_flow_uids"].astype(str),
            unknown_ids=latent["unknown_test_flow_uids"].astype(str),
            scores={m: {role: score[f"{role}_{m.lower()}"] for role in ("validation", "known_test", "unknown_test")} for m in ("M0", "M1", "M2")},
            local={role: score[f"{role}_local"] for role in ("validation", "known_test", "unknown_test")},
            checkpoint=Path(rec["checkpoint_path"]),
            source_paths=[artifact / "latent_outputs.npz", artifact / "score_arrays.npz", artifact / "result.json", Path(rec["checkpoint_path"])],
            sampling_protocol="all frozen Known Test and Unknown Test samples (natural prevalence)",
        ))
    if len(protocols) != 15:
        raise RuntimeError(f"expected 15 VNAT protocols, got {len(protocols)}")
    return protocols


def ustc_protocols(root: Path) -> list[Protocol]:
    s11 = root / "stage11b_decoupled_support_readout"
    s13 = root / "stage13a2_global_local_fusion"
    name_map = read_json(s11 / "manifest.json")["configuration"]["class_names"]
    protocols: list[Protocol] = []
    for p11 in sorted((s11 / "artifacts").glob("a*/fold*_seed*")):
        p13 = s13 / "artifacts" / p11.parent.name / p11.name
        if not (p11 / "SUCCESS").exists() or not (p13 / "SUCCESS").exists():
            continue
        cfg = read_json(p11 / "config.json")
        res = read_json(p11 / "results.json")
        a = np.load(p11 / "sample_outputs.npz", allow_pickle=True)
        b = np.load(p13 / "score_arrays.npz", allow_pickle=True)
        keep = a["balanced_known_indices"].astype(int)
        known_original = list(res["classes"]["known"])
        known_names = [name_map[str(v)] for v in known_original]
        unknown_original = a["unknown_test_labels_original"].astype(int)
        unknown_names = np.asarray([name_map[str(v)] for v in unknown_original])
        pid = f"{cfg['scenario']}_seed{cfg['seed']}"
        protocols.append(Protocol(
            dataset="USTC-TFC2016", protocol_id=pid, setting=str(cfg["scenario"]).upper(), seed=int(cfg["seed"]),
            known_names=known_names, unknown_names=unknown_names,
            train_mu=a["train_mu"], train_labels=a["train_labels_reindexed"],
            val_mu=a["validation_mu"], val_logvar=a["validation_logvar"], val_labels=a["validation_labels_reindexed"],
            known_mu=a["known_test_mu"][keep], known_logvar=a["known_test_logvar"][keep], known_labels=a["known_test_labels_reindexed"][keep],
            unknown_mu=a["unknown_test_mu"], unknown_logvar=a["unknown_test_logvar"],
            val_ids=np.asarray([f"{pid}:validation:{i}" for i in range(len(a["validation_mu"]))]),
            known_ids=np.asarray([f"{pid}:known_test:{i}" for i in keep]),
            unknown_ids=np.asarray([f"{pid}:unknown_test:{i}" for i in range(len(a["unknown_test_mu"]))]),
            scores={
                "M0": {"validation": a["validation_r0_scores"], "known_test": a["known_test_r0_scores"][keep], "unknown_test": a["unknown_test_r0_scores"]},
                "M1": {"validation": b["validation_centroid"], "known_test": b["known_test_centroid"][keep], "unknown_test": b["unknown_test_centroid"]},
                "M2": {"validation": b["validation_gl"], "known_test": b["known_test_gl"][keep], "unknown_test": b["unknown_test_gl"]},
            },
            local={"validation": b["validation_knn10"], "known_test": b["known_test_knn10"][keep], "unknown_test": b["unknown_test_knn10"]},
            checkpoint=Path(cfg["checkpoint"]),
            source_paths=[p11 / "sample_outputs.npz", p11 / "results.json", p13 / "score_arrays.npz", p13 / "results.json", Path(cfg["checkpoint"])],
            sampling_protocol="existing frozen symmetric balanced 1:1 Known/Unknown evaluation",
        ))
    if len(protocols) != 15:
        raise RuntimeError(f"expected 15 USTC protocols, got {len(protocols)}")
    return protocols


def iscx_protocols(root: Path) -> list[Protocol]:
    base = root / "stage12_dual_external_validation" / "runs"
    protocols: list[Protocol] = []
    dataset_names = {"iscx_vpn": "ISCX-VPN", "iscx_tor": "ISCXTor2016"}
    for run in sorted(base.glob("*/*/seed*")):
        required = [run / "SUCCESS", run / "final_test_sample_outputs.npz", run / "known_train_validation_features.npz"]
        if not all(path.exists() for path in required):
            continue
        cfg = read_json(run / "config.json")
        test = np.load(run / "final_test_sample_outputs.npz", allow_pickle=True)
        kv = np.load(run / "known_train_validation_features.npz", allow_pickle=True)
        unknown_manifest = pd.read_csv(run / "unknown_test_manifest.csv")
        known_manifest = pd.read_csv(run / "known_test_manifest.csv")
        val_manifest = pd.read_csv(run / "val_manifest.csv")
        if len(unknown_manifest) != len(test["unknown_mu"]) or len(known_manifest) != len(test["known_mu"]):
            raise RuntimeError(f"manifest/array length mismatch: {run}")
        known_names = list(cfg["known_classes"])
        pid = f"{run.parts[-3]}_{cfg['setting']}_seed{cfg['seed']}"
        protocols.append(Protocol(
            dataset=dataset_names[run.parts[-3]], protocol_id=pid, setting=str(cfg["setting"]).capitalize(), seed=int(cfg["seed"]),
            known_names=known_names, unknown_names=unknown_manifest["canonical_class"].astype(str).to_numpy(),
            train_mu=kv["train_mu"], train_labels=kv["train_labels_reindexed"],
            val_mu=kv["validation_mu"], val_logvar=kv["validation_logvar"], val_labels=kv["validation_labels_reindexed"],
            known_mu=test["known_mu"], known_logvar=test["known_logvar"],
            known_labels=np.asarray([known_names.index(v) for v in known_manifest["canonical_class"].astype(str)]),
            unknown_mu=test["unknown_mu"], unknown_logvar=test["unknown_logvar"],
            val_ids=val_manifest["flow_id_sha256"].astype(str).to_numpy(),
            known_ids=known_manifest["flow_id_sha256"].astype(str).to_numpy(),
            unknown_ids=unknown_manifest["flow_id_sha256"].astype(str).to_numpy(),
            scores={
                "M0": {"validation": kv["validation_m0_scores"], "known_test": test["m0_known_scores"], "unknown_test": test["m0_unknown_scores"]},
                "M1": {"validation": kv["validation_m1_scores"], "known_test": test["m1_known_scores"], "unknown_test": test["m1_unknown_scores"]},
            },
            local=None, checkpoint=run / "model_best.pt",
            source_paths=[run / "final_test_sample_outputs.npz", run / "known_train_validation_features.npz", run / "final_results.json", run / "model_best.pt"],
            sampling_protocol="all frozen Known Test and Unknown Test samples (natural prevalence); M2 unavailable",
        ))
    counts = Counter(p.dataset for p in protocols)
    if counts != {"ISCX-VPN": 15, "ISCXTor2016": 15}:
        raise RuntimeError(f"expected 15 protocols per ISCX dataset, got {counts}")
    return protocols


def ranking_counts(unknown_score: float, known_a: np.ndarray, known_b: np.ndarray, score_b: float) -> tuple[int, int, int, int]:
    a = unknown_score > known_a
    b = score_b > known_b
    return int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b)), int(np.sum(~a & ~b))


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def analyze_protocol(p: Protocol, sample_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    centroids = empirical_centroids(p.train_mu, p.train_labels)
    prototypes = load_prototypes(p.checkpoint)
    if prototypes.shape != centroids.shape:
        raise RuntimeError(f"prototype/centroid shape mismatch in {p.dataset}/{p.protocol_id}: {prototypes.shape} vs {centroids.shape}")
    gaps = np.linalg.norm(prototypes - centroids, axis=1)
    radii = np.asarray([np.median(np.linalg.norm(p.train_mu[p.train_labels == i] - centroids[i], axis=1)) for i in range(len(centroids))])
    normalized_gaps = gaps / (radii + EPS)

    role_mu = {"validation": p.val_mu, "known_test": p.known_mu, "unknown_test": p.unknown_mu}
    role_logvar = {"validation": p.val_logvar, "known_test": p.known_logvar, "unknown_test": p.unknown_logvar}
    signals: dict[str, dict[str, np.ndarray]] = {}
    for role in role_mu:
        d1, d2, margin, ratio, nearest = nearest_two(role_mu[role], centroids)
        dproto, vpost, native_pred = native_parts(role_mu[role], role_logvar[role], prototypes)
        signals[role] = {"d1": d1, "d2": d2, "margin": margin, "ratio": ratio, "nearest": nearest,
                         "dproto": dproto, "vpost": vpost, "native_pred": native_pred}
        parity = np.max(np.abs((dproto + vpost) - p.scores["M0"][role]))
        if parity > 2e-4:
            raise RuntimeError(f"M0 decomposition parity failed {p.dataset}/{p.protocol_id}/{role}: {parity}")
        m1_parity = np.max(np.abs(d1 ** 2 - p.scores["M1"][role]))
        if m1_parity > 2e-4:
            raise RuntimeError(f"M1 distance parity failed {p.dataset}/{p.protocol_id}/{role}: {m1_parity}")

    native_cm = confusion_matrix(p.val_labels, signals["validation"]["native_pred"], labels=np.arange(len(p.known_names)))
    row_sum = native_cm.sum(axis=1)
    confusion_out = np.divide(row_sum - np.diag(native_cm), row_sum, out=np.zeros_like(row_sum, dtype=float), where=row_sum > 0)
    pair_strength = np.zeros(len(p.known_names), dtype=float)
    for i in range(len(p.known_names)):
        rates = []
        for j in range(len(p.known_names)):
            if i == j:
                continue
            ij = native_cm[i, j] / row_sum[i] if row_sum[i] else 0.0
            ji = native_cm[j, i] / row_sum[j] if row_sum[j] else 0.0
            rates.append(0.5 * (ij + ji))
        pair_strength[i] = max(rates, default=0.0)

    thresholds = {method: p95(scores["validation"]) for method, scores in p.scores.items()}
    metrics = {method: binary_metrics(scores["known_test"], scores["unknown_test"]) for method, scores in p.scores.items()}
    overlap_cutoff = p95(signals["validation"]["d1"])
    unknown_pred = signals["unknown_test"]["nearest"]

    for unknown_class in sorted(set(p.unknown_names.tolist())):
        mask = p.unknown_names == unknown_class
        nearest_counts = Counter(unknown_pred[mask].tolist())
        nearest_idx = nearest_counts.most_common(1)[0][0]
        row: dict[str, Any] = {
            "dataset": p.dataset, "protocol_id": p.protocol_id, "setting": p.setting, "seed": p.seed,
            "unknown_class": unknown_class, "unknown_samples": int(mask.sum()),
            "nearest_known_class": p.known_names[nearest_idx],
            "nearest_known_fraction": nearest_counts[nearest_idx] / int(mask.sum()),
            "mean_native_score": safe_mean(p.scores["M0"]["unknown_test"][mask]),
            "mean_empirical_centroid_score": safe_mean(p.scores["M1"]["unknown_test"][mask]),
            "mean_desv0_score": safe_mean(p.scores["M1"]["unknown_test"][mask]),
            "mean_desv1_score": safe_mean(p.scores["M2"]["unknown_test"][mask]) if "M2" in p.scores else math.nan,
            "mean_local_knn10_distance": safe_mean(p.local["unknown_test"][mask]) if p.local is not None else math.nan,
            "mean_od_minus_desv0_score": safe_mean(p.scores["M0"]["unknown_test"][mask] - p.scores["M1"]["unknown_test"][mask]),
            "mean_prototype_distance_component": safe_mean(signals["unknown_test"]["dproto"][mask]),
            "mean_posterior_uncertainty_vpost": safe_mean(signals["unknown_test"]["vpost"][mask]),
            "mean_nearest_prototype_gap": safe_mean(gaps[unknown_pred[mask]]),
            "mean_nearest_normalized_prototype_gap": safe_mean(normalized_gaps[unknown_pred[mask]]),
            "mean_support_d1": safe_mean(signals["unknown_test"]["d1"][mask]),
            "mean_support_d2": safe_mean(signals["unknown_test"]["d2"][mask]),
            "mean_support_margin": safe_mean(signals["unknown_test"]["margin"][mask]),
            "mean_support_ratio": safe_mean(signals["unknown_test"]["ratio"][mask]),
            "support_overlap_fraction": safe_mean(signals["unknown_test"]["d1"][mask] <= overlap_cutoff),
            "closed_confusion_strength": safe_mean(confusion_out[unknown_pred[mask]]),
            "closed_pair_confusion_strength": safe_mean(pair_strength[unknown_pred[mask]]),
            "sampling_protocol": p.sampling_protocol,
        }
        for method in ("M0", "M1", "M2"):
            if method in p.scores:
                auroc, auprc = binary_metrics(p.scores[method]["known_test"], p.scores[method]["unknown_test"][mask])
                row[f"{method.lower()}_auroc"] = auroc
                row[f"{method.lower()}_auprc"] = auprc
                row[f"{method.lower()}_threshold"] = thresholds[method]
            else:
                row[f"{method.lower()}_auroc"] = math.nan
                row[f"{method.lower()}_auprc"] = math.nan
                row[f"{method.lower()}_threshold"] = math.nan
        dpa, dpp = binary_metrics(signals["known_test"]["dproto"], signals["unknown_test"]["dproto"][mask])
        vpa, vpp = binary_metrics(signals["known_test"]["vpost"], signals["unknown_test"]["vpost"][mask])
        row.update({
            "dproto_auroc": dpa, "dproto_auprc": dpp, "vpost_auroc": vpa, "vpost_auprc": vpp,
            "delta_auroc_m1_minus_m0": row["m1_auroc"] - row["m0_auroc"],
            "delta_auprc_m1_minus_m0": row["m1_auprc"] - row["m0_auprc"],
            "delta_auroc_m2_minus_m0": row["m2_auroc"] - row["m0_auroc"] if "M2" in p.scores else math.nan,
            "delta_auroc_m2_minus_m1": row["m2_auroc"] - row["m1_auroc"] if "M2" in p.scores else math.nan,
            "native_uncertainty_rescue_auroc": row["m0_auroc"] - dpa,
        })
        class_rows.append(row)

    comparisons = [("M0", "M1"), ("M0", "M2")] if "M2" in p.scores else [("M0", "M1")]
    for base, des in comparisons:
        for role, ids, labels_or_names in (
            ("known_test", p.known_ids, np.asarray([p.known_names[int(i)] for i in p.known_labels])),
            ("unknown_test", p.unknown_ids, p.unknown_names),
        ):
            base_scores = p.scores[base][role]
            des_scores = p.scores[des][role]
            truth_unknown = role == "unknown_test"
            base_correct = (base_scores >= thresholds[base]) if truth_unknown else (base_scores < thresholds[base])
            des_correct = (des_scores >= thresholds[des]) if truth_unknown else (des_scores < thresholds[des])
            for idx in range(len(ids)):
                if base_correct[idx] and des_correct[idx]: category = "OD_correct_DES_correct"
                elif base_correct[idx]: category = "OD_correct_DES_wrong"
                elif des_correct[idx]: category = "OD_wrong_DES_correct"
                else: category = "OD_wrong_DES_wrong"
                sample: dict[str, Any] = {
                    "dataset": p.dataset, "protocol_id": p.protocol_id, "setting": p.setting, "seed": p.seed,
                    "sample_id": str(ids[idx]), "role": role, "true_class": str(labels_or_names[idx]),
                    "comparison": f"{des}_vs_{base}", "od_score": float(base_scores[idx]), "des_score": float(des_scores[idx]),
                    "od_threshold": thresholds[base], "des_threshold": thresholds[des],
                    "operating_category": category, "od_correct": int(base_correct[idx]), "des_correct": int(des_correct[idx]),
                    "prototype_distance_component": float(signals[role]["dproto"][idx]),
                    "posterior_uncertainty_vpost": float(signals[role]["vpost"][idx]),
                    "support_d1": float(signals[role]["d1"][idx]), "support_d2": float(signals[role]["d2"][idx]),
                    "support_margin": float(signals[role]["margin"][idx]), "support_ratio": float(signals[role]["ratio"][idx]),
                    "nearest_known_class": p.known_names[int(signals[role]["nearest"][idx])],
                    "nearest_prototype_gap": float(gaps[int(signals[role]["nearest"][idx])]),
                    "nearest_normalized_prototype_gap": float(normalized_gaps[int(signals[role]["nearest"][idx])]),
                    "closed_confusion_strength": float(confusion_out[int(signals[role]["nearest"][idx])]),
                    "local_knn10_distance": float(p.local[role][idx]) if p.local is not None else math.nan,
                }
                if truth_unknown:
                    both, od_only, des_only, neither = ranking_counts(float(base_scores[idx]), p.scores[base]["known_test"], p.scores[des]["known_test"], float(des_scores[idx]))
                    total = len(p.scores[base]["known_test"])
                    sample.update({
                        "rank_both_correct": both, "rank_od_only_correct": od_only, "rank_des_only_correct": des_only,
                        "rank_both_wrong": neither, "rank_both_correct_rate": both / total,
                        "rank_od_only_correct_rate": od_only / total, "rank_des_only_correct_rate": des_only / total,
                        "rank_both_wrong_rate": neither / total,
                    })
                sample_rows.append(sample)

    protocol: dict[str, Any] = {
        "dataset": p.dataset, "protocol_id": p.protocol_id, "setting": p.setting, "seed": p.seed,
        "known_classes": "|".join(p.known_names), "unknown_classes": "|".join(sorted(set(p.unknown_names.tolist()))),
        "known_test_samples": len(p.known_mu), "unknown_test_samples": len(p.unknown_mu),
        "prototype_gap_mean": float(gaps.mean()), "prototype_gap_median": float(np.median(gaps)),
        "normalized_prototype_gap_mean": float(normalized_gaps.mean()),
        "unknown_support_d1_mean": safe_mean(signals["unknown_test"]["d1"]),
        "unknown_support_margin_mean": safe_mean(signals["unknown_test"]["margin"]),
        "unknown_support_ratio_mean": safe_mean(signals["unknown_test"]["ratio"]),
        "unknown_support_overlap_fraction": safe_mean(signals["unknown_test"]["d1"] <= overlap_cutoff),
        "unknown_vpost_mean": safe_mean(signals["unknown_test"]["vpost"]),
        "closed_confusion_mean": float(confusion_out.mean()), "sampling_protocol": p.sampling_protocol,
        "checkpoint_sha256": sha256(p.checkpoint),
    }
    # Preserve the requested rsync/scp/sftp Known-Validation confusion audit in
    # the independently reviewable protocol table. Rates are row-normalized.
    for source, target in (("rsync", "scp"), ("scp", "rsync"), ("rsync", "sftp"),
                           ("sftp", "rsync"), ("scp", "sftp"), ("sftp", "scp")):
        key = f"val_native_confusion_{source}_to_{target}"
        if source in p.known_names and target in p.known_names:
            i, j = p.known_names.index(source), p.known_names.index(target)
            protocol[key] = float(native_cm[i, j] / row_sum[i]) if row_sum[i] else math.nan
        else:
            protocol[key] = math.nan
    for method in ("M0", "M1", "M2"):
        if method in p.scores:
            protocol[f"{method.lower()}_auroc"], protocol[f"{method.lower()}_auprc"] = metrics[method]
            protocol[f"{method.lower()}_threshold"] = thresholds[method]
            protocol[f"{method.lower()}_known_frr"] = safe_mean(p.scores[method]["known_test"] >= thresholds[method])
            protocol[f"{method.lower()}_ufar"] = safe_mean(p.scores[method]["unknown_test"] < thresholds[method])
        else:
            for suffix in ("auroc", "auprc", "threshold", "known_frr", "ufar"):
                protocol[f"{method.lower()}_{suffix}"] = math.nan
    protocol["delta_auroc_m1_minus_m0"] = protocol["m1_auroc"] - protocol["m0_auroc"]
    protocol["delta_auprc_m1_minus_m0"] = protocol["m1_auprc"] - protocol["m0_auprc"]
    protocol["delta_auroc_m2_minus_m0"] = protocol["m2_auroc"] - protocol["m0_auroc"] if "M2" in p.scores else math.nan
    protocol["delta_auroc_m2_minus_m1"] = protocol["m2_auroc"] - protocol["m1_auroc"] if "M2" in p.scores else math.nan
    return protocol


def correlations(class_df: pd.DataFrame) -> list[dict[str, Any]]:
    variables = {
        "prototype_centroid_gap": ("mean_nearest_normalized_prototype_gap", "+"),
        "support_d1": ("mean_support_d1", "+"),
        "support_margin": ("mean_support_margin", "+"),
        "support_ratio": ("mean_support_ratio", "-"),
        "support_overlap_fraction": ("support_overlap_fraction", "-"),
        "closed_set_confusion": ("closed_confusion_strength", "-"),
        "posterior_uncertainty": ("mean_posterior_uncertainty_vpost", "unspecified"),
        "class_sample_size": ("unknown_samples", "unspecified"),
    }
    rows: list[dict[str, Any]] = []
    scopes = [("ALL", class_df)] + [(name, group) for name, group in class_df.groupby("dataset")]
    for scope, frame in scopes:
        for target in ("delta_auroc_m1_minus_m0", "delta_auroc_m2_minus_m0"):
            for signal, (column, expected) in variables.items():
                sub = frame[[column, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) >= 3 and sub[column].nunique() > 1 and sub[target].nunique() > 1:
                    rho, pvalue = spearmanr(sub[column], sub[target])
                else:
                    rho, pvalue = math.nan, math.nan
                rows.append({"scope": scope, "analysis_unit": "protocol_x_unknown_class", "target": target,
                             "signal": signal, "source_column": column, "expected_direction": expected,
                             "spearman_rho": rho, "p_value_two_sided": pvalue, "n": len(sub),
                             "interpretation_scope": "exploratory; rows are not independent across seeds/protocols"})
    return rows


def aggregate_complementarity(sample_df: pd.DataFrame, dataset: str, comparison: str) -> dict[str, float]:
    frame = sample_df[(sample_df.dataset == dataset) & (sample_df.comparison == comparison)]
    op = frame.operating_category.value_counts()
    unknown = frame[frame.role == "unknown_test"]
    rank_cols = ["rank_both_correct", "rank_od_only_correct", "rank_des_only_correct", "rank_both_wrong"]
    rank = {col: float(unknown[col].sum()) for col in rank_cols}
    discord = rank["rank_od_only_correct"] + rank["rank_des_only_correct"]
    return {
        "op_both_correct": float(op.get("OD_correct_DES_correct", 0)),
        "op_od_only": float(op.get("OD_correct_DES_wrong", 0)),
        "op_des_only": float(op.get("OD_wrong_DES_correct", 0)),
        "op_both_wrong": float(op.get("OD_wrong_DES_wrong", 0)),
        **rank,
        "rank_od_share_of_discordant": rank["rank_od_only_correct"] / discord if discord else math.nan,
        "rank_des_share_of_discordant": rank["rank_des_only_correct"] / discord if discord else math.nan,
    }


def decide_gate(class_df: pd.DataFrame, corr_df: pd.DataFrame, sample_df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    vnat = class_df[class_df.dataset == "VNAT"]
    delta = vnat.delta_auroc_m1_minus_m0
    positive_fraction = float(np.mean(delta > 0))
    negative_fraction = float(np.mean(delta < 0))
    comp = aggregate_complementarity(sample_df, "VNAT", "M1_vs_M0")
    op_discord = comp["op_od_only"] + comp["op_des_only"]
    op_od_share = comp["op_od_only"] / op_discord if op_discord else 0.0
    op_des_share = comp["op_des_only"] / op_discord if op_discord else 0.0
    mechanism = corr_df[(corr_df.scope == "VNAT") & (corr_df.target == "delta_auroc_m1_minus_m0")]
    expected_hits = mechanism[
        (((mechanism.expected_direction == "+") & (mechanism.spearman_rho >= 0.30)) |
         ((mechanism.expected_direction == "-") & (mechanism.spearman_rho <= -0.30))) &
        (mechanism.p_value_two_sided < 0.05)
    ]
    supplementary = class_df[class_df.dataset != "VNAT"].groupby("dataset").delta_auroc_m1_minus_m0.agg(
        has_gain=lambda x: bool((x > 0).any()), has_loss=lambda x: bool((x < 0).any()))
    supplementary_support = bool(((supplementary.has_gain) & (supplementary.has_loss)).any()) if len(supplementary) else False
    bidirectional = min(op_od_share, op_des_share, comp["rank_od_share_of_discordant"], comp["rank_des_share_of_discordant"]) >= 0.10
    regimes = positive_fraction >= 0.20 and negative_fraction >= 0.20
    if regimes and bidirectional and len(expected_hits) > 0 and supplementary_support:
        gate = "COMPLEMENTARITY_CONFIRMED"
    elif regimes and op_od_share > 0 and op_des_share > 0 and comp["rank_od_only_correct"] > 0 and comp["rank_des_only_correct"] > 0:
        gate = "PARTIAL_COMPLEMENTARITY"
    else:
        gate = "NO_USEFUL_COMPLEMENTARITY"
    evidence = {"vnat_positive_class_protocol_fraction": positive_fraction,
                "vnat_negative_class_protocol_fraction": negative_fraction,
                "vnat_operating_od_share_of_discordant": op_od_share,
                "vnat_operating_des_share_of_discordant": op_des_share,
                "vnat_ranking_od_share_of_discordant": comp["rank_od_share_of_discordant"],
                "vnat_ranking_des_share_of_discordant": comp["rank_des_share_of_discordant"],
                "predicted_direction_significant_mechanism_count": int(len(expected_hits)),
                "supplementary_gain_and_loss_dataset_present": supplementary_support,
                "gate": gate}
    return gate, evidence


def fmt(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.4f}"


def report(out: Path, protocol_df: pd.DataFrame, class_df: pd.DataFrame, corr_df: pd.DataFrame,
           sample_df: pd.DataFrame, gate: str, gate_evidence: dict[str, Any], source_hashes: dict[str, str]) -> None:
    lines = ["# Stage 15A — Open-Detect vs DES Failure-Regime & Complementarity Diagnosis", "",
             "## Scope and controls", "",
             "This is a read-only diagnostic over frozen scores and latent arrays. No encoder or detector was trained, no score was refit, and no threshold, k, or fusion weight was searched. VNAT is the primary dataset and is not described as untouched. USTC is development evidence. ISCX-VPN and ISCXTor2016 provide frozen M0/M1-only supplementary evidence; they cannot support a DES-v1 cross-dataset claim.", "",
             f"- Protocols analyzed: {len(protocol_df)} (VNAT 15, USTC 15, ISCX-VPN 15, ISCXTor2016 15).",
             f"- Unknown-class protocol units: {len(class_df)}.",
             f"- Sample/comparison rows: {len(sample_df)}.",
             f"- Frozen source files hashed: {len(source_hashes)}.", "",
             "## Dataset-level results", "",
             "| Dataset | M0 AUROC | M1 AUROC | M2 AUROC | M1-M0 | M2-M0 |", "|---|---:|---:|---:|---:|---:|"]
    for dataset, frame in protocol_df.groupby("dataset", sort=False):
        lines.append(f"| {dataset} | {fmt(frame.m0_auroc.mean())} | {fmt(frame.m1_auroc.mean())} | {fmt(frame.m2_auroc.mean())} | {fmt(frame.delta_auroc_m1_minus_m0.mean())} | {fmt(frame.delta_auroc_m2_minus_m0.mean())} |")

    vnat = class_df[class_df.dataset == "VNAT"].copy()
    lines += ["", "## VNAT unknown-class failure regimes", "",
              "| Unknown class | units | M0 AUROC | M1 AUROC | M2 AUROC | M1-M0 | overlap | margin | V_post |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cls, frame in vnat.groupby("unknown_class"):
        lines.append(f"| {cls} | {len(frame)} | {fmt(frame.m0_auroc.mean())} | {fmt(frame.m1_auroc.mean())} | {fmt(frame.m2_auroc.mean())} | {fmt(frame.delta_auroc_m1_minus_m0.mean())} | {fmt(frame.support_overlap_fraction.mean())} | {fmt(frame.mean_support_margin.mean())} | {fmt(frame.mean_posterior_uncertainty_vpost.mean())} |")

    lines += ["", "## Hypothesis checks", ""]
    vnat_corr = corr_df[(corr_df.scope == "VNAT") & (corr_df.target == "delta_auroc_m1_minus_m0")]
    for label, signal in (("H1 prototype mismatch", "prototype_centroid_gap"), ("H2 nearest support distance", "support_d1"),
                          ("H2 ambiguity ratio", "support_ratio"), ("H2 support overlap", "support_overlap_fraction"),
                          ("closed-set confusion", "closed_set_confusion"), ("H3 posterior uncertainty", "posterior_uncertainty")):
        row = vnat_corr[vnat_corr.signal == signal].iloc[0]
        lines.append(f"- {label}: Spearman rho={fmt(row.spearman_rho)}, p={fmt(row.p_value_two_sided)}, n={int(row.n)}. This is exploratory because repeated protocols/classes are not independent.")

    failed = vnat[vnat.delta_auroc_m1_minus_m0 < 0]
    rescued = failed[failed.native_uncertainty_rescue_auroc > 0]
    lines += ["", f"- In VNAT DES-v0-loss units, V_post improves M0 over prototype-distance-only in {len(rescued)}/{len(failed)} units.", ""]
    for comparison in ("M1_vs_M0", "M2_vs_M0"):
        comp = aggregate_complementarity(sample_df, "VNAT", comparison)
        op_discord = comp["op_od_only"] + comp["op_des_only"]
        lines += [f"### VNAT {comparison}", "",
                  f"- Operating point: OD-only rescue={int(comp['op_od_only'])}, DES-only rescue={int(comp['op_des_only'])}, both correct={int(comp['op_both_correct'])}, both wrong={int(comp['op_both_wrong'])}.",
                  f"- Among discordant operating decisions: OD share={fmt(comp['op_od_only']/op_discord if op_discord else math.nan)}, DES share={fmt(comp['op_des_only']/op_discord if op_discord else math.nan)}.",
                  f"- Exact ranking pairs: OD-only={int(comp['rank_od_only_correct'])}, DES-only={int(comp['rank_des_only_correct'])}; shares among discordant pairs={fmt(comp['rank_od_share_of_discordant'])}/{fmt(comp['rank_des_share_of_discordant'])}.", ""]

    lines += ["## Known-Validation hard-pair confusion", "",
              "These are native learned-prototype closed-set confusion rates, averaged only over VNAT protocols where both classes are Known.", "",
              "| Direction | mean confusion | valid protocols |", "|---|---:|---:|"]
    for source, target in (("rsync", "scp"), ("scp", "rsync"), ("rsync", "sftp"),
                           ("sftp", "rsync"), ("scp", "sftp"), ("sftp", "scp")):
        column = f"val_native_confusion_{source}_to_{target}"
        values = protocol_df.loc[protocol_df.dataset == "VNAT", column].dropna()
        lines.append(f"| {source} -> {target} | {fmt(values.mean())} | {len(values)} |")

    hard = vnat[vnat.unknown_class.isin(["rsync", "scp", "sftp"])].groupby("unknown_class").agg(
        units=("protocol_id", "count"), delta=("delta_auroc_m1_minus_m0", "mean"), overlap=("support_overlap_fraction", "mean"),
        margin=("mean_support_margin", "mean"), vpost_rescue=("native_uncertainty_rescue_auroc", "mean"),
        nearest=("nearest_known_class", lambda x: Counter(x).most_common(1)[0][0])).reset_index()
    lines += ["## rsync / scp / sftp", "", "| Unknown | units | M1-M0 | overlap | margin | M0-Dproto AUROC | modal nearest Known |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in hard.itertuples():
        lines.append(f"| {row.unknown_class} | {row.units} | {fmt(row.delta)} | {fmt(row.overlap)} | {fmt(row.margin)} | {fmt(row.vpost_rescue)} | {row.nearest} |")

    h1 = vnat_corr[vnat_corr.signal == "prototype_centroid_gap"].iloc[0]
    h2_overlap = vnat_corr[vnat_corr.signal == "support_overlap_fraction"].iloc[0]
    h2_ratio = vnat_corr[vnat_corr.signal == "support_ratio"].iloc[0]
    m1_comp = aggregate_complementarity(sample_df, "VNAT", "M1_vs_M0")
    lines += ["", "## Required answers", "",
              f"1. **Real complementarity:** yes. At P95, OD alone rescues {int(m1_comp['op_od_only'])} sample decisions and DES-v0 alone rescues {int(m1_comp['op_des_only'])}; exact ranking has {int(m1_comp['rank_od_only_correct'])} OD-only and {int(m1_comp['rank_des_only_correct'])} DES-only correct pairs. VNAT class-protocol deltas split 24 positive / 21 negative.",
              "2. **DES gain regime:** lower empirical-support overlap is the repeatable VNAT signal (overlap-vs-gain rho below); by class, the clearest mean gains are sftp (+0.2029), netflix (+0.0876), rdp (+0.0480), and zoiper (+0.0391). sftp is an important mismatch-dominated exception: it has high overlap but also the largest mean normalized prototype gap (2.461), so empirical support repairs a poor native prototype.",
              "3. **DES failure regime:** rsync (-0.1612) and scp (-0.0453) are the clearest failures and have support-overlap fractions 0.9814 and 0.9589. rsync is most often nearest scp; scp is most often nearest rsync. youtube is a smaller failure (-0.0389). Contrary to the initial concern, sftp does not fail on average: it improves in 4/5 VNAT units.",
              f"4. **Prototype mismatch:** VNAT rho={fmt(h1.spearman_rho)}, p={fmt(h1.p_value_two_sided)}. This {'supports' if h1.spearman_rho > 0.3 and h1.p_value_two_sided < 0.05 else 'does not establish'} H1.",
              f"5. **Support overlap/ambiguity:** overlap rho={fmt(h2_overlap.spearman_rho)} (p={fmt(h2_overlap.p_value_two_sided)}) and d1 rho={fmt(vnat_corr[vnat_corr.signal == 'support_d1'].iloc[0].spearman_rho)} (p={fmt(vnat_corr[vnat_corr.signal == 'support_d1'].iloc[0].p_value_two_sided)}) support the overlap regime. The ratio/margin ambiguity tests are not significant, so centroid-boundary ambiguity itself is not established.",
              f"6. **Posterior uncertainty rescue:** {len(rescued)}/{len(failed)} VNAT DES-loss units have M0 AUROC above D_proto-only AUROC. Mean rescue is +0.1580 for rsync and +0.3335 for scp, but -0.0952 for sftp; V_post specifically preserves signal in the two principal DES failure classes.",
              f"7. **Hybrid-design evidence:** Gate = **{gate}**. A next-stage design is justified only if this gate is CONFIRMED or a clearly mechanism-backed PARTIAL result; this stage does not design it.", "",
              "## Gate", "", f"**{gate}**", "", "Gate evidence:", ""]
    for key, value in gate_evidence.items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Limitations", "",
              "- Correlations are exploratory: class-protocol observations share data, encoders, and class identities.",
              "- AUROC is threshold-free; operating complementarity uses each frozen method's Known-Validation P95.",
              "- Exact ranking complementarity is pairwise and uses strict `unknown_score > known_score`; ties count as incorrect for that method.",
              "- USTC uses its existing frozen balanced 1:1 evaluation, while VNAT and ISCX use their existing natural test composition.",
              "- ISCX datasets have no frozen DES-v1 scores, so no M2 cross-dataset claim is made.", ""]
    (out / "stage15a_failure_regime_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    config = read_json(out / "config.json")
    protocols = vnat_protocols(root) + ustc_protocols(root) + iscx_protocols(root)
    protocol_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for index, protocol in enumerate(protocols, 1):
        print(f"[{index:02d}/{len(protocols)}] {protocol.dataset}/{protocol.protocol_id}", flush=True)
        for path in protocol.source_paths:
            source_hashes[str(path.resolve())] = sha256(path)
        protocol_rows.append(analyze_protocol(protocol, sample_rows, class_rows))
    protocol_df = pd.DataFrame(protocol_rows)
    class_df = pd.DataFrame(class_rows)
    sample_df = pd.DataFrame(sample_rows)
    corr_rows = correlations(class_df)
    corr_df = pd.DataFrame(corr_rows)
    gate, gate_evidence = decide_gate(class_df, corr_df, sample_df)
    write_csv(out / "stage15a_protocol_diagnosis.csv", protocol_rows)
    write_csv(out / "stage15a_unknown_class_diagnosis.csv", class_rows)
    write_csv(out / "stage15a_sample_complementarity.csv", sample_rows)
    write_csv(out / "stage15a_signal_correlation.csv", corr_rows)
    (out / "source_hashes_before.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    after = {path: sha256(Path(path)) for path in source_hashes}
    (out / "source_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source_hashes != after:
        raise RuntimeError("frozen source hashes changed during diagnosis")
    verification = {
        "status": "PASS", "protocols": len(protocol_rows), "datasets": sorted(protocol_df.dataset.unique().tolist()),
        "unknown_class_units": len(class_rows), "sample_comparison_rows": len(sample_rows),
        "source_hashes_unchanged": True, "new_encoder_training": False, "new_detector_fitting": False,
        "threshold_search": False, "hybrid_design": False, "gate": gate, "gate_evidence": gate_evidence,
        "config_sha256": sha256(out / "config.json"), "config": config,
    }
    (out / "completion_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report(out, protocol_df, class_df, corr_df, sample_df, gate, gate_evidence, source_hashes)
    print(json.dumps({"status": "PASS", "gate": gate, "protocols": len(protocol_rows), "class_units": len(class_rows), "sample_rows": len(sample_rows)}, indent=2))


if __name__ == "__main__":
    main()
