"""Stage39 service holdout: Known-only calibration, then frozen evaluation."""
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
from sklearn.metrics import accuracy_score, f1_score

from freeze_protocols import PROJECT, ROOT, sha, write_json

S31 = PROJECT / "stage31_four_dataset_three_view_equal"
S38 = PROJECT / "stage38_three_view_des_open_set_transfer"
METHODS = ("msp", "des_v1")
VIEWS = ("trafficformer", "graph", "yatc")
EVAL_ROLES = ("known_test", "unknown_eval", "auxiliary_known_service_unseen_app")
sys.path.insert(0, str(S38))
import run_iscx_detection as prior  # noqa: E402


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"empty result: {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def context(dataset: str, slug: str):
    if dataset not in ("iscx_vpn", "iscx_tor", "VNAT"):
        raise ValueError(dataset)
    fold = ROOT / "settings" / dataset / slug
    p = jload(fold / "protocol.json")
    if p["status"] != "FROZEN_PRETRAIN" or p["checkpoint_reuse"] or \
       sha(fold / "role_manifest.csv") != p["role_manifest_sha256"]:
        raise RuntimeError("Stage39 frozen protocol/hash mismatch")
    rows = list(csv.DictReader((fold / "role_manifest.csv").open(newline="", encoding="utf-8")))
    if Counter(r["role"] for r in rows) != p["counts"]:
        raise RuntimeError("frozen role count mismatch")
    if any(r["service"] in p["unknown_services"] for r in rows
           if r["role"] in ("known_train", "known_validation")):
        raise RuntimeError("Unknown service leaked into fitting roles")
    run = fold / "runs" / ("vnat" if dataset == "VNAT" else dataset) / p["protocol_id"]
    hashes = {"role_manifest.csv": sha(fold / "role_manifest.csv")}
    for view in VIEWS:
        sub = run / view
        meta = jload(sub / "metrics.json")
        if not (sub / "SUCCESS").is_file() or sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"frozen branch invalid: {view}")
        hashes[f"{view}/model_best.pt"] = meta["checkpoint_sha256"]
    fusion = run / "T0_equal"
    meta = jload(fusion / "known_validation_metrics.json")
    if not (fusion / "SUCCESS").is_file():
        raise RuntimeError("frozen fusion head incomplete")
    for name, expected in meta["checkpoint_hashes"].items():
        if sha(fusion / name) != expected:
            raise RuntimeError(f"frozen fusion head hash mismatch: {name}")
        hashes[f"T0_equal/{name}"] = expected
    return fold, p, rows, run, hashes


def model(run: Path, classes: list[str], device: torch.device):
    fusion = run / "T0_equal"
    adapter = prior.Adapters(len(classes)).to(device)
    adapter.load_state_dict(torch.load(fusion / "adapters_best.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    adapter.eval()
    head = prior.EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(fusion / "T0_equal_best.pt", map_location="cpu",
                                    weights_only=False)["state_dict"])
    head.eval()
    with np.load(fusion / "known_train_view_scalers.npz", allow_pickle=False) as file:
        scaler = {key: file[key].copy() for key in file.files}
    return adapter, head, scaler


def known_features(run: Path, rows: list[dict], role: str, classes: list[str]):
    selected = sorted((r for r in rows if r["role"] == role), key=lambda r: r["sample_id"])
    ids = [r["sample_id"] for r in selected]
    features = {}
    for view, dim in (("trafficformer", 768), ("graph", 128), ("yatc", 192)):
        sub = run / view
        saved = np.load(sub / f"{role}_flow_ids.npy", allow_pickle=False).astype(str).tolist()
        if saved != ids:
            raise RuntimeError(f"frozen Known {role}/{view} flow ID mismatch")
        value = np.load(sub / f"{role}_features.npy", allow_pickle=False).astype(np.float32)
        if value.shape != (len(ids), dim) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid Known {role}/{view} features")
        features[view] = value
    labels = np.asarray([classes.index(r["classifier_label"]) for r in selected], dtype=np.int64)
    return ids, labels, features


def calibrate(dataset: str, slug: str):
    torch.set_num_threads(4)
    fold, p, rows, run, hashes = context(dataset, slug)
    out = fold / "detection"
    if out.exists():
        raise FileExistsError(out)
    classes = p["known_classifier_labels"]
    device = torch.device("cpu")
    adapter, head, scaler = model(run, classes, device)
    values = {}
    for role in ("known_train", "known_validation"):
        ids, labels, features = known_features(run, rows, role, classes)
        h, logits = prior.fused(features, device, adapter, head, scaler)
        values[role] = (ids, labels, h, logits)
    train_ids, train_y, train_h, _ = values["known_train"]
    val_ids, val_y, val_h, val_logits = values["known_validation"]
    expected = list(csv.DictReader((run / "T0_equal/known_validation_predictions.csv").open(newline="", encoding="utf-8")))
    if [r["flow_id"] for r in expected] != val_ids or \
       [r["predicted_class"] for r in expected] != [classes[i] for i in val_logits.argmax(1)]:
        raise RuntimeError("Known Validation prediction replay mismatch")
    old = jload(run / "T0_equal/known_validation_metrics.json")["metrics"]
    replay = prior.metrics(val_y, val_logits.argmax(1), len(classes))[0]
    if any(not math.isclose(replay[key], old[key], abs_tol=1e-6)
           for key in ("accuracy", "macro_f1", "weighted_f1")):
        raise RuntimeError("Known Validation metric replay mismatch")
    centers, support = prior.support_models(train_h, train_y, classes)
    val_raw = prior.raw_scores(val_h, val_logits, centers, support)
    median = {key: float(np.median(val_raw[key])) for key in ("centroid", "local")}
    mad = {key: float(np.median(np.abs(val_raw[key] - median[key]))) for key in median}
    prior.add_des_v1(val_raw, median, mad)
    thresholds = {method: {str(q): float(np.percentile(val_raw[method], q, method="higher"))
                           for q in (90, 95, 99)} for method in METHODS}
    out.mkdir(parents=True)
    np.savez(out / "known_train.npz", flow_ids=np.asarray(train_ids, dtype="U80"),
             labels=train_y, h=train_h)
    np.savez(out / "known_validation.npz", flow_ids=np.asarray(val_ids, dtype="U80"),
             labels=val_y, h=val_h, logits=val_logits)
    csv_write(out / "known_validation_scores.csv", [{"sample_id": uid, "role": "known_validation",
        "true_class": classes[int(val_y[i])], "predicted_class": classes[int(val_logits[i].argmax())],
        "score_msp": float(val_raw["msp"][i]), "score_des_v1": float(val_raw["des_v1"][i])}
        for i, uid in enumerate(val_ids)])
    write_json(out / "calibration.json", {"status": "PASS", "dataset": dataset, "fold": slug,
        "known_train": len(train_ids), "known_validation": len(val_ids), "classes": classes,
        "known_validation_metrics": replay, "median": median, "mad": mad,
        "thresholds": thresholds, "methods": list(METHODS), "percentiles": [90, 95, 99],
        "quantile_method": "higher", "decision_rule": "score > threshold", "knn_k": 10,
        "global_local_weights": [0.5, 0.5], "unknown_used_for_fitting": 0,
        "test_used_for_fitting": 0, "checkpoint_hashes": hashes,
        "role_manifest_sha256": p["role_manifest_sha256"]})
    write_json(out / "calibration_verification.json", {"status": "PASS",
        "known_validation_prediction_replay": True, "known_validation_metric_replay": True,
        "unknown_feature_values_loaded": 0, "test_feature_values_loaded": 0})
    print(json.dumps({"status": "PASS", "phase": "calibrate", "dataset": dataset,
                      "fold": slug, "known_validation_macro_f1": replay["macro_f1"]}), flush=True)


def evaluate(dataset: str, slug: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen branch inference")
    torch.set_num_threads(4)
    fold, p, rows, run, hashes = context(dataset, slug)
    out = fold / "detection"
    calibration = jload(out / "calibration.json")
    if calibration["status"] != "PASS" or calibration["checkpoint_hashes"] != hashes or \
       not (out / "calibration_verification.json").is_file() or \
       (out / "sample_scores.csv").exists():
        raise RuntimeError("Known-only calibration absent, changed, or evaluation already exists")
    selected = sorted((r for r in rows if r["role"] in EVAL_ROLES), key=lambda r: r["sample_id"])
    ids = [r["sample_id"] for r in selected]
    prior.ROOT = fold  # reuse the tested Stage38 branch inference with this fold's frozen models/cache
    prior.PROTOCOL = p["protocol_id"]
    adapter, head, scaler = model(run, p["known_classifier_labels"], torch.device("cuda:0"))
    features = prior.test_branches("vnat" if dataset == "VNAT" else dataset,
                                   p["known_classifier_labels"], ids, torch.device("cuda:0"))
    h, logits = prior.fused(features, torch.device("cuda:0"), adapter, head, scaler)
    with np.load(out / "known_train.npz", allow_pickle=False) as train:
        train_h, train_y = train["h"].copy(), train["labels"].copy()
    centers, support = prior.support_models(train_h, train_y, p["known_classifier_labels"])
    raw = prior.raw_scores(h, logits, centers, support)
    prior.add_des_v1(raw, calibration["median"], calibration["mad"])
    classes = p["known_classifier_labels"]
    np.savez(out / "evaluation_vectors.npz", flow_ids=np.asarray(ids, dtype="U80"), h=h, logits=logits)
    score_rows = []
    for i, row in enumerate(selected):
        record = {"sample_id": ids[i], "role": row["role"], "original_role": row["original_role"],
            "service": row["service"], "application": row["application"],
            "domain": row["domain"], "group_id": row["group_id"],
            "true_known_class": row["classifier_label"],
            "predicted_known_class": classes[int(logits[i].argmax())],
            "is_unknown_service": int(row["role"] == "unknown_eval")}
        for method in METHODS:
            record[f"score_{method}"] = float(raw[method][i])
            for q in (90, 95, 99):
                threshold = calibration["thresholds"][method][str(q)]
                record[f"reject_{method}_p{q}"] = int(raw[method][i] > threshold)
        score_rows.append(record)
    csv_write(out / "sample_scores.csv", score_rows)
    metric_rows, per_service, auxiliary = [], [], []
    for method in METHODS:
        for q in (90, 95, 99):
            threshold = calibration["thresholds"][method][str(q)]
            base = [i for i, r in enumerate(selected) if r["role"] in ("known_test", "unknown_eval")]
            truth = np.asarray([int(selected[i]["role"] == "unknown_eval") for i in base], dtype=np.int64)
            score = np.asarray([raw[method][i] for i in base])
            result = prior.metric(truth, score, threshold)
            metric_rows.append({"dataset": dataset, "fold": slug, "method": method,
                "percentile": q, "evaluation_group": "primary_known_test_vs_unknown_service",
                "unknown_prevalence": float(truth.mean()), **result})
            for service in sorted({r["service"] for r in selected if r["role"] == "unknown_eval"}):
                keep = [i for i, r in enumerate(selected) if r["role"] == "known_test" or
                        (r["role"] == "unknown_eval" and r["service"] == service)]
                yt = np.asarray([int(selected[i]["role"] == "unknown_eval") for i in keep], dtype=np.int64)
                per_service.append({"dataset": dataset, "fold": slug, "method": method,
                    "percentile": q, "unknown_service": service, "unknown_prevalence": float(yt.mean()),
                    **prior.metric(yt, raw[method][keep], threshold)})
            aux = [i for i, r in enumerate(selected) if r["role"] == "auxiliary_known_service_unseen_app"]
            known = [i for i, r in enumerate(selected) if r["role"] == "known_test"]
            combined = known + aux
            auxiliary.append({"dataset": dataset, "fold": slug, "method": method, "percentile": q,
                "auxiliary_n": len(aux), "auxiliary_acceptance": float(np.mean(raw[method][aux] <= threshold)) if aux else None,
                "auxiliary_correct_service_and_accepted": float(np.mean([
                    raw[method][i] <= threshold and classes[int(logits[i].argmax())] == selected[i]["classifier_label"]
                    for i in aux])) if aux else None,
                "combined_known_n": len(combined),
                "combined_known_frr": float(np.mean(raw[method][combined] > threshold))})
    csv_write(out / "open_set_metrics.csv", metric_rows)
    csv_write(out / "per_unknown_service.csv", per_service)
    csv_write(out / "auxiliary_known_service.csv", auxiliary)
    known = [i for i, r in enumerate(selected) if r["role"] == "known_test"]
    yt = np.asarray([classes.index(selected[i]["classifier_label"]) for i in known], dtype=np.int64)
    yp = logits[known].argmax(1)
    closed, per = prior.metrics(yt, yp, len(classes))
    threshold = calibration["thresholds"]["des_v1"]["95"]
    rejected = raw["des_v1"][known] > threshold
    after = np.where(rejected, len(classes), yp)
    write_json(out / "known_test_closed_set.json", {"status": "PASS", "known_test": len(known),
        "before_rejection": closed, "after_des_v1_p95": {
            "accuracy": float(accuracy_score(yt, after)),
            "macro_f1_reject_as_error": float(f1_score(yt, after, labels=list(range(len(classes))),
                                                 average="macro", zero_division=0)),
            "known_acceptance": float(np.mean(~rejected))},
        "per_class": [{"class": name, "precision": float(per[0][j]),
                       "recall": float(per[1][j]), "f1": float(per[2][j]),
                       "support": int(per[3][j])} for j, name in enumerate(classes)]})
    write_json(out / "evaluation_audit.json", {"status": "PASS", "dataset": dataset, "fold": slug,
        "checkpoint_hashes_unchanged": hashes == context(dataset, slug)[4],
        "known_train_support_count": calibration["known_train"],
        "known_validation_calibration_count": calibration["known_validation"],
        "unknown_used_for_fitting": 0, "known_test_used_for_fitting": 0,
        "role_counts": dict(Counter(r["role"] for r in selected)), "score_rows": len(score_rows)})
    print(json.dumps({"status": "PASS", "phase": "evaluate", "dataset": dataset,
                      "fold": slug, "known_test_macro_f1": closed["macro_f1"],
                      "des_v1_p95": next(r for r in metric_rows if r["method"] == "des_v1" and r["percentile"] == 95)}), flush=True)


def verify(dataset: str, slug: str):
    fold, p, _, _, hashes = context(dataset, slug)
    out = fold / "detection"
    audit = jload(out / "evaluation_audit.json")
    calibration = jload(out / "calibration.json")
    if audit["status"] != "PASS" or not audit["checkpoint_hashes_unchanged"] or \
       calibration["checkpoint_hashes"] != hashes:
        raise RuntimeError("frozen source/checkpoint audit failed")
    saved = list(csv.DictReader((out / "sample_scores.csv").open(newline="", encoding="utf-8")))
    if len(saved) != audit["score_rows"] or len({r["sample_id"] for r in saved}) != len(saved):
        raise RuntimeError("saved sample scores incomplete or duplicate")
    with np.load(out / "evaluation_vectors.npz", allow_pickle=False) as vectors:
        if vectors["flow_ids"].astype(str).tolist() != [r["sample_id"] for r in saved]:
            raise RuntimeError("saved logits/score flow ID mismatch")
    expected = {(r["method"], int(r["percentile"])): r for r in
                csv.DictReader((out / "open_set_metrics.csv").open(newline="", encoding="utf-8"))}
    base = [r for r in saved if r["role"] in ("known_test", "unknown_eval")]
    truth = np.asarray([int(r["role"] == "unknown_eval") for r in base], dtype=np.int64)
    for method in METHODS:
        for q in (90, 95, 99):
            threshold = calibration["thresholds"][method][str(q)]
            score = np.asarray([float(r[f"score_{method}"]) for r in base])
            if any(int(r[f"reject_{method}_p{q}"]) != int(float(r[f"score_{method}"]) > threshold)
                   for r in saved):
                raise RuntimeError(f"decision replay mismatch: {method}/p{q}")
            replay = prior.metric(truth, score, threshold)
            old = expected[(method, q)]
            if any(not math.isclose(value, float(old[key]), abs_tol=1e-10) for key, value in replay.items()):
                raise RuntimeError(f"metric replay mismatch: {method}/p{q}")
    write_json(out / "completion_verification.json", {"status": "PASS", "dataset": dataset,
        "fold": slug, "saved_score_rows": len(saved), "methods": list(METHODS),
        "thresholds_replayed": 6, "unknown_used_for_fitting": 0,
        "known_test_used_for_fitting": 0, "protected_hashes_unchanged": hashes == context(dataset, slug)[4]})
    print(json.dumps({"status": "PASS", "phase": "verify", "dataset": dataset, "fold": slug}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "evaluate", "verify"))
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "VNAT"), required=True)
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()
    {"calibrate": calibrate, "evaluate": evaluate, "verify": verify}[args.phase](args.dataset, args.fold)
