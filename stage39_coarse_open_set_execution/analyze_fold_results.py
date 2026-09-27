"""Independent Stage39 deployment/domain analysis from frozen per-sample scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from freeze_protocols import ROOT, sha, write_json

METHODS = ("msp", "des_v1")
QUANTILES = (90, 95, 99)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"empty analysis table: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metric(rows, method, threshold):
    truth = np.asarray([int(r["role"] == "unknown_eval") for r in rows], dtype=np.int64)
    if len(set(truth)) != 2:
        raise RuntimeError("both Known and Unknown are required for AUROC")
    score = np.asarray([float(r[f"score_{method}"]) for r in rows], dtype=np.float64)
    reject = score > threshold
    known, unknown = truth == 0, truth == 1
    return {"auroc": float(roc_auc_score(truth, score)),
            "auprc": float(average_precision_score(truth, score)),
            "ufar": float(np.mean(~reject[unknown])),
            "known_frr": float(np.mean(reject[known])),
            "known_acceptance": float(np.mean(~reject[known])),
            "threshold": threshold, "known_test": int(known.sum()),
            "unknown_test": int(unknown.sum()),
            "unknown_prevalence": float(unknown.mean())}


def run(dataset, slug):
    fold = ROOT / "settings" / dataset / slug
    protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
    role_path = fold / "role_manifest.csv"
    if sha(role_path) != protocol["role_manifest_sha256"]:
        raise RuntimeError("frozen Stage39 role manifest changed")
    output = fold / "detection"
    if (output / "deployment_metrics.csv").exists() or (output / "domain_metrics.csv").exists():
        raise FileExistsError("deployment analysis already exists")
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "completion_verification.json").read_text(encoding="utf-8"))
    if calibration["status"] != "PASS" or completion["status"] != "PASS" or \
       calibration["role_manifest_sha256"] != protocol["role_manifest_sha256"] or \
       calibration["unknown_used_for_fitting"] or calibration["test_used_for_fitting"]:
        raise RuntimeError("Known-only calibration/evaluation verification absent")
    samples = load_csv(output / "sample_scores.csv")
    original = {r["sample_id"]: r for r in load_csv(role_path)}
    if len(samples) != len({r["sample_id"] for r in samples}) or \
       set(original[r["sample_id"]]["role"] for r in samples) - {
           "known_test", "unknown_eval", "auxiliary_known_service_unseen_app"}:
        raise RuntimeError("sample ID/role mismatch")
    for row in samples:
        source = original.get(row["sample_id"])
        if source is None or any(row[key] != source[key] for key in
                                 ("role", "service", "application", "domain", "group_id")):
            raise RuntimeError(f"saved score metadata changed: {row['sample_id']}")
    class_services = defaultdict(set)
    for source in original.values():
        if source["role"] == "known_train":
            class_services[source["classifier_label"]].add(source["service"])
    if set(class_services) != set(protocol["known_classifier_labels"]) or \
       any(len(services) != 1 for services in class_services.values()):
        raise RuntimeError("known class to service is not deterministic")
    class_service = {name: next(iter(services)) for name, services in class_services.items()}
    known_services = sorted(set(class_service.values()))
    for row in samples:
        if row["predicted_known_class"] not in class_service:
            raise RuntimeError("predicted class absent from frozen Known mapping")

    deployment, domains = [], []
    for method in METHODS:
        for q in QUANTILES:
            threshold = float(calibration["thresholds"][method][str(q)])
            primary = [r for r in samples if r["role"] in ("known_test", "unknown_eval")]
            base = metric(primary, method, threshold)
            original_primary = next(r for r in load_csv(output / "open_set_metrics.csv")
                                    if r["method"] == method and int(r["percentile"]) == q)
            if any(not math.isclose(base[key], float(original_primary[key]), abs_tol=1e-10)
                   for key in ("auroc", "auprc", "ufar", "known_frr")):
                raise RuntimeError(f"primary metric replay failed: {method}/P{q}")
            combined = [r for r in samples if r["role"] in (
                "known_test", "auxiliary_known_service_unseen_app", "unknown_eval")]
            score = metric(combined, method, threshold)
            known = [r for r in combined if r["role"] != "unknown_eval"]
            truth = [r["service"] for r in known]
            prediction = [class_service[r["predicted_known_class"]] for r in known]
            post = ["__UNKNOWN__" if float(r[f"score_{method}"]) > threshold else
                    class_service[r["predicted_known_class"]] for r in known]
            deployment.append({"dataset": dataset, "fold": slug, "method": method,
                "percentile": q, "evaluation_group": "deployment_known_plus_unseen_app_vs_unknown_service",
                **score, "auxiliary_known_n": sum(r["role"] == "auxiliary_known_service_unseen_app" for r in known),
                "known_service_accuracy_pre_reject": float(accuracy_score(truth, prediction)),
                "known_service_macro_f1_pre_reject": float(f1_score(truth, prediction,
                    labels=known_services, average="macro", zero_division=0)),
                "known_service_accuracy_post_reject": float(accuracy_score(truth, post)),
                "known_service_macro_f1_post_reject": float(f1_score(truth, post,
                    labels=known_services, average="macro", zero_division=0))})
            for domain in sorted({r["domain"] for r in samples}):
                scoped = [r for r in samples if r["domain"] == domain]
                u = [r for r in scoped if r["role"] == "unknown_eval"]
                k = [r for r in scoped if r["role"] == "known_test"]
                a = [r for r in scoped if r["role"] == "auxiliary_known_service_unseen_app"]
                domains.append({"dataset": dataset, "fold": slug, "method": method,
                    "percentile": q, "domain": domain, "unknown_n": len(u),
                    "known_test_n": len(k), "auxiliary_known_n": len(a),
                    "unknown_ufar": float(np.mean([float(r[f"score_{method}"]) <= threshold
                        for r in u])) if u else None,
                    "known_test_frr": float(np.mean([float(r[f"score_{method}"]) > threshold
                        for r in k])) if k else None,
                    "deployment_known_frr": float(np.mean([float(r[f"score_{method}"]) > threshold
                        for r in k + a])) if k or a else None})
    write_csv(output / "deployment_metrics.csv", deployment)
    write_csv(output / "domain_metrics.csv", domains)
    write_json(output / "deployment_verification.json", {"status": "PASS", "dataset": dataset,
        "fold": slug, "score_rows": len(samples), "methods": list(METHODS),
        "quantiles": list(QUANTILES), "primary_metrics_replayed": 6,
        "role_counts": dict(Counter(r["role"] for r in samples)),
        "class_to_service": class_service, "role_manifest_sha256": sha(role_path),
        "unknown_used_for_fitting": 0, "test_used_for_parameter_selection": 0})
    print(json.dumps({"status": "PASS", "dataset": dataset, "fold": slug,
                      "deployment_p95_des_v1": next(r for r in deployment if
                          r["method"] == "des_v1" and r["percentile"] == 95)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "VNAT"), required=True)
    parser.add_argument("--fold", required=True)
    args = parser.parse_args()
    run(args.dataset, args.fold)
