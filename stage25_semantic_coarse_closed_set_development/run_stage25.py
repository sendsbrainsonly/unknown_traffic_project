#!/usr/bin/env python3
"""Frozen-flow semantic-coarse closed-set development; no encoder training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)
ROLES = ("known_validation", "known_test")
METHODS = ("E1", "E3", "OD", "RoNeTC", "YaTC", "TFE", "Trident")
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
STAGE21 = PROJECT / "stage21_coarse_service_ours_e3_benchmark" / "feature_cache"
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "runs"
STAGE23 = PROJECT / "stage23_closed_set_method_table"
STAGE23B = PROJECT / "stage23b_tfe_trident_same_flow_adaptation"
METHOD_REFERENCE = {
    "E1": "TrafficFormer-pretrained", "E3": "OURS-E3-T8-pretrained",
    "OD": "Open-Detect corrected-paper", "RoNeTC": "RoNeTC-runnable-reconstruction",
    "YaTC": "YaTC-official-pretrained-Stage20", "TFE": "TFE-GNN-8-UDP-short",
    "Trident": "Trident-early8-86D",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError(f"empty CSV without schema: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def coarse(fine: str) -> str:
    return "Communication" if fine in {"Chat", "Email", "VoIP"} else fine


def prediction_path(dataset: str, seed: int, method: str, role: str) -> Path:
    s = f"seed{seed}"
    if method in {"E1", "E3"}:
        return STAGE22 / dataset / s / "predictions.csv"
    if method == "OD":
        return STAGE23 / "runs" / "opendetect_corrected_paper" / dataset / s / "predictions.csv"
    if method == "RoNeTC":
        return STAGE23 / "runs" / "ronetc_stage20" / dataset / f"{s}_formal" / "predictions.csv"
    if method == "YaTC":
        name = "validation_predictions.csv" if role == "known_validation" else "test_predictions_cuda.csv"
        return STAGE23 / "runs" / "yatc_stage20" / dataset / f"{s}_formal" / name
    branch = "tfe" if method == "TFE" else "trident"
    run = STAGE23B / "runs" / branch / dataset / s
    if method == "TFE" and role == "known_validation":
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        if selection["status"] != "SELECTION_COMPLETE" or selection["dataset"] != dataset or selection["seed"] != seed:
            raise ValueError(f"invalid frozen TFE selection: {run}")
        return run / f"known_val_epoch{int(selection['best_epoch']):03d}.csv"
    suffix = "known_val" if role == "known_validation" else "known_test"
    return run / f"{suffix}_predictions.csv"


def source_files() -> dict[str, Path]:
    paths = {"stage20_manifest": STAGE20, "historical_result_table":
             STAGE23B / "stage23b_all_methods_run_level.csv"}
    for dataset in DATASETS:
        paths[f"stage21_{dataset}"] = STAGE21 / dataset / "e3_t8_inputs.npz"
        for seed in SEEDS:
            paths[f"stage22_repr_{dataset}_{seed}"] = STAGE22 / dataset / f"seed{seed}" / "representations.npz"
            for method in METHODS:
                if method == "TFE":
                    p = STAGE23B / "runs" / "tfe" / dataset / f"seed{seed}" / "selection.json"
                    paths[str(p.relative_to(PROJECT))] = p
                for role in ROLES:
                    p = prediction_path(dataset, seed, method, role)
                    paths[str(p.relative_to(PROJECT))] = p
    return paths


def metadata() -> tuple[dict[tuple[str, str], dict], dict[str, dict[str, int]]]:
    manifest = {}
    for row in csv_rows(STAGE20):
        key = row["dataset"], row["flow_id"]
        if key in manifest:
            raise ValueError(f"duplicate Stage20 flow: {key}")
        manifest[key] = row
    packets = {}
    for dataset in DATASETS:
        with np.load(STAGE21 / dataset / "e3_t8_inputs.npz", allow_pickle=False) as z:
            ids, counts = z["flow_ids"], z["packet_counts"]
            if len(ids) != len(set(ids.tolist())):
                raise ValueError(f"duplicate cache flow: {dataset}")
            packets[dataset] = dict(zip(ids.tolist(), counts.astype(int).tolist(), strict=True))
        expected = {flow_id for ds, flow_id in manifest if ds == dataset}
        if set(packets[dataset]) != expected:
            raise ValueError(f"cache/manifest mismatch: {dataset}")
    return manifest, packets


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[dict, list[dict]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    result = {"accuracy": float(accuracy_score(y_true, y_pred)),
              "macro_f1": float(np.mean(f1)),
              "weighted_f1": float(np.average(f1, weights=support))}
    per_class = [{"class": name, "precision": float(p), "recall": float(r),
                  "f1": float(f), "support": int(n)}
                 for name, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)]
    return result, per_class


def source_hash_check(before: dict[str, str]) -> dict[str, str]:
    after = {name: digest(path) for name, path in source_files().items()}
    if before != after:
        save_json(OUT / "protected_hashes_after.json", after)
        raise RuntimeError("protected source changed")
    save_json(OUT / "protected_hashes_after.json", after)
    return after


def phase_a() -> None:
    before = {name: digest(path) for name, path in source_files().items()}
    save_json(OUT / "protected_hashes_before.json", before)
    manifest, packets = metadata()
    historical = {(r["dataset"], int(r["seed"]), r["method"]): r
                  for r in csv_rows(STAGE23B / "stage23b_all_methods_run_level.csv")}
    results, class_rows, predictions, coverage = [], [], [], []
    flow_roles = defaultdict(dict)
    for (dataset, flow_id), row in manifest.items():
        flow_roles[(dataset, row["closed_role"])][flow_id] = row
    for dataset in DATASETS:
        fine_labels = sorted({row["service_label"] for (ds, _), row in manifest.items() if ds == dataset})
        coarse_labels = sorted({coarse(label) for label in fine_labels})
        if len(coarse_labels) != (4 if dataset == "iscx_vpn" else 5):
            raise ValueError(f"unexpected coarse class count: {dataset}")
        for role in ("known_train", *ROLES):
            for label in fine_labels:
                rows = [r for r in flow_roles[(dataset, role)].values() if r["service_label"] == label]
                keep = sum(packets[dataset][r["flow_id"]] >= 2 for r in rows)
                coverage.append({"dataset": dataset, "role": role, "fine_class": label,
                                 "coarse_class": coarse(label), "all_flows": len(rows),
                                 "retained_ge2": keep, "excluded_one_packet": len(rows)-keep,
                                 "retention_rate": keep/len(rows) if rows else 0})
        for seed in SEEDS:
            for method in METHODS:
                for role in ROLES:
                    path = prediction_path(dataset, seed, method, role)
                    raw = csv_rows(path)
                    if method in {"E1", "E3"}:
                        raw = [r for r in raw if r["role"] == role and r["encoder"] == method]
                    elif path.name == "predictions.csv":
                        raw = [r for r in raw if r["role"] == role]
                    expected = flow_roles[(dataset, role)]
                    if len(raw) != len(expected) or {r["flow_id"] for r in raw} != set(expected):
                        raise ValueError(f"prediction flow mismatch: {dataset}/{seed}/{method}/{role}")
                    if len({r["flow_id"] for r in raw}) != len(raw):
                        raise ValueError(f"duplicate prediction flow: {path}")
                    aligned = []
                    for row in raw:
                        ref = expected[row["flow_id"]]
                        if row["true_service"] != ref["service_label"] or row["dataset"] != dataset:
                            raise ValueError(f"label/data mismatch: {path}/{row['flow_id']}")
                        if row["predicted_service"] not in fine_labels:
                            raise ValueError(f"invalid prediction class: {path}")
                        if "seed" in row and int(row["seed"]) != seed:
                            raise ValueError(f"seed mismatch: {path}")
                        n = packets[dataset][row["flow_id"]]
                        item = {"dataset": dataset, "seed": seed, "method": method, "role": role,
                                "flow_id": row["flow_id"], "packet_count_capped8": n,
                                "true_fine": ref["service_label"], "pred_fine": row["predicted_service"],
                                "true_coarse": coarse(ref["service_label"]),
                                "pred_coarse": coarse(row["predicted_service"]),
                                "retained_ge2": int(n >= 2)}
                        aligned.append(item)
                        predictions.append(item)
                    for scope in ("all", "ge2"):
                        subset = aligned if scope == "all" else [r for r in aligned if r["retained_ge2"]]
                        for label_level, labels in (("fine", fine_labels), ("coarse", coarse_labels)):
                            y_true = [r[f"true_{label_level}"] for r in subset]
                            y_pred = [r[f"pred_{label_level}"] for r in subset]
                            score, per = metrics(y_true, y_pred, labels)
                            results.append({"dataset": dataset, "seed": seed, "method": method,
                                            "role": role, "scope": scope, "label_level": label_level,
                                            "samples": len(subset), "source_path": str(path.relative_to(PROJECT)),
                                            **score})
                            class_rows.extend({"dataset": dataset, "seed": seed, "method": method,
                                               "role": role, "scope": scope, "label_level": label_level,
                                               **row} for row in per)
                            if role == "known_test" and scope == "all" and label_level == "fine":
                                ref = historical[(dataset, seed, METHOD_REFERENCE[method])]
                                if any(abs(score[k]-float(ref[k])) > 1e-9
                                       for k in ("accuracy", "macro_f1", "weighted_f1")):
                                    raise ValueError(f"historical metric replay failed: {dataset}/{seed}/{method}")
    save_csv(OUT / "remap_run_results.csv", results)
    save_csv(OUT / "remap_per_class.csv", class_rows)
    save_csv(OUT / "remap_predictions.csv", predictions)
    save_csv(OUT / "scope_coverage.csv", coverage)
    source_hash_check(before)
    save_json(OUT / "phase_a_verification.json", {"status": "PASS", "matched_prediction_cells": 56,
              "metric_rows": len(results), "test_metric_replays": 28,
              "protected_assets": len(before), "unknown_usage": 0,
              "label_map": {"Chat": "Communication", "Email": "Communication", "VoIP": "Communication"}})
    (OUT / "PHASE_A_COMPLETE").write_text("verified\n", encoding="utf-8")
    print(f"PHASE_A_COMPLETE predictions={len(predictions)} metrics={len(results)} sources={len(before)}", flush=True)


def phase_b() -> None:
    if not (OUT / "PHASE_A_COMPLETE").is_file():
        raise RuntimeError("run A before B")
    before = json.loads((OUT / "protected_hashes_before.json").read_text())
    manifest, packets = metadata()
    results, per_class, predictions, warning_rows, fit_rows = [], [], [], [], []
    for dataset in DATASETS:
        dataset_rows = {flow_id: row for (ds, flow_id), row in manifest.items() if ds == dataset}
        coarse_labels = sorted({coarse(r["service_label"]) for r in dataset_rows.values()})
        fine_labels = sorted({r["service_label"] for r in dataset_rows.values()})
        for seed in SEEDS:
            path = STAGE22 / dataset / f"seed{seed}" / "representations.npz"
            with np.load(path, allow_pickle=False) as z:
                arrays = {key: z[key].copy() for key in z.files if key.endswith("_z") or key.endswith("_flow_ids")}
            for encoder in ("E1", "E3"):
                key = encoder.lower()
                ids = {role: arrays[f"{role}_flow_ids"].tolist()
                       for role in ("known_train", *ROLES)}
                x = {role: arrays[f"{role}_{key}_z"] for role in ids}
                for role, role_ids in ids.items():
                    expected = {i for i, r in dataset_rows.items() if r["closed_role"] == role}
                    if len(role_ids) != len(expected) or set(role_ids) != expected:
                        raise ValueError(f"representation flow mismatch: {dataset}/{seed}/{encoder}/{role}")
                y_train = [coarse(dataset_rows[i]["service_label"]) for i in ids["known_train"]]
                scaler = StandardScaler().fit(x["known_train"])
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=0)
                start = time.monotonic()
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    model.fit(scaler.transform(x["known_train"]), y_train)
                duration = time.monotonic() - start
                for w in observed:
                    warning_rows.append({"dataset": dataset, "seed": seed, "encoder": encoder,
                                         "category": w.category.__name__, "message": str(w.message)})
                if set(model.classes_) != set(coarse_labels):
                    raise ValueError(f"coarse head class mismatch: {dataset}/{seed}/{encoder}")
                model_dir = OUT / "heads" / dataset / f"seed{seed}" / encoder
                model_dir.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(model_dir / "coarse_head.npz", coef=model.coef_, intercept=model.intercept_,
                                    classes=model.classes_, mean=scaler.mean_, scale=scaler.scale_,
                                    var=scaler.var_, n_iter=model.n_iter_)
                fit_rows.append({"dataset": dataset, "seed": seed, "encoder": encoder,
                                 "train_samples": len(y_train), "val_samples": len(ids["known_validation"]),
                                 "test_samples": len(ids["known_test"]),
                                 "n_iter_max": int(np.max(model.n_iter_)), "fit_seconds": duration,
                                 "warnings": len(observed),
                                 "head_path": str((model_dir / "coarse_head.npz").relative_to(OUT))})
                for role in ROLES:
                    pred = model.predict(scaler.transform(x[role])).tolist()
                    aligned = []
                    for flow_id, predicted in zip(ids[role], pred, strict=True):
                        ref = dataset_rows[flow_id]
                        n = packets[dataset][flow_id]
                        item = {"dataset": dataset, "seed": seed, "encoder": encoder,
                                "role": role, "flow_id": flow_id,
                                "packet_count_capped8": n, "true_fine": ref["service_label"],
                                "true_coarse": coarse(ref["service_label"]), "pred_coarse": predicted,
                                "retained_ge2": int(n >= 2)}
                        aligned.append(item)
                        predictions.append(item)
                    for scope in ("all", "ge2"):
                        subset = aligned if scope == "all" else [r for r in aligned if r["retained_ge2"]]
                        y_true = [r["true_coarse"] for r in subset]
                        y_pred = [r["pred_coarse"] for r in subset]
                        score, per = metrics(y_true, y_pred, coarse_labels)
                        results.append({"dataset": dataset, "seed": seed, "encoder": encoder,
                                        "role": role, "scope": scope, "samples": len(subset), **score})
                        per_class.extend({"dataset": dataset, "seed": seed, "encoder": encoder,
                                          "role": role, "scope": scope, **r} for r in per)
    save_csv(OUT / "coarse_head_fit.csv", fit_rows)
    save_csv(OUT / "coarse_head_results.csv", results)
    save_csv(OUT / "coarse_head_per_class.csv", per_class)
    save_csv(OUT / "coarse_head_predictions.csv", predictions)
    save_csv(OUT / "coarse_head_warnings.csv", warning_rows,
             ["dataset", "seed", "encoder", "category", "message"])
    source_hash_check(before)
    save_json(OUT / "phase_b_verification.json", {"status": "PASS", "heads_fitted": len(fit_rows),
              "metric_rows": len(results), "known_train_only_scaler_and_head": True,
              "unknown_usage": 0, "test_selection_usage": 0, "warnings": len(warning_rows),
              "protected_assets": len(before)})
    (OUT / "PHASE_B_COMPLETE").write_text("verified\n", encoding="utf-8")
    print(f"PHASE_B_COMPLETE heads={len(fit_rows)} metrics={len(results)} warnings={len(warning_rows)}", flush=True)


def phase_finalize() -> None:
    if not (OUT / "PHASE_A_COMPLETE").is_file() or not (OUT / "PHASE_B_COMPLETE").is_file():
        raise RuntimeError("A and B must complete first")
    before = json.loads((OUT / "protected_hashes_before.json").read_text())
    source_hash_check(before)
    remap = csv_rows(OUT / "remap_run_results.csv")
    heads = csv_rows(OUT / "coarse_head_results.csv")
    if len(remap) != 224 or len(heads) != 32:
        raise RuntimeError(f"result row counts: {len(remap)}/{len(heads)}")
    summary = []
    for dataset in DATASETS:
        for scope in ("all", "ge2"):
            for method in METHODS:
                for level in ("fine", "coarse"):
                    rows = [r for r in remap if r["dataset"] == dataset and r["scope"] == scope
                            and r["method"] == method and r["label_level"] == level and r["role"] == "known_test"]
                    if len(rows) != 2:
                        raise RuntimeError("missing paired remap rows")
                    summary.append({"dataset": dataset, "scope": scope, "kind": "hard_remap",
                                    "method": method, "label_level": level,
                                    "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in rows])),
                                    "macro_f1_mean": float(np.mean([float(r["macro_f1"]) for r in rows])),
                                    "macro_f1_std_pop": float(np.std([float(r["macro_f1"]) for r in rows])),
                                    "weighted_f1_mean": float(np.mean([float(r["weighted_f1"]) for r in rows])),
                                    "samples_per_seed": int(rows[0]["samples"])})
            for encoder in ("E1", "E3"):
                rows = [r for r in heads if r["dataset"] == dataset and r["scope"] == scope
                        and r["encoder"] == encoder and r["role"] == "known_test"]
                if len(rows) != 2:
                    raise RuntimeError("missing paired coarse-head rows")
                summary.append({"dataset": dataset, "scope": scope, "kind": "frozen_encoder_coarse_head",
                                "method": encoder, "label_level": "coarse",
                                "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in rows])),
                                "macro_f1_mean": float(np.mean([float(r["macro_f1"]) for r in rows])),
                                "macro_f1_std_pop": float(np.std([float(r["macro_f1"]) for r in rows])),
                                "weighted_f1_mean": float(np.mean([float(r["weighted_f1"]) for r in rows])),
                                "samples_per_seed": int(rows[0]["samples"])})
    save_csv(OUT / "stage25_summary.csv", summary)
    coverage = csv_rows(OUT / "scope_coverage.csv")
    notes = ["# Stage 25 — Semantic-coarse matched-flow closed-set development", "",
             "This is a development comparison; Stage20 Known Test has been exposed before Stage25. No Unknown/open-set claim.",
             "The hard-remap and independently trained coarse-head experiments are different estimands and are not mixed in one ranking.",
             "", "## Frozen label map", "",
             "Chat + Email + VoIP → Communication; File-Transfer, P2P, Streaming remain separate; Tor Browsing remains separate.",
             "Original six/seven-class checkpoints and predictions are unchanged.",
             "", "## Full-flow Known Test Macro-F1, mean of two seeds", "",
             "| Dataset | Method | Original fine | Coarse hard remap | Frozen-encoder coarse head |",
             "| --- | --- | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        for method in METHODS:
            def score(kind: str, level: str) -> str:
                xs = [r for r in summary if r["dataset"] == dataset and r["scope"] == "all"
                      and r["kind"] == kind and r["method"] == method and r["label_level"] == level]
                return f"{xs[0]['macro_f1_mean']:.6f}" if xs else "—"
            notes.append(f"| {dataset} | {method} | {score('hard_remap','fine')} | {score('hard_remap','coarse')} | {score('frozen_encoder_coarse_head','coarse')} |")
    notes.extend(["", "## ≥2-packet conditional Known Test Macro-F1 and coverage", "",
                  "| Dataset | Method | Fine | Coarse hard remap | Coarse head |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        for method in METHODS:
            def score(kind: str, level: str) -> str:
                xs = [r for r in summary if r["dataset"] == dataset and r["scope"] == "ge2"
                      and r["kind"] == kind and r["method"] == method and r["label_level"] == level]
                return f"{xs[0]['macro_f1_mean']:.6f}" if xs else "—"
            notes.append(f"| {dataset} | {method} | {score('hard_remap','fine')} | {score('hard_remap','coarse')} | {score('frozen_encoder_coarse_head','coarse')} |")
    for dataset in DATASETS:
        for role in ("known_train", *ROLES):
            sub = [r for r in coverage if r["dataset"] == dataset and r["role"] == role]
            total = sum(int(r["all_flows"]) for r in sub)
            keep = sum(int(r["retained_ge2"]) for r in sub)
            notes.append(f"- {dataset} {role}: {keep}/{total} retained ({keep/total:.2%}); exact per-original-Service counts in `scope_coverage.csv`.")
    notes.extend(["", "## Integrity and interpretation", "",
                  f"- 28/28 historical fine Known Test metrics independently replayed. {len(before)}/{len(before)} protected input SHA256 checks unchanged.",
                  "- Seven methods were compared on identical flow IDs, roles, coarse mapping and subset rule. The new E1/E3 heads used only Known Train representations for scaler and fit; original encoders/fine heads were not modified.",
                  "- Hard remap maps an existing argmax label; it does not sum class probabilities. A new coarse head is additional training and must be compared separately.",
                  "- Higher coarse Macro-F1 means a different, easier label task, not improved six/seven-class skill. ≥2-packet scores condition on a different population; it is not uniformly easier and must be read with coverage.",
                  "- Capture-derived Service labels are weak labels and Stage20 is not uniformly capture-disjoint; this experiment does not repair those limits.",
                  "", "## Files", "",
                  "`remap_run_results.csv`, `remap_per_class.csv`, `remap_predictions.csv`, `coarse_head_fit.csv`, `coarse_head_results.csv`, `coarse_head_per_class.csv`, `coarse_head_predictions.csv`, `scope_coverage.csv`, `stage25_summary.csv`, source hashes and verification JSON.", ""])
    (OUT / "stage25_report.md").write_text("\n".join(notes), encoding="utf-8")
    save_json(OUT / "completion_verification.json", {"status": "PASS", "hard_remap_prediction_cells": 56,
              "historical_metric_replays": 28, "coarse_heads": 8, "protected_hashes_unchanged": True,
              "protected_assets": len(before), "unknown_usage": 0, "test_fit_or_selection_usage": 0,
              "comparison_scope": "matched Stage20 development; Test previously exposed"})
    print(f"STAGE25_COMPLETE summary_rows={len(summary)} source_hashes={len(before)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("a", "b", "finalize"))
    phase = parser.parse_args().phase
    {"a": phase_a, "b": phase_b, "finalize": phase_finalize}[phase]()


if __name__ == "__main__":
    main()
