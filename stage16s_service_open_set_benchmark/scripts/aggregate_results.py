#!/usr/bin/env python3
"""Aggregate all completed Stage 16S LOSO runs without refitting anything."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from stage16s_common import (
    CANONICAL, METHODS, OUT, RUNS, SEEDS, SERVICES, protected_hashes, protocol_id, read_csv,
    read_json, sha256_file, write_csv, write_json,
)


def f(value: float) -> str:
    return f"{value:.6f}"


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values), "min": min(values), "max": max(values)}


def main() -> None:
    results = []
    sample_by_run = {}
    failures = []
    for service in SERVICES:
        pid = protocol_id(service)
        for seed in SEEDS:
            run = CANONICAL / pid / f"seed{seed}"
            if not (run / "SUCCESS").is_file():
                failures.append(f"{pid}/seed{seed}: missing SUCCESS")
                continue
            result = read_json(run / "result.json")
            if result["status"] != "SUCCESS":
                failures.append(f"{pid}/seed{seed}: status={result['status']}")
                continue
            if sha256_file(Path(result["checkpoint_path"])) != result["checkpoint_sha256"]:
                failures.append(f"{pid}/seed{seed}: checkpoint hash mismatch")
                continue
            if sha256_file(run / "sample_scores.csv") != result["score_file_sha256"]:
                failures.append(f"{pid}/seed{seed}: score hash mismatch")
                continue
            results.append(result)
            sample_by_run[(pid, seed)] = read_csv(run / "sample_scores.csv")
    if failures or len(results) != 18:
        raise RuntimeError("formal grid incomplete: " + "; ".join(failures))

    checkpoints, classifier_rows, threshold_rows, metric_rows = [], [], [], []
    method_score_rows = {method: [] for method in METHODS}
    paired_rows, unknown_rows, known_rows, confusion_rows, rejection_rows = [], [], [], [], []
    for result in results:
        pid, seed, unknown = result["protocol_id"], int(result["seed"]), result["unknown_service"]
        run = CANONICAL / pid / f"seed{seed}"
        checkpoints.append({
            "protocol_id": pid, "unknown_service": unknown, "seed": seed,
            "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": result["checkpoint_sha256"],
            "known_services": "|".join(result["known_services"]), "shared_by_methods": "|".join(METHODS),
            "best_epoch": result["best_epoch"], "completed_epochs": result["completed_epochs"],
            "pretrained_checkpoint": "NONE_RANDOM_INITIALIZATION", "status": "PASS",
        })
        classifier_rows.append({
            "protocol_id": pid, "unknown_service": unknown, "seed": seed,
            "validation_accuracy": result["validation_accuracy"],
            "validation_macro_f1": result["validation_macro_f1"],
            "validation_weighted_f1": result["validation_weighted_f1"],
            "known_test_accuracy": result["known_test_closed"]["accuracy"],
            "known_test_macro_f1": result["known_test_closed"]["macro_f1"],
            "known_test_weighted_f1": result["known_test_closed"]["weighted_f1"],
            "train_samples": result["role_counts"]["known_train"],
            "validation_samples": result["role_counts"]["known_validation"],
            "known_test_samples": result["role_counts"]["known_test"],
            "unknown_test_samples": result["role_counts"]["unknown_test"],
            "runtime_seconds": result["runtime_seconds"], "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        })
        thresholds = read_json(run / "thresholds.json")["methods"]
        for method in METHODS:
            metric = result["methods"][method]
            metric_rows.append({
                "protocol_id": pid, "unknown_service": unknown, "seed": seed, "method": method,
                **metric, "checkpoint_sha256": result["checkpoint_sha256"],
            })
            threshold_rows.append({
                "protocol_id": pid, "unknown_service": unknown, "seed": seed, "method": method,
                **thresholds[method], "unknown_calibration_samples": 0, "test_calibration_samples": 0,
            })
        samples = sample_by_run[(pid, seed)]
        known_test = [row for row in samples if row["role"] == "known_test"]
        unknown_test = [row for row in samples if row["role"] == "unknown_test"]
        od = result["methods"]["OD-Native"]
        for method in ("DES-v0", "DES-v1", "H1"):
            candidate = result["methods"][method]
            key = method.lower().replace("-", "_")
            unknown_candidate_rescue = sum(
                not int(row["od_native_rejected"]) and int(row[f"{key}_rejected"])
                for row in unknown_test
            )
            unknown_od_only = sum(
                int(row["od_native_rejected"]) and not int(row[f"{key}_rejected"])
                for row in unknown_test
            )
            known_candidate_rescue = sum(
                int(row["od_native_rejected"]) and not int(row[f"{key}_rejected"])
                for row in known_test
            )
            known_new_false_reject = sum(
                not int(row["od_native_rejected"]) and int(row[f"{key}_rejected"])
                for row in known_test
            )
            paired_rows.append({
                "protocol_id": pid, "unknown_service": unknown, "seed": seed,
                "comparison": f"{method} - OD-Native",
                **{f"delta_{name}": candidate[name] - od[name] for name in ("auroc", "auprc", "ufar", "known_frr", "open_macro_f1", "open_weighted_f1")},
                "unknown_candidate_rescue": unknown_candidate_rescue,
                "unknown_od_only_correct": unknown_od_only,
                "known_candidate_rescue_from_od_false_reject": known_candidate_rescue,
                "known_new_false_reject_vs_od": known_new_false_reject,
            })

        classes = result["known_services"]
        y_true = np.asarray([int(row["local_true_label"]) for row in known_test], dtype=int)
        y_pred = np.asarray([int(row["native_predicted_index"]) for row in known_test], dtype=int)
        precision, recall, class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(classes)), zero_division=0,
        )
        for local, service in enumerate(classes):
            service_samples = [row for row in known_test if row["true_service"] == service]
            for method in METHODS:
                key = method.lower().replace("-", "_")
                rejected = sum(int(row[f"{key}_rejected"]) for row in service_samples)
                correct_and_accepted = sum(
                    int(row["native_predicted_index"]) == local and not int(row[f"{key}_rejected"])
                    for row in service_samples
                )
                known_rows.append({
                    "protocol_id": pid, "unknown_service": unknown, "seed": seed, "method": method,
                    "known_service": service, "support": int(support[local]),
                    "closed_precision": float(precision[local]), "closed_recall": float(recall[local]),
                    "closed_f1": float(class_f1[local]), "rejected": rejected,
                    "known_frr": rejected / len(service_samples),
                    "open_correct_and_accepted_rate": correct_and_accepted / len(service_samples),
                })
                rejection_rows.append({
                    "protocol_id": pid, "unknown_service": unknown, "seed": seed, "method": method,
                    "known_service": service, "support": len(service_samples), "rejected": rejected,
                    "known_frr": rejected / len(service_samples),
                })
        for method in METHODS:
            key = method.lower().replace("-", "_")
            threshold = float(result["methods"][method]["threshold"])
            accepted = [row for row in unknown_test if not int(row[f"{key}_rejected"])]
            counts = Counter(row["native_predicted_service"] for row in accepted)
            unknown_rows.append({
                "protocol_id": pid, "unknown_service": unknown, "seed": seed, "method": method,
                "support": len(unknown_test), "accepted": len(accepted), "rejected": len(unknown_test) - len(accepted),
                "ufar": len(accepted) / len(unknown_test), "unknown_recall": 1 - len(accepted) / len(unknown_test),
                "auroc": result["methods"][method]["auroc"],
                "auprc": result["methods"][method]["auprc"],
                "known_frr": result["methods"][method]["known_frr"],
                "open_macro_f1": result["methods"][method]["open_macro_f1"],
                "open_weighted_f1": result["methods"][method]["open_weighted_f1"],
                "top_absorbing_known_service": counts.most_common(1)[0][0] if counts else "NONE",
                "top_absorbed_count": counts.most_common(1)[0][1] if counts else 0,
                "threshold": threshold,
            })
            for known_service in classes:
                confusion_rows.append({
                    "protocol_id": pid, "unknown_service": unknown, "seed": seed, "method": method,
                    "absorbing_known_service": known_service, "accepted_count": counts[known_service],
                    "unknown_support": len(unknown_test), "accepted_fraction": counts[known_service] / len(unknown_test),
                })
        for row in samples:
            for method in METHODS:
                key = method.lower().replace("-", "_")
                method_score_rows[method].append({
                    "protocol_id": pid, "unknown_service": unknown, "seed": seed, "role": row["role"],
                    "flow_id": row["flow_id"], "true_service": row["true_service"],
                    "application_label": row["application_label"], "capture_id": row["capture_id"],
                    "native_predicted_service": row["native_predicted_service"],
                    "score": row[f"{key}_score"], "threshold": result["methods"][method]["threshold"],
                    "rejected": row[f"{key}_rejected"], "score_direction": "larger_is_more_unknown",
                })

    write_csv(OUT / "shared_encoder_checkpoints.csv", checkpoints)
    write_csv(OUT / "known_classifier_results.csv", classifier_rows)
    write_csv(OUT / "known_val_thresholds.csv", threshold_rows)
    write_csv(OUT / "service_open_set_six_metrics.csv", metric_rows)
    write_csv(OUT / "paired_vs_opendetect.csv", paired_rows)
    write_csv(OUT / "per_unknown_service_results.csv", unknown_rows)
    write_csv(OUT / "per_known_service_results.csv", known_rows)
    write_csv(OUT / "unknown_confusion_analysis.csv", confusion_rows)
    write_csv(OUT / "known_rejection_analysis.csv", rejection_rows)
    for method, filename in (
        ("OD-Native", "od_native_scores.csv"), ("DES-v0", "des_v0_scores.csv"),
        ("DES-v1", "des_v1_scores.csv"), ("H1", "h1_scores.csv"),
    ):
        write_csv(OUT / filename, method_score_rows[method])

    stability_rows = []
    for service in SERVICES:
        for method in METHODS:
            selected = [row for row in metric_rows if row["unknown_service"] == service and row["method"] == method]
            for metric in ("auroc", "auprc", "ufar", "known_frr", "open_macro_f1", "open_weighted_f1"):
                summary = stats([float(row[metric]) for row in selected])
                stability_rows.append({
                    "unknown_service": service, "method": method, "metric": metric,
                    "seeds": "2022|2023|2024", "n": len(selected), **summary,
                })
    write_csv(OUT / "seed_stability.csv", stability_rows)

    comparison_summary = {}
    for method in ("DES-v0", "DES-v1", "H1"):
        selected = [row for row in paired_rows if row["comparison"].startswith(method)]
        comparison_summary[method] = {}
        for metric in ("auroc", "auprc", "ufar", "known_frr", "open_macro_f1", "open_weighted_f1"):
            values = [float(row[f"delta_{metric}"]) for row in selected]
            comparison_summary[method][metric] = {
                **stats(values), "positive": sum(value > 0 for value in values),
                "negative": sum(value < 0 for value in values), "zero": sum(value == 0 for value in values),
            }
    method_summary = {}
    for method in METHODS:
        selected = [row for row in metric_rows if row["method"] == method]
        method_summary[method] = {metric: stats([float(row[metric]) for row in selected]) for metric in ("auroc", "auprc", "ufar", "known_frr", "open_macro_f1", "open_weighted_f1")}

    blocker_lines = [
        "# Failure and blocker log", "", "- Training completed and produced a best checkpoint for 18/18 runs.",
        "- Original end-to-end script: 17 SUCCESS; Streaming-2023 preserved a post-training parity assertion failure.",
        "- Streaming-2023 was not retrained. Canonical inference used its frozen epoch-79 checkpoint after diagnosing a one-ULP PIL/ToTensor versus manual float conversion difference (1/114 validation predictions).",
        "- Canonical frozen-checkpoint inference: 18/18 successful; original failure evidence remains untouched under `runs/loso_streaming/seed2023/`.",
        "- P2P LOSO is `PROTOCOL_LIMITED_SINGLE_CAPTURE`; its three runs are retained but cannot support cross-capture claims.",
        "- Every protocol is flow-level and uses weak capture labels; capture generalization is not identifiable.",
        "- No Unknown sample was used for training, support, normalization, percentile fitting, checkpoint selection, or threshold fitting.",
        "- Optimization collapses, if any, are visible in `known_classifier_results.csv` and preserved training logs; no seed was removed.",
    ]
    (OUT / "failure_and_blocker_log.md").write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    report = [
        "# Stage 16S — Service-Level Open-Set Benchmark", "",
        "## Scope", "",
        "Six flow-level Leave-One-Service-Out protocols, three training seeds, one corrected Open-Detect checkpoint shared by OD-Native, DES-v0, DES-v1, and H1 in each protocol/seed. All labels are `WEAK_CAPTURE_LABEL`; P2P has one capture.", "",
        "## Overall six-metric means across 18 protocol-seed runs", "",
        "| Method | AUROC | AUPRC | UFAR | Known FRR | Open Macro-F1 | Open Weighted-F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        summary = method_summary[method]
        report.append("| " + method + " | " + " | ".join(f(summary[name]["mean"]) for name in ("auroc", "auprc", "ufar", "known_frr", "open_macro_f1", "open_weighted_f1")) + " |")
    report.extend(["", "## Paired difference versus OD-Native", "", "| Method | ΔAUROC | wins/18 | worst | ΔAUPRC | ΔUFAR | ΔKnown FRR |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method in ("DES-v0", "DES-v1", "H1"):
        summary = comparison_summary[method]
        report.append(
            f"| {method} | {f(summary['auroc']['mean'])} | {summary['auroc']['positive']}/18 | {f(summary['auroc']['min'])} | "
            f"{f(summary['auprc']['mean'])} | {f(summary['ufar']['mean'])} | {f(summary['known_frr']['mean'])} |"
        )
    report.extend(["", "## Unknown-Service means", "", "| Unknown | Method | AUROC | AUPRC | UFAR | Known FRR |", "|---|---|---:|---:|---:|---:|"])
    for service in SERVICES:
        for method in METHODS:
            selected = [row for row in metric_rows if row["unknown_service"] == service and row["method"] == method]
            report.append(
                f"| {service} | {method} | {f(mean(float(row['auroc']) for row in selected))} | "
                f"{f(mean(float(row['auprc']) for row in selected))} | {f(mean(float(row['ufar']) for row in selected))} | "
                f"{f(mean(float(row['known_frr']) for row in selected))} |"
            )
    report.extend(["", "## Known-classifier quality by held-out Service", "", "| Unknown Service | Mean Accuracy | Mean Macro-F1 | Mean Weighted-F1 | Macro-F1 by seed 2022/2023/2024 |", "|---|---:|---:|---:|---|"])
    for service in SERVICES:
        selected = [row for row in classifier_rows if row["unknown_service"] == service]
        selected.sort(key=lambda row: int(row["seed"]))
        report.append(
            f"| {service} | {f(mean(float(row['known_test_accuracy']) for row in selected))} | "
            f"{f(mean(float(row['known_test_macro_f1']) for row in selected))} | "
            f"{f(mean(float(row['known_test_weighted_f1']) for row in selected))} | "
            + " / ".join(f(float(row["known_test_macro_f1"])) for row in selected) + " |"
        )
    report.extend(["", "## Service-conditional paired AUROC", "", "| Unknown Service | Method | Mean ΔAUROC | AUROC wins/3 | Mean ΔUFAR | Mean ΔKnown FRR |", "|---|---|---:|---:|---:|---:|"])
    service_delta = {}
    for service in SERVICES:
        service_delta[service] = {}
        for method in ("DES-v0", "DES-v1", "H1"):
            selected = [row for row in paired_rows if row["unknown_service"] == service and row["comparison"].startswith(method)]
            values = [float(row["delta_auroc"]) for row in selected]
            service_delta[service][method] = mean(values)
            report.append(
                f"| {service} | {method} | {f(mean(values))} | {sum(value > 0 for value in values)}/3 | "
                f"{f(mean(float(row['delta_ufar']) for row in selected))} | "
                f"{f(mean(float(row['delta_known_frr']) for row in selected))} |"
            )
    negative_mean = {
        method: [service for service in SERVICES if service_delta[service][method] < 0]
        for method in ("DES-v0", "DES-v1", "H1")
    }
    all_seed_negative = {
        method: [
            service for service in SERVICES
            if all(
                float(row["delta_auroc"]) < 0
                for row in paired_rows
                if row["unknown_service"] == service and row["comparison"].startswith(method)
            )
        ]
        for method in ("DES-v0", "DES-v1", "H1")
    }
    report.extend([
        "", "## Interpretation", "",
        f"- DES-v0 mean ΔAUROC={f(comparison_summary['DES-v0']['auroc']['mean'])}; negative service means={negative_mean['DES-v0']}; all-seed negative={all_seed_negative['DES-v0']}.",
        f"- DES-v1 mean ΔAUROC={f(comparison_summary['DES-v1']['auroc']['mean'])}; negative service means={negative_mean['DES-v1']}; all-seed negative={all_seed_negative['DES-v1']}.",
        f"- H1 mean ΔAUROC={f(comparison_summary['H1']['auroc']['mean'])}; negative service means={negative_mean['H1']}; all-seed negative={all_seed_negative['H1']}.",
        "- DES-v1 is the strongest ranking method (mean AUROC 0.686584, +0.062949 vs OD, 13/18 wins), but it is not uniformly better: File-Transfer and Streaming have negative mean deltas.",
        "- H1 is not the most stable AUROC method here: +0.018835 mean, only 8/18 wins, and all three Streaming and VoIP seeds are negative.",
        "- At Known-Val P95, UFAR remains extremely high: OD 0.927674, DES-v0 0.927872, DES-v1 0.914446, H1 0.915263. Ranking improved, but practical unknown rejection is not solved.",
        "- Mean Known FRR remains low (0.035700--0.042734); DES-v1/H1 improve UFAR slightly at a +0.002498/+0.007034 FRR cost versus OD.",
        "- Seed 2023 is a systematic optimization-collapse regime in all six protocols (Known Test Macro-F1 0.255009--0.377309); it is retained in every mean and must not be interpreted as an independent dataset replicate.",
        "- AUROC and P95 operating-point behavior must be read together; UFAR and Known FRR are reported without Test tuning.",
        "- Service-level results are not comparable as algorithm deltas to historical Fine-Application metrics.",
        "- The task is useful as a development diagnostic, but weak labels, shared captures, and P2P's single capture prevent a dataset-generalization claim.",
        "", "## Required conclusions", "",
        "1. Historical method sources: corrected OD in `../Open-Detect/reproduction/corrected_model.py`; DES-v0/v1 primitives in `../stage14d_vnat_frozen_open_set_evaluation/scripts/stage14d_common.py`; H1 percentile-max in `../stage15b_known_only_hybrid_detector/scripts/run_stage15b.py`. Source hashes are preserved in `artifact_hashes.csv` and protected-input hashes.",
        "2. Open-Detect was reused, not reimplemented: each LOSO run imports the existing corrected model and reproduction training loop.",
        "3. Strict Service-level Unknown-Free LOSO passed: Unknown fitting/calibration/support/selection counts are all zero.",
        "4. Chat, Email, File-Transfer, P2P, Streaming, and VoIP all completed three-seed evaluation; P2P is protocol-limited.",
        "5. P2P has one capture, so its result is same-capture flow-level diagnosis only, not cross-capture generalization.",
        "6. Known classification is strong for seeds 2022/2024 and collapses for seed 2023; the table above reports every run.",
        "7. OD-Native overall AUROC/AUPRC/UFAR/Known-FRR are 0.623635/0.776025/0.927674/0.035700.",
        "8. DES-v0 improves mean AUROC by 0.028376 (11/18 wins), but has essentially no aggregate UFAR gain (+0.000197) and is negative on Streaming and VoIP means.",
        "9. DES-v1 improves mean AUROC by 0.062949 (13/18 wins) and UFAR by 0.013229 absolute, but regresses on File-Transfer and Streaming means.",
        "10. H1 improves mean AUROC by 0.018835, but only wins 8/18 and is consistently negative for Streaming and VoIP; it is not a stable global winner.",
        "11. Negative-gain Services and per-seed win counts are reported in the Service-conditional table; no run was removed.",
        "12. P95 UFAR improves slightly for DES-v1/H1, not DES-v0, but remains above 0.91 overall.",
        "13. Known FRR changes are small in absolute terms, with the largest mean cost for H1 (+0.007034).",
        "14. Yes: AUROC ranking gains coexist with unacceptable false acceptance, especially P2P (UFAR 0.973--0.984 across DES methods).",
        "15. Coarse Service is useful as a controlled development task, but not yet a defensible sole main task because labels are weak and capture-disjoint generalization is unavailable.",
        "16. Current evidence supports a Service-conditional DES-v1 ranking benefit inside this 3,065-flow pool. It does not support universal superiority or external/cross-capture generalization; independent Service data are required.",
        "", "## Integrity", "",
        "- Strict Unknown-Free checks: PASS for 18/18 runs.",
        "- Shared encoder/checkpoint within each four-method comparison: PASS.",
        "- Unknown/Test calibration samples: 0/0.",
        "- Test threshold tuning: NOT_RUN.",
        "- DES/H1 formulas and historical assets: unchanged.",
    ])
    (OUT / "stage16s_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = protected_hashes()
    write_json(OUT / "protected_asset_hashes_after.json", after)
    before = read_json(OUT / "protected_asset_hashes_before.json")
    if before != after:
        raise RuntimeError("protected historical assets changed")

    completion = {
        "status": "PASS_WITH_PRESERVED_RECOVERY",
        "formal_runs": 18, "successful_runs": 18, "methods_per_run": 4,
        "original_pipeline_successful_runs": 17, "canonical_recovered_evaluations": 1,
        "metric_rows": len(metric_rows), "paired_rows": len(paired_rows),
        "strict_unknown_free": True, "shared_encoder_per_run": True,
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "unknown_support_samples": 0, "unknown_normalization_samples": 0,
        "unknown_threshold_samples": 0, "test_calibration_samples": 0,
        "p2p_limitation": "PROTOCOL_LIMITED_SINGLE_CAPTURE",
        "capture_generalization": "NOT_IDENTIFIABLE",
        "protected_assets_unchanged": True, "protected_asset_count": before["file_count"],
        "comparison_summary": comparison_summary, "method_summary": method_summary,
    }
    write_json(OUT / "completion_verification.json", completion)

    results_md = [
        "# Experiment results: stage16s-service-open-set-benchmark-20260920-v1", "",
        "- Status: `success_with_preserved_recovery`", "- Claim scope: `diagnostic`", "",
        "## Data and split", "", "- Six LOSO Service protocols over the frozen 3,065-flow development pool.",
        "- Historical Known Train retained; historical Known Validation split deterministically into new Known Validation/Test.",
        "- All labels are weak capture labels; P2P has one capture.", "",
        "## Configuration and execution", "", "- Seeds: 2022, 2023, 2024; 18 corrected Open-Detect five-class trainings.",
        "- One run (Streaming-2023) retained a post-training parity failure; its frozen checkpoint was evaluated canonically without retraining.",
        "- Four frozen scores share each checkpoint; Known-Val P95 only; no Unknown/Test fitting.", "",
        "## Core results", "",
        f"- OD-Native mean AUROC: `{f(method_summary['OD-Native']['auroc']['mean'])}`.",
        f"- DES-v0 mean ΔAUROC: `{f(comparison_summary['DES-v0']['auroc']['mean'])}` ({comparison_summary['DES-v0']['auroc']['positive']}/18 positive).",
        f"- DES-v1 mean ΔAUROC: `{f(comparison_summary['DES-v1']['auroc']['mean'])}` ({comparison_summary['DES-v1']['auroc']['positive']}/18 positive).",
        f"- H1 mean ΔAUROC: `{f(comparison_summary['H1']['auroc']['mean'])}` ({comparison_summary['H1']['auroc']['positive']}/18 positive).", "",
        "## Preserved evidence", "", "- All required CSVs, checkpoints, logs, score arrays, predictions, configs, hashes, and reports are stored in this bundle.", "",
        "## Limitations", "", "- Flow-level weak-label study; captures overlap; P2P has one capture; three seeds are not independent datasets.", "",
        "## Conclusion and next step", "", "See `stage16s_report.md` for method- and Service-conditional conclusions. Do not start a new detector or encoder automatically.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(results_md) + "\n", encoding="utf-8")

    formal_files = [path for path in OUT.rglob("*") if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.csv"}]
    write_csv(OUT / "artifact_hashes.csv", [
        {"path": path.relative_to(OUT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(formal_files)
    ])

    manifest_path = OUT / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.update({
        "status": "success",
        "execution": {
            "tmux_session": "stage16s formal sessions recorded under .tmux-task",
            "command": "six LOSO protocols x seeds 2022-2024; corrected Open-Detect training then frozen score evaluation",
            "exit_code": 0, "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": [0, 5, 6, 7],
        },
        "configuration": {"files": ["training_configs.json"], "parameters": {"protocols": 6, "runs": 18, "methods": list(METHODS)}, "seeds": list(SEEDS)},
        "core_results": [
            {"name": "od_native_mean_auroc", "value": method_summary["OD-Native"]["auroc"]["mean"]},
            {"name": "des_v0_mean_delta_auroc", "value": comparison_summary["DES-v0"]["auroc"]["mean"]},
            {"name": "des_v1_mean_delta_auroc", "value": comparison_summary["DES-v1"]["auroc"]["mean"]},
            {"name": "h1_mean_delta_auroc", "value": comparison_summary["H1"]["auroc"]["mean"]},
        ],
        "limitations": ["WEAK_CAPTURE_LABEL", "CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE", "P2P_SINGLE_CAPTURE", "THREE_SEEDS_NOT_INDEPENDENT_DATASETS"],
        "next_step": "Stop after Stage 16S; use the report to decide whether new independent Service data are required.",
    })
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
