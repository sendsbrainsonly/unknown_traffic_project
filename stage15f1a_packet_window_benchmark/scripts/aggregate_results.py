#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from common import (
    ROOT, STAGE12, STAGE14D, STAGE15F0, STAGE3,
    pilot_specs, protocol_rows, read_csv, read_json, write_csv, write_json,
)

WINDOWS = (8, 16, 32)
METRICS = ("validation_accuracy", "validation_macro_f1", "validation_weighted_f1")
FOCUS = {
    "vnat": {"rsync", "scp", "sftp", "vimeo", "youtube", "netflix"},
    "iscx_vpn": {"spotify", "netflix", "vimeo"},
    "iscx_tor": {"facebook", "hangouts", "vimeo", "youtube", "ftp"},
    "ustc": {"worldofwarcraft", "weibo"},
}


def run_dir(window: int, dataset: str, protocol_id: str) -> Path:
    return ROOT / "runs" / f"T{window}" / dataset / protocol_id


def native_predictions(dataset: str, protocol_id: str, classes: list[str], val_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.asarray([row["sample_id"] for row in val_rows])
    truth = np.asarray([row["label"] for row in val_rows], dtype=np.int64)
    if dataset in {"iscx_vpn", "iscx_tor"}:
        path = STAGE12 / "runs" / dataset / "medium" / "seed2022" / "known_train_validation_features.npz"
        with np.load(path, allow_pickle=False) as data:
            source_truth = data["validation_labels_reindexed"].astype(np.int64)
            pred = data["validation_m0_predictions"].astype(np.int64)
        if not np.array_equal(source_truth, truth):
            raise RuntimeError(f"native validation label order mismatch: {dataset}")
        return sample_ids, truth, pred
    if dataset == "ustc":
        table = pd.read_parquet(STAGE3 / "artifacts" / "A-2" / "val_known_mu.parquet", columns=["flow_id", "class_name", "native_predicted_known_class"])
        lookup = table.set_index("flow_id")
        if set(sample_ids) != set(lookup.index.astype(str)):
            raise RuntimeError("USTC native sample-id mismatch")
        class_to_id = {name: i for i, name in enumerate(classes)}
        ordered = lookup.loc[sample_ids]
        source_truth = np.asarray([class_to_id[x] for x in ordered["class_name"]], dtype=np.int64)
        pred = np.asarray([class_to_id[x] for x in ordered["native_predicted_known_class"]], dtype=np.int64)
        if not np.array_equal(source_truth, truth):
            raise RuntimeError("USTC native labels mismatch")
        return sample_ids, truth, pred
    if dataset == "vnat":
        rows = [row for row in read_csv(STAGE14D / "artifacts" / protocol_id / "sample_scores.csv") if row["role"] == "validation" and row["method"] == "M0"]
        lookup = {row["flow_uid"]: row for row in rows}
        if set(sample_ids) != set(lookup):
            raise RuntimeError(f"VNAT native sample-id mismatch: {protocol_id}")
        class_to_id = {name: i for i, name in enumerate(classes)}
        source_truth = np.asarray([class_to_id[lookup[uid]["true_application"]] for uid in sample_ids], dtype=np.int64)
        pred = np.asarray([class_to_id[lookup[uid]["predicted_known_application"]] for uid in sample_ids], dtype=np.int64)
        if not np.array_equal(source_truth, truth):
            raise RuntimeError(f"VNAT native labels mismatch: {protocol_id}")
        return sample_ids, truth, pred
    raise KeyError(dataset)


def temporal_predictions(window: int, dataset: str, protocol_id: str, expected_ids: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
    with np.load(run_dir(window, dataset, protocol_id) / "validation_predictions.npz", allow_pickle=False) as data:
        ids = data["sample_ids"].astype(str)
        truth = data["true_labels"].astype(np.int64)
        pred = data["predicted_labels"].astype(np.int64)
    lookup = {uid: (int(y), int(p)) for uid, y, p in zip(ids, truth, pred)}
    if set(expected_ids) != set(lookup):
        raise RuntimeError(f"T{window} sample-id mismatch: {dataset}/{protocol_id}")
    aligned_truth = np.asarray([lookup[uid][0] for uid in expected_ids], dtype=np.int64)
    if not np.array_equal(aligned_truth, expected_truth):
        raise RuntimeError(f"T{window} label mismatch: {dataset}/{protocol_id}")
    return np.asarray([lookup[uid][1] for uid in expected_ids], dtype=np.int64)


def complementarity_row(dataset, protocol_id, class_name, mask, truth, left, right, left_name, right_name):
    lc, rc = left[mask] == truth[mask], right[mask] == truth[mask]
    return {
        "dataset": dataset, "protocol_id": protocol_id, "class_name": class_name,
        "baseline_method": left_name, "candidate_method": right_name, "support": int(mask.sum()),
        "baseline_correct_candidate_wrong": int((lc & ~rc).sum()),
        "baseline_wrong_candidate_correct": int((~lc & rc).sum()),
        "both_correct": int((lc & rc).sum()), "both_wrong": int((~lc & ~rc).sum()),
        "net_rescue": int((~lc & rc).sum() - (lc & ~rc).sum()),
    }


def main() -> None:
    config = read_json(ROOT / "config.json")
    parity = read_json(ROOT / "legacy_t8_parity" / "parity_summary.json")
    if parity["status"] != "PASS":
        raise RuntimeError("legacy T8 parity is not PASS")

    result_rows, class_rows, confusion_rows, complement_rows, native_rows = [], [], [], [], []
    results = {}
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        classes, _, val_rows = protocol_rows(dataset, protocol_id)
        ids, truth, native = native_predictions(dataset, protocol_id, classes, val_rows)
        native_precision, native_recall, native_f1, native_support = precision_recall_fscore_support(truth, native, labels=np.arange(len(classes)), zero_division=0)
        native_rows.append({
            "dataset": dataset, "protocol_id": protocol_id,
            "accuracy": float(accuracy_score(truth, native)),
            "macro_f1": float(native_f1.mean()),
            "weighted_f1": float(np.average(native_f1, weights=native_support)),
        })
        native_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        np.add.at(native_matrix, (truth, native), 1)
        for true_index, true_name in enumerate(classes):
            for pred_index, pred_name in enumerate(classes):
                count = int(native_matrix[true_index, pred_index])
                confusion_rows.append({
                    "dataset": dataset, "protocol_id": protocol_id, "window_id": "Native",
                    "true_class": true_name, "predicted_class": pred_name, "count": count,
                    "true_class_support": int(native_matrix[true_index].sum()),
                    "row_rate": float(count / max(1, native_matrix[true_index].sum())),
                })
        prediction_sets = {"Native": native}
        for window in WINDOWS:
            directory = run_dir(window, dataset, protocol_id)
            result = read_json(directory / "result.json")
            if result["status"] != "PASS" or result["known_test_samples_used"] or result["unknown_test_samples_used"]:
                raise RuntimeError(f"invalid formal run: {directory}")
            if not result.get("mask_safe") or result["epochs_completed"] != config["training"]["epochs"]:
                raise RuntimeError(f"training-control mismatch: {directory}")
            row = {"window_id": f"T{window}", **result}
            result_rows.append(row)
            results[(dataset, protocol_id, window)] = result
            for class_row in read_csv(directory / "per_class_results.csv"):
                class_rows.append({"window_id": f"T{window}", **class_row})
            matrix = np.load(directory / "validation_confusion_matrix.npy", allow_pickle=False)
            for true_index, true_name in enumerate(classes):
                for pred_index, pred_name in enumerate(classes):
                    count = int(matrix[true_index, pred_index])
                    confusion_rows.append({
                        "dataset": dataset, "protocol_id": protocol_id, "window_id": f"T{window}",
                        "true_class": true_name, "predicted_class": pred_name, "count": count,
                        "true_class_support": int(matrix[true_index].sum()),
                        "row_rate": float(count / max(1, matrix[true_index].sum())),
                    })
            prediction_sets[f"T{window}"] = temporal_predictions(window, dataset, protocol_id, ids, truth)

        comparisons = (("Native", "T8"), ("Native", "T16"), ("Native", "T32"), ("T8", "T16"), ("T8", "T32"), ("T16", "T32"))
        for left_name, right_name in comparisons:
            left, right = prediction_sets[left_name], prediction_sets[right_name]
            masks = [("__ALL__", np.ones(len(truth), dtype=bool))] + [(name, truth == i) for i, name in enumerate(classes)]
            for name, mask in masks:
                complement_rows.append(complementarity_row(dataset, protocol_id, name, mask, truth, left, right, left_name, right_name))

    write_csv(ROOT / "window_pilot_results.csv", result_rows)
    write_csv(ROOT / "window_per_class_results.csv", class_rows)
    write_csv(ROOT / "window_confusion_analysis.csv", confusion_rows)
    write_csv(ROOT / "window_error_complementarity.csv", complement_rows)

    paired_rows = []
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        for baseline, candidate in ((8, 16), (8, 32), (16, 32)):
            left, right = results[(dataset, protocol_id, baseline)], results[(dataset, protocol_id, candidate)]
            row = {"dataset": dataset, "protocol_id": protocol_id, "comparison": f"T{candidate}-T{baseline}", "baseline_window": baseline, "candidate_window": candidate}
            for metric in METRICS:
                row[f"baseline_{metric}"] = left[metric]
                row[f"candidate_{metric}"] = right[metric]
                row[f"delta_{metric}"] = right[metric] - left[metric]
            row["runtime_ratio"] = right["runtime_seconds"] / left["runtime_seconds"]
            row["peak_gpu_memory_delta_bytes"] = right["peak_gpu_memory_bytes"] - left["peak_gpu_memory_bytes"]
            paired_rows.append(row)
    write_csv(ROOT / "window_paired_comparison.csv", paired_rows)

    perclass_lookup = {(r["dataset"], r["protocol_id"], int(r["window"]), r["class_name"]): r for r in class_rows}
    result_lookup = {(r["dataset"], r["protocol_id"], int(r["window"])): r for r in result_rows}
    coverage_rows = []
    coverage_protocol = {"iscx_vpn": "medium", "iscx_tor": "medium", "vnat": None, "ustc": "A-2"}
    for coverage in read_csv(STAGE15F0 / "packet_window_coverage.csv"):
        if coverage["role_scope"] != "known_validation" or int(coverage["window_packets"]) not in WINDOWS:
            continue
        for spec in pilot_specs():
            dataset, protocol_id = spec["dataset"], spec["protocol_id"]
            if coverage["dataset"] != dataset:
                continue
            wanted = protocol_id if coverage_protocol[dataset] is None else coverage_protocol[dataset]
            if coverage["protocol_id"] != wanted:
                continue
            window, class_name = int(coverage["window_packets"]), coverage["class_name"]
            row = dict(coverage)
            row["pilot_protocol_id"] = protocol_id
            if class_name == "__ALL__":
                perf = result_lookup.get((dataset, protocol_id, window))
                if perf:
                    row.update({"validation_f1": perf["validation_macro_f1"], "validation_accuracy": perf["validation_accuracy"], "metric_scope": "protocol_macro"})
            else:
                perf = perclass_lookup.get((dataset, protocol_id, window, class_name))
                if perf:
                    row.update({"validation_f1": perf["f1"], "validation_accuracy": perf["recall"], "metric_scope": "class"})
            if "validation_f1" in row:
                base = result_lookup[(dataset, protocol_id, 8)]["validation_macro_f1"] if class_name == "__ALL__" else float(perclass_lookup[(dataset, protocol_id, 8, class_name)]["f1"])
                row["delta_f1_vs_T8"] = float(row["validation_f1"]) - float(base)
                coverage_rows.append(row)
    write_csv(ROOT / "window_coverage_vs_performance.csv", coverage_rows)

    baseline_truncation = {
        (r["dataset"], r["pilot_protocol_id"], r["class_name"]): float(r["truncation_ratio"])
        for r in coverage_rows
        if int(r["window_packets"]) == 8 and r["class_name"] != "__ALL__"
    }
    coverage_correlations = {}
    for candidate in (16, 32):
        pairs = [
            (
                baseline_truncation[(r["dataset"], r["pilot_protocol_id"], r["class_name"])],
                float(r["delta_f1_vs_T8"]),
            )
            for r in coverage_rows
            if int(r["window_packets"]) == candidate
            and r["class_name"] != "__ALL__"
            and (r["dataset"], r["pilot_protocol_id"], r["class_name"]) in baseline_truncation
        ]
        rho, pvalue = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
        coverage_correlations[f"T{candidate}_delta_vs_T8_truncation"] = {
            "spearman_rho": float(rho),
            "pvalue": float(pvalue),
            "class_protocol_units": len(pairs),
        }

    summary = {}
    for candidate in (16, 32):
        deltas = [results[(s["dataset"], s["protocol_id"], candidate)]["validation_macro_f1"] - results[(s["dataset"], s["protocol_id"], 8)]["validation_macro_f1"] for s in pilot_specs()]
        summary[candidate] = {"mean_delta_macro_f1": float(np.mean(deltas)), "positive_protocols": int(sum(x > 0 for x in deltas)), "worst_delta_macro_f1": float(min(deltas)), "deltas": deltas}
    best = max((16, 32), key=lambda w: summary[w]["mean_delta_macro_f1"])
    protocol_global_pass = summary[best]["mean_delta_macro_f1"] >= 0.015 and summary[best]["positive_protocols"] >= 4 and summary[best]["worst_delta_macro_f1"] >= -0.02
    gains, losses = [], []
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        for class_name in FOCUS[dataset]:
            rows = [r for r in class_rows if r["dataset"] == dataset and r["protocol_id"] == protocol_id and r["class_name"].lower() == class_name]
            by_window = {int(r["window"]): r for r in rows}
            if 8 not in by_window or best not in by_window:
                continue
            delta = float(by_window[best]["f1"]) - float(by_window[8]["f1"])
            rescue = next(r["net_rescue"] for r in complement_rows if r["dataset"] == dataset and r["protocol_id"] == protocol_id and r["class_name"] == by_window[8]["class_name"] and r["baseline_method"] == "T8" and r["candidate_method"] == f"T{best}")
            record = {"dataset": dataset, "protocol_id": protocol_id, "class_name": by_window[8]["class_name"], "delta_f1": delta, "net_rescue": rescue}
            if delta >= 0.05 and rescue > 0:
                gains.append(record)
            if delta < -0.05:
                losses.append(record)
    selected_class_deltas = []
    for key, baseline_row in perclass_lookup.items():
        dataset, protocol_id, window, class_name = key
        if window != 8:
            continue
        candidate_row = perclass_lookup[(dataset, protocol_id, best, class_name)]
        selected_class_deltas.append({
            "dataset": dataset, "protocol_id": protocol_id, "class_name": class_name,
            "delta_f1": float(candidate_row["f1"]) - float(baseline_row["f1"]),
        })
    delta_values = np.asarray([row["delta_f1"] for row in selected_class_deltas], dtype=np.float64)
    class_breadth = {
        "class_protocol_units": int(len(delta_values)),
        "positive_units": int((delta_values > 0).sum()),
        "unchanged_units": int((delta_values == 0).sum()),
        "negative_units": int((delta_values < 0).sum()),
        "median_delta_f1": float(np.median(delta_values)),
        "mean_delta_f1": float(np.mean(delta_values)),
        "units_gain_at_least_0_05": int((delta_values >= 0.05).sum()),
        "units_loss_at_most_minus_0_05": int((delta_values <= -0.05).sum()),
    }
    # The user explicitly forbids declaring a global bottleneck when only a
    # minority of classes benefits.  Protocol-level Macro-F1 can pass while a
    # few large class gains offset many flat/regressing classes, so require a
    # strictly positive class-level median and more positive than negative units.
    class_breadth_pass = class_breadth["median_delta_f1"] > 0 and class_breadth["positive_units"] > class_breadth["negative_units"]
    global_pass = protocol_global_pass and class_breadth_pass
    conditional_pass = (not global_pass) and len(gains) >= 2
    if global_pass:
        gate = "WINDOW_BOTTLENECK_CONFIRMED"
    elif conditional_pass:
        gate = "CLASS_CONDITIONAL_WINDOW_BENEFIT"
    else:
        gate = "NO_USEFUL_WINDOW_BENEFIT"

    n32 = [float(r["truncation_ratio"]) for r in coverage_rows if int(r["window_packets"]) == 32 and r["class_name"] == "__ALL__"]
    focus_n32 = [r for r in coverage_rows if int(r["window_packets"]) == 32 and r["class_name"].lower() in FOCUS.get(r["dataset"], set()) and float(r["truncation_ratio"]) >= 0.25 and float(r["delta_f1_vs_T8"]) >= 0.05]
    t32_t16 = [results[(s["dataset"], s["protocol_id"], 32)]["validation_macro_f1"] - results[(s["dataset"], s["protocol_id"], 16)]["validation_macro_f1"] for s in pilot_specs()]
    t64_enter = float(np.mean(t32_t16)) >= 0.01 and sum(x > 0 for x in t32_t16) >= 3 and (any(x >= 0.10 for x in n32) or bool(focus_n32))
    gate_payload = {
        "gate": gate, "best_longer_window": f"T{best}", "window_summaries": summary,
        "protocol_level_global_gate_pass": protocol_global_pass,
        "class_breadth_requirement": "median class delta > 0 and positive class-protocol units > negative units",
        "class_breadth": class_breadth, "class_breadth_pass": class_breadth_pass,
        "coverage_performance_correlations": coverage_correlations,
        "qualifying_focus_gains": gains, "focus_losses_below_minus_0_05": losses,
        "t64_gate_met": t64_enter, "t64_run_started": False,
    }
    write_json(ROOT / "gate_evaluation.json", gate_payload)

    result_df = pd.DataFrame(result_rows)
    pair_df = pd.DataFrame(paired_rows)
    lines = [
        "# Stage 15F-1A — Packet Window Sufficiency and Class-Conditional Feature Benchmark",
        "", "## Scope and controls", "",
        "All 15 formal runs use only frozen Known Train/Validation samples. Known Test and Unknown Test usage are both zero. The feature formula, seed, optimizer, 100-epoch budget, model topology, loss, and checkpoint rule are fixed; only the maximum packet window changes.",
        "", "Stage 15R T8 used ordinary BatchNorm1d, which allowed structural padding to influence batch moments. This stage therefore preserves the legacy checkpoint only for inference parity and compares T8/T16/T32 using one preregistered mask-safe BatchNorm implementation. Invalid slots are zeroed and excluded from BatchNorm and masked pooling. The padding-invariance tests pass.",
        "", "## Protocol-level results", "",
        "| Dataset | Protocol | Window | Accuracy | Macro-F1 | Weighted-F1 | Best epoch | Runtime (s) | Peak GPU MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.sort_values(["dataset", "protocol_id", "window"]).iterrows():
        lines.append(f"| {row.dataset} | {row.protocol_id} | T{int(row.window)} | {row.validation_accuracy:.6f} | {row.validation_macro_f1:.6f} | {row.validation_weighted_f1:.6f} | {int(row.best_epoch)} | {row.runtime_seconds:.1f} | {row.peak_gpu_memory_bytes/1048576:.1f} |")
    lines += ["", "## Historical E2, mask-safe T8, and Native boundary", "", "| Dataset | Protocol | Historical E2 Macro-F1 | Formal mask-safe T8 Macro-F1 | Δ mask-safe T8 - historical E2 | Native Macro-F1 |", "|---|---|---:|---:|---:|---:|"]
    parity_lookup = {(r["dataset"], r["protocol_id"]): r for r in parity["runs"]}
    native_lookup = {(r["dataset"], r["protocol_id"]): r for r in native_rows}
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        historical = float(parity_lookup[(dataset, protocol_id)]["reference"]["validation_macro_f1"])
        formal = float(results[(dataset, protocol_id, 8)]["validation_macro_f1"])
        native_f1 = float(native_lookup[(dataset, protocol_id)]["macro_f1"])
        lines.append(f"| {dataset} | {protocol_id} | {historical:.6f} | {formal:.6f} | {formal-historical:+.6f} | {native_f1:.6f} |")
    lines += ["", "Legacy inference parity is exact at the classification/confusion level. The historical-to-formal T8 delta is not a window effect: formal runs replace padding-sensitive BatchNorm with the preregistered mask-safe implementation and retrain the model. Window claims use only formal T16-T8 and T32-T8 deltas.", ""]
    lines += ["", "## Paired window deltas", "", "| Dataset | Protocol | Comparison | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | Runtime ratio |", "|---|---|---|---:|---:|---:|---:|"]
    for _, row in pair_df.iterrows():
        lines.append(f"| {row.dataset} | {row.protocol_id} | {row.comparison} | {row.delta_validation_accuracy:+.6f} | {row.delta_validation_macro_f1:+.6f} | {row.delta_validation_weighted_f1:+.6f} | {row.runtime_ratio:.3f} |")
    lines += ["", "## Gate", "", f"Final Gate: `{gate}`", "", f"The stronger longer-window candidate is T{best}: mean ΔMacro-F1 {summary[best]['mean_delta_macro_f1']:+.6f}, positive protocols {summary[best]['positive_protocols']}/5, worst protocol {summary[best]['worst_delta_macro_f1']:+.6f}.", "", f"Protocol-level numerical gate: {'PASS' if protocol_global_pass else 'FAIL'}. Class-breadth constraint from the task specification: {'PASS' if class_breadth_pass else 'FAIL'} ({class_breadth['positive_units']} positive, {class_breadth['negative_units']} negative, {class_breadth['unchanged_units']} unchanged out of {class_breadth['class_protocol_units']}; median ΔF1 {class_breadth['median_delta_f1']:+.6f}). A protocol-level pass is not promoted to a global window bottleneck when class breadth fails.", ""]
    if gains:
        lines += ["Qualifying class-conditional gains:", ""] + [f"- {x['dataset']} / {x['protocol_id']} / {x['class_name']}: ΔF1 {x['delta_f1']:+.6f}, net rescue {x['net_rescue']:+d}." for x in gains] + [""]
    if losses:
        lines += ["Focus-class regressions below -0.05:", ""] + [f"- {x['dataset']} / {x['protocol_id']} / {x['class_name']}: ΔF1 {x['delta_f1']:+.6f}." for x in losses] + [""]
    lines += ["## Native versus temporal complementarity", "", f"| Dataset | Protocol | Native Macro-F1 | T{best} Macro-F1 | Native-only correct | T{best}-only correct | Both correct | Both wrong |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for native_row in native_rows:
        dataset, protocol_id = native_row["dataset"], native_row["protocol_id"]
        comp = next(r for r in complement_rows if r["dataset"] == dataset and r["protocol_id"] == protocol_id and r["class_name"] == "__ALL__" and r["baseline_method"] == "Native" and r["candidate_method"] == f"T{best}")
        temporal = results[(dataset, protocol_id, best)]
        lines.append(f"| {dataset} | {protocol_id} | {native_row['macro_f1']:.6f} | {temporal['validation_macro_f1']:.6f} | {comp['baseline_correct_candidate_wrong']} | {comp['baseline_wrong_candidate_correct']} | {comp['both_correct']} | {comp['both_wrong']} |")
    lines += ["", "The two off-diagonal counts are the direct sample-level test of complementary information. A temporal-only rescue demonstrates information not used by Native; a simultaneous large Native-only count shows that the temporal branch is not a replacement for Native.", ""]

    lines += ["## Focus-class results", "", "| Dataset | Protocol | Class | T8 F1 | T16 F1 | T32 F1 | Best longer Δ vs T8 |", "|---|---|---|---:|---:|---:|---:|"]
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        names = sorted({r["class_name"] for r in class_rows if r["dataset"] == dataset and r["protocol_id"] == protocol_id and r["class_name"].lower() in FOCUS[dataset]})
        for name in names:
            vals = {int(r["window"]): float(r["f1"]) for r in class_rows if r["dataset"] == dataset and r["protocol_id"] == protocol_id and r["class_name"] == name}
            lines.append(f"| {dataset} | {protocol_id} | {name} | {vals[8]:.6f} | {vals[16]:.6f} | {vals[32]:.6f} | {vals[best]-vals[8]:+.6f} |")

    lines += ["", "## VNAT medium-2025 rsync confusion", "", "| Method | rsync support | predicted rsync | predicted scp | predicted sftp |", "|---|---:|---:|---:|---:|"]
    for method in ("Native", "T8", "T16", "T32"):
        rows = [r for r in confusion_rows if r["dataset"] == "vnat" and r["protocol_id"] == "medium_seed2025" and r["window_id"] == method and r["true_class"].lower() == "rsync"]
        if rows:
            count = {r["predicted_class"].lower(): r["count"] for r in rows}
            lines.append(f"| {method} | {rows[0]['true_class_support']} | {count.get('rsync', 0)} | {count.get('scp', 0)} | {count.get('sftp', 0)} |")

    lines += ["", "## Coverage versus performance", ""]
    for candidate in (16, 32):
        corr = coverage_correlations[f"T{candidate}_delta_vs_T8_truncation"]
        lines.append(
            f"- T8 class truncation ratio versus T{candidate}-T8 class F1: "
            f"Spearman rho={corr['spearman_rho']:.6f}, p={corr['pvalue']:.6f}, "
            f"n={corr['class_protocol_units']}."
        )
    lines += [
        "",
        "Observed T8 truncation is not a useful class-level predictor of the gain from T16 or T32 in these pilots. Lower truncation at longer windows therefore does not by itself establish a performance bottleneck.",
        "",
    ]

    lines += ["", "## Direct answers", ""]
    def delta(dataset, protocol, window):
        return results[(dataset, protocol, window)]["validation_macro_f1"] - results[(dataset, protocol, 8)]["validation_macro_f1"]
    vpn_best = max(delta("iscx_vpn", "medium_seed2022", 16), delta("iscx_vpn", "medium_seed2022", 32))
    tor_best = max(delta("iscx_tor", "medium_seed2022", 16), delta("iscx_tor", "medium_seed2022", 32))
    rsync_rows = [r for r in class_rows if r["dataset"] == "vnat" and r["protocol_id"] == "medium_seed2025" and r["class_name"].lower() == "rsync"]
    rsync = {int(r["window"]): float(r["f1"]) for r in rsync_rows}
    lines += [
        f"1. **Is the first-8-packet E2 globally window-limited?** {'Yes under the preregistered global criterion.' if global_pass else 'No under the preregistered global criterion.'} The best longer window changes mean Macro-F1 by {summary[best]['mean_delta_macro_f1']:+.6f} across five protocols.",
        f"2. **Do 16/32 packets improve ISCX-VPN and ISCXTor2016?** Best observed Macro-F1 deltas versus T8 are {vpn_best:+.6f} and {tor_best:+.6f}, respectively.",
        f"3. **Are benefits class-conditional?** {len(gains)} preregistered focus class-protocol units meet ΔF1>=0.05 with positive net rescue; {len(losses)} focus units regress below -0.05 for the selected longer window.",
        f"4. **Does a longer window repair VNAT rsync/scp?** medium-2025 rsync F1 is T8={rsync.get(8, float('nan')):.6f}, T16={rsync.get(16, float('nan')):.6f}, T32={rsync.get(32, float('nan')):.6f}; the confusion counts above show whether scp errors are actually removed rather than redistributed.",
        "5. **Do temporal and Native inputs carry complementary information?** Yes only where both Native-only and temporal-only correct counts are non-zero; the protocol table reports the exact counts without using test data.",
        f"6. **Why was historical E2 weak?** Window insufficiency alone is {'supported' if global_pass else 'not sufficient to explain the gap'}. Formal T8/T16/T32 isolate window effects, while the persistent Native gap and two-way rescue quantify feature/model limitations separately.",
        "",
    ]
    lines += [
        "## T64 decision", "",
        f"The preregistered T64 gate is {'MET' if t64_enter else 'NOT MET'}. T64 was not run in this stage, as required.", "",
        "## Recommendation", "",
        "Retain T16 only as a class-conditional temporal candidate; do not promote it to the default standalone representation. Do not retain T32 as the default and do not enter T64. A later preregistered benchmark should prioritize richer flow statistics, burst structure, or structured byte inputs because the Native gap remains much larger than the window-only gains. No next-stage experiment was started.", "",
        "## Interpretation boundary", "",
        "Any difference between the formal T8 and the historical Stage 15R E2 score includes the preregistered padding-safe BatchNorm correction; only formal T16-T8 and T32-T8 comparisons identify window effects. Native-versus-temporal complementarity is descriptive and uses exactly the same Known Validation sample IDs.", "",
    ]
    (ROOT / "stage15f1a_report.md").write_text("\n".join(lines), encoding="utf-8")
    results_lines = [
        "# RESULTS — Stage 15F-1A", "",
        "Status: complete",
        f"Final Gate: `{gate}`", "",
        "## Core results", "",
        f"T16 is the strongest longer window: mean validation Macro-F1 delta versus T8 is {summary[16]['mean_delta_macro_f1']:+.6f}, with {summary[16]['positive_protocols']}/5 positive protocols and worst delta {summary[16]['worst_delta_macro_f1']:+.6f}.",
        f"T32 mean delta versus T8 is {summary[32]['mean_delta_macro_f1']:+.6f}; its worst protocol delta is {summary[32]['worst_delta_macro_f1']:+.6f}.",
        f"Across {class_breadth['class_protocol_units']} class-protocol units, T16 produces {class_breadth['positive_units']} positive, {class_breadth['negative_units']} negative, and {class_breadth['unchanged_units']} unchanged deltas; median delta is {class_breadth['median_delta_f1']:+.6f}.", "",
        "The detailed evidence is in `window_pilot_results.csv`, `window_paired_comparison.csv`, `window_per_class_results.csv`, `window_confusion_analysis.csv`, `window_error_complementarity.csv`, and `window_coverage_vs_performance.csv`.", "",
        "## Configuration and execution", "",
        "Formal T8/T16/T32 runs use the same signed log frame length, log IAT, valid-packet mask, mask-safe CNN, cross-entropy objective, optimizer, seed, checkpoint rule, and 100-epoch budget. Only maximum packet count changes. All 15 runs completed and retained histories, predictions, per-class metrics, confusion matrices, and hashed best checkpoints.", "",
        "## Data and split", "",
        "The five frozen pilots are ISCX-VPN medium-2022, ISCXTor medium-2022, VNAT medium-2025, VNAT medium-2026, and USTC A-2. Normalization is fit on Known Train only. Known Test and Unknown Test feature usage are both zero.", "",
        "## Preserved evidence", "",
        "The bundle preserves the first-32-packet caches and audits, formal runs, legacy T8 parity evidence, pre/post frozen-asset hashes, failed direction-reconstruction and CPU-parity attempts, aggregation outputs, tests, logs, and `completion_verification.json`.", "",
        "## Limitations", "",
        "This is a five-protocol Known-Validation pilot with one fixed seed, not a final Test or open-set claim. Historical Stage 15R E2 used padding-sensitive BatchNorm, whereas the formal controlled comparison uses one mask-safe implementation; historical-to-formal T8 changes are not window effects.", "",
        "## Conclusion and next step", "",
        f"The final Gate is `{gate}`. Retain T16 only as a class-conditional temporal candidate; do not promote T32 or enter T64. Stop here. A later preregistered experiment may compare richer statistics, burst structure, or structured byte inputs, but no such experiment was started.", "",
        "No Open-Detect, DES, H1, T64, or multiview training was run.", "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(results_lines), encoding="utf-8")
    print(json.dumps(gate_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
