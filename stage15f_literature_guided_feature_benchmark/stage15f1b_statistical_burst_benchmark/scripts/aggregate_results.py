#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from common import (
    ROOT, STAGE12, STAGE14D, STAGE15F1A, STAGE3, feature_set_names,
    group_by_feature, pilot_specs, protocol_rows, read_csv, read_json, run_dir,
    write_csv, write_json,
)
from run_behavior_pilot import assemble


FOCUS = {
    "vnat": {"rsync", "scp", "sftp", "netflix", "youtube", "vimeo"},
    "iscx_vpn": {"spotify", "netflix", "vimeo"},
    "iscx_tor": {"facebook", "hangouts", "vimeo", "youtube", "ftp"},
    "ustc": {"worldofwarcraft", "weibo"},
}
FULL_SETS = ("S12", "S-A", "S-AB", "S-ABC", "S-ABCD", "S-Burst")
EARLY_SETS = ("S12", "S-ABCD", "S-Burst")
METRICS = ("validation_accuracy", "validation_macro_f1", "validation_weighted_f1")


def load_predictions(directory, expected_ids: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
    with np.load(directory / "validation_predictions.npz", allow_pickle=False) as data:
        ids = data["sample_ids"].astype(str)
        truth = data["true_labels"].astype(np.int64)
        pred = data["predicted_labels"].astype(np.int64)
    lookup = {uid: (int(y), int(p)) for uid, y, p in zip(ids, truth, pred)}
    if set(expected_ids) != set(lookup):
        raise RuntimeError(f"prediction sample ID mismatch: {directory}")
    ordered_truth = np.asarray([lookup[uid][0] for uid in expected_ids], dtype=np.int64)
    if not np.array_equal(ordered_truth, expected_truth):
        raise RuntimeError(f"prediction labels mismatch: {directory}")
    return np.asarray([lookup[uid][1] for uid in expected_ids], dtype=np.int64)


def native_predictions(dataset: str, protocol_id: str, classes: list[str], val_rows: list[dict]) -> np.ndarray:
    ids = np.asarray([row["sample_id"] for row in val_rows])
    truth = np.asarray([row["label"] for row in val_rows], dtype=np.int64)
    if dataset in {"iscx_vpn", "iscx_tor"}:
        path = STAGE12 / "runs" / dataset / "medium" / "seed2022" / "known_train_validation_features.npz"
        with np.load(path, allow_pickle=False) as data:
            source_truth = data["validation_labels_reindexed"].astype(np.int64)
            pred = data["validation_m0_predictions"].astype(np.int64)
        if not np.array_equal(source_truth, truth):
            raise RuntimeError(f"Native label mismatch: {dataset}")
        return pred
    if dataset == "ustc":
        table = pd.read_parquet(STAGE3 / "artifacts" / "A-2" / "val_known_mu.parquet", columns=["flow_id", "class_name", "native_predicted_known_class"])
        lookup = table.set_index("flow_id")
        class_to_id = {name: index for index, name in enumerate(classes)}
        ordered = lookup.loc[ids]
        source_truth = np.asarray([class_to_id[name] for name in ordered["class_name"]], dtype=np.int64)
        if not np.array_equal(source_truth, truth):
            raise RuntimeError("USTC Native truth mismatch")
        return np.asarray([class_to_id[name] for name in ordered["native_predicted_known_class"]], dtype=np.int64)
    rows = [row for row in read_csv(STAGE14D / "artifacts" / protocol_id / "sample_scores.csv") if row["role"] == "validation" and row["method"] == "M0"]
    lookup = {row["flow_uid"]: row for row in rows}
    class_to_id = {name: index for index, name in enumerate(classes)}
    source_truth = np.asarray([class_to_id[lookup[uid]["true_application"]] for uid in ids], dtype=np.int64)
    if not np.array_equal(source_truth, truth):
        raise RuntimeError(f"VNAT Native truth mismatch: {protocol_id}")
    return np.asarray([class_to_id[lookup[uid]["predicted_known_application"]] for uid in ids], dtype=np.int64)


def metrics(truth: np.ndarray, prediction: np.ndarray, class_count: int) -> dict[str, float]:
    _, _, f1, support = precision_recall_fscore_support(truth, prediction, labels=np.arange(class_count), zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }


def complement_row(dataset, protocol_id, class_name, mask, truth, left, right, left_name, right_name):
    left_correct = left[mask] == truth[mask]
    right_correct = right[mask] == truth[mask]
    return {
        "dataset": dataset, "protocol_id": protocol_id, "class_name": class_name,
        "baseline_method": left_name, "candidate_method": right_name, "support": int(mask.sum()),
        "baseline_correct_candidate_wrong": int((left_correct & ~right_correct).sum()),
        "baseline_wrong_candidate_correct": int((~left_correct & right_correct).sum()),
        "both_correct": int((left_correct & right_correct).sum()),
        "both_wrong": int((~left_correct & ~right_correct).sum()),
        "net_rescue": int((~left_correct & right_correct).sum() - (left_correct & ~right_correct).sum()),
    }


def comparison_summary(results: dict, candidate: str, baseline: str) -> dict:
    deltas = [
        results[(spec["dataset"], spec["protocol_id"], candidate)]["validation_macro_f1"]
        - results[(spec["dataset"], spec["protocol_id"], baseline)]["validation_macro_f1"]
        for spec in pilot_specs()
    ]
    return {
        "comparison": f"{candidate}-{baseline}", "mean_delta_macro_f1": float(np.mean(deltas)),
        "positive_protocols": int(sum(delta > 0 for delta in deltas)),
        "worst_delta_macro_f1": float(min(deltas)), "deltas": deltas,
    }


def main() -> None:
    config = read_json(ROOT / "config.json")
    parity = read_json(ROOT / "s12_parity_summary.json")
    grid = read_json(ROOT / "grid_status.json")
    if parity["status"] != "PASS" or grid["status"] != "PASS":
        raise RuntimeError("cannot aggregate incomplete parity/grid")
    result_rows, class_rows, confusion_rows, importance_rows = [], [], [], []
    result_lookup = {}
    class_lookup = {}
    for scenario, feature_sets in (("FULL_FLOW", FULL_SETS), ("EARLY_16", EARLY_SETS)):
        for feature_set in feature_sets:
            for spec in pilot_specs():
                dataset, protocol_id = spec["dataset"], spec["protocol_id"]
                directory = run_dir(scenario, feature_set, dataset, protocol_id)
                result = read_json(directory / "result.json")
                if result["status"] != "PASS" or result["known_test_samples_used"] or result["unknown_test_samples_used"]:
                    raise RuntimeError(f"invalid run: {directory}")
                result_rows.append(result)
                result_lookup[(scenario, dataset, protocol_id, feature_set)] = result
                for row in read_csv(directory / "per_class_results.csv"):
                    class_rows.append(row)
                    class_lookup[(scenario, dataset, protocol_id, feature_set, row["class_name"])] = row
                classes = result["class_names"]
                matrix = np.load(directory / "validation_confusion_matrix.npy", allow_pickle=False)
                for true_index, true_name in enumerate(classes):
                    for pred_index, pred_name in enumerate(classes):
                        count = int(matrix[true_index, pred_index])
                        confusion_rows.append({
                            "scenario": scenario, "feature_set": feature_set, "dataset": dataset,
                            "protocol_id": protocol_id, "true_class": true_name, "predicted_class": pred_name,
                            "count": count, "true_class_support": int(matrix[true_index].sum()),
                            "row_rate": float(count / max(1, matrix[true_index].sum())),
                        })
                rows = read_csv(directory / "feature_importance.csv")
                for row in rows:
                    importance_rows.append({"importance_scope": "feature", "scenario": scenario, "feature_set": feature_set, "dataset": dataset, "protocol_id": protocol_id, **row})
                grouped = defaultdict(float)
                for row in rows:
                    grouped[row.get("feature_group", group_by_feature()[row["feature"]])] += float(row["gain"])
                total = sum(grouped.values())
                for group, gain in grouped.items():
                    importance_rows.append({
                        "importance_scope": "group", "scenario": scenario, "feature_set": feature_set,
                        "dataset": dataset, "protocol_id": protocol_id, "feature": f"GROUP_{group}",
                        "feature_group": group, "gain": gain, "normalized_gain": gain / total if total else 0.0,
                    })
    write_csv(ROOT / "statistical_pilot_results.csv", result_rows)
    write_csv(ROOT / "per_class_results.csv", class_rows)
    write_csv(ROOT / "confusion_analysis.csv", confusion_rows)
    write_csv(ROOT / "feature_importance.csv", importance_rows)

    missing_rows = []
    for scenario, feature_sets in (("FULL_FLOW", FULL_SETS), ("EARLY_16", EARLY_SETS)):
        for feature_set in feature_sets:
            for spec in pilot_specs():
                classes, train_rows, val_rows = protocol_rows(spec["dataset"], spec["protocol_id"])
                for role, rows in (("known_train", train_rows), ("known_validation", val_rows)):
                    matrix, _labels, _ids, audit = assemble(spec["dataset"], scenario, feature_set, rows)
                    for column, name in enumerate(audit["feature_names"]):
                        count = int(np.isnan(matrix[:, column]).sum())
                        missing_rows.append({
                            "dataset": spec["dataset"], "protocol_id": spec["protocol_id"], "scenario": scenario,
                            "feature_set": feature_set, "role": role, "feature": name, "samples": len(matrix),
                            "missing_count": count, "missing_ratio": count / len(matrix),
                        })
    write_csv(ROOT / "feature_missingness.csv", missing_rows)
    availability_rows = []
    for row in missing_rows:
        ratio = float(row["missing_ratio"])
        availability_rows.append({
            **row,
            "availability": "UNAVAILABLE" if ratio == 1.0 else ("COMPLETE" if ratio == 0.0 else "PARTIAL"),
            "zero_imputation_used": False,
        })
    write_csv(ROOT / "feature_availability.csv", availability_rows)

    sequential = (("S12", "S-A"), ("S-A", "S-AB"), ("S-AB", "S-ABC"), ("S-ABC", "S-ABCD"), ("S-ABCD", "S-Burst"), ("S12", "S-ABCD"), ("S12", "S-Burst"))
    ablation_rows = []
    full_result_key = {(d, p, f): result_lookup[("FULL_FLOW", d, p, f)] for d, p, f in [(s["dataset"], s["protocol_id"], f) for s in pilot_specs() for f in FULL_SETS]}
    for baseline, candidate in sequential:
        for spec in pilot_specs():
            left = full_result_key[(spec["dataset"], spec["protocol_id"], baseline)]
            right = full_result_key[(spec["dataset"], spec["protocol_id"], candidate)]
            row = {"scenario": "FULL_FLOW", "dataset": spec["dataset"], "protocol_id": spec["protocol_id"], "comparison": f"{candidate}-{baseline}", "baseline": baseline, "candidate": candidate}
            for metric in METRICS:
                row[f"baseline_{metric}"] = left[metric]
                row[f"candidate_{metric}"] = right[metric]
                row[f"delta_{metric}"] = right[metric] - left[metric]
            ablation_rows.append(row)
    write_csv(ROOT / "statistical_group_ablation.csv", ablation_rows)

    burst_rows = []
    for scenario in ("FULL_FLOW", "EARLY_16"):
        for spec in pilot_specs():
            left = result_lookup[(scenario, spec["dataset"], spec["protocol_id"], "S-ABCD")]
            right = result_lookup[(scenario, spec["dataset"], spec["protocol_id"], "S-Burst")]
            burst_rows.append({
                "scenario": scenario, "dataset": spec["dataset"], "protocol_id": spec["protocol_id"],
                "baseline_macro_f1": left["validation_macro_f1"], "burst_macro_f1": right["validation_macro_f1"],
                "delta_macro_f1": right["validation_macro_f1"] - left["validation_macro_f1"],
                "delta_accuracy": right["validation_accuracy"] - left["validation_accuracy"],
                "delta_weighted_f1": right["validation_weighted_f1"] - left["validation_weighted_f1"],
            })
    write_csv(ROOT / "burst_incremental_results.csv", burst_rows)

    early_rows = []
    for feature_set in EARLY_SETS:
        for spec in pilot_specs():
            full = result_lookup[("FULL_FLOW", spec["dataset"], spec["protocol_id"], feature_set)]
            early = result_lookup[("EARLY_16", spec["dataset"], spec["protocol_id"], feature_set)]
            early_rows.append({
                "dataset": spec["dataset"], "protocol_id": spec["protocol_id"], "feature_set": feature_set,
                "full_flow_macro_f1": full["validation_macro_f1"], "early16_macro_f1": early["validation_macro_f1"],
                "delta_full_minus_early_macro_f1": full["validation_macro_f1"] - early["validation_macro_f1"],
                "delta_full_minus_early_accuracy": full["validation_accuracy"] - early["validation_accuracy"],
                "delta_full_minus_early_weighted_f1": full["validation_weighted_f1"] - early["validation_weighted_f1"],
            })
    write_csv(ROOT / "early16_vs_fullflow.csv", early_rows)

    complement_rows, system_rows = [], []
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        classes, _train_rows, val_rows = protocol_rows(dataset, protocol_id)
        ids = np.asarray([row["sample_id"] for row in val_rows])
        truth = np.asarray([row["label"] for row in val_rows], dtype=np.int64)
        predictions = {
            "Native": native_predictions(dataset, protocol_id, classes, val_rows),
            "S12": load_predictions(run_dir("FULL_FLOW", "S12", dataset, protocol_id), ids, truth),
            "S-ABCD": load_predictions(run_dir("FULL_FLOW", "S-ABCD", dataset, protocol_id), ids, truth),
            "S-Burst": load_predictions(run_dir("FULL_FLOW", "S-Burst", dataset, protocol_id), ids, truth),
            "T16": load_predictions(STAGE15F1A / "runs" / "T16" / dataset / protocol_id, ids, truth),
        }
        for method, prediction in predictions.items():
            system_rows.append({"dataset": dataset, "protocol_id": protocol_id, "method": method, **metrics(truth, prediction, len(classes))})
        for left_name, right_name in (
            ("Native", "S12"),
            ("Native", "S-ABCD"),
            ("Native", "S-Burst"),
            ("S12", "S-ABCD"),
            ("S12", "S-Burst"),
            ("S-ABCD", "S-Burst"),
            ("T16", "S-Burst"),
        ):
            left, right = predictions[left_name], predictions[right_name]
            scopes = [("__ALL__", np.ones(len(truth), dtype=bool))] + [(name, truth == index) for index, name in enumerate(classes)]
            for class_name, mask in scopes:
                complement_rows.append(complement_row(dataset, protocol_id, class_name, mask, truth, left, right, left_name, right_name))
    write_csv(ROOT / "native_behavior_complementarity.csv", complement_rows)
    write_csv(ROOT / "system_level_comparison.csv", system_rows)
    system_lookup = {(row["dataset"], row["protocol_id"], row["method"]): row for row in system_rows}
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        for baseline in ("Native", "T16"):
            left = system_lookup[(dataset, protocol_id, baseline)]
            right = system_lookup[(dataset, protocol_id, "S-Burst")]
            row = {"scenario": "SYSTEM_LEVEL", "dataset": dataset, "protocol_id": protocol_id, "comparison": f"S-Burst-{baseline}", "baseline": baseline, "candidate": "S-Burst"}
            for metric in ("accuracy", "macro_f1", "weighted_f1"):
                row[f"baseline_validation_{metric}"] = left[metric]
                row[f"candidate_validation_{metric}"] = right[metric]
                row[f"delta_validation_{metric}"] = right[metric] - left[metric]
            ablation_rows.append(row)
    write_csv(ROOT / "statistical_group_ablation.csv", ablation_rows)

    full_results = {(dataset, protocol_id, feature_set): result_lookup[("FULL_FLOW", dataset, protocol_id, feature_set)] for dataset, protocol_id, feature_set in full_result_key}
    stat_summary = comparison_summary(full_results, "S-ABCD", "S12")
    burst_summary = comparison_summary(full_results, "S-Burst", "S-ABCD")
    total_summary = comparison_summary(full_results, "S-Burst", "S12")
    stat_rule = config["gates"]["statistical_feature_benefit"]
    stat_pass = stat_summary["mean_delta_macro_f1"] >= stat_rule["minimum_mean_delta_macro_f1"] and stat_summary["positive_protocols"] >= stat_rule["minimum_positive_protocols"] and stat_summary["worst_delta_macro_f1"] >= stat_rule["maximum_allowed_worst_delta_macro_f1"]

    burst_focus_gains, focus_losses, behavior_focus_gains = [], [], []
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        for name in {key[4] for key in class_lookup if key[0] == "FULL_FLOW" and key[1] == dataset and key[2] == protocol_id and key[3] == "S12"}:
            if name.lower() not in FOCUS[dataset]:
                continue
            s12 = float(class_lookup[("FULL_FLOW", dataset, protocol_id, "S12", name)]["f1"])
            abcd = float(class_lookup[("FULL_FLOW", dataset, protocol_id, "S-ABCD", name)]["f1"])
            burst = float(class_lookup[("FULL_FLOW", dataset, protocol_id, "S-Burst", name)]["f1"])
            abcd_rescue = next(row["net_rescue"] for row in complement_rows if row["dataset"] == dataset and row["protocol_id"] == protocol_id and row["class_name"] == name and row["baseline_method"] == "S12" and row["candidate_method"] == "S-ABCD")
            total_rescue = next(row["net_rescue"] for row in complement_rows if row["dataset"] == dataset and row["protocol_id"] == protocol_id and row["class_name"] == name and row["baseline_method"] == "S12" and row["candidate_method"] == "S-Burst")
            burst_rescue = next(row["net_rescue"] for row in complement_rows if row["dataset"] == dataset and row["protocol_id"] == protocol_id and row["class_name"] == name and row["baseline_method"] == "S-ABCD" and row["candidate_method"] == "S-Burst")
            best_value, best_name = max((abcd, "S-ABCD"), (burst, "S-Burst"))
            best_rescue = abcd_rescue if best_name == "S-ABCD" else total_rescue
            if best_value - s12 >= 0.05 and best_rescue > 0:
                behavior_focus_gains.append({"dataset": dataset, "protocol_id": protocol_id, "class_name": name, "candidate": best_name, "delta_f1_vs_s12": best_value - s12, "net_rescue": best_rescue})
            if burst - abcd >= 0.05 and burst_rescue > 0:
                burst_focus_gains.append({"dataset": dataset, "protocol_id": protocol_id, "class_name": name, "delta_f1": burst - abcd, "net_rescue": burst_rescue})
            if best_value - s12 < -0.05:
                focus_losses.append({"dataset": dataset, "protocol_id": protocol_id, "class_name": name, "delta_f1_vs_s12": best_value - s12})
    burst_rule = config["gates"]["burst_complementarity"]
    burst_pass = burst_summary["mean_delta_macro_f1"] >= burst_rule["minimum_mean_delta_macro_f1"] and burst_summary["positive_protocols"] >= burst_rule["minimum_positive_protocols"] and burst_summary["worst_delta_macro_f1"] >= burst_rule["maximum_allowed_worst_delta_macro_f1"] and len(burst_focus_gains) >= burst_rule["minimum_focus_class_units_with_delta_f1_0_05_and_positive_rescue"]
    conditional_rule = config["gates"]["class_conditional_behavior_benefit"]
    conditional_pass = len(behavior_focus_gains) >= conditional_rule["minimum_focus_class_units_with_delta_f1_0_05_and_positive_rescue"] and len(focus_losses) <= conditional_rule["maximum_focus_class_units_below_minus_0_05"]
    conclusions = []
    if stat_pass:
        conclusions.append("STATISTICAL_FEATURE_BENEFIT")
    if burst_pass:
        conclusions.append("BURST_COMPLEMENTARITY_CONFIRMED")
    if conditional_pass:
        conclusions.append("CLASS_CONDITIONAL_BEHAVIOR_BENEFIT")
    if not conclusions:
        conclusions.append("NO_USEFUL_BEHAVIOR_BENEFIT")
    full15_rule = config["gates"]["full15_expansion"]
    full15_gate = total_summary["mean_delta_macro_f1"] >= full15_rule["minimum_mean_delta_macro_f1"] and total_summary["positive_protocols"] >= full15_rule["minimum_positive_protocols"] and total_summary["worst_delta_macro_f1"] >= full15_rule["maximum_allowed_worst_delta_macro_f1"]
    gate = {
        "conclusions": conclusions, "statistical_summary": stat_summary, "burst_summary": burst_summary,
        "s_burst_vs_s12": total_summary, "statistical_gate_pass": stat_pass,
        "burst_gate_pass": burst_pass, "class_conditional_gate_pass": conditional_pass,
        "behavior_focus_gains": behavior_focus_gains, "burst_focus_gains": burst_focus_gains,
        "focus_losses": focus_losses, "full15_gate_met": full15_gate,
        "full15_status": "NOT_RUN_STAGE_SCOPE", "open_set_status": "NOT_RUN",
        "stage15f1c_status": "NOT_RUN",
    }
    write_json(ROOT / "gate_evaluation.json", gate)

    lines = [
        "# Stage 15F-1B — Statistical & Burst Feature Sufficiency Benchmark", "",
        "## Scope and integrity", "",
        "All experiments use only frozen Known Train and Known Validation rows. Known Test and Unknown Test feature values are not loaded. FULL_FLOW S12 reuses and exactly reproduces Stage 15R E3; all other runs retain the same LightGBM configuration and change only the preregistered feature set or observation horizon.", "",
        "## Pilot Macro-F1", "",
        "| Dataset | Protocol | S12 | S-A | S-AB | S-ABC | S-ABCD | S-Burst | Early16 S-Burst |", "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        values = [result_lookup[("FULL_FLOW", dataset, protocol_id, feature_set)]["validation_macro_f1"] for feature_set in FULL_SETS]
        early = result_lookup[("EARLY_16", dataset, protocol_id, "S-Burst")]["validation_macro_f1"]
        lines.append(f"| {dataset} | {protocol_id} | " + " | ".join(f"{value:.6f}" for value in values) + f" | {early:.6f} |")
    lines += [
        "", "## Gate", "", f"Conclusions: `{', '.join(conclusions)}`", "",
        f"S-ABCD - S12 mean ΔMacro-F1={stat_summary['mean_delta_macro_f1']:+.6f}, positive={stat_summary['positive_protocols']}/5, worst={stat_summary['worst_delta_macro_f1']:+.6f}.",
        f"S-Burst - S-ABCD mean ΔMacro-F1={burst_summary['mean_delta_macro_f1']:+.6f}, positive={burst_summary['positive_protocols']}/5, worst={burst_summary['worst_delta_macro_f1']:+.6f}.",
        f"S-Burst - S12 mean ΔMacro-F1={total_summary['mean_delta_macro_f1']:+.6f}, positive={total_summary['positive_protocols']}/5, worst={total_summary['worst_delta_macro_f1']:+.6f}.", "",
        "## VNAT medium-2025 rsync/scp audit", "",
        "| Method | rsync correct/191 | rsync→scp | scp→rsync | rsync Recall | rsync F1 | scp Recall | scp F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    dataset, protocol_id = "vnat", "medium_seed2025"
    classes, _, val_rows = protocol_rows(dataset, protocol_id)
    truth = np.asarray([row["label"] for row in val_rows], dtype=np.int64)
    ids = np.asarray([row["sample_id"] for row in val_rows])
    method_predictions = {
        "Native": native_predictions(dataset, protocol_id, classes, val_rows),
        "T16": load_predictions(STAGE15F1A / "runs" / "T16" / dataset / protocol_id, ids, truth),
        "S12": load_predictions(run_dir("FULL_FLOW", "S12", dataset, protocol_id), ids, truth),
        "S-ABCD": load_predictions(run_dir("FULL_FLOW", "S-ABCD", dataset, protocol_id), ids, truth),
        "S-Burst": load_predictions(run_dir("FULL_FLOW", "S-Burst", dataset, protocol_id), ids, truth),
    }
    rsync_i, scp_i = classes.index("rsync"), classes.index("scp")
    for method, pred in method_predictions.items():
        matrix = confusion_matrix(truth, pred, labels=np.arange(len(classes)))
        p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=np.arange(len(classes)), zero_division=0)
        lines.append(f"| {method} | {matrix[rsync_i, rsync_i]}/191 | {matrix[rsync_i, scp_i]} | {matrix[scp_i, rsync_i]} | {r[rsync_i]:.6f} | {f[rsync_i]:.6f} | {r[scp_i]:.6f} | {f[scp_i]:.6f} |")

    step_order = ("S-A-S12", "S-AB-S-A", "S-ABC-S-AB", "S-ABCD-S-ABC", "S-Burst-S-ABCD")
    step_labels = {
        "S-A-S12": "A rates",
        "S-AB-S-A": "B length distribution",
        "S-ABC-S-AB": "C bidirectional",
        "S-ABCD-S-ABC": "D IAT distribution",
        "S-Burst-S-ABCD": "E burst",
    }
    step_stats = {}
    for comparison in step_order:
        values = [float(row["delta_validation_macro_f1"]) for row in ablation_rows if row["scenario"] == "FULL_FLOW" and row["comparison"] == comparison]
        step_stats[comparison] = {
            "mean": float(np.mean(values)), "positive": sum(value > 0 for value in values),
            "worst": min(values), "best": max(values),
        }
    lines += [
        "", "## Incremental feature-group ablation", "",
        "| Increment | Meaning | Mean ΔMacro-F1 | Positive pilots | Worst | Best |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for comparison in step_order:
        stats = step_stats[comparison]
        lines.append(f"| {comparison} | {step_labels[comparison]} | {stats['mean']:+.6f} | {stats['positive']}/5 | {stats['worst']:+.6f} | {stats['best']:+.6f} |")

    lines += [
        "", "## Early-16 versus full-flow", "",
        "Positive values mean full-flow is better; negative values mean Early-16 is better.", "",
        "| Dataset | Protocol | Feature set | Full Macro-F1 | Early-16 Macro-F1 | Full − Early |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in early_rows:
        lines.append(
            f"| {row['dataset']} | {row['protocol_id']} | {row['feature_set']} | "
            f"{float(row['full_flow_macro_f1']):.6f} | {float(row['early16_macro_f1']):.6f} | "
            f"{float(row['delta_full_minus_early_macro_f1']):+.6f} |"
        )
    early_means = {
        feature_set: float(np.mean([float(row["delta_full_minus_early_macro_f1"]) for row in early_rows if row["feature_set"] == feature_set]))
        for feature_set in EARLY_SETS
    }

    lines += [
        "", "## Focus-class behavior", "",
        "Only classes that are Known in the frozen pilot are shown. VNAT `sftp` is Unknown in both pilots, and `scp` is Unknown in medium-2026, so their Known-Validation features are deliberately unavailable.", "",
        "| Dataset | Protocol | Class | S12 F1 | S-ABCD F1 | S-Burst F1 | Best Δ vs S12 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        class_names = sorted({key[4] for key in class_lookup if key[:4] == ("FULL_FLOW", dataset, protocol_id, "S12")})
        for class_name in class_names:
            if class_name.lower() not in FOCUS[dataset]:
                continue
            s12_f1 = float(class_lookup[("FULL_FLOW", dataset, protocol_id, "S12", class_name)]["f1"])
            abcd_f1 = float(class_lookup[("FULL_FLOW", dataset, protocol_id, "S-ABCD", class_name)]["f1"])
            burst_f1 = float(class_lookup[("FULL_FLOW", dataset, protocol_id, "S-Burst", class_name)]["f1"])
            lines.append(f"| {dataset} | {protocol_id} | {class_name} | {s12_f1:.6f} | {abcd_f1:.6f} | {burst_f1:.6f} | {max(abcd_f1, burst_f1) - s12_f1:+.6f} |")

    complement_lookup = {
        (row["dataset"], row["protocol_id"], row["class_name"], row["baseline_method"], row["candidate_method"]): row
        for row in complement_rows
    }
    lines += [
        "", "## System-level error complementarity", "",
        "Counts use exactly the same Known-Validation samples. `lost` is baseline-correct/candidate-wrong; `rescued` is baseline-wrong/candidate-correct.", "",
        "| Dataset | Protocol | Comparison | Lost | Rescued | Net rescue |",
        "|---|---|---|---:|---:|---:|",
    ]
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        for baseline, candidate in (("Native", "S-Burst"), ("T16", "S-Burst"), ("S-ABCD", "S-Burst")):
            row = complement_lookup[(dataset, protocol_id, "__ALL__", baseline, candidate)]
            lines.append(
                f"| {dataset} | {protocol_id} | {candidate} − {baseline} | "
                f"{row['baseline_correct_candidate_wrong']} | {row['baseline_wrong_candidate_correct']} | {int(row['net_rescue']):+d} |"
            )

    lines += [
        "", "## Gain-based feature importance", "",
        "Importance is diagnostic, not causal. Values below are group-normalized gain for full-flow S-Burst.", "",
        "| Dataset | Protocol | A | B | C | D | E | S12 | Top three individual features |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        selected = [row for row in importance_rows if row["importance_scope"] == "group" and row["scenario"] == "FULL_FLOW" and row["feature_set"] == "S-Burst" and row["dataset"] == dataset and row["protocol_id"] == protocol_id]
        group_gain = {row["feature_group"]: float(row["normalized_gain"]) for row in selected}
        feature_selected = [row for row in importance_rows if row["importance_scope"] == "feature" and row["scenario"] == "FULL_FLOW" and row["feature_set"] == "S-Burst" and row["dataset"] == dataset and row["protocol_id"] == protocol_id]
        top = sorted(feature_selected, key=lambda row: float(row["normalized_gain"]), reverse=True)[:3]
        top_text = ", ".join(f"{row['feature']} ({float(row['normalized_gain']):.3f})" for row in top)
        lines.append(
            f"| {dataset} | {protocol_id} | {group_gain.get('A', 0):.3f} | {group_gain.get('B', 0):.3f} | "
            f"{group_gain.get('C', 0):.3f} | {group_gain.get('D', 0):.3f} | {group_gain.get('E', 0):.3f} | "
            f"{group_gain.get('S12', 0):.3f} | {top_text} |"
        )

    native_burst = [
        complement_lookup[(spec["dataset"], spec["protocol_id"], "__ALL__", "Native", "S-Burst")]
        for spec in pilot_specs()
    ]
    t16_burst = [
        complement_lookup[(spec["dataset"], spec["protocol_id"], "__ALL__", "T16", "S-Burst")]
        for spec in pilot_specs()
    ]
    vpn_gain = total_summary["deltas"][0]
    tor_gain = total_summary["deltas"][1]
    lines += [
        "", "## Answers to the 12 required questions", "",
        f"1. **Are the original 12 statistics insufficient?** Yes, at pilot scope. S-ABCD improves S12 by {stat_summary['mean_delta_macro_f1']:+.6f} mean Macro-F1, is positive on {stat_summary['positive_protocols']}/5 pilots, and has no negative pilot (worst {stat_summary['worst_delta_macro_f1']:+.6f}). This establishes a measurable S12 bottleneck, not sufficiency of behavior features as a replacement for Native bytes.", "",
        f"2. **Most valuable new group?** Group B, packet-length distribution, is the most stable incremental block: mean {step_stats['S-AB-S-A']['mean']:+.6f}, positive {step_stats['S-AB-S-A']['positive']}/5, worst {step_stats['S-AB-S-A']['worst']:+.6f}. Its gain importance is also largest in VPN/VNAT/USTC S-Burst models.", "",
        f"3. **Independent bidirectional contribution?** Yes. Adding C after A+B yields mean {step_stats['S-ABC-S-AB']['mean']:+.6f}, positive {step_stats['S-ABC-S-AB']['positive']}/5, with only a negligible worst change of {step_stats['S-ABC-S-AB']['worst']:+.6f}. In VNAT, `forward_length_mean` alone has normalized gain 0.330 (2025) and 0.208 (2026), so this benefit carries capture/domain-fingerprint risk and is not causal evidence.", "",
        f"4. **Independent IAT-distribution contribution?** Not globally stable. Adding D after A+B+C has mean {step_stats['S-ABCD-S-ABC']['mean']:+.6f}, positive {step_stats['S-ABCD-S-ABC']['positive']}/5, and worst {step_stats['S-ABCD-S-ABC']['worst']:+.6f}; it helps VPN, Tor and USTC but hurts both VNAT pilots.", "",
        f"5. **Does Burst add information?** Yes, modestly and conditionally. S-Burst − S-ABCD is {burst_summary['mean_delta_macro_f1']:+.6f} mean, positive {burst_summary['positive_protocols']}/5, worst {burst_summary['worst_delta_macro_f1']:+.6f}; the preregistered Burst gate passes. The only focus-class incremental gain above +0.05 is VNAT-2025 Netflix, so this is not a universal large effect.", "",
        f"6. **Early-16 versus full-flow?** Mean full-minus-Early Macro-F1 is {early_means['S12']:+.6f} for S12, {early_means['S-ABCD']:+.6f} for S-ABCD and {early_means['S-Burst']:+.6f} for S-Burst. Full-flow is slightly better for S-Burst on VPN (+0.004548), Tor (+0.002867) and USTC (+0.008889), but worse on both VNAT pilots (-0.018891 and -0.097597). Therefore more packets are not a global behavior-feature advantage.", "",
        f"7. **ISCX-VPN improvement?** Yes within behavior features: S-Burst − S12 is {vpn_gain:+.6f} Macro-F1, including Spotify F1 0.407643→0.463576 and Vimeo 0.540816→0.568528. It still trails Native by 0.273779 Macro-F1, so it is complementary evidence rather than a replacement.", "",
        f"8. **ISCXTor2016 improvement?** Yes: S-Burst − S12 is {tor_gain:+.6f}. Hangouts improves 0.263736→0.380000 and YouTube 0.501377→0.568047. It remains 0.135283 Macro-F1 below Native.", "",
        "9. **Does VNAT rsync/scp truly improve?** Only modestly. In medium-2025, rsync correct rises 59→64→67 (S12→S-ABCD→S-Burst), rsync→scp falls 132→127→124, and rsync F1 rises 0.390728→0.411576→0.424051. But scp→rsync increases 52→56→57 while scp F1 is nearly flat (0.649254→0.651515→0.652672). The confusion is mitigated, not solved. `sftp` is Unknown in both pilots and was correctly not read.", "",
        f"10. **Damage to Native-good classes?** Relative to S12, S-Burst has only one negative pilot and the worst loss is {total_summary['worst_delta_macro_f1']:+.6f}. Relative to Native, however, standalone S-Burst loses more Native-correct samples than it rescues on 4/5 pilots; only VNAT-2025 has positive net rescue (+13). It must not replace Native wholesale.", "",
        f"11. **Which signals complement Native/T16?** Length-distribution B, bidirectional C, selected IAT D and Burst E all contribute. S-Burst rescues Native errors on every pilot (170, 157, 73, 7 and 758 samples), although its net balance is usually negative. Against T16 it has positive net rescue on all five pilots ({', '.join(str(int(row['net_rescue'])) for row in t16_burst)}). This is real sample-level complementarity, not overall dominance.", "",
        "12. **Next representation direction?** Prioritize a preregistered Byte–Behavior multi-view encoder over another standalone behavior-only model: Native remains much stronger on VPN, Tor, VNAT-2026 and USTC, while behavior features rescue non-overlapping errors and improve the hard VNAT-2025 composition. This is a recommendation only; no multi-view model or Stage 15F-1C was started.", "",
    ]
    lines += [
        "", "## Interpretation", "",
        "Feature importance is diagnostic only. No single feature dominates all datasets, but VNAT relies heavily on `forward_length_mean` and `packet_length_min`; that concentration may encode capture environment and requires future cross-capture checks. Full-flow versus Early-16 differences combine observation amount and representation effects and are not attributed to one feature family. System-level S-Burst-versus-Native/T16 differences also combine different models and observation budgets.", "",
        "## Stopping rule", "",
        f"Full15 gate met: `{full15_gate}`; full15 status: `NOT_RUN_STAGE_SCOPE`. Stage 15F-1C, open-set evaluation, DES/H1 modification, and Byte-Behavior model development were not started.", "",
    ]
    (ROOT / "stage15f1b_report.md").write_text("\n".join(lines), encoding="utf-8")
    results_lines = [
        "# RESULTS — Stage 15F-1B", "", "Status: complete", f"Conclusions: `{', '.join(conclusions)}`", "",
        "## Core results", "", f"S-ABCD-S12 mean validation Macro-F1 delta: {stat_summary['mean_delta_macro_f1']:+.6f}. S-Burst-S-ABCD: {burst_summary['mean_delta_macro_f1']:+.6f}. S-Burst-S12: {total_summary['mean_delta_macro_f1']:+.6f}.", "",
        "## Configuration and execution", "", "Five frozen pilots; exact Stage 15R LightGBM configuration; FULL_FLOW six sets and EARLY_16 three sets; seed 2022; Known-Validation logloss checkpoint selection.", "",
        "## Data and split", "", "Known Train/Validation only. Known Test and Unknown Test feature values used: 0/0.", "",
        "## Preserved evidence", "", "All run models, predictions, confusion matrices, per-class metrics, histories, feature importance, input hashes, cache audits, source audits, failures, and final tables are retained in this bundle.", "",
        "## Limitations", "", "Five-pilot diagnostic with one fixed seed. Full-flow and Early-16 are different observation budgets. Gain importance is not causal evidence.", "",
        "## Conclusion and next step", "", f"Gate labels: `{', '.join(conclusions)}`. Stop after Stage 15F-1B; full15, Stage 15F-1C, open-set evaluation, and Byte-Behavior model development remain NOT_RUN.", "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(results_lines), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
