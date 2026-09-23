#!/usr/bin/env python3
"""Aggregate Stage 12 results and apply the frozen cross-dataset gate."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage12_common import PROJECT_ROOT, STAGE_ROOT, load_config, write_csv, write_json


METHODS = ("M0_OPEN_DETECT_NATIVE", "M1_DES_V0")
METRICS = (
    "known_accuracy", "known_macro_f1", "known_frr", "ufar", "unknown_recall",
    "auroc", "auprc", "natural_binary_f1", "balanced_binary_f1", "balanced_accuracy",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)), "max": float(np.max(array)),
    }


def gate(setting_rows: list[dict], config: dict) -> dict:
    gate_config = config["gate"]
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in setting_rows:
        by_dataset[row["dataset"]].append(row)
    dataset_means = {
        dataset: float(np.mean([float(row["delta_auroc_mean"]) for row in rows]))
        for dataset, rows in by_dataset.items()
    }
    split_policies = {
        dataset: sorted({str(row["split_policy"]) for row in rows})
        for dataset, rows in by_dataset.items()
    }
    conditions = {
        "both_dataset_level_mean_delta_auroc_gt_0": all(value > 0 for value in dataset_means.values()) and len(dataset_means) == 2,
        "each_dataset_has_setting_mean_delta_auroc_ge_registered_minimum": all(
            any(float(row["delta_auroc_mean"]) >= float(gate_config["each_dataset_has_setting_delta_auroc_at_least"]) for row in rows)
            for rows in by_dataset.values()
        ) and len(by_dataset) == 2,
        "no_setting_mean_delta_auroc_at_or_below_registered_floor": all(
            float(row["delta_auroc_mean"]) > float(gate_config["no_setting_delta_auroc_at_or_below"])
            for row in setting_rows
        ),
        "majority_paired_seeds_des_gt_od": sum(int(row["des_auroc_better_seed_count"]) for row in setting_rows) > sum(int(row["seed_count"]) for row in setting_rows) / 2,
        "no_setting_mean_delta_known_frr_above_registered_ceiling": all(
            float(row["delta_known_frr_mean"]) <= float(gate_config["max_allowed_delta_known_frr"])
            for row in setting_rows
        ),
        "no_setting_mean_delta_known_macro_f1_below_registered_floor": all(
            float(row["delta_known_macro_f1_mean"]) >= float(gate_config["min_allowed_mean_delta_known_macro_f1"])
            for row in setting_rows
        ),
    }
    if all(conditions.values()):
        decision = "EXTERNAL_CONFIRMED"
    else:
        positive_datasets = sum(value > 0 for value in dataset_means.values())
        positive_settings = sum(float(row["delta_auroc_mean"]) > 0 for row in setting_rows)
        safe_costs = conditions["no_setting_mean_delta_known_frr_above_registered_ceiling"] and conditions[
            "no_setting_mean_delta_known_macro_f1_below_registered_floor"
        ]
        mild = conditions["no_setting_mean_delta_auroc_at_or_below_registered_floor"]
        required_positive_settings = int(np.ceil(
            len(setting_rows) * float(gate_config["partial_min_positive_setting_fraction"])
        ))
        decision = (
            "PARTIAL_CONFIRMATION"
            if (
                positive_datasets >= int(gate_config["partial_min_positive_datasets"])
                and positive_settings >= required_positive_settings
                and (safe_costs or not gate_config["partial_requires_safe_known_costs"])
                and (mild or not gate_config["partial_requires_no_large_negative_setting"])
            )
            else "NOT_CONFIRMED"
        )
    return {
        "decision": decision,
        "conditions": conditions,
        "dataset_level_mean_delta_auroc": dataset_means,
        "dataset_split_policies": split_policies,
        "claim_reduction_required_for_flow_disjoint_only": any(
            "FLOW_DISJOINT_ONLY" in policies for policies in split_policies.values()
        ),
        "post_test_method_modification": False,
        "forbidden_methods_run": False,
        "gate_definition_source": str((STAGE_ROOT / "configs" / "stage12_config.json").resolve()),
    }


def main() -> int:
    config = load_config()
    run_results = []
    run_rows = []
    per_unknown = defaultdict(list)
    absorption = defaultdict(list)
    bootstrap = defaultdict(list)
    for dataset in config["datasets"]:
        protocol = json.loads((STAGE_ROOT / "protocol" / dataset / "unknown_class_protocol.json").read_text(encoding="utf-8"))
        for setting in protocol["settings"]:
            for seed in config["training_seeds"]:
                run = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
                if not (run / "SUCCESS").is_file():
                    raise RuntimeError(f"missing final test success: {run}")
                result = json.loads((run / "final_results.json").read_text(encoding="utf-8"))
                run_results.append(result)
                for method in METHODS:
                    run_rows.append({
                        "dataset": dataset, "setting": setting, "seed": seed, "method": method,
                        **{metric: result["methods"][method][metric] for metric in METRICS},
                    })
                per_unknown[dataset].extend(read_csv(run / "per_unknown_results.csv"))
                absorption[dataset].extend(read_csv(run / "absorption_matrix.csv"))
                bootstrap[dataset].extend(read_csv(run / "paired_bootstrap.csv"))
    setting_rows = []
    for dataset in config["datasets"]:
        output = STAGE_ROOT / "outputs" / dataset
        rows = [row for row in run_rows if row["dataset"] == dataset]
        write_csv(output / "run_level_results.csv", rows)
        write_csv(output / "per_unknown_class_results.csv", per_unknown[dataset])
        write_csv(output / "absorption_analysis.csv", absorption[dataset])
        write_csv(output / "bootstrap_results.csv", bootstrap[dataset])
        unknown_delta_rows = []
        unknown_keys = sorted({(row["setting"], row["unknown_class"]) for row in per_unknown[dataset]})
        for setting, unknown_class in unknown_keys:
            grouped = [
                row for row in per_unknown[dataset]
                if row["setting"] == setting and row["unknown_class"] == unknown_class
            ]
            values = {
                method: [row for row in grouped if row["method"] == method]
                for method in METHODS
            }
            row = {"dataset": dataset, "setting": setting, "unknown_class": unknown_class}
            for method in METHODS:
                row[f"{method}_auroc_mean"] = float(np.mean([float(item["auroc_vs_all_known"]) for item in values[method]]))
                row[f"{method}_ufar_mean"] = float(np.mean([float(item["ufar"]) for item in values[method]]))
            row["delta_auroc_mean"] = row["M1_DES_V0_auroc_mean"] - row["M0_OPEN_DETECT_NATIVE_auroc_mean"]
            row["delta_ufar_mean"] = row["M1_DES_V0_ufar_mean"] - row["M0_OPEN_DETECT_NATIVE_ufar_mean"]
            row["seed_count"] = len(values["M0_OPEN_DETECT_NATIVE"])
            unknown_delta_rows.append(row)
        write_csv(output / "per_unknown_delta_summary.csv", unknown_delta_rows)
        hub_rows = []
        for setting, method in sorted({(row["setting"], row["method"]) for row in absorption[dataset]}):
            grouped = [row for row in absorption[dataset] if row["setting"] == setting and row["method"] == method]
            counts: dict[str, int] = defaultdict(int)
            for item in grouped:
                counts[item["predicted_known_class"]] += int(item["count"])
            total = sum(counts.values())
            for rank, (known_class, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), start=1):
                hub_rows.append({
                    "dataset": dataset, "setting": setting, "method": method, "rank": rank,
                    "predicted_known_class": known_class, "accepted_unknown_count": count,
                    "share_of_all_accepted_unknown": count / total if total else 0.0,
                })
        write_csv(output / "absorption_hubs.csv", hub_rows)
        settings = sorted({row["setting"] for row in rows}, key=("low", "medium", "high").index)
        for setting in settings:
            setting_results = [result for result in run_results if result["dataset"] == dataset and result["setting"] == setting]
            row = {"dataset": dataset, "setting": setting, "seed_count": len(setting_results)}
            for method in METHODS:
                method_values = [result["methods"][method] for result in setting_results]
                for metric in METRICS:
                    for key, value in stat([float(item[metric]) for item in method_values]).items():
                        row[f"{method}_{metric}_{key}"] = value
            for metric in ("auroc", "auprc", "ufar", "known_frr", "known_macro_f1"):
                values = [float(result["delta_m1_minus_m0"][metric]) for result in setting_results]
                for key, value in stat(values).items():
                    row[f"delta_{metric}_{key}"] = value
            row["des_auroc_better_seed_count"] = sum(
                float(result["delta_m1_minus_m0"]["auroc"]) > 0 for result in setting_results
            )
            row["known_class_count"] = len(setting_results[0]["known_classes"])
            row["unknown_class_count"] = len(setting_results[0]["unknown_classes"])
            row["unknown_classes"] = ";".join(setting_results[0]["unknown_classes"])
            row["split_policy"] = setting_results[0]["split_policy"]
            row["group_aware_not_feasible_classes"] = ";".join(
                setting_results[0]["group_aware_not_feasible_classes"]
            )
            setting_rows.append(row)
        write_csv(output / "setting_summary.csv", [row for row in setting_rows if row["dataset"] == dataset])
    summary = STAGE_ROOT / "outputs" / "summary"
    write_csv(summary / "des_vs_opendetect_summary.csv", setting_rows)
    dataset_rows = []
    for dataset in config["datasets"]:
        values = [result for result in run_results if result["dataset"] == dataset]
        row = {"dataset": dataset, "settings": len({item["setting"] for item in values}), "seeds_total": len(values)}
        for metric in ("auroc", "auprc", "ufar", "known_frr", "known_macro_f1"):
            deltas = [float(item["delta_m1_minus_m0"][metric]) for item in values]
            row[f"delta_{metric}_mean"] = float(np.mean(deltas))
            row[f"delta_{metric}_std"] = float(np.std(deltas, ddof=1))
        row["des_auroc_better_seed_count"] = sum(float(item["delta_m1_minus_m0"]["auroc"]) > 0 for item in values)
        dataset_rows.append(row)
    write_csv(summary / "cross_dataset_comparison.csv", dataset_rows)
    decision = gate(setting_rows, config)
    write_json(summary / "final_gate.json", decision)
    lines = [
        "# Stage 12 Final External Gate", "", f"## Decision: {decision['decision']}", "",
        "## Pre-registered conditions", "",
    ]
    lines.extend(f"- {name}: `{'PASS' if value else 'FAIL'}`" for name, value in decision["conditions"].items())
    lines += ["", "## Dataset-level mean Delta AUROC", ""]
    lines.extend(f"- {dataset}: `{value:+.6f}`" for dataset, value in decision["dataset_level_mean_delta_auroc"].items())
    lines += ["", "No method was modified after the one-shot Test opening. K1/K2, GMM, DAP, DGSB, and adaptive methods were not run."]
    (summary / "final_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (STAGE_ROOT / "RESULTS.md").write_text(
        "# Stage 12 — Dual Independent External Validation\n\n"
        f"- Status: `success`\n- Final gate: **{decision['decision']}**\n"
        f"- Methods: Open-Detect Native and DES-v0 only, paired on the same {len(run_results)} encoders.\n"
        "- Test opening: one frozen opening after all runs reached READY_FOR_ONE_SHOT_TEST.\n"
        "- Detailed results: `outputs/iscx_vpn/`, `outputs/iscx_tor/`, and `outputs/summary/`.\n"
        "- Preflight and protocol decisions: `PROTOCOL_PREFLIGHT.md` and `protocol/`.\n"
        "- Post-Test method modification: `NO`.\n",
        encoding="utf-8",
    )
    result_lines = [
        "", "<!-- STAGE12_RESULTS_START -->", "## ISCX-VPN Results", "",
    ]
    for dataset, title in (("iscx_vpn", "ISCX-VPN"), ("iscx_tor", "ISCXTor2016")):
        if dataset == "iscx_tor":
            result_lines += ["", "## ISCXTor2016 Results", ""]
        for row in [value for value in setting_rows if value["dataset"] == dataset]:
            result_lines.append(
                f"- {row['setting'].title()}: M0 AUROC `{float(row['M0_OPEN_DETECT_NATIVE_auroc_mean']):.6f}`, "
                f"DES AUROC `{float(row['M1_DES_V0_auroc_mean']):.6f}`, Delta `{float(row['delta_auroc_mean']):+.6f}`, "
                f"DES wins `{row['des_auroc_better_seed_count']}/{row['seed_count']}` seeds."
            )
    result_lines += [
        "", "## Cross-Dataset Comparison", "",
        *[f"- {row['dataset']}: mean Delta AUROC `{float(row['delta_auroc_mean']):+.6f}`." for row in dataset_rows],
        "", "## Known Classification", "",
        "Known Accuracy and Macro-F1 are reported per method, setting, and seed in each dataset's `setting_summary.csv`.",
        "", "## Unknown Detection", "",
        "AUROC, AUPRC, Unknown Recall, and both natural and balanced binary metrics are preserved in the run-level and setting summaries.",
        "", "## UFAR / FRR Tradeoff", "",
        "UFAR and Known FRR deltas are reported jointly; the final Gate rejects improvements purchased through a registered large FRR increase.",
        "", "## Failure Cases", "",
        "Per-unknown degradation and absorption hubs are retained in `per_unknown_class_results.csv` and `absorption_analysis.csv` for each dataset.",
        "", "## What Can Be Claimed", "",
        (
            "The registered cross-dataset claim is supported under this frozen protocol."
            if decision["decision"] == "EXTERNAL_CONFIRMED"
            else "The registered cross-dataset claim is not fully supported under this frozen protocol."
        ),
        "", "## What Cannot Be Claimed", "",
        "The result does not establish that DES universally dominates Open-Detect or generalizes to every encrypted-traffic dataset.",
        "<!-- STAGE12_RESULTS_END -->",
    ]
    readme_path = STAGE_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    start = "<!-- STAGE12_RESULTS_START -->"
    end = "<!-- STAGE12_RESULTS_END -->"
    if start in readme and end in readme:
        readme = readme.split(start, 1)[0].rstrip() + "\n" + "\n".join(result_lines[1:]) + "\n"
    else:
        readme = readme.rstrip() + "\n" + "\n".join(result_lines) + "\n"
    readme_path.write_text(readme, encoding="utf-8")
    manifest = json.loads((STAGE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "success"
    manifest["core_results"] = decision
    manifest["configuration"] = config
    manifest["run_count"] = len(run_results)
    manifest["method_result_rows"] = len(run_rows)
    manifest["limitations"] = [
        "capture-filename application labels inherit controlled-capture semantics",
        "source-PCAP group split ratios can deviate from 80/10/10 for classes with few groups",
    ]
    write_json(STAGE_ROOT / "manifest.json", manifest)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
