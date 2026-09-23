#!/usr/bin/env python3
"""Freeze one preregistered DGSB-v2 rule from frozen Known Validation only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import joblib
import numpy as np

from stage8b_common import (
    AccessLedger,
    CONFIG_PATH,
    STAGE8A_ROOT,
    STAGE8B_ROOT,
    VALID_SETTINGS,
    empirical_threshold,
    gate_status,
    load_json,
    read_csv,
    require,
    score_k2_matrices,
    sha256_file,
    true_class_component_assignments,
    verify_all_provenance,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=VALID_SETTINGS)
    return parser.parse_args()


def calibrate(setting: str) -> dict[str, object]:
    config = load_json(CONFIG_PATH)
    require(config["dgsbv2"]["global_quantile"] == 0.025, "global quantile contract changed")
    require(config["dgsbv2"]["local_quantile"] == 0.025, "local quantile contract changed")
    require(config["dgsbv2"]["fallback_min_validation_n"] == 30, "fallback contract changed")
    provenance = STAGE8B_ROOT / "outputs/summary/provenance_verification.json"
    require(provenance.is_file() and load_json(provenance)["status"] == "PASS", "provenance gate missing")
    verify_all_provenance()

    artifact8a = STAGE8A_ROOT / "artifacts" / setting
    output8a = STAGE8A_ROOT / "outputs" / setting
    artifact = STAGE8B_ROOT / "artifacts" / setting
    output = STAGE8B_ROOT / "outputs" / setting
    artifact.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        artifact / "dgsbv2_global_threshold.json",
        artifact / "dgsbv2_local_thresholds.csv",
        artifact / "dgsbv2_fallback_map.csv",
        artifact / "dgsbv2_rule.json",
        artifact / "dgsbv2_rule.md",
        output / "gate_contribution.csv",
        output / "per_class_coverage.csv",
        output / "per_component_coverage.csv",
        output / "known_only_method_comparison.csv",
        output / "calibration_summary.json",
        output / "file_access_audit.json",
    ]
    require(not any(path.exists() for path in required_outputs), f"{setting}: refusing to overwrite Stage 8B outputs")

    ledger = AccessLedger(setting)
    z_path = artifact8a / "z_val_pca64.npy"
    manifest_path = artifact8a / "val_manifest.csv"
    model_path = artifact8a / "multi_k2/models.joblib"
    baseline_path = output8a / "known_val_boundary_summary.csv"
    ledger.record(z_path, "npy", "KNOWN_VALIDATION")
    z_val = np.load(z_path, mmap_mode="r", allow_pickle=False)
    ledger.record(manifest_path, "csv", "KNOWN_VALIDATION")
    manifest = read_csv(manifest_path)
    ledger.record(model_path, "joblib", "FROZEN_KNOWN_TRAIN_MODEL")
    models = joblib.load(model_path)
    ledger.record(baseline_path, "csv", "KNOWN_VALIDATION_DIAGNOSTIC")
    baseline_rows = read_csv(baseline_path)

    require(len(z_val) == len(manifest), f"{setting}: Validation row mismatch")
    require(z_val.ndim == 2 and z_val.shape[1] == 64, f"{setting}: PCA64 shape mismatch")
    require(np.isfinite(z_val).all(), f"{setting}: non-finite Validation representation")
    require(all(row["source_role"] == "KNOWN_VALIDATION" for row in manifest), f"{setting}: role mismatch")
    labels = np.asarray([row["class_name"] for row in manifest], dtype=object)

    names, class_scores, local_scores = score_k2_matrices(models, np.asarray(z_val))
    require(names == list(models), f"{setting}: model order changed")
    predicted_class_indices = np.argmax(class_scores, axis=1)
    predicted_classes = np.asarray([names[index] for index in predicted_class_indices], dtype=object)
    global_scores = class_scores[np.arange(len(z_val)), predicted_class_indices]
    predicted_local_blocks = local_scores.reshape(len(z_val), len(names), 2)[
        np.arange(len(z_val)), predicted_class_indices
    ]
    predicted_components = np.argmax(predicted_local_blocks, axis=1)
    predicted_local_scores = predicted_local_blocks[np.arange(len(z_val)), predicted_components]

    true_components, true_local_scores, _ = true_class_component_assignments(labels, names, local_scores)
    name_to_index = {name: index for index, name in enumerate(names)}
    true_class_local_max = np.empty(len(labels), dtype=np.float64)
    for class_name in names:
        positions = np.flatnonzero(labels == class_name)
        class_index = name_to_index[class_name]
        block = local_scores[positions, 2 * class_index : 2 * class_index + 2]
        true_class_local_max[positions] = np.max(block, axis=1)

    global_threshold, global_acceptance = empirical_threshold(global_scores, 0.975)
    local_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    threshold_lookup: dict[tuple[str, int], float] = {}
    for class_index, class_name in enumerate(names):
        class_positions = np.flatnonzero(labels == class_name)
        class_threshold, class_acceptance = empirical_threshold(true_class_local_max[class_positions], 0.975)
        for component_id in (0, 1):
            positions = class_positions[true_components[class_positions] == component_id]
            validation_n = len(positions)
            raw_threshold = None
            raw_acceptance = None
            if validation_n:
                raw_threshold, raw_acceptance = empirical_threshold(true_local_scores[positions], 0.975)
            fallback = validation_n < 30
            effective = class_threshold if fallback else float(raw_threshold)
            threshold_lookup[(class_name, component_id)] = effective
            row = {
                "setting": setting,
                "class_name": class_name,
                "class_index": class_index,
                "component_id": component_id,
                "validation_n": validation_n,
                "raw_component_p025": raw_threshold,
                "raw_component_acceptance": raw_acceptance,
                "local_class_p025": class_threshold,
                "local_class_acceptance": class_acceptance,
                "effective_threshold": effective,
                "quantile": 0.025,
                "target_acceptance": 0.975,
                "fallback": fallback,
                "fallback_reason": "component validation n < 30" if fallback else "",
                "fallback_min_validation_n": 30,
                "score_semantics": "weighted local log score",
                "calibration_split": "KNOWN_VALIDATION_ONLY",
            }
            local_rows.append(row)
            fallback_rows.append(
                {
                    "setting": setting,
                    "class_name": class_name,
                    "class_index": class_index,
                    "component_id": component_id,
                    "validation_n": validation_n,
                    "fallback": fallback,
                    "fallback_threshold": class_threshold if fallback else "",
                    "effective_threshold": effective,
                    "reason": "component validation n < 30 -> local-class P02.5" if fallback else "",
                }
            )

    local_thresholds = np.asarray(
        [threshold_lookup[(str(name), int(component))] for name, component in zip(predicted_classes, predicted_components)],
        dtype=np.float64,
    )
    global_pass = global_scores >= global_threshold
    local_pass = predicted_local_scores >= local_thresholds
    both_pass = global_pass & local_pass
    group_masks = {
        "A_GLOBAL_PASS_LOCAL_PASS": global_pass & local_pass,
        "B_GLOBAL_FAIL_LOCAL_PASS": ~global_pass & local_pass,
        "C_GLOBAL_PASS_LOCAL_FAIL": global_pass & ~local_pass,
        "D_GLOBAL_FAIL_LOCAL_FAIL": ~global_pass & ~local_pass,
    }
    contribution_rows = [
        {
            "setting": setting,
            "group": group,
            "global_pass": group.startswith("A_") or group.startswith("C_"),
            "local_pass": group.startswith("A_") or group.startswith("B_"),
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
            "validation_n": len(labels),
        }
        for group, mask in group_masks.items()
    ]
    global_exclusive = int(group_masks["B_GLOBAL_FAIL_LOCAL_PASS"].sum())
    local_exclusive = int(group_masks["C_GLOBAL_PASS_LOCAL_FAIL"].sum())
    both_fail = int(group_masks["D_GLOBAL_FAIL_LOCAL_FAIL"].sum())
    status = gate_status(global_exclusive, local_exclusive)

    per_class_rows: list[dict[str, object]] = []
    for class_name in names:
        positions = np.flatnonzero(labels == class_name)
        acceptance = float(both_pass[positions].mean())
        per_class_rows.append(
            {
                "setting": setting,
                "class_name": class_name,
                "support": len(positions),
                "acceptance": acceptance,
                "FRR": 1.0 - acceptance,
            }
        )

    per_component_rows: list[dict[str, object]] = []
    gaps: list[float] = []
    by_class_component: dict[str, list[float]] = defaultdict(list)
    for class_name in names:
        class_positions = np.flatnonzero(labels == class_name)
        for component_id in (0, 1):
            positions = class_positions[true_components[class_positions] == component_id]
            require(len(positions) > 0, f"{setting}/{class_name}/{component_id}: empty Validation component")
            acceptance = float(both_pass[positions].mean())
            by_class_component[class_name].append(acceptance)
            per_component_rows.append(
                {
                    "setting": setting,
                    "class_name": class_name,
                    "component_id": component_id,
                    "n": len(positions),
                    "global_acceptance": float(global_pass[positions].mean()),
                    "local_acceptance": float(local_pass[positions].mean()),
                    "dgsbv2_acceptance": acceptance,
                    "dgsbv2_FRR": 1.0 - acceptance,
                    "assignment_scope": "true-class posterior component assignment",
                }
            )
    for values in by_class_component.values():
        gaps.append(abs(values[0] - values[1]))

    per_class_values = np.asarray([float(row["acceptance"]) for row in per_class_rows])
    fallback_count = sum(bool(row["fallback"]) for row in local_rows)
    summary = {
        "setting": setting,
        "validation_n": len(labels),
        "known_classes": len(names),
        "global_threshold": global_threshold,
        "global_quantile": 0.025,
        "global_target_acceptance": 0.975,
        "global_acceptance": float(global_pass.mean()),
        "global_FRR": float(1.0 - global_pass.mean()),
        "local_acceptance": float(local_pass.mean()),
        "local_FRR": float(1.0 - local_pass.mean()),
        "dgsbv2_acceptance": float(both_pass.mean()),
        "dgsbv2_FRR": float(1.0 - both_pass.mean()),
        "global_exclusive_rejection_count": global_exclusive,
        "global_exclusive_rejection_rate": global_exclusive / len(labels),
        "local_exclusive_rejection_count": local_exclusive,
        "local_exclusive_rejection_rate": local_exclusive / len(labels),
        "both_fail_count": both_fail,
        "both_fail_rate": both_fail / len(labels),
        "gate_status": status,
        "per_class_coverage": {
            "min": float(np.min(per_class_values)),
            "p05": float(np.quantile(per_class_values, 0.05, method="linear")),
            "p25": float(np.quantile(per_class_values, 0.25, method="linear")),
            "median": float(np.median(per_class_values)),
            "p75": float(np.quantile(per_class_values, 0.75, method="linear")),
            "max": float(np.max(per_class_values)),
            "per_class_FRR_std": float(np.std(1.0 - per_class_values, ddof=0)),
        },
        "component_gap": {
            "mean": float(np.mean(gaps)),
            "median": float(np.median(gaps)),
            "max": float(np.max(gaps)),
        },
        "fallback_component_count": fallback_count,
        "component_count": len(local_rows),
        "fallback_rate": fallback_count / len(local_rows),
        "known_rejection_too_high": bool(both_pass.mean() < 0.94),
        "known_collapse": bool(both_pass.sum() == 0 or np.any(per_class_values == 0.0)),
        "quantile_search_performed": False,
        "post_local_global_recalibration_performed": False,
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
    }

    comparison_rows: list[dict[str, object]] = []
    for row in baseline_rows:
        comparison_rows.append(
            {
                "setting": setting,
                "method": row["method"],
                "overall_acceptance": float(row["overall_acceptance"]),
                "FRR": float(row["overall_FRR"]),
                "per_class_coverage_min": float(row["per_class_acceptance_min"]),
                "per_class_coverage_max": float(row["per_class_acceptance_max"]),
                "mean_component_gap": float(row["mean_per_class_component_acceptance_gap"]),
                "max_component_gap": float(row["max_per_class_component_acceptance_gap"]),
                "data_scope": "KNOWN_VALIDATION_ONLY",
            }
        )
    comparison_rows.append(
        {
            "setting": setting,
            "method": "DGSB-v2",
            "overall_acceptance": summary["dgsbv2_acceptance"],
            "FRR": summary["dgsbv2_FRR"],
            "per_class_coverage_min": summary["per_class_coverage"]["min"],
            "per_class_coverage_max": summary["per_class_coverage"]["max"],
            "mean_component_gap": summary["component_gap"]["mean"],
            "max_component_gap": summary["component_gap"]["max"],
            "data_scope": "KNOWN_VALIDATION_ONLY",
        }
    )

    global_payload = {
        "setting": setting,
        "score_definition": "G(z)=S_y*(z), y*=argmax_y log p_K2(z|y)",
        "threshold": global_threshold,
        "quantile": 0.025,
        "target_acceptance": 0.975,
        "empirical_acceptance": global_acceptance,
        "tie_rule": "higher more-conservative threshold",
        "calibration_split": "KNOWN_VALIDATION_ONLY",
        "quantile_search_performed": False,
        "post_local_global_recalibration_performed": False,
        "test_samples_used": 0,
        "unknown_samples_used": 0,
    }
    rule = {
        "setting": setting,
        "name": "DGSB-v2",
        "definition": "Absolute Global Gate AND Predicted-Class Local Support Gate",
        "prediction": "y*=argmax_y S_y(z)",
        "global_score": "G(z)=S_y*(z)=logsumexp_k s_y*k(z)",
        "local_component": "k*=argmax_k s_y*k(z) within predicted class y*",
        "local_score": "L(z)=s_y*k*(z)",
        "known_decision": "G(z)>=tau_global AND L(z)>=tau_local_y*k*",
        "global_threshold": global_threshold,
        "global_quantile": 0.025,
        "local_quantile": 0.025,
        "fallback_min_validation_n": 30,
        "fallback_rule": "component val_n<30 -> predicted class local-max P02.5 threshold",
        "same_class_support_check": True,
        "post_local_global_recalibration": "FORBIDDEN",
        "quantile_search": "FORBIDDEN",
        "unknown_calibration": "FORBIDDEN",
        "stage8a_dependencies": {
            "z_val_pca64_sha256": sha256_file(z_path),
            "k2_model_sha256": sha256_file(model_path),
            "stage8a_evaluation_config_sha256": sha256_file(STAGE8A_ROOT / "configs/evaluation_config.json"),
        },
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "unknown_inference_executed": False,
    }
    rule_md = f"""# DGSB-v2 rule: {setting.title()}

- Prediction: `y*=argmax_y log p_K2(z|y)`.
- Global score: `G(z)=log p_K2(z|y*)`; frozen P02.5 threshold `{global_threshold:.12g}`.
- Local component: `k*=argmax_k s_y*k(z)` inside the same predicted class `y*`.
- Local score: `L(z)=s_y*k*(z)`; each local threshold is frozen at P02.5.
- Sparse fallback: component Validation `n<30` uses the same class's local-max-score P02.5 threshold.
- Decision: Known iff both Global and Local gates pass.
- Global recalibration after Local freeze: **FORBIDDEN**.
- Quantile search and Unknown calibration: **FORBIDDEN**.
- Calibration data: Known Validation only; Known Test/Unknown Test opened: 0/0.
"""

    write_json(artifact / "dgsbv2_global_threshold.json", global_payload)
    write_csv(artifact / "dgsbv2_local_thresholds.csv", local_rows)
    write_csv(artifact / "dgsbv2_fallback_map.csv", fallback_rows)
    write_json(artifact / "dgsbv2_rule.json", rule)
    (artifact / "dgsbv2_rule.md").write_text(rule_md, encoding="utf-8")
    write_csv(output / "gate_contribution.csv", contribution_rows)
    write_csv(output / "per_class_coverage.csv", per_class_rows)
    write_csv(output / "per_component_coverage.csv", per_component_rows)
    write_csv(output / "known_only_method_comparison.csv", comparison_rows)
    write_json(output / "calibration_summary.json", summary)
    ledger.save(output / "file_access_audit.json")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    calibrate(args.setting)


if __name__ == "__main__":
    main()
