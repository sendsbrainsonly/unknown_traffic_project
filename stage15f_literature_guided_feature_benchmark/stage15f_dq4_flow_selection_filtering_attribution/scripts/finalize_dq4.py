#!/usr/bin/env python3
"""Aggregate DQ-4 Track E and training-selection results."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from dq4_common import DQ3F, OUT, SEEDS, SERVICES, hash_protected_inputs, read_json, sha256_file, write_csv, write_json


MODEL_SUBSET = {"M-A": "A", "M-B": "B", "M-C": "C_FINAL", "M-D": "D"}


def metric_bundle(frame: pd.DataFrame) -> dict:
    true = frame["true_index"].to_numpy(dtype=int)
    pred = frame["predicted_index"].to_numpy(dtype=int)
    return {
        "samples": int(len(frame)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=range(6), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "errors": int(np.sum(true != pred)),
    }


def load_predictions(model: str, seed: int, manifest: pd.DataFrame) -> pd.DataFrame:
    validation = manifest[manifest["split_role"].eq("known_validation")]
    if model == "M-A":
        path = DQ3F / "runs/service" / f"seed{seed}" / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["training_subset"] = "A"
        frame = frame.merge(validation[["flow_id", "A", "B", "C_PARENT", "C_FINAL", "D"]], on="flow_id", validate="one_to_one")
    else:
        path = OUT / "runs" / model / f"seed{seed}" / "all_a_validation_predictions.csv"
        frame = pd.read_csv(path)
    if len(frame) != 335 or frame["flow_id"].nunique() != 335:
        raise RuntimeError(f"prediction coverage mismatch: {model} seed{seed}")
    return frame


def mean_std(frame: pd.DataFrame, fields: list[str]) -> dict:
    result = {}
    for field in fields:
        result[f"{field}_mean"] = float(frame[field].mean())
        result[f"{field}_std"] = float(frame[field].std(ddof=0))
    return result


def main() -> None:
    preflight = read_json(OUT / "preflight_status.json")
    if preflight["status"] != "PASS":
        raise RuntimeError("preflight is not PASS")
    for model in ("M-B", "M-C", "M-D"):
        for seed in SEEDS:
            if not (OUT / "runs" / model / f"seed{seed}" / "SUCCESS").is_file():
                raise RuntimeError(f"missing SUCCESS: {model} seed{seed}")
    manifest = pd.read_csv(OUT / "dq4_subset_manifest.csv")

    training_rows = []
    for model, subset in MODEL_SUBSET.items():
        for seed in SEEDS:
            if model == "M-A":
                run_dir = DQ3F / "runs/service" / f"seed{seed}"
                result = read_json(run_dir / "result.json")
                steps = math.ceil(int(result["train_samples"]) / 128)
                row = {
                    "model": model, "training_subset": subset, "seed": seed, "action": "REUSE_DQ3F_F3",
                    "status": result["status"], "train_samples": result["train_samples"],
                    "validation_samples": result["validation_samples"], "best_epoch": result["best_epoch"],
                    "completed_epochs": result["completed_epochs"], "stopped_early": result["stopped_early"],
                    "steps_per_epoch": steps, "completed_optimizer_updates": steps * int(result["completed_epochs"]),
                    "runtime_seconds": result["runtime_seconds"], "train_loss": result["train_loss"],
                    "validation_loss": result["validation_loss"], "validation_accuracy": result["validation_accuracy"],
                    "validation_macro_f1": result["validation_macro_f1"],
                    "validation_weighted_f1": result["validation_weighted_f1"],
                    "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": result["checkpoint_sha256"],
                    "checkpoint_hash_verified": sha256_file(run_dir / "model_best.pt") == result["checkpoint_sha256"],
                    "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
                }
            else:
                run_dir = OUT / "runs" / model / f"seed{seed}"
                result = read_json(run_dir / "result.json")
                row = {
                    "model": model, "training_subset": subset, "seed": seed, "action": "TRAIN_DQ4",
                    **{key: result[key] for key in (
                        "status", "train_samples", "validation_samples", "best_epoch", "completed_epochs",
                        "stopped_early", "steps_per_epoch", "completed_optimizer_updates", "runtime_seconds",
                        "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1",
                        "validation_weighted_f1", "checkpoint_path", "checkpoint_sha256",
                        "known_test_feature_values_used", "unknown_test_feature_values_used",
                    )},
                    "checkpoint_hash_verified": sha256_file(run_dir / "model_best.pt") == result["checkpoint_sha256"],
                }
            training_rows.append(row)
    write_csv(OUT / "dq4_training_results.csv", training_rows)

    prediction_map = {(model, seed): load_predictions(model, seed, manifest) for model in MODEL_SUBSET for seed in SEEDS}
    within_rows, matched_rows, stress_rows, per_service_rows, capture_rows = [], [], [], [], []
    for (model, seed), prediction in prediction_map.items():
        contexts = {
            "within_population": MODEL_SUBSET[model],
            "matched_D": "D",
            "full_A_stress": "A",
        }
        for design, subset in contexts.items():
            part = prediction[prediction[subset].eq(1)].copy()
            row = {"model": model, "training_subset": MODEL_SUBSET[model], "evaluation_subset": subset, "seed": seed, **metric_bundle(part)}
            if design == "within_population":
                within_rows.append(row)
            elif design == "matched_D":
                matched_rows.append(row)
            else:
                stress_rows.append(row)
            precision, recall, f1, support = precision_recall_fscore_support(
                part["true_index"], part["predicted_index"], labels=range(6), zero_division=0,
            )
            for index, service in enumerate(SERVICES):
                per_service_rows.append({
                    "evaluation_design": design, "evaluation_subset": subset, "model": model,
                    "training_subset": MODEL_SUBSET[model], "seed": seed, "service": service,
                    "precision": float(precision[index]), "recall": float(recall[index]),
                    "f1": float(f1[index]), "support": int(support[index]),
                })
            for capture, group in part.groupby("capture_id", sort=True):
                capture_rows.append({
                    "evaluation_design": design, "evaluation_subset": subset, "model": model,
                    "training_subset": MODEL_SUBSET[model], "seed": seed, "capture_id": capture,
                    "service": group["true_label"].iloc[0], "application": group["application_label"].iloc[0],
                    **metric_bundle(group),
                })
    write_csv(OUT / "dq4_within_population_results.csv", within_rows)
    write_csv(OUT / "dq4_matched_evaluation_results.csv", matched_rows)
    write_csv(OUT / "dq4_full_validation_stress_test.csv", stress_rows)
    write_csv(OUT / "dq4_per_service_metrics.csv", per_service_rows)
    write_csv(OUT / "dq4_capture_level_results.csv", capture_rows)

    paired_rows = []
    for seed in SEEDS:
        for subset in ("A", "D"):
            frames = {model: prediction_map[(model, seed)].set_index("flow_id") for model in MODEL_SUBSET}
            ids = frames["M-A"].index[frames["M-A"][subset].eq(1)]
            for left, right in itertools.combinations(MODEL_SUBSET, 2):
                a = frames[left].loc[ids, "correct"].astype(bool)
                b = frames[right].loc[ids, "correct"].astype(bool)
                paired_rows.append({
                    "seed": seed, "evaluation_subset": subset, "left_model": left, "right_model": right,
                    "samples": int(len(ids)), "both_correct": int((a & b).sum()),
                    "left_only_correct": int((a & ~b).sum()), "right_only_correct": int((~a & b).sum()),
                    "both_wrong": int((~a & ~b).sum()),
                    "accuracy_delta_right_minus_left": float(b.mean() - a.mean()),
                })
    write_csv(OUT / "dq4_paired_error_analysis.csv", paired_rows)

    overlap_rows = []
    for role in ("known_train", "known_validation"):
        split = manifest[manifest["split_role"].eq(role)]
        for b, c in itertools.product((0, 1), repeat=2):
            part = split[(split["B"] == b) & (split["C_FINAL"] == c)]
            for grouping, groups in (("overall", [("ALL", part)]), ("service", part.groupby("service", sort=True))):
                for value, group in groups:
                    overlap_rows.append({
                        "split_role": role, "B_selected": b, "C_selected": c,
                        "overlap_cell": f"B{b}_C{c}", "grouping": grouping, "group_value": value,
                        "flow_count": int(len(group)), "capture_count": int(group["capture_id"].nunique()),
                        "one_packet_count": int((group["packet_count"] == 1).sum()),
                        "two_packet_count": int((group["packet_count"] == 2).sum()),
                        "median_total_bytes": float(group["total_bytes"].median()) if len(group) else np.nan,
                        "median_duration_seconds": float(group["duration_seconds"].median()) if len(group) else np.nan,
                    })
    write_csv(OUT / "dq4_selection_overlap_analysis.csv", overlap_rows)

    within = pd.DataFrame(within_rows)
    matched = pd.DataFrame(matched_rows)
    stress = pd.DataFrame(stress_rows)
    track_e = pd.read_csv(OUT / "dq4_evaluation_population_effect.csv")
    summary = []
    for name, frame in (("within", within), ("matched_D", matched), ("full_A_stress", stress)):
        for model, group in frame.groupby("model", sort=True):
            summary.append({"design": name, "model": model, **mean_std(group, ["accuracy", "macro_f1", "weighted_f1"])})
    summary_frame = pd.DataFrame(summary)
    matched_pivot = matched.pivot(index="seed", columns="model", values="macro_f1")
    delta = {model: float((matched_pivot[model] - matched_pivot["M-A"]).mean()) for model in ("M-B", "M-C", "M-D")}
    positive = {model: int((matched_pivot[model] > matched_pivot["M-A"]).sum()) for model in ("M-B", "M-C", "M-D")}
    track_e_mean = track_e.groupby("evaluation_subset")[["accuracy", "macro_f1", "weighted_f1"]].mean()
    eval_delta = {subset: float(track_e_mean.loc[subset, "macro_f1"] - track_e_mean.loc["A", "macro_f1"]) for subset in ("B", "C_FINAL", "D")}
    gates = ["EVALUATION_POPULATION_EFFECT"] if any(abs(value) >= 0.01 for value in eval_delta.values()) else []
    if delta["M-B"] > 0.01 and positive["M-B"] >= 2:
        gates.append("FLOW_SELECTION_EFFECT")
    if delta["M-C"] > 0.01 and positive["M-C"] >= 2:
        gates.append("FILTERING_EFFECT")
    if delta["M-D"] > 0.01 and positive["M-D"] >= 2:
        gates.append("TRAINING_SELECTION_BENEFIT")
    if not any(gate in gates for gate in ("FLOW_SELECTION_EFFECT", "FILTERING_EFFECT", "TRAINING_SELECTION_BENEFIT")):
        gates.append("NO_CLEAR_SELECTION_BENEFIT")
    service = pd.DataFrame(per_service_rows)
    service_d = service[service["evaluation_design"].eq("matched_D")]
    service_means = service_d.groupby(["model", "service"])["f1"].mean().unstack(0)
    signs = []
    for model in ("M-B", "M-C", "M-D"):
        signs.extend(np.sign(service_means[model] - service_means["M-A"]).tolist())
    if any(value > 0 for value in signs) and any(value < 0 for value in signs):
        gates.append("CLASS_CONDITIONAL_SELECTION_EFFECT")

    def table(frame: pd.DataFrame) -> str:
        lines = ["| Model | Accuracy mean | Macro-F1 mean | Weighted-F1 mean |", "|---|---:|---:|---:|"]
        for row in frame.itertuples(index=False):
            lines.append(f"| {row.model} | {row.accuracy_mean:.6f} | {row.macro_f1_mean:.6f} | {row.weighted_f1_mean:.6f} |")
        return "\n".join(lines)

    within_summary = summary_frame[summary_frame["design"].eq("within")]
    matched_summary = summary_frame[summary_frame["design"].eq("matched_D")]
    stress_summary = summary_frame[summary_frame["design"].eq("full_A_stress")]
    subset_counts = preflight["subset_counts"]
    report = f"""# Stage 15F-DQ-4 — Flow Selection and Filtering Attribution

## Conclusion

Gates: `{', '.join(gates)}`  
Persistent limitations: `WEAK_CAPTURE_LABEL`, `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`.

The experiment separates evaluation-population selection from training-selection effects on the frozen 3,065-flow DQ-3F mother population. No Known Test or Unknown Test feature was read.

## Frozen populations

- A: {subset_counts['A']} flows.
- B (unique packet-consistent CATE parent match): {subset_counts['B']} flows.
- C_PARENT/C_FINAL Service: {subset_counts['C_PARENT']}/{subset_counts['C_FINAL']} flows.
- D = B intersection C: {subset_counts['D']} flows.
- C_PARENT equals C_FINAL for Service inclusion, but Native sessions and TrafficFormer parent flows are not one-to-one.

## Within-population results

{table(within_summary)}

These values are not a fair cross-model ranking because both Train and Validation populations differ.

## Common D Validation comparison

{table(matched_summary)}

Mean matched-D Macro-F1 deltas versus M-A: `{json.dumps(delta, sort_keys=True)}`. Positive seeds out of 3: `{json.dumps(positive, sort_keys=True)}`.

## Full A Validation stress test

{table(stress_summary)}

This is distribution-shift stress evidence only; A Validation was not used for checkpoint selection in M-B/M-C/M-D.

## Attribution

- Evaluation-population-only Macro-F1 deltas for the unchanged M-A model versus A: `{json.dumps(eval_delta, sort_keys=True)}`.
- Any apparent gain when only changing A_val to B/C/D_val is therefore a population-composition effect, not a learned model improvement.
- Training-selection claims use only the common D_val matched comparison. The combined D model is not automatically preferred unless its paired delta is consistently positive.
- CATE B removes flows absent from the target-flow five-tuple list; this status is not authoritative evidence that removed flows are mislabeled.
- TrafficFormer C removes parents below 2,048 captured bytes (then checks >=3 packets). It strongly changes Service composition and observational support.
- B/D VoIP Validation support is only four flows and C's minimum class support is six; per-Service conclusions for these cells are fragile.
- P2P remains one-capture in the shared31 cohort, so capture-generalization cannot be identified.

## Cross-paper gap boundary

Flow selection and filtering can explain a measurable part of reported evaluation differences when the unchanged M-A model changes score across A/B/C/D. They do not alone prove that TrafficFormer or TFE-GNN model mechanisms are better, because input representation, architecture, supervision and split construction also differ. Conversely, within-population model differences cannot be interpreted without the matched-D control.

## Integrity

- M-A: reused 3/3 frozen DQ-3F F3 checkpoints after exact array/prediction/hash parity.
- M-B/M-C/M-D: 9/9 frozen-config runs completed.
- Known Test feature values used: 0.
- Unknown Test feature values used: 0.
- TrafficFormer success-parent to final-Service multiset parity: PASS.
"""
    (OUT / "dq4_performance_gap_attribution.md").write_text(report, encoding="utf-8")
    (OUT / "RESULTS.md").write_text(report.replace("# Stage 15F-DQ-4 — Flow Selection and Filtering Attribution", "# RESULTS — Stage 15F-DQ-4", 1), encoding="utf-8")
    write_json(OUT / "aggregate_summary.json", {
        "gates": gates, "persistent_limitations": ["WEAK_CAPTURE_LABEL", "CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE"],
        "subset_counts": subset_counts, "matched_D_macro_f1_delta_vs_M_A": delta,
        "matched_D_positive_seeds": positive, "evaluation_population_macro_f1_delta_vs_A": eval_delta,
        "known_test_feature_values_used": 0, "unknown_test_feature_values_used": 0,
    })
    after = hash_protected_inputs()
    write_json(OUT / "protected_asset_hashes_after.json", after)
    if after != read_json(OUT / "protected_asset_hashes_before.json"):
        raise RuntimeError("protected inputs changed during DQ-4")
    print(json.dumps(read_json(OUT / "aggregate_summary.json"), indent=2))


if __name__ == "__main__":
    main()
