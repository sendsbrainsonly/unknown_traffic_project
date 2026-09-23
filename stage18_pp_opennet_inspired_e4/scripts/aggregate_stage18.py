#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage18_common import OUT, PROJECT, SEEDS, SERVICES, VARIANT_ORDER, protocol_id, write_json


STAGE17 = PROJECT / "stage17_encoder_recovery_and_open_set_pilot"


def load_run_tables(filename: str) -> pd.DataFrame:
    frames = []
    for service in SERVICES:
        for seed in SEEDS:
            path = OUT / "runs" / protocol_id(service) / f"seed{seed}" / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def mean_std_table(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(groups, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        row["runs"] = len(part)
        for value in values:
            row[f"{value}_mean"] = float(part[value].mean())
            row[f"{value}_std"] = float(part[value].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(STAGE17 / "encoder_pilot_results.csv")
    e0 = source[(source.encoder == "E0") & (source.detector == "OD-Native")].copy()
    e3 = source[(source.encoder == "E3") & (source.detector == "DES-v1")].copy()
    return e0, e3


def pair_against_baselines(closed: pd.DataFrame, opened: pd.DataFrame) -> pd.DataFrame:
    e4c = closed[closed.variant == "E4_FULL"].copy()
    e4o = opened[(opened.variant == "E4_FULL") & (opened.score == "feature_distance") & np.isclose(opened.known_acceptance_target, 0.95)].copy()
    e0, e3 = baseline_rows()
    rows = []
    for _, crow in e4c.iterrows():
        key = (crow.protocol_id, int(crow.seed))
        orow = e4o[(e4o.protocol_id == key[0]) & (e4o.seed == key[1])].iloc[0]
        for baseline_name, base in (("E0_OD_NATIVE", e0), ("E3_DES_V1", e3)):
            brow = base[(base.protocol_id == key[0]) & (base.seed == key[1])].iloc[0]
            rows.append({
                "protocol_id": key[0],
                "unknown_service": crow.unknown_service,
                "seed": key[1],
                "baseline": baseline_name,
                "e4_accuracy": crow.known_test_accuracy,
                "baseline_accuracy": brow.known_test_accuracy,
                "delta_accuracy": crow.known_test_accuracy - brow.known_test_accuracy,
                "e4_macro_f1": crow.known_test_macro_f1,
                "baseline_macro_f1": brow.known_test_macro_f1,
                "delta_macro_f1": crow.known_test_macro_f1 - brow.known_test_macro_f1,
                "e4_weighted_f1": crow.known_test_weighted_f1,
                "baseline_weighted_f1": brow.known_test_weighted_f1,
                "delta_weighted_f1": crow.known_test_weighted_f1 - brow.known_test_weighted_f1,
                "e4_auroc": orow.auroc,
                "baseline_auroc": brow.auroc,
                "delta_auroc": orow.auroc - brow.auroc,
                "e4_auprc": orow.auprc,
                "baseline_auprc": brow.auprc,
                "delta_auprc": orow.auprc - brow.auprc,
                "e4_ufar95": orow.ufar,
                "baseline_ufar95": brow.ufar,
                "delta_ufar95": orow.ufar - brow.ufar,
                "e4_known_acceptance95": orow.known_test_acceptance,
                "baseline_known_acceptance95": 1.0 - brow.known_frr,
                "delta_known_acceptance95": orow.known_test_acceptance - (1.0 - brow.known_frr),
            })
    return pd.DataFrame(rows)


def component_ablation(closed: pd.DataFrame, opened: pd.DataFrame) -> pd.DataFrame:
    primary = opened[(opened.score == "feature_distance") & np.isclose(opened.known_acceptance_target, 0.95)]
    comparisons = {
        "multi_scale_conditional_on_recurrent": ("E4_FULL", "E4_NO_MS"),
        "recurrent_conditional_on_multiscale": ("E4_FULL", "E4_NO_RNN"),
        "joint_modules_vs_base": ("E4_FULL", "E4_BASE"),
        # Use the direction-free Time+Length branch as the left-hand side for
        # strict single-feature attribution.  Comparing E4_FULL here would
        # confound Time/Length with the direction channel.
        "time_information_vs_length_only": ("E4_TIME_LENGTH", "E4_LENGTH"),
        "length_information_vs_time_only": ("E4_TIME_LENGTH", "E4_TIME"),
        "direction_information_vs_time_length": ("E4_FULL", "E4_TIME_LENGTH"),
    }
    rows = []
    for service in SERVICES:
        for seed in SEEDS:
            for name, (left_variant, reference_variant) in comparisons.items():
                left_c = closed[(closed.unknown_service == service) & (closed.seed == seed) & (closed.variant == left_variant)].iloc[0]
                left_o = primary[(primary.unknown_service == service) & (primary.seed == seed) & (primary.variant == left_variant)].iloc[0]
                other_c = closed[(closed.unknown_service == service) & (closed.seed == seed) & (closed.variant == reference_variant)].iloc[0]
                other_o = primary[(primary.unknown_service == service) & (primary.seed == seed) & (primary.variant == reference_variant)].iloc[0]
                rows.append({
                    "protocol_id": protocol_id(service), "unknown_service": service, "seed": seed,
                    "comparison": name, "full_variant": left_variant, "reference_variant": reference_variant,
                    "delta_accuracy": left_c.known_test_accuracy - other_c.known_test_accuracy,
                    "delta_macro_f1": left_c.known_test_macro_f1 - other_c.known_test_macro_f1,
                    "delta_weighted_f1": left_c.known_test_weighted_f1 - other_c.known_test_weighted_f1,
                    "delta_auroc": left_o.auroc - other_o.auroc,
                    "delta_auprc": left_o.auprc - other_o.auprc,
                    "delta_ufar95": left_o.ufar - other_o.ufar,
                })
    return pd.DataFrame(rows)


def confusion_table(predictions: pd.DataFrame) -> pd.DataFrame:
    known = predictions[(predictions.variant == "E4_FULL") & (predictions.role == "known_test")].copy()
    known.true_local_label = known.true_local_label.astype(int)
    rows = []
    for (service, seed), part in known.groupby(["unknown_service", "seed"]):
        for true_label in range(5):
            for pred_label in range(5):
                rows.append({
                    "unknown_service": service, "seed": seed, "true_local_label": true_label,
                    "predicted_local_label": pred_label,
                    "count": int(((part.true_local_label == true_label) & (part.predicted_local_label == pred_label)).sum()),
                })
    return pd.DataFrame(rows)


def complementarity(predictions: pd.DataFrame) -> pd.DataFrame:
    source_path = STAGE17 / "encoder_pilot_predictions.csv"
    if not source_path.is_file():
        return pd.DataFrame(columns=["unknown_service", "seed", "comparison", "both_correct", "e4_only_correct", "baseline_only_correct", "both_wrong"])
    old = pd.read_csv(source_path)
    old = old[(old.encoder == "E3") & (old.role.isin(["known_test", "unknown_test"]))]
    new = predictions[(predictions.variant == "E4_FULL") & (predictions.role.isin(["known_test", "unknown_test"]))]
    rows = []
    for service in SERVICES:
        for seed in SEEDS:
            a = new[(new.unknown_service == service) & (new.seed == seed)].copy()
            b = old[(old.unknown_service == service) & (old.seed == seed)].copy()
            joined = a.merge(b, on=["protocol_id", "unknown_service", "seed", "role", "flow_id"], suffixes=("_e4", "_e3"), validate="one_to_one")
            known = joined.role == "known_test"
            e4_closed = joined.predicted_local_label_e4.astype(int) == joined.true_local_label_e4.astype(int)
            e3_closed = joined.predicted_local_label_e3.astype(int) == joined.true_local_label_e3.astype(int)
            # Stage 17 is the only input table with a generic ``rejected``
            # column, so pandas correctly leaves it unsuffixed during the
            # merge.  Accept a suffixed name as well to keep this audit robust
            # if a future frozen E4 table adds the same generic column.
            e3_rejected = joined["rejected_e3"] if "rejected_e3" in joined else joined["rejected"]
            for comparison, mask, e4_correct, e3_correct in (
                ("known_classification", known, e4_closed, e3_closed),
                (
                    "binary_open_detection",
                    np.ones(len(joined), dtype=bool),
                    np.where(known, joined.feature_distance_rejected_95 == 0, joined.feature_distance_rejected_95 == 1),
                    np.where(known, e3_rejected == 0, e3_rejected == 1),
                ),
            ):
                ec = np.asarray(e4_correct)[mask]
                bc = np.asarray(e3_correct)[mask]
                rows.append({
                    "unknown_service": service, "seed": seed, "comparison": comparison,
                    "samples": int(mask.sum()),
                    "both_correct": int(np.sum(ec & bc)),
                    "e4_only_correct": int(np.sum(ec & ~bc)),
                    "baseline_only_correct": int(np.sum(~ec & bc)),
                    "both_wrong": int(np.sum(~ec & ~bc)),
                })
    return pd.DataFrame(rows)


def gate_decision(closed: pd.DataFrame, opened: pd.DataFrame, paired: pd.DataFrame) -> dict:
    e0 = paired[paired.baseline == "E0_OD_NATIVE"]
    service_checks = []
    for service in SERVICES:
        part = e0[e0.unknown_service == service]
        service_checks.append({
            "service": service,
            "mean_delta_macro_f1": float(part.delta_macro_f1.mean()),
            "closed_pass": bool(part.delta_macro_f1.mean() >= -0.02),
            "mean_delta_auroc": float(part.delta_auroc.mean()),
            "auroc_pass": bool(part.delta_auroc.mean() >= -0.005),
            "mean_ufar95": float(part.e4_ufar95.mean()),
            "absolute_ufar_pass": bool(part.e4_ufar95.mean() <= 0.85),
        })
    conditions = {
        "both_services_closed": all(item["closed_pass"] for item in service_checks),
        "both_services_auroc": all(item["auroc_pass"] for item in service_checks),
        "overall_positive_auroc": bool(e0.delta_auroc.mean() > 0),
        "positive_auroc_runs_at_least_4": int((e0.delta_auroc > 0).sum()) >= 4,
        "both_services_absolute_ufar": all(item["absolute_ufar_pass"] for item in service_checks),
        "no_closed_catastrophe": bool(closed[closed.variant == "E4_FULL"].known_test_macro_f1.min() >= 0.50),
    }
    passed = all(conditions.values())
    return {
        "final_gate": "E4_SINGLE_BRANCH_PASS" if passed else "E4_FAIL_NO_FUSION",
        "fusion_started": False,
        "conditions": conditions,
        "service_checks": service_checks,
        "positive_auroc_runs_vs_e0": int((e0.delta_auroc > 0).sum()),
        "overall_mean_delta_auroc_vs_e0": float(e0.delta_auroc.mean()),
        "minimum_full_macro_f1": float(closed[closed.variant == "E4_FULL"].known_test_macro_f1.min()),
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    closed = load_run_tables("closed_set_results.csv")
    opened = load_run_tables("open_set_results.csv")
    per_class = load_run_tables("per_class_results.csv")
    predictions = load_run_tables("predictions.csv")
    dynamics = []
    for service in SERVICES:
        for seed in SEEDS:
            run = OUT / "runs" / protocol_id(service) / f"seed{seed}"
            for variant in VARIANT_ORDER:
                frame = pd.read_csv(run / f"{variant}_history.csv")
                frame.insert(0, "variant", variant); frame.insert(0, "seed", seed); frame.insert(0, "unknown_service", service); frame.insert(0, "protocol_id", protocol_id(service))
                dynamics.append(frame)
    dynamics = pd.concat(dynamics, ignore_index=True)
    closed.to_csv(OUT / "e4_closed_set_results.csv", index=False)
    opened.to_csv(OUT / "e4_open_set_results.csv", index=False)
    per_class.to_csv(OUT / "e4_per_class_results.csv", index=False)
    predictions.to_csv(OUT / "e4_predictions.csv", index=False)
    dynamics.to_csv(OUT / "e4_training_dynamics.csv", index=False)
    closed_summary = mean_std_table(closed, ["unknown_service", "variant"], ["known_test_accuracy", "known_test_macro_f1", "known_test_weighted_f1", "best_val_macro_f1"])
    open_summary = mean_std_table(opened, ["unknown_service", "variant", "score", "known_acceptance_target"], ["auroc", "auprc", "ufar", "known_test_acceptance"])
    class_summary = mean_std_table(per_class, ["unknown_service", "variant", "service"], ["precision", "recall", "f1"])
    closed_summary.to_csv(OUT / "closed_set_summary.csv", index=False)
    open_summary.to_csv(OUT / "open_set_summary.csv", index=False)
    class_summary.to_csv(OUT / "per_class_summary.csv", index=False)
    paired = pair_against_baselines(closed, opened)
    paired.to_csv(OUT / "paired_comparison.csv", index=False)
    ablation = component_ablation(closed, opened)
    ablation.to_csv(OUT / "component_ablation.csv", index=False)
    mean_std_table(ablation, ["comparison"], ["delta_accuracy", "delta_macro_f1", "delta_weighted_f1", "delta_auroc", "delta_auprc", "delta_ufar95"]).to_csv(OUT / "component_ablation_summary.csv", index=False)
    confusion_table(predictions).to_csv(OUT / "confusion_matrices.csv", index=False)
    comp = complementarity(predictions)
    comp.to_csv(OUT / "e4_e3_error_complementarity.csv", index=False)
    gate = gate_decision(closed, opened, paired)
    write_json(OUT / "gate_decision.json", gate)

    full_closed = closed_summary[closed_summary.variant == "E4_FULL"]
    full_open = open_summary[(open_summary.variant == "E4_FULL") & (open_summary.score == "feature_distance") & np.isclose(open_summary.known_acceptance_target, 0.95)]
    e0_pairs = paired[paired.baseline == "E0_OD_NATIVE"]
    e3_pairs = paired[paired.baseline == "E3_DES_V1"]
    lines = [
        "# Stage 18 PP-OpenNet-inspired E4 report", "", "## Scope", "",
        "This is an independent 30-packet PP-OpenNet-inspired reconstruction on frozen Stage17 protocols, not an author-code or paper-dataset reproduction. Official code was unavailable. No Open-Detect component was used to train E4.", "",
        "## E4_FULL mean results", "", "| Unknown | Accuracy | Macro-F1 | Weighted-F1 | AUROC | AUPRC | UFAR@Val95 | Known acceptance |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for service in SERVICES:
        c = full_closed[full_closed.unknown_service == service].iloc[0]
        o = full_open[full_open.unknown_service == service].iloc[0]
        lines.append(f"| {service} | {fmt(c.known_test_accuracy_mean)} | {fmt(c.known_test_macro_f1_mean)} | {fmt(c.known_test_weighted_f1_mean)} | {fmt(o.auroc_mean)} | {fmt(o.auprc_mean)} | {fmt(o.ufar_mean)} | {fmt(o.known_test_acceptance_mean)} |")
    lines += ["", "## Paired comparison", ""]
    for name, part in (("E0 Open-Detect Native", e0_pairs), ("E3 Fusion DES-v1", e3_pairs)):
        lines.append(f"- E4 minus {name}: mean delta Macro-F1 `{fmt(part.delta_macro_f1.mean())}`, AUROC `{fmt(part.delta_auroc.mean())}`, AUPRC `{fmt(part.delta_auprc.mean())}`, UFAR@95 `{fmt(part.delta_ufar95.mean())}`; AUROC wins `{int((part.delta_auroc > 0).sum())}/6`.")
    ab_summary = pd.read_csv(OUT / "component_ablation_summary.csv")
    lines += ["", "## Module and input contributions", ""]
    for _, row in ab_summary.iterrows():
        lines.append(f"- `{row.comparison}`: mean delta Macro-F1 `{fmt(row.delta_macro_f1_mean)}`, AUROC `{fmt(row.delta_auroc_mean)}`, UFAR@95 `{fmt(row.delta_ufar95_mean)}`.")
    lines += [
        "", "## Five requested questions", "",
        "1. E4 single-branch quality is reported jointly through closed-set metrics, ranking metrics and absolute UFAR; the gate below does not use Macro-F1 alone.",
        "2. Extra value is assessed by paired E4-vs-E0/E3 deltas and sample-level E4/E3 rescue counts, not inferred from architecture names.",
        "3. Multi-scale and recurrent contributions are the preregistered conditional ablations E4_FULL−E4_NO_MS and E4_FULL−E4_NO_RNN.",
        "4. Representation versus rejection is separated by comparing the same E4 embedding under feature-distance, MSP and Energy, and by reporting 90/95/99 Known-Validation thresholds.",
        "5. E3 fusion is not run in this stage. It is eligible only if every preregistered single-branch gate condition passes.",
        "", "## Gate", "", f"Final gate: **{gate['final_gate']}**.", "",
        f"Conditions: `{json.dumps(gate['conditions'], sort_keys=True)}`.", "",
        "Fusion started: **NO**.", "", "## Limitations", "",
        "- Only Email and Streaming development protocols were executed.",
        "- The packet cache is capped at 30 packets rather than the paper's 1,000-packet/two-second retroactive slices.",
        "- Labels remain weak capture-activity labels and capture-disjoint generalization is not identifiable.",
        "- The paper's background class, ARPL and stochastic PVRP are intentionally excluded to preserve Strict Unknown-Free comparability.",
    ]
    (OUT / "stage18_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "aggregation_summary.json", {
        "status": "PASS", "closed_rows": len(closed), "open_rows": len(opened), "per_class_rows": len(per_class),
        "prediction_rows": len(predictions), "training_dynamic_rows": len(dynamics), "gate": gate,
    })
    print(json.dumps({"status": "PASS", "closed_rows": len(closed), "open_rows": len(opened), "prediction_rows": len(predictions), "gate": gate["final_gate"]}), flush=True)


if __name__ == "__main__":
    main()
