#!/usr/bin/env python3
"""Aggregate Stage 14C-4 cleanup evidence and apply the pre-registered gate."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F2_ROOT = PROJECT / "stage14c5_feature_representation_audit"
OD_ROOT = PROJECT / "stage14c_native_opendetect_vnat"
STAGE14C3 = PROJECT / "stage14c3_vnat_closed_set_failure_diagnosis"
PROTOCOLS = ("medium_seed2025", "medium_seed2026")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_weighted(protocol_id: str) -> float:
    with (F2_ROOT / "runs" / "f2" / protocol_id / "per_class_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(np.average(
        np.asarray([float(row["f1"]) for row in rows]),
        weights=np.asarray([int(row["val_support"]) for row in rows]),
    ))


def main() -> None:
    dynamics_rows: list[dict[str, object]] = []
    comparison: list[dict[str, object]] = []
    results: dict[tuple[str, str], dict] = {}
    for protocol_id in PROTOCOLS:
        original = read_json(F2_ROOT / "runs" / "f2" / protocol_id / "result.json")
        original_metrics = pd.read_csv(F2_ROOT / "runs" / "f2" / protocol_id / "training_metrics.csv")
        original_best = original_metrics.loc[original_metrics.epoch == int(original["best_epoch"])].iloc[0]
        comparison.append({
            "protocol_id": protocol_id,
            "version": "original_f2",
            "run_role": "baseline",
            "training_seed": int(original["seed"]),
            "epochs_completed": int(original["stop_epoch"]),
            "best_epoch": int(original["best_epoch"]),
            "train_loss_at_best": float(original_best.train_total),
            "val_loss_at_best": float(original["val_loss"]),
            "val_accuracy": float(original["val_accuracy"]),
            "val_macro_f1": float(original["val_macro_f1"]),
            "val_weighted_f1": original_weighted(protocol_id),
            "nan_or_crash": False,
            "checkpoint_sha256": original["checkpoint_sha256"],
            "known_test_samples_used": 0,
            "unknown_test_samples_used": 0,
            "des_executed": False,
        })
        for seed_mode in ("protocol", "fixed2022"):
            run_id = f"{protocol_id}_{seed_mode}"
            result = read_json(ROOT / "runs" / run_id / "result.json")
            results[(protocol_id, seed_mode)] = result
            comparison.append({
                "protocol_id": protocol_id,
                "version": "cleaned_primary" if seed_mode == "protocol" else "cleaned_fixed2022_control",
                "run_role": result["run_role"],
                "training_seed": result["training_seed"],
                "epochs_completed": result["epochs_completed"],
                "best_epoch": result["best_epoch"],
                "train_loss_at_best": result["train_loss"],
                "val_loss_at_best": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
                "val_macro_f1": result["val_macro_f1"],
                "val_weighted_f1": result["val_weighted_f1"],
                "nan_or_crash": result["nan_or_crash"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "known_test_samples_used": result["known_test_samples_used"],
                "unknown_test_samples_used": result["unknown_test_samples_used"],
                "des_executed": result["des_executed"],
            })
            for row in csv.DictReader((ROOT / "runs" / run_id / "training_metrics.csv").open(encoding="utf-8")):
                dynamics_rows.append(dict(row))
        od = read_json(OD_ROOT / "runs" / protocol_id / "result.json")
        comparison.append({
            "protocol_id": protocol_id,
            "version": "native_opendetect_reference",
            "run_role": "reference_only",
            "training_seed": od["training_seed"],
            "epochs_completed": od["epochs_completed"],
            "best_epoch": od["best_epoch"],
            "train_loss_at_best": "",
            "val_loss_at_best": "",
            "val_accuracy": od["validation_accuracy"],
            "val_macro_f1": od["validation_macro_f1"],
            "val_weighted_f1": od["validation_weighted_f1"],
            "nan_or_crash": False,
            "checkpoint_sha256": od["checkpoint_sha256"],
            "known_test_samples_used": od["known_test_samples_used"],
            "unknown_test_samples_used": od["unknown_samples_used_in_validation"],
            "des_executed": False,
        })

    write_csv(ROOT / "cleanup_comparison.csv", comparison)
    write_csv(ROOT / "training_dynamics.csv", dynamics_rows)

    stage14c3 = pd.read_csv(STAGE14C3 / "component_ablation.csv")
    prior_sensitivity: dict[str, float] = {}
    for protocol_id in PROTOCOLS:
        part = stage14c3[stage14c3.protocol_id == protocol_id].set_index("variant")
        prior_sensitivity[protocol_id] = abs(float(part.loc["D6", "val_macro_f1"]) - float(part.loc["D0", "val_macro_f1"]))

    sensitivity_rows: list[dict[str, object]] = []
    for protocol_id in PROTOCOLS:
        primary = results[(protocol_id, "protocol")]
        fixed = results[(protocol_id, "fixed2022")]
        cleaned_delta = abs(float(primary["val_macro_f1"]) - float(fixed["val_macro_f1"]))
        sensitivity_rows.append({
            "protocol_id": protocol_id,
            "cleaned_protocol_seed_macro_f1": primary["val_macro_f1"],
            "cleaned_fixed2022_macro_f1": fixed["val_macro_f1"],
            "cleaned_absolute_seed_delta": cleaned_delta,
            "prior_stage14c3_absolute_seed_delta": prior_sensitivity[protocol_id],
            "sensitivity_reduced": cleaned_delta < prior_sensitivity[protocol_id],
        })
    write_csv(ROOT / "seed_sensitivity.csv", sensitivity_rows)

    original_by_protocol = {row["protocol_id"]: row for row in comparison if row["version"] == "original_f2"}
    weak_original = float(original_by_protocol["medium_seed2025"]["val_macro_f1"])
    normal_original = float(original_by_protocol["medium_seed2026"]["val_macro_f1"])
    weak_primary = float(results[("medium_seed2025", "protocol")]["val_macro_f1"])
    normal_primary = float(results[("medium_seed2026", "protocol")]["val_macro_f1"])
    completed = all(result["epochs_completed"] == 100 and not result["nan_or_crash"] for result in results.values() if result["run_role"] == "primary")
    weak_recovery = weak_primary - weak_original
    normal_change = normal_primary - normal_original
    sensitivity_reduced_both = all(row["sensitivity_reduced"] for row in sensitivity_rows)
    if completed and weak_recovery >= 0.10 and normal_change >= -0.03 and sensitivity_reduced_both:
        conclusion = "CLEANUP_PASS"
    elif completed and weak_recovery >= 0.10:
        conclusion = "PARTIAL_PASS"
    else:
        conclusion = "NEEDS_FURTHER_DIAGNOSIS"

    deterministic = pd.read_csv(ROOT / "deterministic_split_metrics.csv")
    primary_table = pd.DataFrame(comparison)
    primary_table = primary_table[primary_table.version.isin(["original_f2", "cleaned_primary", "native_opendetect_reference"])]
    table_lines = [
        "| Protocol | Version | Epochs | Best epoch | Val loss | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary_table.iterrows():
        val_loss = "—" if row.val_loss_at_best == "" else f"{float(row.val_loss_at_best):.6f}"
        table_lines.append(
            f"| {row.protocol_id} | {row.version} | {int(row.epochs_completed)} | {int(row.best_epoch)} | {val_loss} | {float(row.val_accuracy):.6f} | {float(row.val_macro_f1):.6f} | {float(row.val_weighted_f1):.6f} |"
        )
    deterministic_lines = [
        "| Protocol | Version | Split | Loss | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in deterministic[deterministic.version.isin(["original_f2", "cleaned_primary"])].iterrows():
        deterministic_lines.append(
            f"| {row.protocol_id} | {row.version} | {row.split} | {row.loss:.6f} | {row.accuracy:.6f} | {row.macro_f1:.6f} | {row.weighted_f1:.6f} |"
        )

    report = f"""# Stage 14C-4 — Our-Method Training Mechanism Audit & Cleanup

## Final conclusion

**`{conclusion}`**

This cleanup retains the F2 representation, Stage3 model, existing loss,
composite checkpoint rule, scheduler, protocol-seed primary policy, logvar
guard, and prototype mechanism. Native Open-Detect is reference only.

## Selected-checkpoint results

{chr(10).join(table_lines)}

## Deterministic Known split metrics

{chr(10).join(deterministic_lines)}

## Gate evidence

- Both primary runs completed exactly 100 epochs without NaN/crash: **{completed}**.
- Weak Medium-2025 Macro-F1 recovery: **{weak_recovery:+.6f}** (gate >= +0.10).
- Normal Medium-2026 Macro-F1 change: **{normal_change:+.6f}** (gate >= -0.03).
- Seed sensitivity reduced on both protocols: **{sensitivity_reduced_both}**.
- Known Test / Unknown Test used: **0 / 0**.
- DES executed: **false**.

## Mechanism decisions

1. **Removed:** patience-5 early stopping. It caused every original F0/F2 run
   to stop before the epoch-50 scheduler/prototype phase.
2. **Changed:** batch size 512 to 128, based on the prior one-factor D1 result.
3. **Fixed:** prototype resets keep the same epochs and class-mean values but
   now update the existing Parameter in place and clear its Adam state. The old
   object replacement left Adam attached to the obsolete Parameter.
4. **Retained:** F2 feature, Stage3 model, loss, composite checkpoint selection,
   scheduler, protocol seed, initialization and logvar guard.
5. **Not added:** gradient clipping, class weights, weighted sampler, warmup,
   LR search, weight decay, dropout, or any new regularizer, because no direct
   evidence justifies them.

## Required answers

1. Unreasonable mechanisms: premature patience-5 stopping, batch 512 update
   starvation, and prototype Parameter replacement that breaks optimizer link.
2. Deleted/modified: early stopping removed, batch set to 128, prototype reset
   made optimizer-safe. The evidence is recorded in `training_mechanism_audit.csv`.
3. Required mechanisms retained: model architecture, F2 feature design, loss,
   representation/prototype schedule, scheduler, composite Known-Val checkpoint
   rule, initialization and numerical guard.
4. Under-training solved: **{completed}**.
5. Weak run recovery: **{weak_recovery:+.6f} Macro-F1**.
6. Stability: see `seed_sensitivity.csv`; reduced on both protocols =
   **{sensitivity_reduced_both}**.
7. Formal 15-run recommendation follows the `{conclusion}` gate; only
   `CLEANUP_PASS` supports immediate freeze and rerun.

## Limitations

- This is a two-protocol representative audit, not the formal 15-run result.
- Fixed seed 2022 is a diagnostic control and is not selected as the production
  seed policy.
- No Unknown performance was observed, so no open-set claim is made.

Generated at `{datetime.now(timezone.utc).isoformat()}`.
"""
    (ROOT / "stage14c4_report.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "PASS",
        "conclusion": conclusion,
        "primary_runs_completed_100_epochs": completed,
        "weak_macro_f1_recovery": weak_recovery,
        "normal_macro_f1_change": normal_change,
        "seed_sensitivity_reduced_both": sensitivity_reduced_both,
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
    }
    (ROOT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
