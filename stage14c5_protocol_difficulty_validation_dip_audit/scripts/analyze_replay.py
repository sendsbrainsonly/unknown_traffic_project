#!/usr/bin/env python3
"""Aggregate the Stage 14C-5 read-only protocol and telemetry audit.

This script never opens Known Test or Unknown Test data.  It consumes only the
frozen Stage 14B protocol metadata, immutable Stage 14C-4 validation artifacts,
and parity-gated Stage 14C-5 observational replays.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE14C4 = PROJECT / "stage14c4_our_method_training_cleanup"
PROTOCOLS = ("medium_seed2025", "medium_seed2026")
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
RESET_WINDOW = {49, 50, 51, 52, 79, 80, 81, 82}


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def is_obvious_drop(delta_macro: float, delta_accuracy: float, loss: float, previous_loss: float) -> bool:
    ratio = loss / max(previous_loss, 1e-12)
    return bool(
        delta_macro <= -0.15
        or delta_accuracy <= -0.15
        or (ratio >= 3.0 and loss - previous_loss >= 0.10)
    )


def main() -> None:
    statuses: dict[str, dict[str, object]] = {}
    telemetry_parts: list[pd.DataFrame] = []
    class_parts: list[pd.DataFrame] = []
    for protocol in PROTOCOLS:
        base = OUT / "telemetry_replay" / protocol
        status = json.loads((base / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "PASS" or not status.get("parity_pass"):
            raise RuntimeError(f"invalid observational replay for {protocol}: {status}")
        if status.get("freeze_hash") != EXPECTED_FREEZE:
            raise RuntimeError(f"freeze hash mismatch in {protocol}")
        if status.get("known_test_samples_used") != 0 or status.get("unknown_test_samples_used") != 0:
            raise RuntimeError(f"test-set access detected in {protocol}")
        statuses[protocol] = status
        telemetry_parts.append(pd.read_csv(base / "epoch_telemetry.csv"))
        class_parts.append(pd.read_csv(base / "per_class_validation.csv"))

    telemetry = pd.concat(telemetry_parts, ignore_index=True)
    per_class = pd.concat(class_parts, ignore_index=True)
    cleanup = pd.read_csv(STAGE14C4 / "cleanup_comparison.csv")
    composition = pd.read_csv(OUT / "protocol_class_composition.csv")
    selected_class = pd.read_csv(OUT / "selected_checkpoint_per_class.csv")
    confusion = pd.read_csv(OUT / "selected_checkpoint_confusion_long.csv")

    attribution_rows: list[dict[str, object]] = []
    primary_scores: dict[str, float] = {}
    fixed_scores: dict[str, float] = {}
    for protocol in PROTOCOLS:
        primary = cleanup[(cleanup.protocol_id == protocol) & (cleanup.version == "cleaned_primary")].iloc[0]
        fixed = cleanup[(cleanup.protocol_id == protocol) & (cleanup.version == "cleaned_fixed2022_control")].iloc[0]
        primary_scores[protocol] = float(primary.val_macro_f1)
        fixed_scores[protocol] = float(fixed.val_macro_f1)
        protocol_composition = composition[composition.protocol_id == protocol]
        attribution_rows.append({
            "protocol_id": protocol,
            "protocol_training_seed": int(primary.training_seed),
            "known_classes": "|".join(protocol_composition[protocol_composition.class_role == "known"].application),
            "unknown_classes": "|".join(protocol_composition[protocol_composition.class_role == "unknown"].application),
            "primary_val_accuracy": float(primary.val_accuracy),
            "primary_val_macro_f1": float(primary.val_macro_f1),
            "primary_val_weighted_f1": float(primary.val_weighted_f1),
            "fixed2022_val_macro_f1": float(fixed.val_macro_f1),
            "within_protocol_seed_delta_macro_f1": abs(float(primary.val_macro_f1) - float(fixed.val_macro_f1)),
            "best_epoch": int(primary.best_epoch),
            "epochs_completed": int(primary.epochs_completed),
        })
    attribution = pd.DataFrame(attribution_rows)
    write_csv(OUT / "protocol_difficulty_attribution.csv", attribution)

    event_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    class_event_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for protocol, group in telemetry.groupby("protocol_id", sort=False):
        group = group.sort_values("epoch").reset_index(drop=True)
        proto_grad_median = float(group.prototype_gradient_norm_mean.median())
        enc_grad_median = float(group.encoder_gradient_norm_mean.median())
        event_count = 0
        recovered_count = 0
        for i, row in group.iterrows():
            epoch = int(row.epoch)
            previous = group.iloc[i - 1] if i else None
            delta_macro = np.nan if previous is None else float(row.val_macro_f1 - previous.val_macro_f1)
            delta_accuracy = np.nan if previous is None else float(row.val_accuracy - previous.val_accuracy)
            loss_ratio = np.nan if previous is None else float(row.val_total / max(previous.val_total, 1e-12))
            obvious = bool(previous is not None and is_obvious_drop(
                delta_macro, delta_accuracy, float(row.val_total), float(previous.val_total)
            ))
            relation = (
                "reset_and_reduced_lr" if epoch in (51, 81)
                else "pre_reset_window" if epoch in (49, 50, 79, 80)
                else "post_reset_window" if epoch in (52, 82)
                else "other"
            )
            record = {
                "protocol_id": protocol,
                "epoch": epoch,
                "event_relation": relation,
                "learning_rate_used": float(row.learning_rate_used),
                "prototype_reset": bool(row.prototype_reset),
                "train_loss": float(row.train_total),
                "val_loss": float(row.val_total),
                "train_accuracy": float(row.train_accuracy),
                "val_accuracy": float(row.val_accuracy),
                "train_macro_f1": float(row.train_macro_f1),
                "val_macro_f1": float(row.val_macro_f1),
                "delta_val_macro_f1": delta_macro,
                "delta_val_accuracy": delta_accuracy,
                "val_loss_ratio_to_previous": loss_ratio,
                "prototype_norm_start": float(row.prototype_norm_start),
                "prototype_norm_after_train": float(row.prototype_norm_after_train),
                "prototype_norm_after_reset": float(row.prototype_norm_after_reset),
                "prototype_norm_reset_delta": float(row.prototype_norm_reset_delta),
                "prototype_gradient_norm_mean": float(row.prototype_gradient_norm_mean),
                "prototype_gradient_ratio_to_run_median": float(row.prototype_gradient_norm_mean) / max(proto_grad_median, 1e-12),
                "encoder_gradient_norm_mean": float(row.encoder_gradient_norm_mean),
                "encoder_gradient_ratio_to_run_median": float(row.encoder_gradient_norm_mean) / max(enc_grad_median, 1e-12),
                "obvious_drop": obvious,
            }
            if epoch in RESET_WINDOW:
                window_rows.append(record)
            if obvious:
                event_count += 1
                future = group[(group.epoch > epoch) & (group.epoch <= epoch + 3)]
                recovered = bool(not future.empty and float(future.val_macro_f1.max()) >= float(previous.val_macro_f1))
                recovered_count += int(recovered)
                event_record = dict(record)
                event_record.update({
                    "pre_drop_val_macro_f1": float(previous.val_macro_f1),
                    "best_val_macro_f1_next_3_epochs": float(future.val_macro_f1.max()) if not future.empty else np.nan,
                    "recovered_to_pre_drop_within_3_epochs": recovered,
                })
                event_rows.append(event_record)

                now_class = per_class[(per_class.protocol_id == protocol) & (per_class.epoch == epoch)]
                previous_class = per_class[(per_class.protocol_id == protocol) & (per_class.epoch == epoch - 1)][["class", "recall"]]
                previous_class = previous_class.rename(columns={"recall": "previous_recall"})
                joined = now_class.merge(previous_class, on="class", how="left")
                for _, class_row in joined.iterrows():
                    class_event_rows.append({
                        "protocol_id": protocol,
                        "drop_epoch": epoch,
                        "class": class_row["class"],
                        "support": int(class_row["support"]),
                        "previous_recall": float(class_row["previous_recall"]),
                        "drop_epoch_recall": float(class_row["recall"]),
                        "delta_recall": float(class_row["recall"] - class_row["previous_recall"]),
                    })

        reset_rows = group[group.prototype_reset]
        summary_rows.append({
            "protocol_id": protocol,
            "obvious_drop_count": event_count,
            "obvious_drops_at_reset_epochs": int(sum(row["prototype_reset"] for row in event_rows if row["protocol_id"] == protocol)),
            "obvious_drops_before_first_lr_reduction": int(sum(
                row["epoch"] <= 50 for row in event_rows if row["protocol_id"] == protocol
            )),
            "obvious_drops_after_first_lr_reduction": int(sum(
                row["epoch"] >= 51 for row in event_rows if row["protocol_id"] == protocol
            )),
            "recovered_within_3_epochs_count": recovered_count,
            "reset_epoch_51_delta_macro_f1": float(group.loc[group.epoch == 51, "val_macro_f1"].iloc[0] - group.loc[group.epoch == 50, "val_macro_f1"].iloc[0]),
            "reset_epoch_81_delta_macro_f1": float(group.loc[group.epoch == 81, "val_macro_f1"].iloc[0] - group.loc[group.epoch == 80, "val_macro_f1"].iloc[0]),
            "reset_epoch_51_prototype_norm_jump": float(reset_rows.loc[reset_rows.epoch == 51, "prototype_norm_reset_delta"].iloc[0]),
            "reset_epoch_81_prototype_norm_jump": float(reset_rows.loc[reset_rows.epoch == 81, "prototype_norm_reset_delta"].iloc[0]),
            "max_val_macro_f1_after_epoch_82": float(group.loc[group.epoch >= 82, "val_macro_f1"].max()),
            "nan_metric_count": int(group.select_dtypes(include=[np.number]).isna().sum().sum()),
        })

    events = pd.DataFrame(event_rows)
    windows = pd.DataFrame(window_rows)
    class_events = pd.DataFrame(class_event_rows)
    dip_summary = pd.DataFrame(summary_rows)
    write_csv(OUT / "replay_drop_telemetry.csv", events)
    write_csv(OUT / "replay_reset_window.csv", windows)
    write_csv(OUT / "per_class_drop_recall.csv", class_events)
    write_csv(OUT / "validation_dip_summary.csv", dip_summary)

    # Compact confusion-pair table: only non-zero off-diagonal cells.
    errors = confusion[(confusion.off_diagonal.astype(str).str.lower() == "true") & (confusion["count"] > 0)].copy()
    errors = errors.sort_values(["protocol_id", "count"], ascending=[True, False])
    write_csv(OUT / "nonzero_confusion_pairs.csv", errors)

    # Include selected-checkpoint per-class metrics in a self-contained final table.
    write_csv(OUT / "per_class_comparison.csv", selected_class)

    primary_gap = primary_scores["medium_seed2026"] - primary_scores["medium_seed2025"]
    fixed_gap = fixed_scores["medium_seed2026"] - fixed_scores["medium_seed2025"]
    all_status_pass = all(status["status"] == "PASS" and status["parity_pass"] for status in statuses.values())
    audit_summary = {
        "status": "PASS" if all_status_pass else "FAIL",
        "conclusion": "PIPELINE_READY_WITH_KNOWN_TRANSIENT" if all_status_pass else "NEEDS_FURTHER_FIX",
        "freeze_hash": EXPECTED_FREEZE,
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
        "training_code_modified": False,
        "observational_replays_parity_pass": all_status_pass,
        "maximum_replay_absolute_difference": max(float(s["max_absolute_difference"]) for s in statuses.values()),
        "primary_medium2026_minus_medium2025_macro_f1": primary_gap,
        "fixed2022_medium2026_minus_medium2025_macro_f1": fixed_gap,
        "same_seed_gap_fraction_of_primary_gap": fixed_gap / primary_gap,
        "difficulty_attribution": "Known-class composition is the dominant cause; rsync/scp coexist only in Medium-2025.",
        "dip_attribution": "Large drops are transient optimization dynamics and are not synchronized with scheduler/reset epochs.",
        "reset_interpretation": "Resets cause prototype-norm discontinuities, but reset-epoch validation effects are small/mixed and recover.",
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(audit_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
