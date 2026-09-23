#!/usr/bin/env python3
"""Read-only audit of frozen protocol composition and Stage 14C-4 evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
STAGE14C4 = PROJECT / "stage14c4_our_method_training_cleanup"
PROTOCOL_IDS = ("medium_seed2025", "medium_seed2026")
PRIMARY_RUNS = {pid: f"{pid}_protocol" for pid in PROTOCOL_IDS}
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
DIFFICULTY_CLASSES = {"rsync", "scp", "sftp", "rdp"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    frozen = json.loads((STAGE14B / "vnat_open_set_protocol.json").read_text(encoding="utf-8"))
    if frozen["freeze_hash"] != EXPECTED_FREEZE:
        raise RuntimeError("Stage 14B freeze hash changed")
    protocols = {row["protocol_id"]: row for row in frozen["protocols"]}

    composition_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for protocol_id in PROTOCOL_IDS:
        protocol = protocols[protocol_id]
        protocol_rows.append({
            "protocol_id": protocol_id,
            "protocol_seed": protocol["seed"],
            "known_classes": "|".join(protocol["known_applications"]),
            "unknown_classes": "|".join(protocol["unknown_applications"]),
            "known_class_count": len(protocol["known_applications"]),
            "unknown_class_count": len(protocol["unknown_applications"]),
            "known_train_samples": protocol["split_statistics"]["train"]["flows"],
            "known_val_samples": protocol["split_statistics"]["validation"]["flows"],
            "known_test_samples_opened": 0,
            "unknown_test_samples_opened": 0,
        })
        for application in frozen["retained_applications"]:
            role = "known" if application in protocol["known_applications"] else "unknown"
            counts = protocol["known_class_split_counts"].get(application, {})
            composition_rows.append({
                "protocol_id": protocol_id,
                "protocol_seed": protocol["seed"],
                "application": application,
                "class_role": role,
                "train_samples": counts.get("train", 0),
                "validation_samples": counts.get("validation", 0),
                "difficulty_focus": application in DIFFICULTY_CLASSES,
            })

        run_id = PRIMARY_RUNS[protocol_id]
        per_class = pd.read_csv(STAGE14C4 / "runs" / run_id / "per_class_metrics.csv")
        for row in per_class.to_dict("records"):
            per_class_rows.append({"protocol_id": protocol_id, **row})

        matrix = pd.read_csv(STAGE14C4 / "confusion_matrices" / f"{run_id}.csv")
        true_col = matrix.columns[0]
        for _, row in matrix.iterrows():
            for predicted in matrix.columns[1:]:
                confusion_rows.append({
                    "protocol_id": protocol_id,
                    "true_class": row[true_col],
                    "predicted_class": predicted,
                    "count": int(row[predicted]),
                    "off_diagonal": row[true_col] != predicted,
                })

    write_csv(OUT / "protocol_summary.csv", protocol_rows)
    write_csv(OUT / "protocol_class_composition.csv", composition_rows)
    write_csv(OUT / "selected_checkpoint_per_class.csv", per_class_rows)
    write_csv(OUT / "selected_checkpoint_confusion_long.csv", confusion_rows)

    dynamics = pd.read_csv(STAGE14C4 / "training_dynamics.csv")
    event_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for (protocol_id, seed_mode), group in dynamics.groupby(["protocol_id", "seed_mode"], sort=True):
        group = group.sort_values("epoch").reset_index(drop=True)
        for index, row in group.iterrows():
            previous = group.iloc[index - 1] if index else None
            delta_macro = np.nan if previous is None else row["val_macro_f1"] - previous["val_macro_f1"]
            delta_accuracy = np.nan if previous is None else row["val_accuracy"] - previous["val_accuracy"]
            loss_ratio = np.nan if previous is None else row["val_total"] / max(previous["val_total"], 1e-12)
            obvious = bool(
                previous is not None
                and (
                    delta_macro <= -0.15
                    or delta_accuracy <= -0.15
                    or (loss_ratio >= 3.0 and row["val_total"] - previous["val_total"] >= 0.10)
                )
            )
            epoch = int(row["epoch"])
            relation = (
                "reset_epoch" if epoch in (51, 81)
                else "pre_reset_window" if epoch in (49, 50, 79, 80)
                else "post_reset_window" if epoch in (52, 82)
                else "other"
            )
            common = {
                "protocol_id": protocol_id,
                "seed_mode": seed_mode,
                "training_seed": int(row["training_seed"]),
                "epoch": epoch,
                "event_relation": relation,
                "learning_rate_used": row["learning_rate"],
                "prototype_reset": bool(row["prototype_reset"]),
                "train_loss": row["train_total"],
                "val_loss": row["val_total"],
                "train_accuracy": row["train_accuracy"],
                "val_accuracy": row["val_accuracy"],
                "train_macro_f1": row["train_macro_f1"],
                "val_macro_f1": row["val_macro_f1"],
                "delta_val_macro_f1": delta_macro,
                "delta_val_accuracy": delta_accuracy,
                "val_loss_ratio_to_previous": loss_ratio,
                "obvious_drop": obvious,
            }
            if obvious:
                event_rows.append(common)
            if epoch in (49, 50, 51, 52, 79, 80, 81, 82):
                window_rows.append(common)

    if not event_rows:
        event_rows.append({
            "protocol_id": "none", "seed_mode": "none", "training_seed": -1,
            "epoch": -1, "event_relation": "none", "learning_rate_used": np.nan,
            "prototype_reset": False, "train_loss": np.nan, "val_loss": np.nan,
            "train_accuracy": np.nan, "val_accuracy": np.nan, "train_macro_f1": np.nan,
            "val_macro_f1": np.nan, "delta_val_macro_f1": np.nan,
            "delta_val_accuracy": np.nan, "val_loss_ratio_to_previous": np.nan,
            "obvious_drop": False,
        })
    write_csv(OUT / "validation_drop_events.csv", event_rows)
    write_csv(OUT / "reset_window_dynamics.csv", window_rows)

    availability = [
        {
            "field": "learning_rate",
            "historical_stage14c4_available": True,
            "granularity": "per epoch",
            "audit_action": "reuse immutable training_dynamics.csv",
        },
        {
            "field": "train_val_loss_accuracy_macro_f1",
            "historical_stage14c4_available": True,
            "granularity": "per epoch",
            "audit_action": "reuse immutable training_dynamics.csv",
        },
        {
            "field": "per_class_recall",
            "historical_stage14c4_available": False,
            "granularity": "selected checkpoint only",
            "audit_action": "exact-config instrumented replay required for epoch-level evidence",
        },
        {
            "field": "prototype_norm",
            "historical_stage14c4_available": False,
            "granularity": "not recorded",
            "audit_action": "exact-config instrumented replay required",
        },
        {
            "field": "prototype_gradient_norm",
            "historical_stage14c4_available": False,
            "granularity": "not recorded",
            "audit_action": "exact-config instrumented replay required",
        },
        {
            "field": "encoder_gradient_norm",
            "historical_stage14c4_available": False,
            "granularity": "not recorded",
            "audit_action": "exact-config instrumented replay required",
        },
    ]
    write_csv(OUT / "telemetry_availability.csv", availability)
    print(json.dumps({
        "status": "PASS",
        "freeze_hash": frozen["freeze_hash"],
        "protocols": list(PROTOCOL_IDS),
        "obvious_drop_events": len([row for row in event_rows if row["obvious_drop"]]),
        "historical_norm_telemetry_available": False,
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
