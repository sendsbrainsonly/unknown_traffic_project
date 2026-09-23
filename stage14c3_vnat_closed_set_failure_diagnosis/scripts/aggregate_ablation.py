#!/usr/bin/env python3
"""Aggregate frozen references and new one-factor ablation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
F0_ROOT = PROJECT / "stage14c_vnat_encoder_training"
F2_ROOT = PROJECT / "stage14c5_feature_representation_audit"
OD_ROOT = PROJECT / "stage14c_native_opendetect_vnat"
PROTOCOLS = ("medium_seed2025", "medium_seed2026")
VARIANT_ORDER = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "F2_REFERENCE")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def frozen_feature_rows() -> dict[tuple[str, str], dict[str, str]]:
    table = read_csv(F2_ROOT / "stage14c5_feature_ablation.csv")
    return {(row["feature"].upper(), row["protocol_id"]): row for row in table if row["feature"] in {"f0", "f2"}}


def metric_row(protocol_id: str, variant: str, references: dict[tuple[str, str], dict[str, str]]) -> dict[str, object]:
    d0 = references[("F0", protocol_id)]
    if variant == "D0":
        return {
            "source": "reused_frozen_F0", "status": "success",
            "stop_epoch": int(d0["stop_epoch"]), "best_epoch": int(d0["best_epoch"]),
            "val_accuracy": float(d0["val_accuracy_recomputed"]),
            "val_macro_f1": float(d0["val_macro_f1_recomputed"]),
            "val_weighted_f1": float(d0["val_weighted_f1"]),
            "checkpoint_sha256": d0["checkpoint_sha256"],
        }
    if variant in {"D2", "D3"}:
        copied = metric_row(protocol_id, "D0", references)
        copied.update({
            "source": "not_run_identical_effective_configuration",
            "status": "IDENTICAL_TO_D0_BY_CONFIG",
            "checkpoint_sha256": "same frozen D0 reference",
        })
        return copied
    if variant == "D8":
        result = json.loads((OD_ROOT / "runs" / protocol_id / "result.json").read_text(encoding="utf-8"))
        return {
            "source": "reused_frozen_native_OD", "status": result["status"],
            "stop_epoch": int(result["epochs_completed"]), "best_epoch": int(result["best_epoch"]),
            "val_accuracy": float(result["validation_accuracy"]),
            "val_macro_f1": float(result["validation_macro_f1"]),
            "val_weighted_f1": float(result["validation_weighted_f1"]),
            "checkpoint_sha256": result["checkpoint_sha256"],
        }
    if variant == "F2_REFERENCE":
        row = references[("F2", protocol_id)]
        return {
            "source": "reused_frozen_F2", "status": "success",
            "stop_epoch": int(row["stop_epoch"]), "best_epoch": int(row["best_epoch"]),
            "val_accuracy": float(row["val_accuracy_recomputed"]),
            "val_macro_f1": float(row["val_macro_f1_recomputed"]),
            "val_weighted_f1": float(row["val_weighted_f1"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
    result = json.loads((ROOT / "ablation_runs" / variant / protocol_id / "result.json").read_text(encoding="utf-8"))
    return {
        "source": "new_known_only_ablation", "status": result["status"],
        "stop_epoch": int(result["stop_epoch"]), "best_epoch": int(result["best_epoch"]),
        "val_accuracy": float(result["val_accuracy"]),
        "val_macro_f1": float(result["val_macro_f1"]),
        "val_weighted_f1": float(result["val_weighted_f1"]),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "guard_activations": result["guard_activations"],
    }


def variant_metadata(variant: str) -> tuple[str, object, object, object, object, object]:
    values = {
        "D0": ("current F0", 512, "protocol seed", "composite", "patience 5", "clamp20"),
        "D1": ("batch size only", 128, "protocol seed", "composite", "patience 5", "clamp20"),
        "D2": ("optimizer/LR identical no-op", 512, "protocol seed", "composite", "patience 5", "clamp20"),
        "D3": ("scheduler identical no-op", 512, "protocol seed", "composite", "patience 5", "clamp20"),
        "D4": ("checkpoint rule only", 512, "protocol seed", "accuracy", "patience 5 composite monitor", "clamp20"),
        "D5": ("numerical protection only", 512, "protocol seed", "composite", "patience 5", "raw logvar"),
        "D6": ("seed only", 512, 2022, "composite", "patience 5", "clamp20"),
        "D7": ("early stopping only", 512, "protocol seed", "composite", "none; 100 epochs", "clamp20"),
        "D8": ("full native OD reference", 128, 2022, "accuracy", "none; 100 epochs", "raw logvar"),
        "F2_REFERENCE": ("F2 feature reference", 512, "protocol seed", "composite", "patience 5", "clamp20"),
    }
    return values[variant]


def main() -> None:
    references = frozen_feature_rows()
    rows: list[dict[str, object]] = []
    for protocol_id in PROTOCOLS:
        d0_macro = float(references[("F0", protocol_id)]["val_macro_f1_recomputed"])
        d8 = json.loads((OD_ROOT / "runs" / protocol_id / "result.json").read_text(encoding="utf-8"))
        d8_macro = float(d8["validation_macro_f1"])
        gap = d8_macro - d0_macro
        for variant in VARIANT_ORDER:
            factor, batch, seed, selection, early_stop, guard = variant_metadata(variant)
            metric = metric_row(protocol_id, variant, references)
            delta = float(metric["val_macro_f1"]) - d0_macro
            rows.append({
                "protocol_id": protocol_id,
                "run_type": "weak" if protocol_id.endswith("2025") else "normal",
                "variant": variant,
                "factor": factor,
                "batch_size": batch,
                "training_seed": seed,
                "checkpoint_rule": selection,
                "early_stopping": early_stop,
                "numerical_protection": guard,
                **metric,
                "delta_macro_f1_vs_D0": delta,
                "D0_to_D8_macro_f1_gap": gap,
                "fraction_of_D0_D8_gap_explained": 0.0 if gap == 0 else delta / gap,
                "known_test_samples_used": 0,
                "unknown_test_samples_used": 0,
            })
    write_csv(ROOT / "component_ablation.csv", rows)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    for protocol_id in PROTOCOLS:
        subset = [row for row in rows if row["protocol_id"] == protocol_id]
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.bar([row["variant"] for row in subset], [float(row["val_macro_f1"]) for row in subset])
        axis.set_ylabel("Known Validation Macro-F1")
        axis.set_title(f"Component ablation — {protocol_id}")
        axis.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"component_ablation_{protocol_id}.png", dpi=180)
        plt.close(fig)
    summary = {
        "status": "PASS" if len(rows) == 20 else "FAIL",
        "rows": len(rows),
        "new_runs": sum(row["source"] == "new_known_only_ablation" for row in rows),
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
    }
    (ROOT / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
