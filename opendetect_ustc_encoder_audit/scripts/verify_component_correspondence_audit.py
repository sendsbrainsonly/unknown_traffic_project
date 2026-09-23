#!/usr/bin/env python3
"""Independent structural verifier for the component correspondence audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = AUDIT_ROOT / "outputs/component_correspondence_audit"
CLASSES = {"FTP", "Cridex", "Miuref", "Outlook"}
REQUIRED = {
    "tf_component_assignments.parquet",
    "od_component_assignments.parquet",
    "cross_encoder_correspondence.csv",
    "cross_encoder_confusion.csv",
    "shortcut_continuous_tests.csv",
    "shortcut_categorical_tests.csv",
    "metadata_predictability.csv",
    "train_val_stability.csv",
    "component_profile_comparison.csv",
    "audit_summary.md",
    "run_metadata.json",
}


def main() -> None:
    missing = sorted(name for name in REQUIRED if not (OUTPUT / name).is_file())
    if missing:
        raise AssertionError(f"missing required outputs: {missing}")
    tf = pd.read_parquet(OUTPUT / "tf_component_assignments.parquet")
    od = pd.read_parquet(OUTPUT / "od_component_assignments.parquet")
    expected_columns = {
        "flow_id",
        "class_name",
        "split",
        "component_id",
        "component_probability",
        "max_posterior",
    }
    for name, frame in (("TF", tf), ("OD", od)):
        if not expected_columns.issubset(frame.columns):
            raise AssertionError(f"{name}: assignment columns incomplete")
        if frame.flow_id.duplicated().any():
            raise AssertionError(f"{name}: duplicate flow_id")
        if set(frame.class_name) != CLASSES or set(frame.split) != {"train", "val"}:
            raise AssertionError(f"{name}: wrong classes or splits")
        if not set(frame.component_id).issubset({0, 1}):
            raise AssertionError(f"{name}: component outside fixed K=2")
        if not frame.max_posterior.between(0.5, 1.0).all():
            raise AssertionError(f"{name}: invalid maximum posterior")
    for class_name in CLASSES:
        for split in ("train", "val"):
            left = set(tf[(tf.class_name == class_name) & (tf.split == split)].flow_id)
            right = set(od[(od.class_name == class_name) & (od.split == split)].flow_id)
            if left != right:
                raise AssertionError(f"{class_name}/{split}: cross-encoder flow mismatch")
    correspondence = pd.read_csv(OUTPUT / "cross_encoder_correspondence.csv")
    if len(correspondence) != 8:
        raise AssertionError("expected four classes x train/validation correspondence rows")
    for column in ("nmi", "ami", "ari", "hungarian_accuracy"):
        if not np.isfinite(correspondence[column]).all():
            raise AssertionError(f"non-finite correspondence metric: {column}")
    if not correspondence.nmi.between(0, 1).all():
        raise AssertionError("NMI outside [0,1]")
    if not correspondence.ari.between(-1, 1).all() or not correspondence.ami.between(-1, 1).all():
        raise AssertionError("ARI/AMI outside [-1,1]")
    if not correspondence.hungarian_accuracy.between(0.5, 1).all():
        raise AssertionError("K=2 Hungarian accuracy outside [0.5,1]")
    stability = pd.read_csv(OUTPUT / "train_val_stability.csv")
    if len(stability) != 16 or stability.validation_component_empty.astype(bool).any():
        raise AssertionError("component stability rows incomplete or validation component empty")
    predictability = pd.read_csv(OUTPUT / "metadata_predictability.csv")
    if len(predictability) != 8 or set(predictability.model) != {"S", "S+"}:
        raise AssertionError("Model S/S+ output incomplete")
    metadata = json.loads((OUTPUT / "run_metadata.json").read_text(encoding="utf-8"))
    protocol = metadata["formal_protocol"]
    forbidden_true = [
        "test_loaded_or_used",
        "encoder_retrained",
        "split_changed",
        "sampling_changed",
        "k_searched",
        "bic_used",
        "unknown_detection_run",
    ]
    if any(protocol[field] for field in forbidden_true):
        raise AssertionError("a forbidden protocol flag is true")
    if protocol["K"] != 2 or protocol["seed"] != 0 or protocol["pca_dim"] != 64:
        raise AssertionError("formal K/PCA/seed protocol changed")
    for class_name, check in metadata["trafficformer"]["checks"].items():
            if check["train_absolute_difference"] > 1e-6:
                raise AssertionError(f"trafficformer/{class_name}: train NLL reproduction failed")
            if check["validation_absolute_difference"] > 1e-6:
                raise AssertionError(f"trafficformer/{class_name}: val NLL reproduction failed")
    for class_name, check in metadata["opendetect"]["checks"].items():
        if check["validation_absolute_difference"] > 1e-6:
            raise AssertionError(f"opendetect/{class_name}: val NLL reproduction failed")
        if not check["train_component_counts_exact"]:
            raise AssertionError(f"opendetect/{class_name}: train component counts changed")
    readme = (AUDIT_ROOT / "README.md").read_text(encoding="utf-8")
    if "## Cross-Encoder Component Correspondence Audit" not in readme:
        raise AssertionError("README section missing")
    print(
        json.dumps(
            {
                "status": "PASS",
                "assignment_rows_per_encoder": len(tf),
                "correspondence_rows": len(correspondence),
                "stability_rows": len(stability),
                "predictability_rows": len(predictability),
                "gate": metadata["overall_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
