#!/usr/bin/env python3
"""Independent acceptance checks for the completed latent Gaussian audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = AUDIT_ROOT / "outputs/latent_gaussian_audit"
ARTIFACT = AUDIT_ROOT / "artifacts/latent_gaussian_audit"
CLASSES = {"FTP", "Cridex", "Miuref", "Outlook"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required_outputs = {
        "checkpoint_selection.md",
        "latent_extraction_manifest.csv",
        "latent_quality_report.md",
        "gaussian_results.csv",
        "gaussian_summary_by_class.csv",
        "component_weights.csv",
        "covariance_diagnostics.csv",
        "encoder_comparison.csv",
        "audit_summary.md",
        "run_metadata.json",
    }
    missing = sorted(name for name in required_outputs if not (OUTPUT / name).is_file())
    if missing:
        raise AssertionError(f"missing required outputs: {missing}")
    for split, expected in (("train", 391_280), ("val", 48_910)):
        path = ARTIFACT / f"{split}_mu.parquet"
        frame = pd.read_parquet(path)
        mu_columns = [column for column in frame if column.startswith("mu_")]
        assert len(frame) == expected
        assert len(mu_columns) == 128
        assert frame["flow_id"].is_unique
        assert set(frame["split"].unique()) == {split}
        assert np.isfinite(frame[mu_columns].to_numpy()).all()
    selection = json.loads((OUTPUT / "checkpoint_selection.json").read_text())
    assert sha256_file(ARTIFACT / "best_checkpoint.pt") == selection["checkpoint_sha256"]
    assertions = pd.read_csv(OUTPUT / "flow_id_equality_assertions.csv")
    assert len(assertions) == 8 and assertions["flow_id_sets_equal"].astype(bool).all()
    gaussian = pd.read_csv(OUTPUT / "gaussian_results.csv")
    assert len(gaussian) == 36
    assert set(gaussian["class_name"]) == CLASSES
    assert set(gaussian["K"]) == {1, 2, 3}
    assert set(gaussian["seed"]) == {0, 1, 2}
    assert set(gaussian["covariance_type"]) == {"full"}
    assert set(gaussian["reg_covar"]) == {1e-3}
    assert np.isfinite(gaussian["validation_avg_nll"]).all()
    components = pd.read_csv(OUTPUT / "component_weights.csv")
    covariance = pd.read_csv(OUTPUT / "covariance_diagnostics.csv")
    assert len(components) == 72 and len(covariance) == 72
    assert np.isfinite(covariance[["min_eigenvalue", "max_eigenvalue", "condition_number"]]).all().all()
    comparison = pd.read_csv(OUTPUT / "encoder_comparison.csv")
    assert len(comparison) == 4 and set(comparison["class_name"]) == CLASSES
    metadata = json.loads((OUTPUT / "run_metadata.json").read_text())
    assert metadata["test_loaded_or_used"] is False
    assert "## Latent Gaussian Audit" in (AUDIT_ROOT / "README.md").read_text()
    print(
        json.dumps(
            {
                "status": "PASS",
                "required_outputs": len(required_outputs),
                "gaussian_fits": len(gaussian),
                "component_rows": len(components),
                "covariance_rows": len(covariance),
                "comparison_classes": len(comparison),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
