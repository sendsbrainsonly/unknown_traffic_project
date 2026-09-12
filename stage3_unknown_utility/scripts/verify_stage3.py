#!/usr/bin/env python3
"""Independent observable-evidence verifier for the completed Stage 3 task."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from common import STAGE3_ROOT, VALID_SETTINGS, latent_columns, load_fold, sha256_file, verify_frozen_inputs
from prepare_setting import FORBIDDEN_UNKNOWN_FLAGS


REQUIRED_OUTPUTS = (
    "RESULTS.md",
    "manifest.json",
    "unknown_leakage_audit.md",
    "data_manifest.csv",
    "leakage_audit.json",
    "smoke_report.md",
    "training_config.json",
    "training_metrics.csv",
    "checkpoint_selection.json",
    "latent_export_manifest.json",
    "latent_quality_gate.json",
    "density_fit_warnings.csv",
    "density_validation_loglik.csv",
    "k2_stability.csv",
    "threshold_calibration.csv",
    "frozen_model_hashes.json",
    "detector_metrics.csv",
    "per_unknown_class_results.csv",
    "decision_transition_matrix.csv",
    "unknown_absorption_by_known_class.csv",
    "paired_bootstrap_samples.csv",
    "paired_bootstrap_ci.csv",
    "run_metadata.json",
)
REQUIRED_ARTIFACTS = (
    "best_checkpoint.pt",
    "standard_scaler.joblib",
    "pca64.joblib",
    "single_full_k1_models.joblib",
    "multi_full_k2_models.joblib",
    "train_known_mu.parquet",
    "val_known_mu.parquet",
    "test_known_mu.parquet",
    "test_unknown_mu.parquet",
    "frozen_final_predictions.parquet",
)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    verify_frozen_inputs()
    checks: list[dict[str, object]] = []
    for setting in VALID_SETTINGS:
        fold = load_fold(setting)
        output_dir = STAGE3_ROOT / "outputs" / setting
        artifact_dir = STAGE3_ROOT / "artifacts" / setting
        missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
        missing += [name for name in REQUIRED_ARTIFACTS if not (artifact_dir / name).exists()]
        if missing:
            raise RuntimeError(f"{setting}: missing required files: {missing}")
        audit = json.loads((output_dir / "leakage_audit.json").read_text(encoding="utf-8"))
        if audit["status"] != "PASS" or any(audit["unknown_forbidden_usage_counts"].values()):
            raise RuntimeError(f"{setting}: leakage audit failed")
        if sha256_file(output_dir / "data_manifest.csv") != audit["data_manifest_sha256"]:
            raise RuntimeError(f"{setting}: data manifest hash changed")
        unknown_forbidden = {flag: 0 for flag in FORBIDDEN_UNKNOWN_FLAGS}
        unknown_final_wrong_split = 0
        with (output_dir / "data_manifest.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["known_or_unknown"] != "Unknown":
                    continue
                for flag in FORBIDDEN_UNKNOWN_FLAGS:
                    unknown_forbidden[flag] += row[flag] == "true"
                if row["used_final_test"] == "true" and row["original_split"] != "test":
                    unknown_final_wrong_split += 1
        if any(unknown_forbidden.values()) or unknown_final_wrong_split:
            raise RuntimeError(f"{setting}: manifest-level Unknown leakage")
        config = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
        if config["unknown_samples_loaded"] != 0 or config["num_classes"] != fold["known_count"]:
            raise RuntimeError(f"{setting}: training config violates fold")
        if config["warm_start"] or config["forbidden_20class_checkpoint_used"]:
            raise RuntimeError(f"{setting}: forbidden checkpoint initialization")
        checkpoint = json.loads((output_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
        if checkpoint["unknown_samples_used"] != 0:
            raise RuntimeError(f"{setting}: Unknown used in checkpoint selection")
        if sha256_file(artifact_dir / "best_checkpoint.pt") != checkpoint["checkpoint_sha256"]:
            raise RuntimeError(f"{setting}: checkpoint hash mismatch")
        frozen = json.loads((output_dir / "frozen_model_hashes.json").read_text(encoding="utf-8"))
        for item in frozen["files"]:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"{setting}: frozen artifact mismatch: {item['path']}")
        frozen_mtime = (output_dir / "frozen_model_hashes.json").stat().st_mtime_ns
        if any((artifact_dir / name).stat().st_mtime_ns <= frozen_mtime for name in ("test_known_mu.parquet", "test_unknown_mu.parquet", "frozen_final_predictions.parquet")):
            raise RuntimeError(f"{setting}: final-test artifact does not postdate model freeze")
        id_sets = []
        for filename, expected_rows in (
            ("train_known_mu.parquet", fold["formal_counts"]["known_train"]),
            ("val_known_mu.parquet", fold["formal_counts"]["known_validation"]),
            ("test_known_mu.parquet", fold["formal_counts"]["known_final_test"]),
            ("test_unknown_mu.parquet", fold["formal_counts"]["unknown_final_test"]),
        ):
            frame = pd.read_parquet(artifact_dir / filename)
            if len(frame) != expected_rows or frame["flow_id"].duplicated().any():
                raise RuntimeError(f"{setting}: {filename} row/uniqueness check failed")
            latent_columns(frame.columns)
            id_sets.append(set(frame["flow_id"].astype(str)))
        if any(id_sets[left] & id_sets[right] for left in range(4) for right in range(left + 1, 4)):
            raise RuntimeError(f"{setting}: latent split overlap")
        thresholds = csv_rows(output_dir / "threshold_calibration.csv")
        if len(thresholds) != 3 or any(int(row["unknown_samples_used"]) for row in thresholds):
            raise RuntimeError(f"{setting}: threshold calibration leakage")
        metrics = csv_rows(output_dir / "detector_metrics.csv")
        if {row["detector"] for row in metrics} != {"Native", "Single-Full", "Multi-Full-K2"}:
            raise RuntimeError(f"{setting}: detector set mismatch")
        bootstrap = csv_rows(output_dir / "paired_bootstrap_samples.csv")
        if len(bootstrap) != 1000:
            raise RuntimeError(f"{setting}: bootstrap replicate count mismatch")
        checks.append(
            {
                "setting": setting,
                "status": "PASS",
                "known_classes": fold["known_count"],
                "unknown_classes": fold["unknown_count"],
                "leakage": "PASS",
                "latent_dim": 128,
                "bootstrap_resamples": 1000,
            }
        )
    for required in (
        STAGE3_ROOT / "outputs/summary/cross_setting_results.csv",
        STAGE3_ROOT / "outputs/summary/openness_trend.md",
        STAGE3_ROOT / "outputs/summary/final_gate.json",
        STAGE3_ROOT / "outputs/summary/final_gate.md",
        STAGE3_ROOT / "README.md",
    ):
        if not required.exists():
            raise RuntimeError(f"missing summary artifact: {required}")
    result = {"status": "PASS", "settings": checks}
    path = STAGE3_ROOT / "outputs/summary/completion_verification.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
