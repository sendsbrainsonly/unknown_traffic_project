#!/usr/bin/env python3
"""Freeze evaluation configuration, per-setting bundles, summary, and final gate."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (
    CONFIG_PATH,
    EXPECTED_OPEN_SET_PROTOCOL_SHA256,
    EXPECTED_STAGE7,
    PROJECT_ROOT,
    STAGE8_ROOT,
    VALID_SETTINGS,
    load_fold,
    require,
    sha256_file,
    stage7_run,
    verify_all_provenance,
    write_csv,
    write_json,
)


METHODS = (
    "Native",
    "Single-Full-K1",
    "Multi-Global-K2",
    "Class-P05-K2",
    "Component-P05-K2",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def method_paths(setting: str) -> dict[str, object]:
    artifact = STAGE8_ROOT / "artifacts" / setting
    checkpoint = stage7_run(setting) / "artifacts/training_best_checkpoint.pt"
    shared = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "scaler_path": str((artifact / "scaler.joblib").resolve()),
        "scaler_sha256": sha256_file(artifact / "scaler.joblib"),
        "pca_path": str((artifact / "pca64.joblib").resolve()),
        "pca_sha256": sha256_file(artifact / "pca64.joblib"),
        "representation": "deterministic mu_x -> train-only StandardScaler -> train-only PCA64",
    }
    return {
        "Native": {
            "score_definition": "negative minimum KL divergence to frozen Open-Detect prototypes",
            "threshold_path": str((artifact / "native_threshold.json").resolve()),
            "threshold_sha256": sha256_file(artifact / "native_threshold.json"),
            "checkpoint_path": shared["checkpoint_path"],
            "checkpoint_sha256": shared["checkpoint_sha256"],
        },
        "Single-Full-K1": {
            **shared,
            "score_definition": "max_y log p_K1(x|y)",
            "model_path": str((artifact / "single_k1/models.joblib").resolve()),
            "model_sha256": sha256_file(artifact / "single_k1/models.joblib"),
            "threshold_path": str((artifact / "single_global_threshold.json").resolve()),
            "threshold_sha256": sha256_file(artifact / "single_global_threshold.json"),
        },
        "Multi-Global-K2": {
            **shared,
            "score_definition": "max_y log p_K2(x|y)",
            "model_path": str((artifact / "multi_k2/models.joblib").resolve()),
            "model_sha256": sha256_file(artifact / "multi_k2/models.joblib"),
            "threshold_path": str((artifact / "multi_global_threshold.json").resolve()),
            "threshold_sha256": sha256_file(artifact / "multi_global_threshold.json"),
        },
        "Class-P05-K2": {
            **shared,
            "score_definition": "max_y(log p_K2(x|y) - tau_class_y)",
            "model_path": str((artifact / "multi_k2/models.joblib").resolve()),
            "model_sha256": sha256_file(artifact / "multi_k2/models.joblib"),
            "threshold_path": str((artifact / "class_p05_thresholds.csv").resolve()),
            "threshold_sha256": sha256_file(artifact / "class_p05_thresholds.csv"),
        },
        "Component-P05-K2": {
            **shared,
            "score_definition": "max_yk(log w_yk + log N(x|mu_yk,Sigma_yk) - tau_yk_effective)",
            "model_path": str((artifact / "multi_k2/models.joblib").resolve()),
            "model_sha256": sha256_file(artifact / "multi_k2/models.joblib"),
            "threshold_path": str((artifact / "component_p05_thresholds.csv").resolve()),
            "threshold_sha256": sha256_file(artifact / "component_p05_thresholds.csv"),
            "fallback_rule": "component validation n < 30 -> class P05 threshold",
        },
    }


def build_evaluation_config() -> dict[str, object]:
    return {
        "stage": "Stage 8A frozen configuration for the next Final Test stage",
        "created_at_utc": utc_now(),
        "created_before_any_test_access": True,
        "stage6_protocol_sha256": EXPECTED_OPEN_SET_PROTOCOL_SHA256,
        "method_order": list(METHODS),
        "excluded_methods": ["DGSB-v1", "DGSB-v2"],
        "settings": {
            setting: {
                "known_classes": load_fold(setting)["known_classes"],
                "unknown_classes": load_fold(setting)["unknown_classes"],
                "methods": method_paths(setting),
            }
            for setting in VALID_SETTINGS
        },
        "frozen_choices": {
            "latent_dimension": 128,
            "pca_dimension": 64,
            "single_components": 1,
            "multi_components": 2,
            "covariance_type": "full",
            "reg_covar": 0.001,
            "class_quantile": 0.05,
            "component_quantile": 0.05,
            "component_fallback_min_validation_n": 30,
        },
        "test_execution_performed": False,
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "unknown_inference_executed": False,
    }


def frozen_setting_files(setting: str, evaluation_config: Path) -> list[Path]:
    artifact = STAGE8_ROOT / "artifacts" / setting
    output = STAGE8_ROOT / "outputs" / setting
    paths = [
        stage7_run(setting) / "artifacts/training_best_checkpoint.pt",
        artifact / "mu_train.npy",
        artifact / "mu_val.npy",
        artifact / "train_manifest.csv",
        artifact / "val_manifest.csv",
        artifact / "native_val_known_scores.npy",
        artifact / "native_val_predictions.npy",
        artifact / "scaler.joblib",
        artifact / "pca64.joblib",
        artifact / "z_train_pca64.npy",
        artifact / "z_val_pca64.npy",
        artifact / "single_k1/models.joblib",
        artifact / "multi_k2/models.joblib",
        artifact / "native_threshold.json",
        artifact / "single_global_threshold.json",
        artifact / "multi_global_threshold.json",
        artifact / "class_p05_thresholds.csv",
        artifact / "component_p05_thresholds.csv",
        output / "representation_audit.json",
        output / "known_val_per_class_metrics.csv",
        output / "known_val_confusion_matrix.csv",
        output / "scaler_pca_audit.json",
        output / "k2_component_diagnostics.csv",
        output / "component_assignments_train.csv",
        output / "component_assignments_val.csv",
        output / "component_calibration_capacity.csv",
        output / "fallback_summary.csv",
        output / "known_val_boundary_summary.csv",
        output / "k1_k2_density_comparison.csv",
        output / "file_access_audit.json",
        evaluation_config,
    ]
    paths.extend(sorted((artifact / "single_k1/classes").glob("*.npz")))
    paths.extend(sorted((artifact / "multi_k2/classes").glob("*.npz")))
    require(all(path.is_file() for path in paths), f"{setting}: incomplete frozen file set")
    return paths


def freeze_setting_bundle(setting: str, evaluation_config: Path) -> dict[str, object]:
    artifact = STAGE8_ROOT / "artifacts" / setting
    manifest_path = artifact / "bundle_manifest.json"
    hashes_path = artifact / "bundle_hashes.sha256"
    if manifest_path.exists() or hashes_path.exists():
        require(manifest_path.is_file() and hashes_path.is_file(), f"{setting}: partial bundle freeze exists")
        manifest = load_json(manifest_path)
        lines = hashes_path.read_text(encoding="utf-8").splitlines()
        require(len(lines) == len(manifest["files"]), f"{setting}: existing bundle hash count mismatch")
        for line, item in zip(lines, manifest["files"]):
            expected, relative = line.split(maxsplit=1)
            require(expected == item["sha256"] and relative == item["path"], f"{setting}: existing bundle ledger mismatch")
            require(sha256_file((artifact / relative).resolve()) == expected, f"{setting}: existing frozen file changed")
        return {"setting": setting, "files": len(lines), "bundle_manifest_sha256": sha256_file(manifest_path), "bundle_hashes_sha256": sha256_file(hashes_path), "status": "PASS"}
    files = frozen_setting_files(setting, evaluation_config)
    records = [
        {
            "path": os.path.relpath(path.resolve(), artifact.resolve()),
            "absolute_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "external_reference": not path.resolve().is_relative_to(artifact.resolve()),
        }
        for path in files
    ]
    payload = {
        "setting": setting,
        "status": "FROZEN",
        "created_before_test_access": True,
        "checkpoint_sha256": EXPECTED_STAGE7[setting]["checkpoint_sha256"],
        "protocol_sha256": EXPECTED_OPEN_SET_PROTOCOL_SHA256,
        "fit_data": "KNOWN_TRAIN_ONLY",
        "threshold_data": "KNOWN_VALIDATION_ONLY",
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "unknown_inference_executed": False,
        "files": records,
    }
    write_json(manifest_path, payload)
    hashes_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in records),
        encoding="utf-8",
    )
    for item in records:
        require(sha256_file((artifact / str(item["path"])).resolve()) == item["sha256"], f"{setting}: frozen hash replay failed")
    return {"setting": setting, "files": len(records), "bundle_manifest_sha256": sha256_file(manifest_path), "bundle_hashes_sha256": sha256_file(hashes_path), "status": "PASS"}


def main() -> None:
    provenance = verify_all_provenance()
    require(provenance["status"] == "PASS", "provenance gate failed")
    evaluation_path = STAGE8_ROOT / "configs/evaluation_config.json"
    if evaluation_path.exists():
        evaluation = load_json(evaluation_path)
        require(evaluation["method_order"] == list(METHODS), "existing evaluation method order changed")
        require(evaluation["created_before_any_test_access"] is True, "existing evaluation config lacks pre-test freeze")
    else:
        evaluation = build_evaluation_config()
        write_json(evaluation_path, evaluation)

    summaries: list[dict[str, object]] = []
    for setting in VALID_SETTINGS:
        fold = load_fold(setting)
        output = STAGE8_ROOT / "outputs" / setting
        artifact = STAGE8_ROOT / "artifacts" / setting
        metadata = load_json(output / "density_boundary_metadata.json")
        representation = load_json(output / "representation_audit.json")
        access = load_json(output / "file_access_audit.json")
        boundary = {row["method"]: row for row in csv_rows(output / "known_val_boundary_summary.csv")}
        native = load_json(artifact / "native_threshold.json")
        single = load_json(artifact / "single_global_threshold.json")
        multi = load_json(artifact / "multi_global_threshold.json")
        require(access["status"] == "PASS" and int(access["known_test_opened"]) == 0 and int(access["unknown_test_opened"]) == 0, f"{setting}: access isolation failed")
        require(representation["train"]["shape"] == [int(fold["known_train_count"]), 128], f"{setting}: train representation incomplete")
        require(representation["validation"]["shape"] == [int(fold["known_validation_count"]), 128], f"{setting}: validation representation incomplete")
        summaries.append(
            {
                "setting": setting,
                "known_classes": len(fold["known_classes"]),
                "train_n": int(fold["known_train_count"]),
                "val_n": int(fold["known_validation_count"]),
                "Native_threshold": float(native["threshold"]),
                "Single_threshold": float(single["threshold"]),
                "Multi_threshold": float(multi["threshold"]),
                "mean_DeltaNLL2": float(metadata["mean_DeltaNLL2"]),
                "median_DeltaNLL2": float(metadata["median_DeltaNLL2"]),
                "classes_K2_better_count": int(metadata["classes_K2_validation_NLL_better"]),
                "min_component_weight": float(metadata["minimum_k2_component_weight"]),
                "n_component_val_lt_30": int(metadata["component_val_n_lt_30"]),
                "fallback_rate": float(metadata["component_fallback_rate"]),
                "Native_val_acceptance": float(boundary["Native"]["overall_acceptance"]),
                "Single_val_acceptance": float(boundary["Single-Full-K1"]["overall_acceptance"]),
                "Multi_val_acceptance": float(boundary["Multi-Global-K2"]["overall_acceptance"]),
                "Class_val_acceptance": float(boundary["Class-P05-K2"]["overall_acceptance"]),
                "Component_val_acceptance": float(boundary["Component-P05-K2"]["overall_acceptance"]),
                "Class_per_class_acceptance_min": float(boundary["Class-P05-K2"]["per_class_acceptance_min"]),
                "Class_per_class_acceptance_max": float(boundary["Class-P05-K2"]["per_class_acceptance_max"]),
                "Component_per_class_acceptance_min": float(boundary["Component-P05-K2"]["per_class_acceptance_min"]),
                "Component_per_class_acceptance_max": float(boundary["Component-P05-K2"]["per_class_acceptance_max"]),
                "Class_mean_component_coverage_gap": float(boundary["Class-P05-K2"]["mean_per_class_component_acceptance_gap"]),
                "Component_mean_component_coverage_gap": float(boundary["Component-P05-K2"]["mean_per_class_component_acceptance_gap"]),
                "known_test_opened": 0,
                "unknown_test_opened": 0,
            }
        )
    write_csv(STAGE8_ROOT / "outputs/summary/cross_setting_summary.csv", summaries)
    bundle_results = [freeze_setting_bundle(setting, evaluation_path) for setting in VALID_SETTINGS]

    prohibited = [
        STAGE8_ROOT / f"artifacts/{setting}/{name}"
        for setting in VALID_SETTINGS
        for name in ("mu_test.npy", "mu_unknown.npy", "test_manifest.csv", "unknown_manifest.csv", "z_test_pca64.npy", "z_unknown_pca64.npy")
    ]
    gate_checks = {
        "stage6_hash_pass": provenance["stage6"]["verification_status"] == "PASS",
        "stage7_checkpoint_hash_pass": all(item["verification_status"] == "PASS" for item in provenance["settings"]),
        "train_val_mu_complete": all((STAGE8_ROOT / f"artifacts/{setting}/mu_train.npy").is_file() and (STAGE8_ROOT / f"artifacts/{setting}/mu_val.npy").is_file() for setting in VALID_SETTINGS),
        "test_mu_absent": not any(path.exists() for path in prohibited if "test" in path.name),
        "unknown_mu_absent": not any(path.exists() for path in prohibited if "unknown" in path.name),
        "scaler_train_only": all(load_json(STAGE8_ROOT / f"outputs/{setting}/scaler_pca_audit.json")["scaler_fit_split"] == "KNOWN_TRAIN_ONLY" for setting in VALID_SETTINGS),
        "pca_train_only": all(load_json(STAGE8_ROOT / f"outputs/{setting}/scaler_pca_audit.json")["pca_fit_split"] == "KNOWN_TRAIN_ONLY" for setting in VALID_SETTINGS),
        "k1_k2_train_only": all(load_json(STAGE8_ROOT / f"outputs/{setting}/density_boundary_metadata.json")["k1_fit_split"] == "KNOWN_TRAIN_ONLY" and load_json(STAGE8_ROOT / f"outputs/{setting}/density_boundary_metadata.json")["k2_fit_split"] == "KNOWN_TRAIN_ONLY" for setting in VALID_SETTINGS),
        "thresholds_validation_only": all(load_json(STAGE8_ROOT / f"outputs/{setting}/density_boundary_metadata.json")["threshold_fit_split"] == "KNOWN_VALIDATION_ONLY" for setting in VALID_SETTINGS),
        "k2_convergence_or_anomaly_record_complete": all((STAGE8_ROOT / f"outputs/{setting}/k2_component_diagnostics.csv").is_file() and len(csv_rows(STAGE8_ROOT / f"outputs/{setting}/k2_component_diagnostics.csv")) == len(load_fold(setting)["known_classes"]) for setting in VALID_SETTINGS),
        "fallback_map_frozen": all((STAGE8_ROOT / f"artifacts/{setting}/component_p05_thresholds.csv").is_file() for setting in VALID_SETTINGS),
        "evaluation_config_frozen": evaluation_path.is_file(),
        "known_test_usage_zero": all(int(load_json(STAGE8_ROOT / f"outputs/{setting}/file_access_audit.json")["known_test_opened"]) == 0 for setting in VALID_SETTINGS),
        "unknown_test_usage_zero": all(int(load_json(STAGE8_ROOT / f"outputs/{setting}/file_access_audit.json")["unknown_test_opened"]) == 0 for setting in VALID_SETTINGS),
        "bundle_hashes_pass": all(result["status"] == "PASS" for result in bundle_results),
    }
    gate = "READY_FOR_FINAL_TEST" if all(gate_checks.values()) else "NOT_READY"
    final_gate = {
        "gate": gate,
        "checks": gate_checks,
        "bundle_verification": bundle_results,
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
    }
    write_json(STAGE8_ROOT / "outputs/summary/final_gate.json", final_gate)
    (STAGE8_ROOT / "outputs/summary/final_gate.md").write_text(
        "# Stage 8A Final Gate\n\n"
        f"**{gate}**\n\n"
        + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gate_checks.items())
        + "\n\nKnown Test opened: 0. Unknown Test opened: 0. Unknown inference executed: false.\n",
        encoding="utf-8",
    )
    require(gate == "READY_FOR_FINAL_TEST", "Stage 8A final gate is NOT_READY")

    original_manifest = load_json(STAGE8_ROOT / "manifest.json")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    manifest = {
        **original_manifest,
        "updated_at_utc": utc_now(),
        "status": "success",
        "inputs": [
            {"path": str((STAGE8_ROOT.parent / "stage6_cipherspectrum_protocol/outputs/protocol/cipherspectrum_open_set_protocol.json").resolve()), "sha256": EXPECTED_OPEN_SET_PROTOCOL_SHA256},
            *[
                {"path": str((stage7_run(setting) / "artifacts/training_best_checkpoint.pt").resolve()), "sha256": EXPECTED_STAGE7[setting]["checkpoint_sha256"]}
                for setting in VALID_SETTINGS
            ],
        ],
        "code": {"revision": revision, "dirty": True, "changes": ["stage8a_cipherspectrum_known_density"]},
        "execution": {
            "tmux_session": "stage8a_finalize_20260913",
            "related_sessions": [
                "stage8a_provenance_freeze_20260913",
                *[f"stage8a_extract_{setting}_20260913" for setting in VALID_SETTINGS],
                *[f"stage8a_fit_{setting}_20260913" for setting in VALID_SETTINGS],
            ],
            "command": "python stage8a_cipherspectrum_known_density/scripts/finalize_stage8a.py",
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": sorted({int(load_json(STAGE8_ROOT / f"outputs/{setting}/extraction_metadata.json")["physical_gpu"]) for setting in VALID_SETTINGS}),
        },
        "configuration": {
            "files": ["configs/stage8a_config.json", "configs/evaluation_config.json"],
            "parameters": load_json(CONFIG_PATH),
            "seeds": [0],
        },
        "core_results": [
            {"metric": f"{row['setting']}_mean_DeltaNLL2", "value": row["mean_DeltaNLL2"]}
            for row in summaries
        ] + [
            {"metric": f"{row['setting']}_component_fallback_rate", "value": row["fallback_rate"]}
            for row in summaries
        ],
        "artifacts": [],
        "limitations": [
            "Known Validation diagnostics only; no Known Test or Unknown Test evaluation was performed.",
            "K2 density-fit improvement is not evidence of Unknown utility.",
        ],
        "next_step": "Stop. A separately authorized final-test stage may consume only the frozen evaluation_config.json and bundle hashes.",
    }
    write_json(STAGE8_ROOT / "manifest.json", manifest)

    table_lines = [
        "| Setting | mu train/val | PCA64 variance | K2 min weight | val_n<30 / fallback | mean/median DeltaNLL2 | K2 better | Native/Single/Multi/Class/Component acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        meta = load_json(STAGE8_ROOT / f"outputs/{row['setting']}/density_boundary_metadata.json")
        table_lines.append(
            f"| {str(row['setting']).title()} | {int(row['train_n']):,}/{int(row['val_n']):,} | {float(meta['pca_cumulative_explained_variance']):.6f} | {float(row['min_component_weight']):.6f} | {int(row['n_component_val_lt_30'])}/{float(row['fallback_rate']):.6f} | {float(row['mean_DeltaNLL2']):.6f}/{float(row['median_DeltaNLL2']):.6f} | {int(row['classes_K2_better_count'])}/{int(row['known_classes'])} | {float(row['Native_val_acceptance']):.6f}/{float(row['Single_val_acceptance']):.6f}/{float(row['Multi_val_acceptance']):.6f}/{float(row['Class_val_acceptance']):.6f}/{float(row['Component_val_acceptance']):.6f} |"
        )
    results = f"""# Experiment results: stage8a-cipherspectrum-known-density-20260913

- Status: `success`
- Experiment type: `evaluation`
- Claim scope: `diagnostic`
- Completed (UTC): `{utc_now()}`
- Final Gate: `{gate}`

## Data and split

- Stage 6 canonical 120k Low/Medium/High frozen folds.
- Source data opened: Known Train and Known Validation only.
- Known Test opened: 0; Unknown Test opened: 0; Unknown inference: false.

## Configuration and execution

- Deterministic Stage 7 best-checkpoint `mu_x` (128D) -> train-only StandardScaler -> train-only PCA64.
- Per-class full K1/K2 (`reg_covar=1e-3`, K2 `n_init=3`, `max_iter=300`, seed 0).
- Native/global/class/component thresholds use Known Validation only; component fallback is fixed at n<30.

## Core results

{chr(10).join(table_lines)}

## Preserved evidence

- `artifacts/{{low,medium,high}}/`: representations, transforms, density models, thresholds, bundle manifests and hashes.
- `outputs/{{low,medium,high}}/`: representation/classification/component/calibration diagnostics and access ledgers.
- `outputs/summary/`: provenance, cross-setting summary, and final gate.
- `configs/evaluation_config.json`: frozen next-stage method configuration.

## Limitations

- These are Known Validation calibration diagnostics, not Test Accuracy, UFAR, AUROC, AUPRC, or Unknown utility.
- Positive DeltaNLL2 demonstrates only held-out density-fit improvement.

## Conclusion and next step

- `{gate}`. Stop here; do not open Test without a separate next-stage instruction.
"""
    (STAGE8_ROOT / "RESULTS.md").write_text(results, encoding="utf-8")
    print(json.dumps(final_gate, sort_keys=True))


if __name__ == "__main__":
    main()
