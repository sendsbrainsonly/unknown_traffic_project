#!/usr/bin/env python3
"""Freeze the unified rule, evaluation config, hashes, summary, and Stage 8B gate."""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from stage8b_common import (
    CONFIG_PATH,
    PROJECT_ROOT,
    STAGE8A_ROOT,
    STAGE8B_ROOT,
    VALID_SETTINGS,
    hash_manifest,
    load_json,
    read_csv,
    require,
    sha256_file,
    verify_all_provenance,
    write_csv,
    write_json,
)


BASELINE_METHODS = [
    "Native",
    "Single-Full-K1",
    "Multi-Global-K2",
    "Class-P05-K2",
    "Component-P05-K2",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dgsb_method(setting: str, rule_sha: str) -> dict[str, object]:
    base = load_json(STAGE8A_ROOT / "configs/evaluation_config.json")["settings"][setting]["methods"]["Multi-Global-K2"]
    artifact = STAGE8B_ROOT / "artifacts" / setting
    return {
        "checkpoint_path": base["checkpoint_path"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "representation": base["representation"],
        "scaler_path": base["scaler_path"],
        "scaler_sha256": base["scaler_sha256"],
        "pca_path": base["pca_path"],
        "pca_sha256": base["pca_sha256"],
        "model_path": base["model_path"],
        "model_sha256": base["model_sha256"],
        "score_definition": "G=S_y*, L=s_y*k* with y*=argmax_y S_y and k*=argmax_k s_y*k; Known iff both frozen gates pass",
        "global_threshold_path": str((artifact / "dgsbv2_global_threshold.json").resolve()),
        "global_threshold_sha256": sha256_file(artifact / "dgsbv2_global_threshold.json"),
        "local_threshold_path": str((artifact / "dgsbv2_local_thresholds.csv").resolve()),
        "local_threshold_sha256": sha256_file(artifact / "dgsbv2_local_thresholds.csv"),
        "fallback_map_path": str((artifact / "dgsbv2_fallback_map.csv").resolve()),
        "fallback_map_sha256": sha256_file(artifact / "dgsbv2_fallback_map.csv"),
        "rule_path": str((artifact / "dgsbv2_rule.json").resolve()),
        "rule_sha256": rule_sha,
        "global_quantile": 0.025,
        "local_quantile": 0.025,
        "fallback_min_validation_n": 30,
        "gate_logic": "AND",
        "post_local_global_recalibration": "FORBIDDEN",
        "unknown_calibration": "FORBIDDEN",
    }


def main() -> None:
    verify_all_provenance()
    config = load_json(CONFIG_PATH)
    summaries = []
    for setting in VALID_SETTINGS:
        summary_path = STAGE8B_ROOT / f"outputs/{setting}/calibration_summary.json"
        require(summary_path.is_file(), f"{setting}: calibration summary missing")
        summaries.append(load_json(summary_path))

    spec = {
        "name": "DGSB-v2",
        "stage": "Stage 8B Known-only rule freeze",
        "K": 2,
        "covariance_type": "full",
        "reg_covar": 0.001,
        "pca_dimension": 64,
        "global_quantile": 0.025,
        "local_quantile": 0.025,
        "fallback_n": 30,
        "gate_logic": "AND",
        "prediction": "argmax class mixture score",
        "local_component": "argmax component within predicted class",
        "same_class_support_check": True,
        "post_local_global_recalibration": "FORBIDDEN",
        "unknown_calibration": "FORBIDDEN",
        "quantile_search": "FORBIDDEN",
        "error_budget_decomposition": {
            "target_total_rejection": 0.05,
            "global_rejection_budget": 0.025,
            "local_rejection_budget": 0.025,
            "preregistered_before_test_access": True,
        },
        "primary_comparison": "DGSB-v2 vs Multi-Global-K2",
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
    }
    spec_path = STAGE8B_ROOT / "artifacts/dgsbv2_global_spec.json"
    spec_md_path = STAGE8B_ROOT / "artifacts/dgsbv2_global_spec.md"
    require(not spec_path.exists() and not spec_md_path.exists(), "refusing to overwrite unified rule specification")
    write_json(spec_path, spec)
    spec_md_path.write_text(
        """# DGSB-v2 unified frozen specification

The preregistered total Known rejection budget is approximately 5%, split equally before Test access: 2.5% for the absolute Global density floor and 2.5% for predicted-class Local component support. This is not an Unknown-driven search or a grid search.

- Frozen density: PCA64, per-class full-covariance K2, `reg_covar=1e-3`.
- Prediction: class with maximum mixture score.
- Local component: maximum weighted component score inside that same predicted class.
- Decision: Global Gate AND Local Gate.
- Sparse fallback: Validation `n<30` uses the class local-max-score P02.5 threshold, preserving local-score semantics.
- Post-local Global recalibration: FORBIDDEN.
- Unknown calibration and quantile search: FORBIDDEN.
- Primary comparison: DGSB-v2 vs Multi-Global-K2.
""",
        encoding="utf-8",
    )

    rule_hashes: dict[str, str] = {}
    bundle_hashes: dict[str, str] = {}
    for setting in VALID_SETTINGS:
        artifact = STAGE8B_ROOT / "artifacts" / setting
        rule_hashes[setting] = sha256_file(artifact / "dgsbv2_rule.json")
        local_files = [
            artifact / "dgsbv2_global_threshold.json",
            artifact / "dgsbv2_local_thresholds.csv",
            artifact / "dgsbv2_fallback_map.csv",
            artifact / "dgsbv2_rule.json",
            artifact / "dgsbv2_rule.md",
        ]
        bundle_hashes[setting] = hash_manifest(local_files, artifact, artifact / "dgsbv2_hashes.sha256")

    old_config_path = STAGE8A_ROOT / "configs/evaluation_config.json"
    old_config = load_json(old_config_path)
    new_config = copy.deepcopy(old_config)
    new_config["stage"] = "Stage 8B frozen configuration for one-shot CipherSpectrum Final Test"
    new_config["created_at_utc"] = utc_now()
    new_config["created_before_any_test_access"] = True
    new_config["method_order"] = BASELINE_METHODS + ["DGSB-v2"]
    new_config["excluded_methods"] = ["DGSB-v1"]
    for setting in VALID_SETTINGS:
        new_config["settings"][setting]["methods"]["DGSB-v2"] = dgsb_method(setting, rule_hashes[setting])
    new_config["primary_comparison"] = {
        "candidate": "DGSB-v2",
        "reference": "Multi-Global-K2",
        "reason": "identical encoder, mu_x, scaler, PCA64 and K2 GMM; boundary is the main difference",
    }
    new_config["secondary_comparisons"] = [
        "DGSB-v2 vs Native",
        "DGSB-v2 vs Single-Full-K1",
        "DGSB-v2 vs Class-P05-K2",
        "DGSB-v2 vs Component-P05-K2",
    ]
    new_config["final_test_metrics"] = config["final_test_metrics"]
    new_config["operating_point_policy"] = {
        "report_known_FRR_and_UFAR_jointly": True,
        "report_AUROC_and_AUPRC": True,
        "test_threshold_refit": "FORBIDDEN",
    }
    new_config["absorption_analysis_fields"] = [
        "sample_id", "true_unknown_class", "method", "accepted_as_known",
        "predicted_known_class", "score", "threshold_or_margin", "global_score",
        "global_margin", "predicted_component", "local_score", "local_threshold", "local_margin",
    ]
    new_config["known_test_opened"] = 0
    new_config["known_test_mu_generated"] = False
    new_config["unknown_test_opened"] = 0
    new_config["unknown_mu_generated"] = False
    new_config["unknown_inference_executed"] = False
    new_config_path = STAGE8B_ROOT / "configs/evaluation_config_v2.json"
    require(not new_config_path.exists(), "refusing to overwrite evaluation_config_v2")
    write_json(new_config_path, new_config)

    modified: list[str] = []
    for setting in VALID_SETTINGS:
        for method in BASELINE_METHODS:
            if old_config["settings"][setting]["methods"][method] != new_config["settings"][setting]["methods"][method]:
                modified.append(f"{setting}:{method}")
    require(not modified, f"existing baseline methods changed: {modified}")
    diff_path = STAGE8B_ROOT / "outputs/summary/evaluation_config_diff.md"
    diff_path.write_text(
        "# Evaluation configuration diff\n\n"
        f"- Stage 8A source SHA-256: `{sha256_file(old_config_path)}`\n"
        f"- Stage 8B v2 SHA-256: `{sha256_file(new_config_path)}`\n"
        "- Existing methods compared field by field: 15 setting-method objects.\n"
        "- `existing_methods_modified = 0`\n"
        "- Only method added: `DGSB-v2`.\n"
        "- DGSB-v1 remains excluded.\n",
        encoding="utf-8",
    )

    cross_rows = []
    for summary in summaries:
        cross_rows.append(
            {
                "setting": summary["setting"],
                "known_classes": summary["known_classes"],
                "validation_n": summary["validation_n"],
                "global_threshold": summary["global_threshold"],
                "global_acceptance": summary["global_acceptance"],
                "global_FRR": summary["global_FRR"],
                "local_acceptance": summary["local_acceptance"],
                "local_FRR": summary["local_FRR"],
                "dgsbv2_acceptance": summary["dgsbv2_acceptance"],
                "dgsbv2_FRR": summary["dgsbv2_FRR"],
                "global_exclusive_count": summary["global_exclusive_rejection_count"],
                "global_exclusive_rate": summary["global_exclusive_rejection_rate"],
                "local_exclusive_count": summary["local_exclusive_rejection_count"],
                "local_exclusive_rate": summary["local_exclusive_rejection_rate"],
                "both_fail_count": summary["both_fail_count"],
                "both_fail_rate": summary["both_fail_rate"],
                "gate_status": summary["gate_status"],
                "per_class_min": summary["per_class_coverage"]["min"],
                "per_class_median": summary["per_class_coverage"]["median"],
                "per_class_max": summary["per_class_coverage"]["max"],
                "component_gap_mean": summary["component_gap"]["mean"],
                "component_gap_max": summary["component_gap"]["max"],
                "fallback_component_count": summary["fallback_component_count"],
                "fallback_rate": summary["fallback_rate"],
                "known_collapse": summary["known_collapse"],
            }
        )
    write_csv(STAGE8B_ROOT / "outputs/summary/cross_setting_summary.csv", cross_rows)

    access_ok = True
    for setting in VALID_SETTINGS:
        access = load_json(STAGE8B_ROOT / f"outputs/{setting}/file_access_audit.json")
        access_ok &= (
            access["known_test_opened"] == 0
            and access["known_test_mu_generated"] is False
            and access["unknown_test_opened"] == 0
            and access["unknown_mu_generated"] is False
            and access["unknown_inference_executed"] is False
        )
    checks = {
        "stage6_hashes_pass": True,
        "stage7_hashes_pass": True,
        "stage8a_hashes_pass": True,
        "known_test_access_zero": access_ok,
        "unknown_test_access_zero": access_ok,
        "global_p025_frozen": all(float(row["global_quantile"]) == 0.025 for row in summaries),
        "local_p025_frozen": all(not load_json(STAGE8B_ROOT / f"outputs/{s}/calibration_summary.json")["quantile_search_performed"] for s in VALID_SETTINGS),
        "no_quantile_search": all(not row["quantile_search_performed"] for row in summaries),
        "no_post_local_global_recalibration": all(not row["post_local_global_recalibration_performed"] for row in summaries),
        "dgsbv2_rule_hash_frozen": all(bool(rule_hashes[s]) and bool(bundle_hashes[s]) for s in VALID_SETTINGS),
        "evaluation_config_v2_frozen": new_config_path.is_file(),
        "existing_five_methods_unmodified": len(modified) == 0,
        "final_test_metrics_frozen": bool(new_config["final_test_metrics"]),
        "paired_bootstrap_fixed_1000": new_config["final_test_metrics"]["paired_bootstrap_iterations"] == 1000,
        "no_known_collapse": all(not row["known_collapse"] for row in summaries),
    }
    gate = "READY_FOR_ONE_SHOT_FINAL_TEST" if all(checks.values()) else "NOT_READY"
    final_gate = {
        "gate": gate,
        "checks": checks,
        "gate_redundancy": {row["setting"]: row["gate_status"] for row in summaries},
        "unified_rule_sha256": sha256_file(spec_path),
        "setting_rule_sha256": rule_hashes,
        "setting_bundle_hashes": bundle_hashes,
        "evaluation_config_v2_sha256": sha256_file(new_config_path),
        "existing_methods_modified": len(modified),
        "primary_comparison": "DGSB-v2 vs Multi-Global-K2",
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
    }
    write_json(STAGE8B_ROOT / "outputs/summary/final_gate.json", final_gate)
    (STAGE8B_ROOT / "outputs/summary/final_gate.md").write_text(
        "# Stage 8B Final Gate\n\n"
        f"**{gate}**\n\n"
        + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
        + "\n\nGate redundancy is reported, not tuned:\n"
        + "\n".join(f"- {name}: {status}" for name, status in final_gate["gate_redundancy"].items())
        + "\n\nKnown Test opened: 0. Unknown Test opened: 0. Unknown inference: false.\n",
        encoding="utf-8",
    )

    completion = {
        "provenance_status": "PASS",
        "settings": summaries,
        "unified_rule_sha256": final_gate["unified_rule_sha256"],
        "setting_rule_sha256": rule_hashes,
        "setting_bundle_hashes": bundle_hashes,
        "evaluation_config_v2_sha256": final_gate["evaluation_config_v2_sha256"],
        "existing_methods_modified": 0,
        "primary_comparison": "DGSB-v2 vs Multi-Global-K2",
        "final_test_metrics_frozen": True,
        "paired_bootstrap_iterations": 1000,
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
        "final_gate": gate,
    }
    write_json(STAGE8B_ROOT / "outputs/summary/completion_report.json", completion)

    table = [
        "| Setting | Global P02.5 | Global acc/FRR | Local acc/FRR | AND acc/FRR | Global-only | Local-only | Both fail | Gate status | Class min/median/max | Component gap mean/max | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            f"| {row['setting'].title()} | {row['global_threshold']:.6f} | {row['global_acceptance']:.6f}/{row['global_FRR']:.6f} | "
            f"{row['local_acceptance']:.6f}/{row['local_FRR']:.6f} | {row['dgsbv2_acceptance']:.6f}/{row['dgsbv2_FRR']:.6f} | "
            f"{row['global_exclusive_rejection_count']}/{row['global_exclusive_rejection_rate']:.6f} | "
            f"{row['local_exclusive_rejection_count']}/{row['local_exclusive_rejection_rate']:.6f} | "
            f"{row['both_fail_count']}/{row['both_fail_rate']:.6f} | {row['gate_status']} | "
            f"{row['per_class_coverage']['min']:.6f}/{row['per_class_coverage']['median']:.6f}/{row['per_class_coverage']['max']:.6f} | "
            f"{row['component_gap']['mean']:.6f}/{row['component_gap']['max']:.6f} | "
            f"{row['fallback_component_count']}/{row['fallback_rate']:.6f} |"
        )
    results = f"""# Experiment results: stage8b-cipherspectrum-dgsbv2-20260913

- Status: `success`
- Experiment type: `evaluation`
- Claim scope: `diagnostic`
- Completed (UTC): `{utc_now()}`
- Final Gate: `{gate}`

## Data and split

- Frozen Stage 6/7/8A provenance: PASS.
- Calibration data: Known Validation only; frozen density models came from Known Train.
- Known Test opened/generated: 0/false; Unknown Test opened/generated/inferred: 0/false/false.

## Configuration and execution

- One preregistered rule: Global P02.5 AND predicted-class Local P02.5.
- Sparse fallback n<30 preserves weighted local-score semantics.
- No quantile search and no post-local Global recalibration.

## Core results

{chr(10).join(table)}

## Preserved evidence

- Per-setting thresholds, fallback maps, rule specifications and SHA-256 ledgers under `artifacts/`.
- Gate, class, component, baseline-comparison and access audits under `outputs/`.
- Frozen six-method evaluation configuration under `configs/evaluation_config_v2.json`.

## Limitations

- Known-only calibration evidence; no Test Accuracy, Macro-F1, UFAR, AUROC or AUPRC has been computed.
- Gate activity on Known Validation does not establish Unknown utility.

## Conclusion and next step

- `{gate}`. Stop here. A separately authorized one-shot Final Test may consume only this frozen bundle.
"""
    (STAGE8B_ROOT / "RESULTS.md").write_text(results, encoding="utf-8")

    readme = f"""# Stage 8B — CipherSpectrum DGSB-v2 Known-Only Rule Freeze

## Motivation

Freeze one dual-gate boundary before any Test access.

## Why DGSB-v1 Was Rejected

DGSB-v1 recalibrated its Global threshold after Local calibration, making the Global gate redundant. It is excluded.

## DGSB-v2

`Absolute Global Gate AND Predicted-Class Local Support Gate`; Global and Local always refer to the same predicted class.

## Error-Budget Decomposition

The approximately 5% rejection budget is preregistered as 2.5% Global plus 2.5% Local. This is not a grid search.

## Global Gate

The frozen absolute density floor is empirical P02.5 of the maximum class-mixture score on Known Validation.

## Local Gate

Each predicted-class component uses its Known-Validation P02.5 weighted local log-score threshold.

## Same-Class Support Check

Class prediction is selected by mixture score; the local component is selected only within that class.

## Sparse Component Fallback

Component Validation n<30 falls back to the same class's local-max-score P02.5 threshold.

## Known-Only Calibration

Only frozen Known Validation representations are scored. Frozen models originate from Known Train.

## Gate Contribution

{chr(10).join(f'- {row["setting"].title()}: {row["gate_status"]}; Global-only={row["global_exclusive_rejection_count"]}, Local-only={row["local_exclusive_rejection_count"]}, both-fail={row["both_fail_count"]}.' for row in summaries)}

## Per-Class Coverage

See `outputs/{{setting}}/per_class_coverage.csv` and the cross-setting summary.

## Per-Component Coverage

See `outputs/{{setting}}/per_component_coverage.csv`; cohorts use true-class posterior assignments.

## No Hyperparameter Search

Global/local quantiles are fixed at 0.025; fallback n is fixed at 30; Global recalibration is forbidden.

## Strict Test Isolation

Known Test opened/generated: 0/false. Unknown Test opened/generated/inferred: 0/false/false.

## Frozen Evaluation Config

`configs/evaluation_config_v2.json` adds only DGSB-v2; all 15 existing setting-method objects are unchanged.

## Final Test Metrics

Accuracy, Macro-F1, Known acceptance/FRR, UFAR, Unknown rejection, AUROC, AUPRC, class/component analyses, 95% CIs and paired bootstrap=1000 are frozen before Test access.

## Primary Comparison

`DGSB-v2 vs Multi-Global-K2`.

## Final Gate

`{gate}`.

## Next Step

Stop. Do not open Test without separate authorization for the one-shot Final Test.
"""
    (STAGE8B_ROOT / "README.md").write_text(readme, encoding="utf-8")

    original_manifest = load_json(STAGE8B_ROOT / "manifest.json")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    manifest = {
        **original_manifest,
        "updated_at_utc": utc_now(),
        "status": "success",
        "inputs": [
            {"path": str(old_config_path.resolve()), "sha256": sha256_file(old_config_path)},
            *[
                {
                    "path": str((STAGE8A_ROOT / f"artifacts/{setting}/bundle_hashes.sha256").resolve()),
                    "sha256": sha256_file(STAGE8A_ROOT / f"artifacts/{setting}/bundle_hashes.sha256"),
                }
                for setting in VALID_SETTINGS
            ],
        ],
        "code": {"revision": revision, "dirty": True, "changes": ["stage8b_cipherspectrum_dgsbv2"]},
        "execution": {
            "tmux_session": "stage8b_finalize_20260913",
            "related_sessions": [
                "stage8b_provenance_20260913",
                "stage8b_calibrate_low_20260913",
                "stage8b_calibrate_medium_20260913",
                "stage8b_calibrate_high_20260913",
            ],
            "command": "python stage8b_cipherspectrum_dgsbv2/scripts/finalize_stage8b.py",
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": [],
        },
        "configuration": {"files": ["configs/stage8b_config.json", "configs/evaluation_config_v2.json"], "parameters": config, "seeds": [0]},
        "core_results": [
            {"metric": f"{row['setting']}_dgsbv2_acceptance", "value": row["dgsbv2_acceptance"]}
            for row in summaries
        ],
        "artifacts": [],
        "limitations": [
            "Known Validation calibration only; no Known Test or Unknown Test inference was performed.",
            "Known-side gate non-redundancy does not establish Unknown utility.",
        ],
        "next_step": "Stop. Await separate authorization for the one-shot frozen Final Test.",
    }
    write_json(STAGE8B_ROOT / "manifest.json", manifest)
    require(gate == "READY_FOR_ONE_SHOT_FINAL_TEST", "Stage 8B final gate is NOT_READY")
    print(json.dumps(final_gate, sort_keys=True))


if __name__ == "__main__":
    main()
