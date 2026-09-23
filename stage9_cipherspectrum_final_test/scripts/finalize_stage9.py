#!/usr/bin/env python3
"""Aggregate Low/Medium/High, classify the final gate, document, and freeze Stage 9."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage9_common import (
    METHOD_ORDER,
    PROJECT_ROOT,
    STAGE8A_ROOT,
    STAGE8B_ROOT,
    STAGE9_ROOT,
    VALID_SETTINGS,
    assert_test_open_record,
    file_manifest,
    load_json,
    require,
    sha256_file,
    verify_all_provenance,
    write_csv,
    write_hash_lines,
    write_json,
)


def cross_gate(primary_rows: list[dict[str, object]]) -> tuple[str, str]:
    conclusions = [str(row["primary_conclusion"]) for row in primary_rows]
    deltas = [float(row["DeltaUFAR"]) for row in primary_rows]
    any_supported_frr_harm = any(float(row.get("Delta Known FRR CI low", float("-inf"))) > 0 for row in primary_rows)
    if any(value < 0 for value in deltas) and any(value > 0 for value in deltas):
        return "E", "MIXED EXTERNAL RESULT"
    if conclusions.count("WIN") >= 2 and all(value <= 0 for value in deltas) and not any_supported_frr_harm:
        return "A", "CONSISTENT EXTERNAL UTILITY"
    if conclusions.count("HARM") >= 2:
        return "D", "EXTERNAL HARM"
    if "TRADEOFF" in conclusions and "HARM" not in conclusions:
        return "B", "OPERATING-POINT TRADEOFF"
    if "WIN" not in conclusions and "HARM" not in conclusions:
        return "C", "NO EXTERNAL GAIN"
    return "E", "MIXED EXTERNAL RESULT"


def setting_bundle(setting: str) -> dict[str, object]:
    artifact = STAGE9_ROOT / "artifacts" / setting
    output = STAGE9_ROOT / "outputs" / setting
    manifest_path = artifact / "test_bundle_manifest.json"
    hash_path = artifact / "test_bundle_hashes.sha256"
    require(not manifest_path.exists() and not hash_path.exists(), f"{setting}: refusing to overwrite Test bundle freeze")
    paths = [path for root in (artifact, output) for path in root.rglob("*") if path.is_file()]
    manifest = {
        "setting": setting,
        "status": "FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_test_tuning_performed": False,
        "methods_modified_after_test_open": False,
        "files": file_manifest(paths, STAGE9_ROOT),
    }
    write_json(manifest_path, manifest)
    write_hash_lines(paths + [manifest_path], STAGE9_ROOT, hash_path)
    return {
        "setting": setting,
        "status": "PASS",
        "files_in_manifest": len(paths),
        "test_bundle_manifest_sha256": sha256_file(manifest_path),
        "test_bundle_hash_file_sha256": sha256_file(hash_path),
    }


def coverage_consistency(setting: str, cross: pd.DataFrame) -> list[dict[str, object]]:
    source = pd.read_csv(STAGE8A_ROOT / f"outputs/{setting}/known_val_boundary_per_class.csv")
    dgsb_path = STAGE8B_ROOT / f"outputs/{setting}/per_class_coverage.csv"
    if dgsb_path.exists():
        dgsb = pd.read_csv(dgsb_path)
        frr_column = "FRR" if "FRR" in dgsb.columns else "dgsbv2_frr"
        if frr_column in dgsb.columns:
            source = pd.concat([
                source,
                pd.DataFrame({"method": "DGSB-v2", "FRR": dgsb[frr_column].astype(float)}),
            ], ignore_index=True)
    rows = []
    multi_std = float(source[source["method"] == "Multi-Global-K2"]["FRR"].astype(float).std(ddof=0))
    multi_ufar = float(cross[cross["method"] == "Multi-Global-K2"]["UFAR"].iloc[0])
    for method in ("Class-P05-K2", "Component-P05-K2", "DGSB-v2"):
        subset = source[source["method"] == method]
        if subset.empty:
            continue
        method_std = float(subset["FRR"].astype(float).std(ddof=0))
        method_ufar = float(cross[cross["method"] == method]["UFAR"].iloc[0])
        rows.append({
            "setting": setting,
            "method": method,
            "multi_validation_per_class_FRR_std": multi_std,
            "method_validation_per_class_FRR_std": method_std,
            "coverage_consistency_improved": method_std < multi_std,
            "multi_test_UFAR": multi_ufar,
            "method_test_UFAR": method_ufar,
            "unknown_utility_worsened": method_ufar > multi_ufar,
            "consistency_improved_but_utility_worsened": method_std < multi_std and method_ufar > multi_ufar,
        })
    return rows


def append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in current:
        path.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def main() -> None:
    verify_all_provenance()
    test_open = assert_test_open_record()
    summary_dir = STAGE9_ROOT / "outputs/summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    require(not (summary_dir / "cross_setting_results.csv").exists(), "refusing to overwrite Stage 9 summary")

    cross_rows = []
    primary_rows = []
    analysis: dict[str, object] = {"settings": {}}
    for setting in VALID_SETTINGS:
        setting_summary = load_json(STAGE9_ROOT / f"outputs/{setting}/setting_summary.json")
        require(setting_summary["status"] == "COMPLETE", f"{setting}: evaluation incomplete")
        known = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/known_test_metrics.csv")
        unknown = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/unknown_test_metrics.csv")
        auc = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/detection_auc_metrics.csv")
        per_unknown = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/per_unknown_class_ufar.csv")
        merged = known.merge(unknown, on=["setting", "method"]).merge(auc, on=["setting", "method"])
        for _, row in merged.iterrows():
            classes = per_unknown[per_unknown["method"] == row["method"]]
            cross_rows.append({
                "setting": setting,
                "method": row["method"],
                "Known Accuracy": row["known_accuracy"],
                "Known Macro-F1": row["known_macro_f1"],
                "Known acceptance": row["known_acceptance"],
                "Known FRR": row["known_FRR"],
                "UFAR": row["UFAR"],
                "Unknown rejection": row["unknown_rejection_rate"],
                "AUROC": row["AUROC"],
                "AUPRC": row["AUPRC"],
                "per-unknown macro UFAR": classes["UFAR"].mean(),
                "per-unknown worst-class UFAR": classes["UFAR"].max(),
            })
        primary = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/primary_comparison.csv").iloc[0].to_dict()
        bootstrap = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/paired_bootstrap_primary.csv")
        for metric in ("DeltaUFAR", "Delta Known FRR", "DeltaAUROC", "DeltaAUPRC"):
            row = bootstrap[bootstrap["metric"] == metric].iloc[0]
            primary[f"{metric} CI low"] = float(row["ci_low"])
            primary[f"{metric} CI high"] = float(row["ci_high"])
        primary["bootstrap_iterations"] = 1000
        primary["seed"] = 0
        primary_rows.append(primary)

        dgsb_unknown = per_unknown[per_unknown["method"] == "DGSB-v2"].sort_values(["UFAR", "unknown_class"])
        absorption = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/unknown_to_known_absorption_matrix.csv")
        hubs = absorption.groupby(["method", "predicted_known_class"], as_index=False)["count"].sum().sort_values(["method", "count"], ascending=[True, False])
        difficulty = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/known_difficulty_vs_absorption.csv")
        dgsb_difficulty = difficulty[difficulty["method"] == "DGSB-v2"]
        density = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/density_vs_unknown_utility.csv")
        gate_u = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/dgsbv2_unknown_gate_attribution.csv")
        gate_k = pd.read_csv(STAGE9_ROOT / f"outputs/{setting}/dgsbv2_known_gate_attribution.csv")
        setting_cross = pd.DataFrame([row for row in cross_rows if row["setting"] == setting])
        analysis["settings"][setting] = {
            "best_unknown_classes_dgsbv2": dgsb_unknown.head(3)[["unknown_class", "UFAR"]].to_dict("records"),
            "worst_unknown_classes_dgsbv2": dgsb_unknown.tail(3).sort_values(["UFAR", "unknown_class"], ascending=[False, True])[["unknown_class", "UFAR"]].to_dict("records"),
            "unknown_class_ufar_ge_0_95": dgsb_unknown[dgsb_unknown["UFAR"] >= 0.95][["unknown_class", "UFAR"]].to_dict("records"),
            "top_absorption_hubs": {method: hubs[hubs["method"] == method].head(5)[["predicted_known_class", "count"]].to_dict("records") for method in METHOD_ORDER},
            "dgsbv2_known_f1_vs_absorption_rho": float(dgsb_difficulty["rho_f1_vs_count"].iloc[0]),
            "dgsbv2_known_accuracy_vs_absorption_rho": float(dgsb_difficulty["rho_accuracy_vs_count"].iloc[0]),
            "density_vs_unknown_utility": {comparison: {
                "rho": float(group["spearman_rho_count"].iloc[0]),
                "p": float(group["spearman_p_count"].iloc[0]),
            } for comparison, group in density.groupby("comparison")},
            "dgsbv2_unknown_gate_attribution": gate_u.to_dict("records"),
            "dgsbv2_known_gate_attribution": gate_k.to_dict("records"),
            "coverage_consistency_vs_utility": coverage_consistency(setting, setting_cross),
        }

    write_csv(summary_dir / "cross_setting_results.csv", cross_rows)
    write_csv(summary_dir / "primary_dgsbv2_vs_multi.csv", primary_rows)
    gate_code, gate_name = cross_gate(primary_rows)
    write_json(summary_dir / "analysis_summary.json", analysis)
    final_gate = {
        "gate": gate_code,
        "name": gate_name,
        "setting_conclusions": {row["setting"]: row["primary_conclusion"] for row in primary_rows},
        "test_open_started_at": test_open["test_open_started_at"],
        "settings_complete": list(VALID_SETTINGS),
        "methods_frozen": list(METHOD_ORDER),
        "bootstrap_iterations": 1000,
        "bootstrap_seed": 0,
        "post_test_tuning_performed": False,
        "methods_modified_after_test_open": False,
    }
    write_json(summary_dir / "final_gate.json", final_gate)
    (summary_dir / "final_gate.md").write_text(
        f"# Stage 9 Final Gate\n\n- Gate: `{gate_code} — {gate_name}`\n- Low: `{final_gate['setting_conclusions']['low']}`\n- Medium: `{final_gate['setting_conclusions']['medium']}`\n- High: `{final_gate['setting_conclusions']['high']}`\n- Post-Test tuning: `NO`\n- Methods remained frozen: `YES`\n",
        encoding="utf-8",
    )

    bundles = [setting_bundle(setting) for setting in VALID_SETTINGS]
    completion = {
        "status": "COMPLETE",
        "stage6_stage7_stage8a_stage8b_hashes": "PASS",
        "test_open_started_at": test_open["test_open_started_at"],
        "settings": list(VALID_SETTINGS),
        "methods": list(METHOD_ORDER),
        "final_gate": final_gate,
        "setting_bundles": bundles,
        "post_test_tuning_performed": False,
    }
    write_json(summary_dir / "completion_report.json", completion)

    cross = pd.DataFrame(cross_rows)
    primary_frame = pd.DataFrame(primary_rows)
    result_lines = [
        "# Experiment results: stage9-cipherspectrum-final-test-20260913",
        "",
        "- Status: `success`",
        "- Experiment type: `evaluation`",
        "- Claim scope: `approximate` (strict local frozen split; not author-equivalent)",
        f"- Test opened (UTC): `{test_open['test_open_started_at']}`",
        "- Objective: one-shot frozen external evaluation of six methods across Low, Medium, and High.",
        "",
        "## Data and split",
        "",
    ]
    for setting in VALID_SETTINGS:
        row = cross[(cross["setting"] == setting) & (cross["method"] == "DGSB-v2")].iloc[0]
        summary = load_json(STAGE9_ROOT / f"outputs/{setting}/setting_summary.json")
        result_lines.append(f"- {setting.title()}: Known Test n={summary['known_test_n']}; Unknown Test n={summary['unknown_test_n']}.")
    result_lines += [
        "",
        "## Configuration and execution",
        "",
        "- Six upstream-frozen methods; deterministic `mu_x`; frozen scaler/PCA64/K1/K2/thresholds.",
        "- AUROC/AUPRC positive class is Known; higher detector score means more Known.",
        "- Primary paired class-stratified bootstrap: 1000 iterations, seed 0.",
        "- No Test recalibration, search, retraining, or post-Test method change.",
        "",
        "## Core results",
        "",
        f"- Final Gate: `{gate_code} — {gate_name}`.",
    ]
    for _, row in primary_frame.iterrows():
        result_lines.append(
            f"- {str(row['setting']).title()}: Multi UFAR={float(row['Multi UFAR']):.6f}, DGSB UFAR={float(row['DGSB UFAR']):.6f}, DeltaUFAR={float(row['DeltaUFAR']):+.6f} "
            f"(95% CI [{float(row['DeltaUFAR CI low']):+.6f}, {float(row['DeltaUFAR CI high']):+.6f}]); conclusion `{row['primary_conclusion']}`."
        )
    result_lines += [
        "",
        "## Preserved evidence",
        "",
        "- `outputs/summary/cross_setting_results.csv`",
        "- `outputs/summary/primary_dgsbv2_vs_multi.csv`",
        "- `outputs/summary/final_gate.json`",
        "- `artifacts/{low,medium,high}/test_bundle_manifest.json`",
        "- `manifest.json`",
        "",
        "## Limitations",
        "",
        "- CipherSpectrum labels are directory-derived and the local frozen folds are not official author fold identities.",
        "- Domain/endpoint shortcuts remain a known validity risk; results do not establish semantic open-world generalization.",
        "- Correlation and absorption analyses are post-Test explanations only and were not used to modify methods.",
        "",
        "## Conclusion and next step",
        "",
        "- Stage 9 is frozen. No further dataset or method stage was started.",
    ]
    (STAGE9_ROOT / "RESULTS.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    readme = f"""# Stage 9 — CipherSpectrum One-Shot Final Test

## Purpose

Run the first strict external Known Test plus Unknown Test evaluation for six methods frozen in Stages 6–8B.

## One-Shot Test Principle

Test opened once at `{test_open['test_open_started_at']}`. No method, threshold, split, or bootstrap rule changed afterward.

## Frozen Upstream Assets

Stage 6 protocol, three Stage 7 checkpoints, Stage 8A baseline bundles, Stage 8B DGSB-v2 bundles, and `evaluation_config_v2.json` all passed SHA-256 verification before Test access.

## Low / Medium / High

All three primary settings ran in the fixed order Low, Medium, High and are reported separately.

## Known Test

See `outputs/{{setting}}/known_test_metrics.csv` and `known_test_per_class_metrics.csv`.

## Unknown Test

See `outputs/{{setting}}/unknown_test_metrics.csv`.

## Six Frozen Methods

Native, Single-Full-K1, Multi-Global-K2, Class-P05-K2, Component-P05-K2, and DGSB-v2.

## Primary Comparison

DGSB-v2 versus Multi-Global-K2. Per-setting conclusions are `{final_gate['setting_conclusions']}`.

## Known FRR

Known FRR is reported jointly with UFAR; it is never omitted from operating-point interpretation.

## UFAR

Overall and per-Unknown-class false acceptance rates are preserved for every method and setting.

## AUROC / AUPRC

Known is the positive class and higher score means more Known. DGSB-v2 uses the frozen AND-equivalent score `min(global_margin, local_margin)`.

## Bootstrap

Primary paired class-stratified bootstrap used exactly 1000 iterations, seed 0, and 95% percentile intervals.

## Per-Unknown-Class Results

See `outputs/{{setting}}/per_unknown_class_ufar.csv`.

## Absorption Analysis

Complete Unknown-to-Known matrices and difficulty/density correlations are preserved per setting.

## DGSB Gate Attribution

Known and Unknown Test are each decomposed into Global-only fail, Local-only fail, Both fail, and Both pass.

## Operating-Point Interpretation

Known FRR, UFAR, AUROC, and AUPRC must be interpreted together. Final outcome: `{gate_code} — {gate_name}`.

## Final Gate

`{gate_code} — {gate_name}`.

## Validity Boundary

This is a strict local frozen-split evaluation, not an author-exact reproduction. Directory-derived labels and endpoint/domain shortcuts limit external validity.

## No Post-Test Tuning

No retraining, scaler/PCA/GMM fitting, threshold calibration, quantile/K search, or method redesign occurred.

## Next Step

Stop at Stage 9. No CSTNET or CICIDS task was started.
"""
    (STAGE9_ROOT / "README.md").write_text(readme, encoding="utf-8")
    append_once(
        PROJECT_ROOT / "README.md",
        "## Stage 9 — CipherSpectrum One-Shot Final Test",
        f"## Stage 9 — CipherSpectrum One-Shot Final Test\n\nCompleted strict frozen Low/Medium/High external evaluation. Final result: `{gate_code} — {gate_name}`. Evidence: [`stage9_cipherspectrum_final_test/README.md`](stage9_cipherspectrum_final_test/README.md).",
    )
    append_once(
        PROJECT_ROOT / "EXPERIMENT_RESULTS.md",
        "stage9-cipherspectrum-final-test-20260913",
        f"- `stage9-cipherspectrum-final-test-20260913` — status `success`; strict frozen CipherSpectrum Low/Medium/High final Test; Gate `{gate_code} — {gate_name}`; evidence: [`stage9_cipherspectrum_final_test/RESULTS.md`](stage9_cipherspectrum_final_test/RESULTS.md).",
    )

    excluded = {summary_dir / "stage9_final_manifest.json", summary_dir / "stage9_final_hashes.sha256"}
    final_paths = [
        path for root in (STAGE9_ROOT / "artifacts", STAGE9_ROOT / "outputs")
        for path in root.rglob("*") if path.is_file() and path not in excluded
    ]
    final_manifest_path = summary_dir / "stage9_final_manifest.json"
    final_hash_path = summary_dir / "stage9_final_hashes.sha256"
    write_json(final_manifest_path, {
        "status": "FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_gate": gate_code,
        "files": file_manifest(final_paths, STAGE9_ROOT),
        "post_test_tuning_performed": False,
    })
    write_hash_lines(final_paths + [final_manifest_path], STAGE9_ROOT, final_hash_path)
    print(json.dumps({"status": "COMPLETE", "gate": gate_code, "gate_name": gate_name, "setting_bundles": bundles, "final_files": len(final_paths)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
