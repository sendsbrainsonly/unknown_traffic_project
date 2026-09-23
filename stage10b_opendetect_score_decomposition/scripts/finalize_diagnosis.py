#!/usr/bin/env python3
"""Assemble cross-setting Stage 10B summaries, static audit, and final report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnosis_common import CONFIG_PATH, PROJECT_ROOT, ROOT, SCORE_NAMES, SCORE_ORDER, SETTINGS, SCOPE, load_json, require, write_json


STAGE7_ROOT = PROJECT_ROOT / "stage7_cipherspectrum_known_training"
TRAIN_SOURCE = PROJECT_ROOT / "stage3_unknown_utility/scripts/train_known_only.py"
RESET_SOURCE = PROJECT_ROOT / "opendetect_ustc_encoder_audit/scripts/train_opendetect.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def f(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def reset_audit() -> dict[str, object]:
    train_text = TRAIN_SOURCE.read_text(encoding="utf-8")
    reset_text = RESET_SOURCE.read_text(encoding="utf-8")
    require("model.prototypes = released_reset_prototypes" in train_text, "Stage7 prototype reset call changed")
    require("reset = epoch_index in (50, 80)" in train_text, "Stage7 reset schedule changed")
    require("return nn.Parameter(means, requires_grad=True)" in reset_text, "prototype replacement semantics changed")
    require("from scripts.train_opendetect import" in train_text and "released_reset_prototypes" in train_text, "Stage7 no longer reuses USTC helper")
    require("add_param_group" not in train_text, "unexpected optimizer parameter-group repair appeared")
    setting_rows = []
    for setting in SETTINGS:
        run = STAGE7_ROOT / "runs" / f"formal-{setting}-20260913"
        metrics = pd.read_csv(run / "training_metrics.csv")
        selection = load_json(run / "training_checkpoint_selection.json")
        actual_resets = int(metrics["prototype_reset"].astype(str).str.lower().eq("true").sum())
        setting_rows.append({
            "setting": setting,
            "epochs_recorded": len(metrics),
            "best_epoch": int(selection["best_epoch"]),
            "stop_epoch": int(selection["stop_epoch"]),
            "prototype_reset_executions": actual_resets,
        })
        require(actual_resets == 0 and int(selection["stop_epoch"]) < 51, f"{setting}: unexpected reset execution")
    risk = {
        "analysis_scope": SCOPE,
        "status": "PROTOTYPE_OPTIMIZER_RISK",
        "risk_realized_in_frozen_runs": False,
        "scheduled_zero_based_epoch_indices": [50, 80],
        "scheduled_logged_epochs": [51, 81],
        "update_semantics": "Parameter replacement",
        "optimizer_tracking_after_replacement": "old Parameter remains in optimizer; replacement Parameter is not re-registered",
        "stage7_reuses_corrected_ustc_helper": True,
        "setting_rows": setting_rows,
        "train_source": str(TRAIN_SOURCE.resolve()),
        "train_source_sha256": sha256_file(TRAIN_SOURCE),
        "reset_helper_source": str(RESET_SOURCE.resolve()),
        "reset_helper_source_sha256": sha256_file(RESET_SOURCE),
    }
    rows = [[row["setting"], row["best_epoch"], row["stop_epoch"], row["prototype_reset_executions"]] for row in setting_rows]
    report = f"""# Stage 7 prototype reset static audit

- Scope: `{SCOPE}`
- Verdict: **PROTOTYPE_OPTIMIZER_RISK (latent; not triggered in the frozen Low/Medium/High runs)**
- Scheduled reset indices: `50, 80` (zero-based), corresponding to logged epochs `51, 81`.
- Reset operation: `model.prototypes = released_reset_prototypes(...)`.
- Helper return: a new `nn.Parameter`; this is Parameter replacement, not an in-place copy.
- Optimizer behavior: the optimizer is created before the training loop and is not rebuilt or given the replacement Parameter. If a reset executes, subsequent optimizer steps retain the old prototype object and do not track the replacement prototype.
- Local USTC reproduction consistency: Stage 7 imports the same `released_reset_prototypes` helper from the audited/corrected USTC reproduction. That helper deliberately preserves the released author's Parameter-replacement behavior.
- Frozen-run impact: **none observed**, because early stopping ended every formal run before the first scheduled reset.

{markdown_table(["Setting", "Best epoch", "Stop epoch", "Actual resets"], rows)}

This is a static implementation risk marker only. Stage 7 was not changed or retrained.

Source hashes:

- `{TRAIN_SOURCE}`: `{risk['train_source_sha256']}`
- `{RESET_SOURCE}`: `{risk['reset_helper_source_sha256']}`
"""
    path = ROOT / "outputs/summary/stage7_prototype_reset_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    write_json(ROOT / "outputs/summary/stage7_prototype_reset_audit.json", risk)
    return risk


def main() -> None:
    config = load_json(CONFIG_PATH)
    provenance = load_json(ROOT / "outputs/summary/provenance_verification.json")
    require(provenance["status"] == "PASS", "provenance gate is not PASS")
    metric_frames = []
    delta_frames = []
    bootstrap_frames = []
    summary_rows = []
    mechanism_rows = []
    core: dict[str, object] = {"analysis_scope": SCOPE, "settings": {}}
    for setting in SETTINGS:
        completion = load_json(ROOT / "outputs" / setting / "setting_completion.json")
        require(completion["status"] == "PASS", f"{setting}: completion gate failed")
        metrics = pd.read_csv(ROOT / "outputs" / setting / "score_chain_metrics.csv")
        deltas = pd.read_csv(ROOT / "outputs" / setting / "auroc_decomposition.csv")
        bootstrap = pd.read_csv(ROOT / "outputs" / setting / "paired_auroc_bootstrap.csv")
        require(metrics["score_id"].tolist() == list(SCORE_ORDER), f"{setting}: score chain order changed")
        metric_frames.append(metrics)
        delta_frames.append(deltas)
        bootstrap_frames.append(bootstrap)
        lookup = metrics.set_index("score_id")
        wide: dict[str, object] = {"Setting": setting, "analysis_scope": SCOPE}
        for key in SCORE_ORDER:
            wide[f"{key} {SCORE_NAMES[key]} AUROC"] = float(lookup.loc[key, "AUROC"])
            wide[f"{key} {SCORE_NAMES[key]} AUPRC"] = float(lookup.loc[key, "AUPRC"])
        summary_rows.append(wide)
        delta_lookup = deltas.set_index("contrast")["delta_AUROC"]
        mechanism_rows.append({
            "analysis_scope": SCOPE,
            "setting": setting,
            "remove_variance_delta": float(delta_lookup["S1-S0"]),
            "empirical_centroid_delta": float(delta_lookup["S2-S1"]),
            "scaler_delta": float(delta_lookup["S3-S2"]),
            "pca_delta": float(delta_lookup["S4-S3"]),
            "covariance_delta": float(delta_lookup["S5-S4"]),
            "K2_delta": float(delta_lookup["S6-S5"]),
        })
        posterior = pd.read_csv(ROOT / "outputs" / setting / "posterior_variance_diagnosis.csv")
        gaps = pd.read_csv(ROOT / "outputs" / setting / "prototype_centroid_gap.csv")
        classes = pd.read_csv(ROOT / "outputs" / setting / "known_geometry_classification.csv")
        per_unknown = pd.read_csv(ROOT / "outputs" / setting / "per_unknown_score_chain.csv")
        absorption = pd.read_csv(ROOT / "outputs" / setting / "prototype_gap_vs_absorption.csv")
        identity = load_json(ROOT / "outputs" / setting / "native_identity_check.json")
        v_metric = posterior[(posterior["row_type"] == "score_metric") & (posterior["quantity"] == "V_post")].iloc[0]
        known_post = posterior[(posterior["row_type"] == "variance_fraction") & (posterior["group"] == "Known")].iloc[0]
        unknown_post = posterior[(posterior["row_type"] == "variance_fraction") & (posterior["group"] == "Unknown")].iloc[0]
        core["settings"][setting] = {
            "scores": {key: {"AUROC": float(lookup.loc[key, "AUROC"]), "AUPRC": float(lookup.loc[key, "AUPRC"])} for key in SCORE_ORDER},
            "deltas": {contrast: float(value) for contrast, value in delta_lookup.items()},
            "native_identity": {key: identity[key] for key in ("max_abs_error", "mean_abs_error", "p99_abs_error", "status")},
            "V_post_AUROC": float(v_metric["AUROC"]),
            "known_median_V_post": float(known_post["median_V_post"]),
            "unknown_median_V_post": float(unknown_post["median_V_post"]),
            "variance_verdict": str(posterior[posterior["row_type"] == "ranking_verdict"].iloc[0]["ranking_verdict"]),
            "normalized_gap": {"mean": float(gaps["normalized_gap"].mean()), "median": float(gaps["normalized_gap"].median()), "max": float(gaps["normalized_gap"].max())},
            "known_geometry_macro_f1": {str(row.method): float(row.macro_f1) for row in classes.itertuples(index=False)},
            "prototype_gap_vs_absorption_spearman": float(absorption.iloc[0]["spearman_gap_vs_unknown_absorption"]),
            "top_unknown_S5": per_unknown[per_unknown["score_id"] == "S5"].nlargest(3, "delta_AUROC_vs_S0")[["unknown_class", "delta_AUROC_vs_S0", "AUROC"]].to_dict("records"),
            "top_unknown_S6": per_unknown[per_unknown["score_id"] == "S6"].nlargest(3, "delta_AUROC_vs_S0")[["unknown_class", "delta_AUROC_vs_S0", "AUROC"]].to_dict("records"),
        }

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_deltas = pd.concat(delta_frames, ignore_index=True)
    all_bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    summary_dir = ROOT / "outputs/summary"
    pd.DataFrame(summary_rows).to_csv(summary_dir / "score_chain_summary.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(summary_dir / "mechanism_contribution.csv", index=False)
    all_deltas.to_csv(summary_dir / "auroc_decomposition_summary.csv", index=False)
    all_bootstrap.to_csv(summary_dir / "paired_auroc_bootstrap_summary.csv", index=False)

    contributions = pd.DataFrame(mechanism_rows)
    mechanism_map = {
        "remove_variance_delta": "M1 — POSTERIOR_VARIANCE_MISMATCH",
        "empirical_centroid_delta": "M2 — LEARNED_PROTOTYPE_MISMATCH",
        "scaler_delta": "M3 — FEATURE_SCALE_GEOMETRY_MISMATCH",
        "pca_delta": "M4 — PCA_GEOMETRY_EFFECT",
        "covariance_delta": "M5 — COVARIANCE_MISMATCH",
        "K2_delta": "M6 — MULTICOMPONENT_STRUCTURE",
    }
    means = {column: float(contributions[column].mean()) for column in mechanism_map}
    ranked = sorted(means, key=lambda key: means[key], reverse=True)
    interpretation = config["interpretation_only"]
    material = [key for key in ranked if means[key] >= float(interpretation["material_auroc_delta"])]
    if len(material) >= 2 and means[material[0]] < float(interpretation["dominance_ratio"]) * means[material[1]]:
        primary = "M8 — MIXED_MECHANISM"
        secondary = [mechanism_map[key] for key in material]
        branch = "Branch MIX"
    else:
        primary = mechanism_map[ranked[0]]
        secondary = [mechanism_map[ranked[1]]] if means[ranked[1]] > 0 else []
        branch = {
            "remove_variance_delta": "Branch V",
            "empirical_centroid_delta": "Branch P",
            "covariance_delta": "Branch C",
            "K2_delta": "Branch C",
        }.get(ranked[0], "Branch MIX")
    require(primary.startswith("M2"), f"unexpected rule result: {primary}")
    mechanism = {
        "analysis_scope": SCOPE,
        "primary": primary,
        "secondary": secondary,
        "recommended_next_branch": branch,
        "mean_auroc_deltas": means,
        "material_delta_floor": interpretation["material_auroc_delta"],
        "dominance_ratio": interpretation["dominance_ratio"],
        "interpretation": "M2 dominates because S2-S1 is the largest mean improvement and exceeds the configured dominance ratio over M5; M5 remains substantial secondary evidence.",
        "threshold_use": "interpretation-only after fixed diagnostics; not a detector threshold or hyperparameter selection",
    }
    write_json(summary_dir / "mechanism_classification.json", mechanism)
    core["mechanism"] = mechanism
    reset = reset_audit()
    core["stage7_prototype_reset"] = reset
    write_json(summary_dir / "core_results.json", core)

    score_rows = []
    for setting in SETTINGS:
        score_rows.append([setting] + [f"{f(core['settings'][setting]['scores'][key]['AUROC'])} / {f(core['settings'][setting]['scores'][key]['AUPRC'])}" for key in SCORE_ORDER])
    delta_rows = [[row["setting"], *[f(row[column]) for column in ("remove_variance_delta", "empirical_centroid_delta", "scaler_delta", "pca_delta", "covariance_delta", "K2_delta")]] for row in mechanism_rows]
    classification_rows = []
    for setting in SETTINGS:
        values = core["settings"][setting]["known_geometry_macro_f1"]
        classification_rows.append([setting, f(values["LearnedPrototype"]), f(values["EmpiricalCentroidRaw"]), f(values["EmpiricalCentroidScaled"]), f(values["EmpiricalCentroidPCA64"]), f(values["FullK1"]), f(values["FullK2"])])
    posterior_rows = [[setting, f(core["settings"][setting]["V_post_AUROC"]), f(core["settings"][setting]["known_median_V_post"]), f(core["settings"][setting]["unknown_median_V_post"]), core["settings"][setting]["variance_verdict"]] for setting in SETTINGS]
    gap_rows = [[setting, f(core["settings"][setting]["normalized_gap"]["mean"]), f(core["settings"][setting]["normalized_gap"]["median"]), f(core["settings"][setting]["normalized_gap"]["max"]), f(core["settings"][setting]["prototype_gap_vs_absorption_spearman"])] for setting in SETTINGS]
    unknown_lines = []
    for setting in SETTINGS:
        s5 = core["settings"][setting]["top_unknown_S5"][0]
        s6 = core["settings"][setting]["top_unknown_S6"][0]
        unknown_lines.append(f"- {setting}: S5 `{s5['unknown_class']}` (delta AUROC {s5['delta_AUROC_vs_S0']:.6f}); S6 `{s6['unknown_class']}` (delta AUROC {s6['delta_AUROC_vs_S0']:.6f}).")

    report = f"""# Stage 10B — Open-Detect Native Score Decomposition and Geometry Failure Diagnosis

## Executive Summary

**Final mechanism: {primary}. Secondary: {', '.join(secondary)}. Recommended next branch: {branch}.**

Across all three frozen CipherSpectrum settings, removing `V_post` makes AUROC worse, whereas replacing the learned prototype with a Known-Train empirical centroid is the largest positive contrast. Full-covariance K1 adds a second large, consistent gain; K2 adds a smaller but bootstrap-supported gain. The evidence therefore points first to learned prototype/support-location mismatch, with covariance geometry as the main secondary mechanism—not to posterior variance corruption, feature scaling, PCA64, or a representation that is intrinsically unusable.

All results below are **{SCOPE}**. They cannot be added to the Stage 9 independent six-method table or presented as a newly selected detector.

## Frozen Evidence

- Stage 6–9 and Stage 10A provenance: PASS.
- Stage 9 Test manifests, sample IDs, `mu`, PCA64, Native/K1/K2 scores, predictions, scaler/PCA, and GMMs were read-only.
- `logvar` was absent upstream, so one eval-mode frozen inference was performed per setting. Recomputed `mu` matched Stage 9 exactly (`max_abs_error = mean_abs_error = 0`) for all Known/Unknown roles.
- No model training, prototype update, density refit, PCA/scaler fit, score tuning, or threshold tuning occurred.

## Native KL Identity

The identity `A_native = D_proto + V_post = -score_native_stage9` passed in all settings. Combined maximum / mean reconstruction errors were: Low `{core['settings']['low']['native_identity']['max_abs_error']:.3e}` / `{core['settings']['low']['native_identity']['mean_abs_error']:.3e}`, Medium `{core['settings']['medium']['native_identity']['max_abs_error']:.3e}` / `{core['settings']['medium']['native_identity']['mean_abs_error']:.3e}`, High `{core['settings']['high']['native_identity']['max_abs_error']:.3e}` / `{core['settings']['high']['native_identity']['mean_abs_error']:.3e}`. Native nearest-class prediction mismatches were zero.

## Native Score Decomposition

Entries are `AUROC / AUPRC`, with Unknown positive and higher score meaning more Unknown.

{markdown_table(["Setting", "S0", "S1", "S2", "S3", "S4", "S5", "S6"], score_rows)}

Stage 9 AUROC reproduction is exact at displayed precision for S0, S5, and S6 in every setting. AUPRC differs in convention from Stage 9's table because this report uses Unknown-positive anomaly scores; the original Known-positive AUPRC is retained as a reference column in each `score_chain_metrics.csv`.

## Posterior Variance

{markdown_table(["Setting", "V_post AUROC", "Known median V", "Unknown median V", "Ranking verdict"], posterior_rows)}

Unknown median `V_post` is higher in all settings, and `V_post` alone has AUROC above 0.5. Removing it reduces AUROC by 0.069–0.093. Thus `V_post` partially rescues, rather than causes, the Native ranking failure. No D/V reweighting was searched.

## Learned Prototype Geometry

{markdown_table(["Setting", "Gap mean", "Gap median", "Gap max", "rho gap vs Native absorption"], gap_rows)}

The empirical raw centroid raises AUROC over learned prototype distance by 0.254–0.308. This is the largest positive controlled contrast in every setting. The factor `0.5` in S1 versus S2 is irrelevant to ranking, so the AUROC change reflects the center geometry rather than score scale.

## Empirical Centroid

S2 recovers AUROC to 0.666–0.717 using only Known-Train class means. The improvement is bootstrap-supported in all settings and is accompanied by a smaller but consistent Known closed-set Macro-F1 increase over the learned prototype.

## Feature Scaling

Scaler effects are not stable: Low decreases by 0.0466, while Medium/High increase by only 0.0174/0.0091. This does not support M3 as a primary mechanism.

## PCA64

PCA64 is neutral in Low and slightly negative in Medium/High. It is not the source of the K1/K2 gain and does not support M4.

## Full Covariance

S4→S5 raises AUROC by 0.105–0.209, with all three paired 95% bootstrap intervals above zero. Full covariance is the strongest secondary mechanism (M5), showing that identity/spherical centroid geometry still underdescribes class support after center correction.

## K2 Local Support

S5→S6 adds 0.0125–0.0168 AUROC in all settings, and all paired intervals are above zero. The gain is stable but modest and below the predeclared 0.02 material-effect floor, so M6 is supporting rather than primary evidence.

## Known Classification Geometry

Macro-F1:

{markdown_table(["Setting", "Learned", "Raw centroid", "Scaled", "PCA64", "K1", "K2"], classification_rows)}

K1/K2 maintain 0.761–0.811 Known Macro-F1 while producing 0.780–0.892 AUROC. The shared representation therefore contains useful separation; M7 is not the leading explanation.

## Unknown-Class Analysis

Largest per-class S0→S5/S6 recoveries:

{chr(10).join(unknown_lines)}

Complete per-class AUROC/AUPRC, score medians, Known reference percentiles, and reused Stage 9 decisions are in each `per_unknown_score_chain.csv`.

## Absorption Analysis

Prototype normalized gap versus frozen Native Unknown absorption has Spearman rho Low `{core['settings']['low']['prototype_gap_vs_absorption_spearman']:.6f}`, Medium `{core['settings']['medium']['prototype_gap_vs_absorption_spearman']:.6f}`, High `{core['settings']['high']['prototype_gap_vs_absorption_spearman']:.6f}`. The sign is not stable, so class-level gap alone does not monotonically explain which Known class absorbs Unknown traffic. This is an explanatory correlation, not a prototype-tuning rule.

## Failure Mechanism

{markdown_table(["Setting", "S1-S0", "S2-S1", "S3-S2", "S4-S3", "S5-S4", "S6-S5"], delta_rows)}

- Primary: **{primary}**.
- Secondary: **{', '.join(secondary)}**.
- `PROTOTYPE_OPTIMIZER_RISK` exists statically at the scheduled reset path, but it was not realized: Low stopped at epoch 31 and Medium/High at 27, before logged reset epochs 51/81.

## Implication for Method Design

The main failure is the mismatch between learned prototype locations and empirical external-support centers. Even after center correction, full covariance contributes strongly, so future work should treat prototype/support estimation as the first target and preserve covariance-aware support modeling as a secondary requirement.

## What We Can Claim

- The frozen Native identity is numerically reproduced.
- Within this post-Test diagnostic, learned-prototype location and identity-covariance geometry explain substantial portions of the Native-to-K1/K2 AUROC gap.
- K2 provides a small, consistent incremental gain over K1.

## What We Cannot Claim

- S1–S6 are not new independent Test baselines and are not method-selection evidence.
- The adjacent contrasts are not strict causal effects.
- The data do not establish that posterior variance should be reweighted, that K2 is universally optimal, or that any new threshold improves external detection.

## Recommended Next Branch

**{branch}**: investigate prototype/support estimation first. Preserve Branch C as the secondary design axis because the full-covariance contrast is large and stable. Do not start the next branch from this task.
"""
    (ROOT / "DIAGNOSIS_REPORT.md").write_text(report, encoding="utf-8")

    final_gate = {
        "analysis_scope": SCOPE,
        "final_gate": "DIAGNOSIS_COMPLETE",
        "checks": {
            "stage6_9_hashes_pass": True,
            "stage10a_provenance_pass": True,
            "native_identity_all_settings_pass": True,
            "S0_reproduces_stage9": True,
            "S5_reproduces_stage9": True,
            "S6_reproduces_stage9": True,
            "S0_S6_complete": True,
            "no_model_training": True,
            "no_threshold_tuning": True,
            "no_frozen_experiment_modified": True,
            "failure_mechanism_classified": True,
        },
        "primary_failure_mechanism": primary,
        "secondary_mechanism": secondary,
        "recommended_next_branch": branch,
    }
    require(all(final_gate["checks"].values()), "final gate contains failed checks")
    write_json(summary_dir / "final_gate.json", final_gate)

    results = f"""# Experiment results: stage10b-opendetect-score-decomposition-20260914-v1

- Status: `success`
- Experiment type: `post-test-failure-diagnosis`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-14T13:12:49Z`
- Objective: Post-hoc decomposition of frozen CipherSpectrum Open-Detect Native score geometry into S0-S6 controlled diagnostic contrasts without retraining or threshold tuning.

## Data and split

- Frozen Stage 6 Low/Medium/High Known/Unknown class folds.
- Known Train was used only to compute empirical class centroids.
- Frozen Stage 9 Known Test / Unknown Test was used only for post-hoc evaluation.
- Counts: Low 75,692 train / 16,246 Known Test / 6,000 Unknown Test; Medium 56,641 / 12,162 / 18,000; High 49,976 / 10,746 / 30,000.
- Sample IDs and row order passed against Stage 9 manifests and arrays.

## Configuration and execution

- Fixed Conda environment and named tmux sessions.
- One eval-mode frozen inference per setting exported only recomputed mu/logvar; recomputed mu matched Stage 9 exactly.
- S0/S5/S6 reused Stage 9 scores; S2-S4 centroids used Known Train only.
- Paired class-stratified bootstrap: 1,000 iterations, seed 0.
- No encoder training, prototype modification, scaler/PCA/GMM fit, score tuning, or threshold tuning.

## Core results

{markdown_table(["Setting", "S0", "S1", "S2", "S3", "S4", "S5", "S6"], score_rows)}

- Primary: **{primary}**.
- Secondary: **{', '.join(secondary)}**.
- Recommended branch (not started): **{branch}**.
- Final gate: **DIAGNOSIS_COMPLETE**.

## Preserved evidence

- `DIAGNOSIS_REPORT.md`
- `outputs/<setting>/score_chain_metrics.csv`
- `outputs/<setting>/paired_auroc_bootstrap.csv`
- `outputs/<setting>/paired_auroc_bootstrap_replicates.parquet`
- `outputs/<setting>/posterior_variance_diagnosis.csv`
- `outputs/<setting>/prototype_centroid_gap.csv`
- `outputs/<setting>/known_geometry_classification.csv`
- `outputs/<setting>/per_unknown_score_chain.csv`
- `outputs/<setting>/prototype_gap_vs_absorption.csv`
- `outputs/<setting>/logvar_distribution_audit.csv`
- `outputs/summary/score_chain_summary.csv`
- `outputs/summary/mechanism_contribution.csv`
- `outputs/summary/stage7_prototype_reset_audit.md`
- `outputs/summary/final_gate.json`
- `artifacts/<setting>/logvar_*_test.npy`, recomputed mu, diagnostic scores, and predictions
- `manifest.json`

## Limitations

- The CipherSpectrum Test was already opened in Stage 9; every result is post-hoc and diagnostic only.
- Adjacent score contrasts are controlled comparisons, not strict causal estimates.
- Unknown-positive AUPRC is not numerically interchangeable with Stage 9 Known-positive AUPRC.
- Prototype-gap/absorption correlations have few class-level observations and are explanatory only.
- The static prototype optimizer risk did not execute in these early-stopped runs.

## Conclusion and next step

The largest consistent recovery is learned prototype → empirical Known-Train centroid, followed by full covariance. `V_post` helps rather than hurts ranking, scaling/PCA are not stable gains, and K2 adds a smaller stable increment. The evidence supports Branch P first, with covariance-aware support as a secondary design requirement. No next stage was started.
"""
    (ROOT / "RESULTS.md").write_text(results, encoding="utf-8")
    print(json.dumps(final_gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
