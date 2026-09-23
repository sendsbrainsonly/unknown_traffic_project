# Experiment results: stage10b-opendetect-score-decomposition-20260914-v1

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

| Setting | S0 | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|---|
| low | 0.500830 / 0.277508 | 0.431597 / 0.284080 | 0.717132 / 0.512625 | 0.670549 / 0.479113 | 0.670901 / 0.474392 | 0.880005 / 0.761724 | 0.892480 / 0.807311 |
| medium | 0.497970 / 0.585791 | 0.412627 / 0.523474 | 0.666464 / 0.697699 | 0.683861 / 0.710331 | 0.674294 / 0.701215 | 0.779753 / 0.857264 | 0.795385 / 0.869747 |
| high | 0.472133 / 0.695744 | 0.379376 / 0.659542 | 0.686885 / 0.831246 | 0.695970 / 0.831664 | 0.687189 / 0.826026 | 0.804478 / 0.923875 | 0.821250 / 0.929708 |

- Primary: **M2 — LEARNED_PROTOTYPE_MISMATCH**.
- Secondary: **M5 — COVARIANCE_MISMATCH**.
- Recommended branch (not started): **Branch P**.
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
