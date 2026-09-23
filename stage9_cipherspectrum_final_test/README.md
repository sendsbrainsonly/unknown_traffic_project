# Stage 9 — CipherSpectrum One-Shot Final Test

## Purpose

Run the first strict external Known Test plus Unknown Test evaluation for six methods frozen in Stages 6–8B.

## One-Shot Test Principle

Test opened once at `2026-09-13T14:30:55.571231+00:00`. No method, threshold, split, or bootstrap rule changed afterward.

## Frozen Upstream Assets

Stage 6 protocol, three Stage 7 checkpoints, Stage 8A baseline bundles, Stage 8B DGSB-v2 bundles, and `evaluation_config_v2.json` all passed SHA-256 verification before Test access.

## Low / Medium / High

All three primary settings ran in the fixed order Low, Medium, High and are reported separately.

## Known Test

See `outputs/{setting}/known_test_metrics.csv` and `known_test_per_class_metrics.csv`.

## Unknown Test

See `outputs/{setting}/unknown_test_metrics.csv`.

## Six Frozen Methods

Native, Single-Full-K1, Multi-Global-K2, Class-P05-K2, Component-P05-K2, and DGSB-v2.

## Primary Comparison

DGSB-v2 versus Multi-Global-K2. Per-setting conclusions are `{'low': 'TRADEOFF', 'medium': 'TRADEOFF', 'high': 'TRADEOFF'}`.

## Known FRR

Known FRR is reported jointly with UFAR; it is never omitted from operating-point interpretation.

## UFAR

Overall and per-Unknown-class false acceptance rates are preserved for every method and setting.

## AUROC / AUPRC

Known is the positive class and higher score means more Known. DGSB-v2 uses the frozen AND-equivalent score `min(global_margin, local_margin)`.

## Bootstrap

Primary paired class-stratified bootstrap used exactly 1000 iterations, seed 0, and 95% percentile intervals.

## Per-Unknown-Class Results

See `outputs/{setting}/per_unknown_class_ufar.csv`.

## Absorption Analysis

Complete Unknown-to-Known matrices and difficulty/density correlations are preserved per setting.

## DGSB Gate Attribution

Known and Unknown Test are each decomposed into Global-only fail, Local-only fail, Both fail, and Both pass.

## Operating-Point Interpretation

Known FRR, UFAR, AUROC, and AUPRC must be interpreted together. Final outcome: `B — OPERATING-POINT TRADEOFF`.

## Final Gate

`B — OPERATING-POINT TRADEOFF`.

## Validity Boundary

This is a strict local frozen-split evaluation, not an author-exact reproduction. Directory-derived labels and endpoint/domain shortcuts limit external validity.

## No Post-Test Tuning

No retraining, scaler/PCA/GMM fitting, threshold calibration, quantile/K search, or method redesign occurred.

## Next Step

Stop at Stage 9. No CSTNET or CICIDS task was started.
