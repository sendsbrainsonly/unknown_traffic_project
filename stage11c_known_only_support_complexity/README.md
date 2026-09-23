# Stage 11C — Known-Only Support Complexity Diagnosis

## Goal

Diagnose whether frozen Known Train/Validation statistics can indicate when the Stage 11B Full-K1 support readout is preferable to the empirical-centroid readout.

## Stage11B Motivation

Stage 11B found context-dependent covariance value: Full-K1 harmed A1, materially helped A2, and only slightly helped A3. This stage explains that difference without creating a new detector.

## Why A1/A2/A3 Differ

The analysis contrasts covariance geometry, sample adequacy, bootstrap stability, Known-Validation fit, margins, classification and coverage over the same 15 frozen runs.

## Known-Only Constraint

Candidate features use only Known Train and Known Validation. Known/Unknown Test arrays are never loaded as features. Stage 11B R2-R1 outcomes are read only after feature construction and marked `OUTCOME_ONLY`.

## Covariance Geometry

Raw 128D and the frozen Stage 11B scaler plus PCA64 spaces report condition number, anisotropy, effective rank, spectral concentration, log determinant, trace and sample-to-dimension adequacy.

## Covariance Stability

PCA64 covariance stability uses exactly 100 train-only bootstrap resamples with seed 0 and fixed top-5/top-10 subspace diagnostics.

## Validation Density Fit

Train-MLE spherical Gaussian and the already-frozen Stage 11B Full-K1 model are compared on Known Validation. No new Full-K1 model is fitted.

## Validation Classification

Known-Validation centroid and frozen Full-K1 classification accuracy/Macro-F1 are compared; the fixed F1 rule margin is 0.005.

## Margin Analysis

Known Validation own-centroid versus nearest-other-centroid margins and ratios are measured in raw128 and PCA64 spaces.

## Coverage Analysis

The frozen Stage 11B Known-Validation P95 thresholds define per-class R1/R2 coverage; no threshold is recomputed or searched.

## Candidate Rules

Only the preregistered Rule-NLL, Rule-NLL-1pct, Rule-F1 and Rule-Stability are evaluated. They select between existing frozen R1/R2 outcomes retrospectively.

## Leave-One-Scenario-Out

All three held-scenario rounds are evaluated. The Rule-Stability median is computed from the two training scenarios only.

## A1 Diagnosis

Generated from Known-only evidence in `outputs/summary/a1_failure_diagnosis.md`.

## A2 Diagnosis

Generated from Known-only evidence in `outputs/summary/a2_covariance_benefit_diagnosis.md`.

## A3 Diagnosis

Generated from Known-only evidence in `outputs/summary/a3_diagnosis.md`.

## Mechanism

The final mechanism is restricted to C1–C5 and recorded after the fixed analyses.

## Final Gate

The final decision is restricted to `SIGNAL_FOUND`, `WEAK_SIGNAL`, or `NO_SIGNAL`.

## Limitations

This is a 15-run USTC development diagnosis. Correlations are exploratory, not independent confirmation or causal evidence.

## Next Step

No adaptive selector, retraining, USTC rerun, CipherSpectrum rerun or Stage 12 is started here.
## Recorded Final Result

# Stage 11C final gate

**WEAK_SIGNAL**

Mechanism: **C4**. No new training, detector fitting, threshold search, Unknown-derived criterion, CipherSpectrum Test access, or next-stage execution occurred.

