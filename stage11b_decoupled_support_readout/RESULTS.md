# Experiment results: stage11b-decoupled-support-readout-20260915-v1

## Configuration and execution

- Status: `success`
- Experiment type: `method-development-evaluation`
- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`
- New encoder training: `NO`
- Frozen B0 checkpoints reused: `15/15`
- Final gate: **NO_GO**
- Mechanism: **D5**
- Covariance verdict: **COVARIANCE_BENEFIT_CONTEXT_DEPENDENT**
- K2 verdict: **MULTICOMPONENT_SECONDARY**

## Data and split

Fifteen frozen Open-Detect v6 B0 checkpoints were evaluated across A1/A2/A3 and seeds 2022-2026. Support fitting used Known Train only; all thresholds used Known Validation P95 only; Test was evaluation-only.

## Core results

- A1: R2-R0 `-0.024100`
- A2: R2-R0 `0.108791`
- A3: R2-R0 `0.023048`

## Limitations

This is a USTC development result, not independent external validation. Local corrected v6 splits are not author-exact folds. CipherSpectrum Stage9 sample-level Test was not used.

## Preserved evidence

Per-run support models, thresholds, latent/score/prediction arrays, class/absorption tables, results, hashes, RESULTS, and manifests are retained under `artifacts/`; aggregate tables and decisions are retained under `outputs/summary/`.

## Conclusion and next step

The fixed decision is NO_GO with mechanism D5. Per-run evidence is preserved under `artifacts/`; no next stage was started.
