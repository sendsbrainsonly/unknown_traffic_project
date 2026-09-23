# Experiment results: stage9-cipherspectrum-final-test-20260913

- Status: `success`
- Experiment type: `evaluation`
- Claim scope: `approximate` (strict local frozen split; not author-equivalent)
- Test opened (UTC): `2026-09-13T14:30:55.571231+00:00`
- Objective: one-shot frozen external evaluation of six methods across Low, Medium, and High.

## Data and split

- Low: Known Test n=16246; Unknown Test n=6000.
- Medium: Known Test n=12162; Unknown Test n=18000.
- High: Known Test n=10746; Unknown Test n=30000.

## Configuration and execution

- Six upstream-frozen methods; deterministic `mu_x`; frozen scaler/PCA64/K1/K2/thresholds.
- AUROC/AUPRC positive class is Known; higher detector score means more Known.
- Primary paired class-stratified bootstrap: 1000 iterations, seed 0.
- No Test recalibration, search, retraining, or post-Test method change.

## Core results

- Final Gate: `B — OPERATING-POINT TRADEOFF`.
- Low: Multi UFAR=0.397000, DGSB UFAR=0.536000, DeltaUFAR=+0.139000 (95% CI [+0.130167, +0.147675]); conclusion `TRADEOFF`.
- Medium: Multi UFAR=0.543833, DGSB UFAR=0.591500, DeltaUFAR=+0.047667 (95% CI [+0.043832, +0.051833]); conclusion `TRADEOFF`.
- High: Multi UFAR=0.496100, DGSB UFAR=0.583367, DeltaUFAR=+0.087267 (95% CI [+0.084132, +0.090637]); conclusion `TRADEOFF`.

## Preserved evidence

- `outputs/summary/cross_setting_results.csv`
- `outputs/summary/primary_dgsbv2_vs_multi.csv`
- `outputs/summary/final_gate.json`
- `artifacts/{low,medium,high}/test_bundle_manifest.json`
- `manifest.json`

## Limitations

- CipherSpectrum labels are directory-derived and the local frozen folds are not official author fold identities.
- Domain/endpoint shortcuts remain a known validity risk; results do not establish semantic open-world generalization.
- Correlation and absorption analyses are post-Test explanations only and were not used to modify methods.

## Conclusion and next step

- Stage 9 is frozen. No further dataset or method stage was started.
