# Stage 8B — CipherSpectrum DGSB-v2 Known-Only Rule Freeze

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

- Low: DUAL_GATE_ACTIVE; Global-only=266, Local-only=111, both-fail=140.
- Medium: DUAL_GATE_ACTIVE; Global-only=200, Local-only=74, both-fail=104.
- High: DUAL_GATE_ACTIVE; Global-only=154, Local-only=65, both-fail=115.

## Per-Class Coverage

See `outputs/{setting}/per_class_coverage.csv` and the cross-setting summary.

## Per-Component Coverage

See `outputs/{setting}/per_component_coverage.csv`; cohorts use true-class posterior assignments.

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

`READY_FOR_ONE_SHOT_FINAL_TEST`.

## Next Step

Stop. Do not open Test without separate authorization for the one-shot Final Test.
