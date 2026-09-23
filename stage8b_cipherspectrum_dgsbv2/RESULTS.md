# Experiment results: stage8b-cipherspectrum-dgsbv2-20260913

- Status: `success`
- Experiment type: `evaluation`
- Claim scope: `diagnostic`
- Completed (UTC): `2026-09-13T13:47:17Z`
- Final Gate: `READY_FOR_ONE_SHOT_FINAL_TEST`

## Data and split

- Frozen Stage 6/7/8A provenance: PASS.
- Calibration data: Known Validation only; frozen density models came from Known Train.
- Known Test opened/generated: 0/false; Unknown Test opened/generated/inferred: 0/false/false.

## Configuration and execution

- One preregistered rule: Global P02.5 AND predicted-class Local P02.5.
- Sparse fallback n<30 preserves weighted local-score semantics.
- No quantile search and no post-local Global recalibration.

## Core results

| Setting | Global P02.5 | Global acc/FRR | Local acc/FRR | AND acc/FRR | Global-only | Local-only | Both fail | Gate status | Class min/median/max | Component gap mean/max | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Low | -104.159317 | 0.975003/0.024997 | 0.984546/0.015454 | 0.968169/0.031831 | 266/0.016377 | 111/0.006834 | 140/0.008620 | DUAL_GATE_ACTIVE | 0.917778/0.977222/0.990610 | 0.029916/0.143801 | 2/0.026316 |
| Medium | -91.527687 | 0.975016/0.024984 | 0.985371/0.014629 | 0.968935/0.031065 | 200/0.016437 | 74/0.006082 | 104/0.008547 | DUAL_GATE_ACTIVE | 0.932203/0.973414/0.989170 | 0.027481/0.100000 | 2/0.029412 |
| High | -90.361076 | 0.974953/0.025047 | 0.983240/0.016760 | 0.968901/0.031099 | 154/0.014339 | 65/0.006052 | 115/0.010708 | DUAL_GATE_ACTIVE | 0.935096/0.973333/0.995074 | 0.034724/0.190588 | 4/0.066667 |

## Preserved evidence

- Per-setting thresholds, fallback maps, rule specifications and SHA-256 ledgers under `artifacts/`.
- Gate, class, component, baseline-comparison and access audits under `outputs/`.
- Frozen six-method evaluation configuration under `configs/evaluation_config_v2.json`.

## Limitations

- Known-only calibration evidence; no Test Accuracy, Macro-F1, UFAR, AUROC or AUPRC has been computed.
- Gate activity on Known Validation does not establish Unknown utility.

## Conclusion and next step

- `READY_FOR_ONE_SHOT_FINAL_TEST`. Stop here. A separately authorized one-shot Final Test may consume only this frozen bundle.
