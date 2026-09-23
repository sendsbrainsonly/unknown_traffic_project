# Experiment results: stage8a-cipherspectrum-known-density-20260913

- Status: `success`
- Experiment type: `evaluation`
- Claim scope: `diagnostic`
- Completed (UTC): `2026-09-13T09:31:31Z`
- Final Gate: `READY_FOR_FINAL_TEST`

## Data and split

- Stage 6 canonical 120k Low/Medium/High frozen folds.
- Source data opened: Known Train and Known Validation only.
- Known Test opened: 0; Unknown Test opened: 0; Unknown inference: false.

## Configuration and execution

- Deterministic Stage 7 best-checkpoint `mu_x` (128D) -> train-only StandardScaler -> train-only PCA64.
- Per-class full K1/K2 (`reg_covar=1e-3`, K2 `n_init=3`, `max_iter=300`, seed 0).
- Native/global/class/component thresholds use Known Validation only; component fallback is fixed at n<30.

## Core results

| Setting | mu train/val | PCA64 variance | K2 min weight | val_n<30 / fallback | mean/median DeltaNLL2 | K2 better | Native/Single/Multi/Class/Component acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 75,692/16,242 | 0.918283 | 0.067240 | 2/0.026316 | 18.240958/17.130226 | 37/38 | 0.949945/0.950006/0.950006/0.968415/0.971925 |
| Medium | 56,641/12,168 | 0.966317 | 0.070724 | 2/0.029412 | 17.787341/17.094158 | 34/34 | 0.949951/0.950033/0.950033/0.974112/0.976085 |
| High | 49,976/10,740 | 0.973541 | 0.065744 | 4/0.066667 | 17.978591/17.007143 | 30/30 | 0.950000/0.950000/0.950000/0.967225/0.968622 |

## Preserved evidence

- `artifacts/{low,medium,high}/`: representations, transforms, density models, thresholds, bundle manifests and hashes.
- `outputs/{low,medium,high}/`: representation/classification/component/calibration diagnostics and access ledgers.
- `outputs/summary/`: provenance, cross-setting summary, and final gate.
- `configs/evaluation_config.json`: frozen next-stage method configuration.

## Limitations

- These are Known Validation calibration diagnostics, not Test Accuracy, UFAR, AUROC, AUPRC, or Unknown utility.
- Positive DeltaNLL2 demonstrates only held-out density-fit improvement.

## Conclusion and next step

- `READY_FOR_FINAL_TEST`. Stop here; do not open Test without a separate next-stage instruction.
