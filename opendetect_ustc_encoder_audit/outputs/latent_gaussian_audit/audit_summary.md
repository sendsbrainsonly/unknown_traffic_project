# Open-Detect USTC Latent Gaussian Audit

## Frozen formal checkpoint

- Training stop epoch: 37
- Best epoch: 32
- Validation Accuracy: 0.986158250
- Validation Macro-F1: 0.987371196
- Combined score: 0.986764350
- Checkpoint SHA-256: `f8e53d36bd6dc8333451b2f159300bc5a2cd875fccaff01a2c6dc43356951b36`
- Stop reason: validation combined score did not improve for 5 consecutive epochs

## Protocol

Deterministic `mu_x` was extracted for the complete fixed train and validation splits. StandardScaler and PCA64 were fit only on all 20-class train `mu_x`; validation was transform/evaluation-only. Class-specific full-covariance GMMs used K=1/2/3, seeds 0/1/2, reg_covar=1e-3, n_init=3, max_iter=300 and float64. Test was not loaded.

## Encoder comparison

| class | TF K1 | TF K2 | TF K3 | TF Delta2 | TF Delta3 | OD K1 | OD K2 | OD K3 | OD Delta2 | OD Delta3 | case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FTP | -53.976444 | -69.765320 | -75.507957 | 15.788876 | 21.531513 | -91.094227 | -98.101579 | -103.468403 | 7.007352 | 12.374176 | Case A |
| Cridex | -99.045807 | -111.781731 | -114.306702 | 12.735924 | 15.260895 | -98.397549 | -108.319648 | -110.537789 | 9.922098 | 12.140239 | Case A |
| Miuref | -61.720963 | -74.841499 | -79.644180 | 13.120537 | 17.923218 | -79.223667 | -90.673427 | -99.929099 | 11.449760 | 20.705432 | Case A |
| Outlook | -91.338013 | -112.926254 | -118.512474 | 21.588242 | 27.174461 | -91.074622 | -107.727576 | -108.664861 | 16.652955 | 17.590239 | Case A |

## Open-Detect compactness

| class | assigned prototype mean/median/p95 | within-class variance | nearest-wrong mean | margin |
|---|---:|---:|---:|---:|
| FTP | 0.974919 / 0.980521 / 1.097564 | 0.049854 | 6.586681 | 5.611762 |
| Cridex | 0.913145 / 0.897352 / 1.070214 | 0.166052 | 5.161216 | 4.248071 |
| Miuref | 1.394938 / 1.410479 / 1.534080 | 0.305621 | 3.697485 | 2.302548 |
| Outlook | 1.852590 / 1.875140 / 2.079304 | 0.276054 | 3.909477 | 2.056886 |

Compactness is descriptive and is not used as proof that K=1 is adequate.

## Stability and degeneracy

- Gaussian fits: 36
- Captured warnings/non-convergence records: 0
- Tiny component rows (<1% train assignment): 0/72
- Very tiny component rows (<0.5%): 0/72
- Validation-empty component rows: 0/72
- Cholesky failures: 0/72
- Maximum covariance condition number: 12152.1731
- Seed-specific NLL, DeltaNLL, component counts/weights, train-validation TV, eigenvalues and iterations are preserved in the CSV outputs.

## Per-class cases

- FTP: Case A; residual representation-level heterogeneity persists across encoders.
- Cridex: Case A; residual representation-level heterogeneity persists across encoders.
- Miuref: Case A; residual representation-level heterogeneity persists across encoders.
- Outlook: Case A; residual representation-level heterogeneity persists across encoders.

## Overall Gate

**A. CROSS-ENCODER PERSISTENCE**

Explicit encoder dependence observed: no.

This result concerns class-conditional density fit in learned representations. It does not prove natural Gaussian modes, semantic traffic subtypes, Open-Detect detection failure, unknown false acceptance, or that a multi-Gaussian detector will improve Unknown Detection.

## Known limitations

- At the epoch-5 GPU migration, the old checkpoint lacked optimizer/RNG state; the discontinuity cannot be repaired retrospectively.
- Batch 1024 produced `logvar.exp()` overflow; formal training used batch 512.
- Open-Detect released loss, decoder and prototype reset differ from the paper as documented in the source audit.
- TrafficFormer comparison uses its immutable seed-0 Stage 2.5 fit; it was not rerun.
