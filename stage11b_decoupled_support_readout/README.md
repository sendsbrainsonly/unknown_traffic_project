# Stage 11B — Frozen Open-Detect Encoder / Decoupled Empirical Support Readout

## Goal

Test whether open-set detection support should be estimated after training from a frozen Open-Detect representation, without changing the encoder, checkpoint, training prototype, loss, or training process.

## Stage11A NO-GO

Stage 11A reduced the normalized prototype-to-centroid gap from approximately 1.0–1.2 to approximately `1e-5`, but reduced Known Macro-F1 and Native AUROC. Its fixed decision was `P3 / NO_GO`: hard training-time prototype anchoring is harmful or insufficient.

## Why Decouple Training and Detection

The training prototype may still help learn a discriminative representation even when it is a poor density model for detection. Stage 11B therefore leaves representation learning untouched and replaces only the post-hoc detection readout.

## Frozen B0 Encoders

The only encoders are the 15 frozen USTC v6 corrected B0 best checkpoints: A1/A2/A3 crossed with seeds 2022–2026. Each run verifies checkpoint, split-array, class-set, and historical metric parity before its result can receive a `SUCCESS` marker.

No optimizer, backward pass, loss update, checkpoint update, DAP checkpoint, or prototype write-back exists in this stage.

## R0 Native

R0 is the unchanged forward Gaussian KL from the frozen B0 model:

`min_y KL[N(mu_x, diag(exp(logvar_x))) || N(p_y, I)]`.

## R1 Empirical Centroid

R1 computes one centroid per class from Known Train deterministic `mu_x` and scores a sample by its minimum squared Euclidean distance to those centroids.

## R2 Full Gaussian K1

R2 uses the frozen Stage 11A B0 diagnostic support models, which were fit only on Known Train deterministic `mu_x` after a train-only StandardScaler and train-only randomized PCA64. Each class has one full-covariance Gaussian with `reg_covar=1e-3`.

## R3 Full Gaussian K2

R3 uses the corresponding frozen two-component full-covariance GMM per class with `reg_covar=1e-3`, `n_init=3`, `max_iter=300`, and `random_state=0`. K is not searched.

## Validation-Only Calibration

R0/R1/R2/R3 each use a separate global threshold equal to the Known Validation score P95 (`numpy method=higher`). Unknown labels and test labels are never used for calibration or model selection.

## Five-Seed Evaluation

All four readouts use the same checkpoint, split, Known samples, Unknown samples, and symmetric balanced 1:1 evaluation draw within every scenario/seed. Scenario summaries report mean, median, standard deviation, minimum, maximum, and paired direction counts.

## Known Classification

Known Test Accuracy and Macro-F1 are reported for every readout. Readout changes can change class predictions even though the encoder remains frozen.

## Unknown Detection

The development-only metrics are AUROC, AUPRC, binary Accuracy, binary F1, UFAR, and Known FRR. Results are explicitly labelled `DEVELOPMENT_RESULT` and `NOT_INDEPENDENT_EXTERNAL_VALIDATION`.

## Absorption Analysis

For each true Unknown class and predicted Known class, the outputs record accepted counts and absorption rates. Per-class support evidence records the learned prototype, empirical centroid, full-Gaussian location/covariance diagnostics, Known F1, and Unknown absorption totals.

## Mechanism Interpretation

The fixed mechanism label is selected only after all 15 paired runs are complete: D1–D6 as defined in the task specification. Class/run-level gap associations are diagnostic and are not used for tuning.

## Final Gate

The fixed gate is one of `GO`, `CONDITIONAL_GO`, or `NO_GO`. Primary evidence is R2 versus R0; R1 versus R0, R2 versus R1, and R3 versus R2 are secondary mechanism diagnostics.

## Limitations

- This is USTC method development, not independent external validation.
- The local repeated grouped-image-disjoint v6 splits are not author-exact folds.
- Stage 9 CipherSpectrum sample-level Test representations, scores, and predictions are not opened, read, or rescored.
- Reusing frozen Stage 11A K1/K2 models avoids redundant fitting but retains that exact diagnostic protocol.

## Next Step

No later stage is automatically started. The final gate only records whether a separate future design study would be justified.
## Recorded Final Result

# Stage 11B Final Gate

## Decision

**NO_GO**

## Primary R2 versus R0

| Scenario | Mean ΔAUROC | R2 better seeds | Mean ΔKnown Macro-F1 | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|
| A1 | -0.024100 | 0/5 | 0.014308 | 0.011000 |
| A2 | 0.108791 | 5/5 | 0.001215 | -0.000667 |
| A3 | 0.023048 | 5/5 | -0.001102 | -0.000000 |

## Fixed interpretations

- Mechanism: **D5**.
- R1 stable over R0: `False`.
- R2 stable over R0: `False`.
- Covariance: **COVARIANCE_BENEFIT_CONTEXT_DEPENDENT**.
- K2: **MULTICOMPONENT_SECONDARY**.

## Scope

`DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. No encoder training, no CipherSpectrum Stage9 sample-level Test access, and no next stage.

