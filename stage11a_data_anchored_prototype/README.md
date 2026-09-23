# Stage 11A — Data-Anchored Prototype Feasibility Study

## Goal

Test, on the USTC development benchmark only, whether forcing Open-Detect prototypes to equal current Known-Train empirical class centroids prevents prototype drift and improves Native unknown detection.

## Stage10B Motivation

Frozen Stage 10B diagnostics identified learned-prototype mismatch as the primary mechanism and full-covariance mismatch as the secondary mechanism. Stage 11A tests the prototype mechanism prospectively; it does not reuse CipherSpectrum Test samples.

## Why Prototype Anchoring

For each known class, the prototype is tied directly to the support represented by deterministic encoder means instead of being allowed to drift under optimizer gradients.

## USTC Development Status

All unknown-detection results are `DEVELOPMENT_RESULT` and `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. The benchmark uses the existing v6 local-PCAP repeated grouped-image-disjoint splits, not the authors' exact published folds.

## B0 Baseline

B0 is the frozen 15-run v6 corrected Open-Detect reproduction. Its checkpoints and results remain read-only and are never overwritten.

## DAP

DAP uses the same corrected model, data, architecture, latent dimension, optimizer, learning rate, loss, scheduler, early stopping, and validation-accuracy checkpoint rule as B0. The sole training-mechanism change is prototype anchoring.

## Prototype Update Rule

Before epoch 1 and after every complete training epoch, the encoder runs in evaluation mode without gradients over deterministic Known-Train inputs. For class `y`, `p_y <- mean(mu_x | y)`. The encoder/model then returns to its prior training mode. Anchoring frequency is fixed at `ANCHOR_EVERY_EPOCH` and is not searched.

The DAP prototype tensor is a persistent registered buffer. It is absent from `named_parameters()`, has `requires_grad=False`, and is absent from every optimizer parameter group. Only the anchoring function updates it, so the stale-Parameter failure mode cannot occur.

## No Unknown Training

Prototype initialization, all prototype updates, density fitting, and checkpoint selection use no Unknown data. Checkpoint selection uses Known Validation Accuracy only. Unknown Test is read only after the best checkpoint is frozen for development evaluation.

## Primary Comparison

The primary paired comparison is DAP-Native versus frozen B0-Native for A1/A2/A3 and seeds 2022–2026. The Native score remains the original forward Gaussian KL, and its threshold is the Known Validation P95 with NumPy's `higher` quantile method.

## Prototype Alignment

For each frozen best checkpoint, normalized prototype gap is Euclidean prototype-to-centroid distance divided by the median within-class radius, computed from Known-Train deterministic `mu` only.

## Known Classification

Known Validation and Known Test Accuracy and Macro-F1 are reported for B0 and DAP. Unknown data never enter checkpoint selection.

## Native Unknown Detection

The symmetric balanced 1:1 v6 evaluation reports Accuracy, Binary F1, AUROC, Unknown-positive AUPRC, Known FRR, and UFAR using only the validation-calibrated P95 threshold.

## K1/K2 Diagnostics

Each frozen B0/DAP checkpoint is diagnosed with train-only StandardScaler, train-only randomized PCA64, and per-class full-covariance GMMs. K1 and K2 are fixed; `reg_covar=1e-3`, `n_init=3`, `max_iter=300`, and `random_state=0`. No K search is performed.

## Five-Seed Analysis

All conclusions use paired scenario/seed comparisons and scenario summaries over all five seeds (mean, median, sample standard deviation, minimum, maximum, and direction counts).

## Mechanism Interpretation

The final result is classified as P1/P2/P3/P4 from the joint change in prototype gap and Native AUROC. K1 and K2 gaps are interpreted separately as covariance and multi-component diagnostics.

## Limitations

This is method development on USTC scenarios already used during development. It is not independent external validation and cannot be added to the frozen CipherSpectrum Stage 9 comparison. B0 is a local v6 reproduction rather than an author-exact fold reproduction.

## Final Gate

The final gate will be exactly one of `GO`, `CONDITIONAL_GO`, or `NO_GO`, using the preregistered +0.02 AUROC materiality rule and Known-utility/tradeoff conditions.

## Next Step

Stage 11A stops after the gate. Even if the result is GO, no covariance-aware final method or subsequent stage is started automatically.

## Recorded Final Results


## Decision

**NO_GO**

## Primary evidence

| Scenario | Mean ΔAUROC | Median ΔAUROC | DAP better seeds | Gap-lower seeds | Mean ΔKnown Macro-F1 | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|---:|---:|
| A1 | -0.167800 | -0.153725 | 0/5 | 5/5 | -0.060538 | 0.010000 |
| A2 | -0.068712 | -0.046122 | 0/5 | 5/5 | -0.036661 | 0.005667 |
| A3 | -0.036409 | -0.043832 | 1/5 | 5/5 | -0.094302 | 0.001978 |


## Fixed gate checks

- Prototype alignment decreases 5/5 in every scenario: `True`.
- DAP-Native scenario mean AUROC is non-negative versus B0 in all scenarios: `False`.
- Scenarios meeting material `+0.02` AUROC: `0`.
- Known Test Macro-F1 has no scenario mean drop of 0.02 or more: `False`.
- Mean Known FRR does not rise by 0.02 or more: `True`.
- DAP AUROC improves in at least 4/5 paired seeds per scenario: `False`.

## Covariance diagnostics

- DAP K1 minus Native mean AUROC: A1=-0.001560, A2=0.078767, A3=0.034826.
- Verdict: **COVARIANCE_REQUIREMENT_NOT_STABLE**.
- DAP K2 minus K1 mean AUROC: A1=-0.067470, A2=0.013793, A3=0.003361.
- Verdict: **MULTICOMPONENT_SECONDARY**.

## Scope boundary

`DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. CipherSpectrum Stage 9 sample-level Test data were not opened, read, or rescored. No next stage was started.
