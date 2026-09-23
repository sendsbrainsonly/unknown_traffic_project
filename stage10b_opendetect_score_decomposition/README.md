# Stage 10B — Open-Detect Native Score Decomposition

This directory contains a **POST_HOC_DIAGNOSTIC_ONLY** analysis of the frozen
CipherSpectrum Test opened in Stage 9. It is not an independent Test baseline
and it does not modify the Stage 9 six-method result.

All upstream inputs are read-only. The encoder, learned prototypes,
StandardScaler, PCA64, K1/K2 GMMs, Test manifests, Test representations,
scores, predictions, and thresholds remain frozen. The only GPU operation is
an eval-mode frozen forward pass that exports `mu_x` and `logvar_x`; its
recomputed `mu_x` must pass a numerical equality gate against Stage 9 before a
setting is analyzed. No model is trained and no detector threshold is searched
or recalibrated.

## Fixed score chain

All diagnostic scores use `higher = more Unknown` and Unknown is the positive
class for AUROC/AUPRC.

| Contrast | Fixed interpretation |
|---|---|
| S0 → S1 | Remove the posterior-uncertainty term |
| S1 → S2 | Learned prototype → empirical Known-Train centroid |
| S2 → S3 | Apply the frozen feature standardization |
| S3 → S4 | Apply the frozen PCA64 projection/truncation |
| S4 → S5 | Spherical centroid geometry → frozen full-covariance Gaussian |
| S5 → S6 | Frozen single Gaussian → frozen two-component local support |

These are controlled diagnostic contrasts, not strict causal effects. S0,
S5, and S6 are required to reproduce the corresponding Stage 9 AUROCs before
mechanistic interpretation is allowed. Any interpretation thresholds in the
configuration classify evidence after the fact; they are not detector
hyperparameters and are never used to select or tune a Test score.

See [`DIAGNOSIS_REPORT.md`](DIAGNOSIS_REPORT.md) for results after completion.
