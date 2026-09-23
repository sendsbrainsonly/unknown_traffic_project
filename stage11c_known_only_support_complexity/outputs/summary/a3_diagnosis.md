# A3 covariance diagnosis

## Verified Known-only evidence

- PCA64 centroid median margin: `12.851757`; negative-margin fraction: `0.022662`.
- PCA64 covariance condition-number median: `9109.747385`; effective-rank median: `8.134246`.
- Median PCA64 n/d: `25.000000`.
- Train-only covariance bootstrap relative instability: `0.096008`.
- Known-Validation Full-vs-Spherical DeltaNLL: `86.122883`.
- Known-Validation Full-K1 minus centroid Macro-F1: `-0.005047`.
- R1/R2 coverage variance: `0.070937` / `0.064340`.

## Post-hoc outcome

- Stage11B R2-R1 AUROC: `0.002600` (`OUTCOME_ONLY`).

## Interpretation

CENTROID_ALREADY_SUFFICIENT. This is an exploratory mechanism inference, not causal proof. Overall Stage11C gate: `WEAK_SIGNAL`.
