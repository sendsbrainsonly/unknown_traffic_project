# A1 Full-K1 failure diagnosis

## Verified Known-only evidence

- PCA64 centroid median margin: `11.002230`; negative-margin fraction: `0.063673`.
- PCA64 covariance condition-number median: `9361.611581`; effective-rank median: `8.715156`.
- Median PCA64 n/d: `25.000000`.
- Train-only covariance bootstrap relative instability: `0.111365`.
- Known-Validation Full-vs-Spherical DeltaNLL: `85.510456`.
- Known-Validation Full-K1 minus centroid Macro-F1: `-0.012012`.
- R1/R2 coverage variance: `0.068478` / `0.047832`.

## Post-hoc outcome

- Stage11B R2-R1 AUROC: `-0.024045` (`OUTCOME_ONLY`).

## Interpretation

Known-only covariance fit does not transfer to open-set ranking; centroid geometry is already sufficient. This is an exploratory mechanism inference, not causal proof. Overall Stage11C gate: `WEAK_SIGNAL`.
