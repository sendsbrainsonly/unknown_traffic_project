# A2 covariance benefit diagnosis

## Verified Known-only evidence

- PCA64 centroid median margin: `11.187387`; negative-margin fraction: `0.072412`.
- PCA64 covariance condition-number median: `9908.243143`; effective-rank median: `7.892466`.
- Median PCA64 n/d: `25.000000`.
- Train-only covariance bootstrap relative instability: `0.102138`.
- Known-Validation Full-vs-Spherical DeltaNLL: `86.317375`.
- Known-Validation Full-K1 minus centroid Macro-F1: `-0.005346`.
- R1/R2 coverage variance: `0.083390` / `0.059487`.

## Post-hoc outcome

- Stage11B R2-R1 AUROC: `0.034821` (`OUTCOME_ONLY`).

## Interpretation

Full covariance captures useful Known-side structure. This is an exploratory mechanism inference, not causal proof. Overall Stage11C gate: `WEAK_SIGNAL`.
