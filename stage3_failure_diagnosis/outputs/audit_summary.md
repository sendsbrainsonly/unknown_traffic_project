# Stage 3 Failure Diagnosis Audit Summary

## Scope and validity

This is post-hoc diagnosis of an already observed Final Test. It did not fit a scaler, PCA, Gaussian/GMM, threshold, encoder, K, or Adaptive detector, and did not rerun Final Test. Frozen Unknown test mu_x was loaded only to recover missing K2 component assignments; replayed Single/Multi scores were required to match the frozen scores row by row.

## Attribution

- A-1: Htbot accounts for +86 of the net +86 absorbed Unknown samples; all other Known classes have zero delta.
- All 86 A-1 Single-reject/Multi-accept Tinba samples use Htbot K2 component 1 in the frozen-score replay.
- A-2: Miuref contributes +16 and Shifu -1, for a net +15 over 5,579 Unknown samples; the large common Geodo absorption remains almost unchanged.
- A-3: Virut accounts for -236; these are 236 recovered Neris samples, while Shifu remains unchanged at two accepted Htbot samples.
- The 236 recovered A-3 Neris samples use Virut K2 component 0/1 in counts 179/57; A-3 has no Single-reject/Multi-accept regression.

## Known-only geometry

- A-1/Htbot: DeltaNLL2=14.075196, weights=(0.736541,0.263459), TV=0.003103, Euclidean/Mahalanobis separation=8.690270/19.202526, max condition=2673.54.
- A-3/Virut: DeltaNLL2=24.334647, weights=(0.218307,0.781693), TV=0.008581, Euclidean/Mahalanobis separation=1.755157/6.534201, max condition=1273.68.
- Degeneracy flags over 51 setting-class cells: tiny components=0, validation-empty components=0. The A-1/Htbot failure class has neither flag.

## Density versus utility

- 2 class-setting cells improve Known-validation NLL while increasing Unknown absorption: A-1/Htbot (+86), A-2/Miuref (+16).
- Therefore density-fit improvement does not imply open-set utility.
- Largest absolute pooled post-hoc Spearman coefficient: `component_mean_euclidean_distance` rho=0.194685, p=0.170998, n=51. This is exploratory under a sparse, zero-inflated outcome.

## Simple statistics

- The strongest measured association among non-zero contributors was A-1/Htbot train `backward_packets` (epsilon-squared=0.588646); direction alone did not identify utility.
- Because the Top-3 lists contain zero-delta ties, only Htbot (A-1) and Virut (A-3) are causal non-zero contributors; the other ranked rows are reference comparisons.

## Final diagnosis

**Diagnosis A — DENSITY-UTILITY MISMATCH.** K2 improves Known-validation density likelihood for the failure class while expanding acceptance of Unknown traffic. The mixture is not tiny, validation-empty, or train/validation-unstable, so simple mixture degeneration is not the primary explanation. Known-only correlations remain exploratory and do not establish a reliable class-selection rule.

## Independent validation boundary

Any rule proposed after observing A-1/A-2/A-3 must be frozen and evaluated first on an untouched benchmark. Priority: CSTNET-TLS1.3, then CipherSpectrum. No claim that an adaptive rule beats Single on USTC is made here.

## Reproducibility checks

- Frozen binary prediction parity: PASS for 16,735 Unknown setting-sample rows.
- Frozen Unknown score replay parity: PASS; maximum Single/Multi absolute errors=0/0.
- Official transition and per-Unknown-class aggregate parity: PASS.
- Frozen scaler/PCA/GMM manifest hashes: PASS.
- Stage 3 output tree before/after hashes: PASS.
