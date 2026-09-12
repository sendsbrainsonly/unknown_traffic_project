# Stage 3 — Strict Unknown-Free Utility Test Protocol

Protocol status: **FROZEN_ADAPTED_NON_AUTHOR_EQUIVALENT**

Created before Unknown Test: **true**

Dataset: **USTC-TFC2016, 20 logical classes**

Class scenarios: **official Open-Detect A-1/A-2/A-3**

## Scientific question

At matched Known acceptance, does fixed multi-local-support modeling reduce Unknown false acceptance relative to one full Gaussian per Known class?

This tests task utility. It does not estimate a true K and does not interpret components semantically.

## Frozen split

The official class scenarios are reused by class name:

- A-1: 19 Known / 1 Unknown; Unknown = Tinba.
- A-2: 17 Known / 3 Unknown; Unknown = Geodo, Htbot, Tinba.
- A-3: 15 Known / 5 Unknown; Unknown = Geodo, Htbot, Tinba, Miuref, Neris.

Within each scenario, use the pre-existing `data/splits/compatible_min1/` stratified 80/10/10 flow split, created with seeds 41 and 42. Known Train and Known Validation contain only Known classes. Known Final Test contains Known test rows. Unknown Final Test contains only source test rows of held-out classes. Held-out-class source train/validation rows are ignored and must not be loaded.

The paper reports five folds but does not publish sample-level fold assignments or seeds. Therefore the formal local study uses three official class-held-out scenarios and one immutable sample split, and must not be described as author-exact five-fold reproduction.

## Strict Unknown-Free constraints

Unknown classes are forbidden from:

- encoder training;
- encoder validation and checkpoint selection;
- StandardScaler fitting;
- PCA fitting;
- Gaussian/GMM fitting;
- K selection;
- threshold calibration;
- hyperparameter tuning.

Unknown is permitted only in Final Test. Unknown labels are used only after scores are frozen, for evaluation. Each A-1/A-2/A-3 scenario requires a separately trained Known-only encoder. The existing 20-class Open-Detect checkpoint is diagnosis-only.

## Representation and models

For every scenario, train a Known-only Open-Detect encoder and extract deterministic evaluation representation `mu_x`.

Controlled density route:

```text
Known Train mu_x
-> train-only StandardScaler
-> train-only PCA64
-> class-conditional support model
```

Compare:

1. **Baseline A — Open-Detect Native:** official nearest learned-prototype KL score; no Unknown-derived threshold.
2. **Baseline B — Single-Full:** one full-covariance Gaussian per Known class after the frozen scaler/PCA route.
3. **Method Diagnostic — Multi-Full-K2:** fixed K=2 full-covariance GMM per Known class after the identical scaler/PCA route.

Primary controlled comparison: **Single-Full vs Multi-Full-K2**. The only intended change is the class-conditional support model. Fixed GMM settings: `reg_covar=1e-3`, `n_init=3`, `max_iter=300`, seed 0. No BIC, Adaptive K, test-selected K, or post-hoc class-dependent K.

## Threshold and metrics

Each method calibrates its own score threshold on Known Validation only to accept 95% of Known Validation. No threshold is selected from Unknown or final-test labels.

Primary metrics:

- Unknown False Acceptance Rate (Unknown predicted Known / all Unknown);
- Unknown Rejection Rate (Unknown predicted Unknown / all Unknown);
- Known Test False Rejection Rate (Known predicted Unknown / all Known);
- AUROC, with Unknown as positive;
- AUPRC-Unknown, with Unknown as positive;
- FPR@95% Unknown TPR, where false positives are Known rejected as Unknown; diagnostic only;
- Unknown TPR at 95% Known Validation acceptance.

The primary claim must compare Single-Full and Multi-Full-K2 at the preregistered matched Known-validation acceptance. Report per-scenario and pooled stratified-bootstrap 95% confidence intervals over final-test flows; bootstrap seed 0 and 10,000 resamples, stratified by Known/Unknown and class. The Unknown test is not used to select a model.

## Decision gates

- **Gate A — TASK UTILITY CONFIRMED:** Multi materially lowers Unknown FAR at matched Known rejection and the paired bootstrap 95% CI for the FAR difference excludes zero in the favorable direction.
- **Gate B — DENSITY ONLY:** K>1 improves density fit but not Unknown Detection materially.
- **Gate C — MULTI HURTS:** Multi worsens Unknown Detection or Known FRR.
- **Gate D — MIXED:** effect differs materially across Known/Unknown class scenarios; move to class-dependent support complexity without claiming universal superiority.

## Deferred work

Adaptive K, prototype-specific boundaries (RQ2), and Unknown Class Discovery (RQ3) remain blocked until Gate A or a scientifically justified Gate D follow-up. No Stage 3 result exists yet.
