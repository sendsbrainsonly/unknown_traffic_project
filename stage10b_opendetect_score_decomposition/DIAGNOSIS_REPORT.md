# Stage 10B — Open-Detect Native Score Decomposition and Geometry Failure Diagnosis

## Executive Summary

**Final mechanism: M2 — LEARNED_PROTOTYPE_MISMATCH. Secondary: M5 — COVARIANCE_MISMATCH. Recommended next branch: Branch P.**

Across all three frozen CipherSpectrum settings, removing `V_post` makes AUROC worse, whereas replacing the learned prototype with a Known-Train empirical centroid is the largest positive contrast. Full-covariance K1 adds a second large, consistent gain; K2 adds a smaller but bootstrap-supported gain. The evidence therefore points first to learned prototype/support-location mismatch, with covariance geometry as the main secondary mechanism—not to posterior variance corruption, feature scaling, PCA64, or a representation that is intrinsically unusable.

All results below are **POST_HOC_DIAGNOSTIC_ONLY**. They cannot be added to the Stage 9 independent six-method table or presented as a newly selected detector.

## Frozen Evidence

- Stage 6–9 and Stage 10A provenance: PASS.
- Stage 9 Test manifests, sample IDs, `mu`, PCA64, Native/K1/K2 scores, predictions, scaler/PCA, and GMMs were read-only.
- `logvar` was absent upstream, so one eval-mode frozen inference was performed per setting. Recomputed `mu` matched Stage 9 exactly (`max_abs_error = mean_abs_error = 0`) for all Known/Unknown roles.
- No model training, prototype update, density refit, PCA/scaler fit, score tuning, or threshold tuning occurred.

## Native KL Identity

The identity `A_native = D_proto + V_post = -score_native_stage9` passed in all settings. Combined maximum / mean reconstruction errors were: Low `9.537e-06` / `1.504e-06`, Medium `1.192e-05` / `1.582e-06`, High `1.144e-05` / `1.424e-06`. Native nearest-class prediction mismatches were zero.

## Native Score Decomposition

Entries are `AUROC / AUPRC`, with Unknown positive and higher score meaning more Unknown.

| Setting | S0 | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|---|
| low | 0.500830 / 0.277508 | 0.431597 / 0.284080 | 0.717132 / 0.512625 | 0.670549 / 0.479113 | 0.670901 / 0.474392 | 0.880005 / 0.761724 | 0.892480 / 0.807311 |
| medium | 0.497970 / 0.585791 | 0.412627 / 0.523474 | 0.666464 / 0.697699 | 0.683861 / 0.710331 | 0.674294 / 0.701215 | 0.779753 / 0.857264 | 0.795385 / 0.869747 |
| high | 0.472133 / 0.695744 | 0.379376 / 0.659542 | 0.686885 / 0.831246 | 0.695970 / 0.831664 | 0.687189 / 0.826026 | 0.804478 / 0.923875 | 0.821250 / 0.929708 |

Stage 9 AUROC reproduction is exact at displayed precision for S0, S5, and S6 in every setting. AUPRC differs in convention from Stage 9's table because this report uses Unknown-positive anomaly scores; the original Known-positive AUPRC is retained as a reference column in each `score_chain_metrics.csv`.

## Posterior Variance

| Setting | V_post AUROC | Known median V | Unknown median V | Ranking verdict |
|---|---|---|---|---|
| low | 0.563337 | 9.266171 | 10.276074 | VARIANCE_TERM_HELPS_RANKING |
| medium | 0.595660 | 9.086911 | 11.430264 | VARIANCE_TERM_HELPS_RANKING |
| high | 0.615454 | 9.039632 | 11.086539 | VARIANCE_TERM_HELPS_RANKING |

Unknown median `V_post` is higher in all settings, and `V_post` alone has AUROC above 0.5. Removing it reduces AUROC by 0.069–0.093. Thus `V_post` partially rescues, rather than causes, the Native ranking failure. No D/V reweighting was searched.

## Learned Prototype Geometry

| Setting | Gap mean | Gap median | Gap max | rho gap vs Native absorption |
|---|---|---|---|---|
| low | 1.407130 | 1.342672 | 4.028912 | -0.464913 |
| medium | 1.388972 | 1.236462 | 3.901726 | 0.125754 |
| high | 1.480266 | 1.247257 | 5.189286 | -0.241379 |

The empirical raw centroid raises AUROC over learned prototype distance by 0.254–0.308. This is the largest positive controlled contrast in every setting. The factor `0.5` in S1 versus S2 is irrelevant to ranking, so the AUROC change reflects the center geometry rather than score scale.

## Empirical Centroid

S2 recovers AUROC to 0.666–0.717 using only Known-Train class means. The improvement is bootstrap-supported in all settings and is accompanied by a smaller but consistent Known closed-set Macro-F1 increase over the learned prototype.

## Feature Scaling

Scaler effects are not stable: Low decreases by 0.0466, while Medium/High increase by only 0.0174/0.0091. This does not support M3 as a primary mechanism.

## PCA64

PCA64 is neutral in Low and slightly negative in Medium/High. It is not the source of the K1/K2 gain and does not support M4.

## Full Covariance

S4→S5 raises AUROC by 0.105–0.209, with all three paired 95% bootstrap intervals above zero. Full covariance is the strongest secondary mechanism (M5), showing that identity/spherical centroid geometry still underdescribes class support after center correction.

## K2 Local Support

S5→S6 adds 0.0125–0.0168 AUROC in all settings, and all paired intervals are above zero. The gain is stable but modest and below the predeclared 0.02 material-effect floor, so M6 is supporting rather than primary evidence.

## Known Classification Geometry

Macro-F1:

| Setting | Learned | Raw centroid | Scaled | PCA64 | K1 | K2 |
|---|---|---|---|---|---|---|
| low | 0.768938 | 0.781533 | 0.772046 | 0.770138 | 0.811083 | 0.810713 |
| medium | 0.710081 | 0.731342 | 0.730191 | 0.729527 | 0.760870 | 0.767607 |
| high | 0.721287 | 0.742409 | 0.740157 | 0.739822 | 0.764047 | 0.768270 |

K1/K2 maintain 0.761–0.811 Known Macro-F1 while producing 0.780–0.892 AUROC. The shared representation therefore contains useful separation; M7 is not the leading explanation.

## Unknown-Class Analysis

Largest per-class S0→S5/S6 recoveries:

- low: S5 `segment.com` (delta AUROC 0.500771); S6 `segment.com` (delta AUROC 0.532950).
- medium: S5 `ubuntu.com` (delta AUROC 0.475203); S6 `naver.com` (delta AUROC 0.487406).
- high: S5 `onetrust.com` (delta AUROC 0.616784); S6 `onetrust.com` (delta AUROC 0.621027).

Complete per-class AUROC/AUPRC, score medians, Known reference percentiles, and reused Stage 9 decisions are in each `per_unknown_score_chain.csv`.

## Absorption Analysis

Prototype normalized gap versus frozen Native Unknown absorption has Spearman rho Low `-0.464913`, Medium `0.125754`, High `-0.241379`. The sign is not stable, so class-level gap alone does not monotonically explain which Known class absorbs Unknown traffic. This is an explanatory correlation, not a prototype-tuning rule.

## Failure Mechanism

| Setting | S1-S0 | S2-S1 | S3-S2 | S4-S3 | S5-S4 | S6-S5 |
|---|---|---|---|---|---|---|
| low | -0.069233 | 0.285535 | -0.046584 | 0.000352 | 0.209104 | 0.012475 |
| medium | -0.085343 | 0.253837 | 0.017397 | -0.009566 | 0.105459 | 0.015632 |
| high | -0.092757 | 0.307509 | 0.009085 | -0.008780 | 0.117289 | 0.016772 |

- Primary: **M2 — LEARNED_PROTOTYPE_MISMATCH**.
- Secondary: **M5 — COVARIANCE_MISMATCH**.
- `PROTOTYPE_OPTIMIZER_RISK` exists statically at the scheduled reset path, but it was not realized: Low stopped at epoch 31 and Medium/High at 27, before logged reset epochs 51/81.

## Implication for Method Design

The main failure is the mismatch between learned prototype locations and empirical external-support centers. Even after center correction, full covariance contributes strongly, so future work should treat prototype/support estimation as the first target and preserve covariance-aware support modeling as a secondary requirement.

## What We Can Claim

- The frozen Native identity is numerically reproduced.
- Within this post-Test diagnostic, learned-prototype location and identity-covariance geometry explain substantial portions of the Native-to-K1/K2 AUROC gap.
- K2 provides a small, consistent incremental gain over K1.

## What We Cannot Claim

- S1–S6 are not new independent Test baselines and are not method-selection evidence.
- The adjacent contrasts are not strict causal effects.
- The data do not establish that posterior variance should be reweighted, that K2 is universally optimal, or that any new threshold improves external detection.

## Recommended Next Branch

**Branch P**: investigate prototype/support estimation first. Preserve Branch C as the secondary design axis because the full-covariance contrast is large and stable. Do not start the next branch from this task.
