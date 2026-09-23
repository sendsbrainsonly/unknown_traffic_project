# Stage 10A Open-Detect Paper-Code-Reproduction Protocol Audit

## Executive Summary

The CipherSpectrum Native result cannot yet be attributed directly to an Open-Detect generalization failure.  The official code and Stage9 agree on inference `mu_x`, one fixed-identity Gaussian prototype per class, forward Gaussian KL, and nearest-KL prediction.  They do **not** agree on threshold calibration: released code tunes a global Youden-J threshold on labeled Known+Unknown test data, whereas Stage9 freezes a global P05 threshold using Known Validation only.  Stage7 also uses a max-logvar stability wrapper and a much stricter group-aware external protocol.  Final decision: **Gate B — HIGH-PRIORITY PROTOCOL MISMATCH**; claim status: `EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE`.

## Paper Protocol

The paper uses 20-class USTC-TFC2016 and 24-class Malicious TLS, an 8:1:1 split, and eight open-world scenarios.  It does not specify capture/session/group isolation, exact fold memberships, or seeds.  Input is the first eight IP packets, each contributing 80 header and 48 payload bytes, zero-padded/truncated to a 1024-byte 32x32 image.  Full evidence is in `outputs/paper/paper_protocol.csv`.

## Official Code

The released source is complete enough to recover model, score, threshold, metrics, loader, and preprocessing behavior.  It is not complete enough for author-exact reproduction because exact folds and raw-to-NPZ provenance are absent.  The source tree also exposes paper-code differences: training discriminative classification uses sampled-z squared Euclidean distance, while the paper writes a KL-derived class probability; train.py uses the released test NPZ as validation.

## USTC Reproduction

The reproduction layer is the **sibling method project** `Projects/Open-Detect`, not `unknown_traffic_project/opendetect_ustc_encoder_audit` and not `stage3_unknown_utility`.  Its authoritative final matrix records zero paper-exact reproductions, eight local approximations, one missing-data block, and one scope exclusion.

The latest USTC evidence is the v6 local-PCAP campaign: one 34,585-image pool, five independently seeded class-stratified grouped-image-disjoint 8:1:1 splits, and A-1/A-2/A-3 training on every split (15 complete checkpoints).  All five full-pool splits contain 27,667/3,457/3,461 train/validation/test images and record zero cross-split flow and exact-image overlap.  These are repeated local splits, **not the authors' exact five folds**.  The local `corrected-paper` model subclasses released Open-Detect but repairs the decoder path, uses the paper KL discriminative family, preserves optimizer linkage at prototype reset, and clamps log variance.

With a global Known-validation P95 KL threshold and balanced 1:1 binary test composition, five-repeat Accuracy/F1/AUROC are A-1 0.9785/0.9789/0.9952, A-2 0.8253/0.7980/0.8534, and A-3 0.9204/0.9167/0.9699.  The separate all-known USTC approximation records Accuracy 0.993909 and weighted F1 0.993832.  Per-run checkpoint hashes and metrics are in `outputs/ustc_reproduction/ustc_open_set_reproduction.csv`.

The earlier 0.986158 validation Accuracy / 0.987371 validation Macro-F1 record belongs to a nested audit inside `unknown_traffic_project`; it is not the sibling Open-Detect reproduction requested here and is excluded from layer C.  The sibling v6 trainer selects checkpoints by validation Accuracy and does not record validation Macro-F1, so no replacement validation Macro-F1 is invented.

## CipherSpectrum Protocol

Stage6 freezes group-aware Low/Medium/High splits.  Stage7 uses the same custom Open-Detect structural architecture through a local subclass, but adds a max-logvar=20 stability path and local early-stopping/data wrappers.  The recorded stability guard did not activate on Stage7 train/validation.  Stage8A freezes deterministic `mu_x`, thresholds, scaler/PCA, and density models using Known train/validation only.  Stage9 performs the one-shot frozen test without refitting.

## Latent Mode

- Paper: training is `SAMPLED_Z`; both inference modes are `UNCLEAR` because no inference pseudocode chooses sampled z versus mu_x.
- Official code: training is `SAMPLED_Z`; classification and unknown detection inference are `MU_X`.
- Sibling USTC v6 reproduction: training is `SAMPLED_Z`; evaluation classification uses `MU_X`, and Native detection uses `mu_x` plus `logvar_x` in the forward KL.
- CipherSpectrum: Stage7 training sampled; Stage8A/9 classification/detection uses deterministic `MU_X`.

Therefore there is **no sampled-z versus mu_x mismatch** between official inference and Stage9.

## Prototype Definition

Each Known class has one 128-dimensional mean `mu_y`.  Prototype covariance is fixed identity.  Means are trainable, then reset to deterministic training-class means at epochs 50/80; the released parameter replacement disconnects the new parameter from the original optimizer.  See `outputs/official_code/prototype_definition.md`.

## Native Score

Official Native anomaly score is:

`s_unknown(x)=min_y 0.5*(||mu_x-mu_y||^2 + sum(exp(logvar_x)-logvar_x-1))`.

It is forward `KL[q_x || N(mu_y,I)]`; it is not Euclidean-only, Mahalanobis refitting, symmetric KL, or a GMM.  Stage9 stores the negative of the same minimum KL so higher means more Known.  This sign/label reversal preserves ranking when applied consistently.  Stage9 uses the stability-clamped logvar path.

## Threshold Calibration

The paper states a global threshold chosen so 95% of overall Known validation is accepted.  It does not state per-class 95%.  Released code instead maximizes Youden J on labeled Known+Unknown test scores.  Stage9 uses a global linear P05 of Known Validation Native scores, achieving 0.94994/0.94995/0.95000 validation acceptance for Low/Medium/High.  This is paper-aligned and leakage-resistant, but not an exact reproduction of the released evaluator.

## Metric Definitions

Paper F1 averaging is not explicitly defined.  Released closed-set code uses weighted multiclass F1; released open-set code uses binary F1 with Unknown positive.  Stage9 reports Known multiclass Macro-F1 separately from UFAR and Known-positive AUROC.  Paper open-world F1 and Stage9 Known Macro-F1 are `NOT_COMPARABLE`.

## Data Split

Paper: 8:1:1, five-fold, group isolation unspecified.  Released code: prebuilt train/test NPZ, with test reused for validation and threshold selection.  The sibling USTC v6 reproduction uses five independently seeded grouped-image-disjoint 8:1:1 splits over one locally reconstructed flow-image pool; it is not capture-isolated and is not the author fold assignment.  CipherSpectrum uses a frozen group-aware split.  Cross-group generalization is a **potential explanation** for lower performance, not a causally proven explanation.

## Preprocessing

The four layers agree on the central 8x(80+48) 32x32 byte representation, IP masking, port retention in released/local code, padding, truncation, and training crop/flip.  Important uncertainty/differences remain in original flow construction, invalid/non-IP handling, direction/capture structure, and split provenance.  See `outputs/summary/preprocessing_comparison.csv`.

## Same-Encoder Detector Comparison

Native, Single-Full-K1, and Multi-Global-K2 all reuse the same Stage7 checkpoint and the same deterministic `mu_x`.  Native additionally uses `logvar_x`; K1/K2 apply the frozen train-only StandardScaler/PCA64 to `mu_x` and different frozen density scores.

| Setting | Native AUROC | K1 AUROC | K2 AUROC |
|---|---:|---:|---:|
| Low | 0.500830 | 0.880005 | 0.892480 |
| Medium | 0.497970 | 0.779753 | 0.795385 |
| High | 0.472133 | 0.804478 | 0.821250 |

This is strong evidence that score/distribution modeling, not only the encoder, explains a substantial part of the external gap.

## Failure Decomposition

| Factor | Evidence |
|---|---|
| F1 Representation limitation | EVIDENCE_MODERATE |
| F2 Native detector-score mismatch | EVIDENCE_STRONG |
| F3 Dataset/domain shift | EVIDENCE_MODERATE |
| F4 Protocol difficulty shift | EVIDENCE_MODERATE |
| F5 Implementation mismatch | EVIDENCE_STRONG |
| F6 Metric-definition mismatch | EVIDENCE_STRONG |

The strongest direct explanation is that the Native fixed-identity prototype-posterior KL score is poorly aligned/calibrated for CipherSpectrum geometry: changing only the frozen post-encoder density model raises AUROC by roughly 0.28-0.39.  Domain and group-aware protocol shifts plausibly compound this, but were not causally isolated.

## What Can Be Claimed

- The sibling USTC project has a complete 15-run v6 local-PCAP A-scenario campaign and a separate high all-known approximation; its final matrix correctly labels both as local approximations rather than paper-exact results.
- A-1 is close to the paper locally, while A-2 and A-3 show material protocol-level gaps; the local campaign does not support a blanket statement that every USTC open-world scenario was reproduced closely.
- Official inference and Stage9 both use deterministic `mu_x`; sampled-z inference mismatch is not the issue.
- Stage9 Native shares the official prototype/KL family but is not an exact released-evaluator reproduction.
- Under the frozen CipherSpectrum protocol, Native ranking is near random and K1/K2 density scores over the same `mu_x` are substantially stronger.
- The paper/code/local protocols and F1 definitions have material comparability limits.

## What Cannot Be Claimed

- It cannot be claimed that Open-Detect universally fails to generalize.
- It cannot be claimed that paper open-world F1 equals Stage9 Known Macro-F1.
- It cannot be claimed that group-aware splitting, domain shift, representation quality, or score mismatch alone is the proven cause.
- It cannot be claimed that the old nested 98.6158%/98.7371% result is the sibling Open-Detect reproduction.
- It cannot be claimed that the sibling v6 result is author-exact: author flow identities, folds and seeds are unavailable, and local source-fragment selection was partly inferred.
- It cannot be claimed that Stage9 is bit-for-bit identical to the released evaluator.

## Final Gate

**Gate B — HIGH-PRIORITY PROTOCOL MISMATCH**

`EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE`

The decisive mismatch is threshold calibration (official mixed labeled test/Youden-J versus Stage9 Known-validation/P05), compounded by the stability wrapper, split difficulty, and metric-definition differences.  No retraining or test rerun was performed.
