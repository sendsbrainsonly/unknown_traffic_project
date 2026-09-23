# Stage 10A — Open-Detect Protocol Audit

## Goal

Audit the Open-Detect paper, released implementation, sibling `Projects/Open-Detect` USTC reproduction, and frozen CipherSpectrum Stage6-9 pipeline without training, inference, refitting, threshold tuning, or upstream changes.

## Why This Audit Was Needed

Open-Detect is strong in the paper and has substantial local USTC reproduction evidence, while its Native detector is nearly random on frozen CipherSpectrum.  Before attributing that gap, the method-project boundary, latent mode, score, prototype, threshold, split, preprocessing, and metrics must be aligned.

## Sources

The local 2025 TIFS PDF, nested official checkout, sibling Open-Detect reproduction artifacts, 15 v6 checkpoints, and frozen Stage6-9 code/results are enumerated and hashed in `sources/source_manifest.json`.  Extracted paper text is preserved in `sources/open_detect_paper.txt`.  `Projects/Open-Detect` is read-only and is a peer of `Projects/unknown_traffic_project`.

## Paper vs Code

The core Gaussian prototype and KL equations are represented in code.  Released inference deterministically uses `mu_x`.  Important differences include the sampled-z Euclidean discriminative training term, test-NPZ validation, oracle test Youden-J threshold, decoder/entropy implementation issues, and prototype optimizer disconnection after reset.

## Paper vs USTC Reproduction

Overall `PARTIAL_MATCH`: the 20-class taxonomy, target count, central preprocessing, latent dimension, scenario classes, prototype family, and paper-style Known-validation threshold largely align.  The local v6 campaign uses corrected-paper code, locally inferred flow/source reconstruction, and five repeated grouped-image-disjoint splits rather than author folds.  Its final matrix records zero paper-exact reproductions.  The old nested 98.6158%/98.7371% result is explicitly excluded from this layer.

## Paper vs CipherSpectrum

CipherSpectrum preserves the central byte image, structural encoder, deterministic inference mean, and prototype KL family, but uses a different domain, group-aware split, local numerical wrapper, validation-only threshold, and different reporting metrics.

## Key Protocol Differences

The highest-priority difference is official test-oracle Youden-J calibration versus Stage9 Known-validation P05.  Group-aware versus unspecified/random-flow splitting and under-specified F1 definitions further limit direct comparison.

## Native Score Audit

Native is minimum forward KL from `N(mu_x,diag(exp(logvar_x)))` to one `N(mu_y,I)` prototype per Known class.  Stage9 stores its negative, an equivalent orientation reversal.  It is not a GMM or symmetric KL.

## Latent Mode Audit

Paper inference is `UNCLEAR`; official and Stage9 inference are both `MU_X`.  No sampled-z/mu_x mismatch was found.

## Metric Definition Audit

Paper F1 is under-specified.  Released open-set F1 is binary Unknown-positive, while Stage9 Known Macro-F1 is multiclass.  These are `NOT_COMPARABLE`.

## Final Gate

**Gate B — HIGH-PRIORITY PROTOCOL MISMATCH**; `EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE`.

## Next Step

No next stage is launched by this audit.  Any future action requires a separate pre-registered task and must not tune against the already-opened CipherSpectrum test set.
