# CipherSpectrum Strict Unknown-Free Open-Set Protocol

`created_before_unknown_evaluation = true`

## Canonical corpus

- 40 official classes, 120,000 PCAPs, 3,000/class.
- Each class contains 1,000 AES-128, 1,000 AES-256, and 1,000 CHACHA samples.
- MIX=0; getpocket.com=0; `split_group_id` coverage=100%.
- MIX remains `PARTIALLY_OVERLAPPING` and is excluded because 16,596/18,260 collection groups overlap canonical sources.

## Frozen openness and splits

| Setting | Seed | Known classes | Unknown classes | Known Train | Known Validation | Known Test | Unknown Test | Purged Known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 42 | 38 | 2 | 75,692 | 16,242 | 16,246 | 6,000 | 5,820 |
| Medium | 43 | 34 | 6 | 56,641 | 12,168 | 12,162 | 18,000 | 21,029 |
| High | 44 | 30 | 10 | 49,976 | 10,740 | 10,746 | 30,000 | 18,538 |

Unknown class candidates use NumPy PCG64 on lexicographically sorted class names. Entire candidate sets are rejected if fixed metadata-only group-split constraints fail. No model output participates in selection.

Known samples use a pre-registered 70/15/15 group-aware assignment. Its fixed objective contains only total, per-class, and per-class×cipher-source ratio errors. Every Unknown-touched group purges its Known rows before assignment. Active Train/Validation/Test and Unknown Test group overlaps are exactly zero.

## Input preprocessing

- First 8 packets per flow.
- Each packet: 80 masked-IPv4/header bytes + 48 Raw payload bytes.
- Per-region truncate/right-zero-pad; missing packets zero-pad.
- 1,024 bytes reshaped to 32×32.
- IP masking frozen; ports retained as `KNOWN_SHORTCUT_RISK`.
- Audited model input leakage: domain 17/400, SNI 0/400, token 18/400, combined 18/400 — `LOW_DIRECT_DOMAIN_LEAKAGE`.

## Strict Unknown-Free boundary

Unknown classes are forbidden from encoder train/validation, checkpoint selection, scaler/PCA/K1/K2 fitting, all boundary calibration, global threshold calibration, and hyperparameter selection. They are allowed only at the future final test. Low/Medium/High must each train an independent Known-only encoder; full-40 and cross-setting checkpoint reuse are forbidden.

## Future comparison matrix

Frozen candidates: Open-Detect Native, Single-Full K1, Multi-Global K2, Class-P05 K2, Component-P05 K2. DGSB-v1 is **NOT YET CLEARED FOR EXTERNAL EVALUATION** and is not part of the formal external comparison. A future Known-only DGSB-v2 may be added only if frozen before the first CipherSpectrum Unknown inference.

No training, embedding extraction, scaler/PCA/GMM fit, threshold calibration, prediction, UFAR, AUROC, AUPRC, or Unknown inference was executed in Stage 6.

## Final Gate

**READY**, subject to the independent hash and invariant verifier. All twelve protocol conditions were satisfied by the generated metadata-only freeze.
