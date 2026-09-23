# Stage 6 — CipherSpectrum Strict Unknown-Free Open-Set Protocol Freeze

## Goal

Freeze the external CipherSpectrum corpus, Low/Medium/High class-held-out settings, group-aware Known splits, Open-Detect input representation, and Strict Unknown-Free boundary before any model training or Unknown evaluation.

## Why CipherSpectrum Is Primary External Benchmark

CipherSpectrum has 40 official website/SNI-oriented classes with a balanced canonical three-cipher view. It has not been used to obtain the USTC method conclusions, so it is the primary independent external-validation candidate. This role is conditional on respecting `split_group_id` and the frozen Unknown boundary.

## Canonical 120k Corpus

- Source: existing `cipherspectrum_official40_manifest.csv` only.
- 120,000 readable PCAPs, 40 classes, 3,000 samples/class.
- Every class contains 1,000 `aes-128-gcm`, 1,000 `aes-256-gcm`, and 1,000 `chacha20-poly1305` samples.
- `split_group_id` coverage: 120,000/120,000.
- The 3,919 existing `manual_review_required` flags are retained as metadata; labels are not redefined.

## Why MIX Is Excluded

MIX has no exact SHA-256 or normalized packet-fingerprint duplicates against the canonical sources, but 16,596/18,260 MIX groups (90.89%) share `split_group_id` with them. It is therefore `PARTIALLY_OVERLAPPING`, not an independent fourth source. MIX and `getpocket.com` contribute zero rows to the canonical corpus and all frozen splits.

## Official40 Labels

The existing official40 directory-label mapping is reused without guessing or relabeling. `getpocket.com`, macOS resource forks, missing/invalid PCAPs, and noncanonical source groups are excluded before protocol construction.

## Group Dependency

`split_group_id` is the indivisible collection/session proxy. When a selected Unknown class touches a group, every Known-class row in that group is assigned `PURGED_GROUP_OVERLAP`. Active Known Train/Validation/Test groups are mutually disjoint and disjoint from Unknown Test.

## Openness Settings

| Setting | Seed | Known | Unknown | Openness |
|---|---:|---:|---:|---:|
| Low | 42 | 38 | 2 | 5% |
| Medium | 43 | 34 | 6 | 15% |
| High | 44 | 30 | 10 | 25% |

All three are mandatory future settings; none may be selected or discarded using model results.

## Unknown Class Selection

Unknown candidates are complete class sets drawn by `numpy.random.default_rng` (PCG64) over lexicographically sorted class names. Rejected candidates remain recorded in `fold_selection_audit.csv`.

- Low candidate 3: `coinbase.com`, `segment.com`.
- Medium candidate 15: `doubleclick.net`, `hotjar.com`, `hs-scripts.com`, `naver.com`, `onetrust.com`, `ubuntu.com`.
- High candidate 15: `cookielaw.org`, `doubleclick.net`, `gmx.net`, `hsadspixel.net`, `naver.com`, `onetrust.com`, `sharethis.com`, `twitter.com`, `wistia.com`, `xnxx-cdn.com`.

## Known Group-Aware Split

| Setting | Known Train | Known Validation | Known Test | Unknown Test | Purged Known |
|---|---:|---:|---:|---:|---:|
| Low | 75,692 | 16,242 | 16,246 | 6,000 | 5,820 |
| Medium | 56,641 | 12,168 | 12,162 | 18,000 | 21,029 |
| High | 49,976 | 10,740 | 10,746 | 30,000 | 18,538 |

The fixed assignment objective uses only total sample, per-class, and per-class×cipher-source ratio errors. It does not use embeddings, density, scores, or model metrics. Every Known class has at least 500 Train, 100 Validation, 100 Test, and at least two cipher sources in each active split.

## 70/15/15 Rationale

The target ratios are pre-registered as 70/15/15 because future class- and component-specific Known-only calibration needs more validation support. Actual aggregate ratios differ by less than 0.07 percentage points in every setting. The protocol must not be changed to 80/10/10 after results are observed.

## Calibration Capacity

Validation min/P25/median/P75/max:

- Low: 249/423/450/450/460.
- Medium: 130/276.25/402.5/450/461.
- High: 186/276.75/374.5/450/458.

For the fixed `component validation n < 30` fallback, 50/50 produces no fallback. At 80/20 the fallback counts are Low 0/38, Medium 2/34, High 0/30; at 90/10 they are 1/38, 11/34, 9/30; at 95/5 all classes fall back.

## Input Leakage Risk

The frozen Open-Detect representation uses the first eight packets, 80 masked IPv4/header bytes plus 48 Raw payload bytes per packet, zero padding/truncation, and a 32×32 uint8 image. The pre-protocol 400-PCAP audit found domain 4.25%, SNI 0%, class token 4.50%, combined 4.50%: `LOW_DIRECT_DOMAIN_LEAKAGE`. IP masking passed 3,200/3,200 checks. Ports remain visible 3,200/3,200 and are frozen as `PORT_SHORTCUT_RISK = PRESENT`; Stage 6 does not change the representation.

## Strict Unknown-Free Rules

Unknown data are forbidden from encoder training/validation, checkpoint selection, scaler/PCA/K1/K2 fitting, class/component/global boundary calibration, and hyperparameter selection. Unknown classes are allowed only in the future final test. Low/Medium/High require separately trained Known-only encoders; full-40 checkpoints and cross-setting warm starts are prohibited.

## Future Baselines

The pre-registered comparison candidates are Open-Detect Native, Single-Full K1, Multi-Global K2, Class-P05 K2, and Component-P05 K2. Stage 6 freezes this matrix but executes none of it.

## DGSB Status

`DGSB-v1 = NOT YET CLEARED FOR EXTERNAL EVALUATION`. It is excluded from the CipherSpectrum formal comparison because Stage 5.5 found no independent Global-Gate effect in USTC A-3. A Known-only DGSB-v2 may be added only if separately frozen before the first CipherSpectrum Unknown inference.

## Frozen Protocol

The canonical manifest, three fold JSON files, 360,000-row setting/sample manifest, candidate audit, calibration audit, preprocessing identity, Unknown-Free ledger, and 16 protocol hashes are frozen. Independent verification passes all counts, group-overlap constraints, minimum class sizes, source coverage, and hashes. `created_before_unknown_evaluation = true`.

## Final Gate

**READY.** No training, embedding extraction, scaler/PCA/GMM fitting, calibration, prediction, UFAR, AUROC, AUPRC, or Unknown inference was executed.

## Next Step

Stop at the protocol boundary. A future explicitly authorized task may train three independent Known-only Open-Detect encoders, but it must not inspect CipherSpectrum Unknown results until all method choices intended for external comparison are frozen.
