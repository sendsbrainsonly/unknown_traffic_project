# Experiment results: stage14d-vnat-frozen-open-set-mechanism-evaluation-20260918-v1

- Status: `success`
- Experiment type: `evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-18T12:40:17Z`
- Completed (UTC): `2026-09-18T13:00:02Z`
- Objective: compare Native Open-Detect, DES-v0, and fixed DES-v1 on all 15 frozen VNAT protocols using the identical Stage 14C Native checkpoint for each protocol.

## Data and split

- Dataset: VNAT clean pool, 23,449 flows.
- Protocol: frozen Stage 14B `Low/Medium/High × seeds 2022–2026`.
- Stage 14B freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` before and after.
- Known Train constructs empirical support only; Known Validation fits M2 median/MAD and all P95 thresholds; Known Test and Unknown Test are final evaluation only.
- Unknown support, normalization, and threshold-calibration samples: `0 / 0 / 0`.

## Configuration and execution

- M0: frozen learned-prototype minimum Gaussian KL score.
- M1: Known-Train deterministic-`mu` class centroid, minimum squared Euclidean distance.
- M2: fixed class-conditional `k=10`, Known-Val median/MAD, `0.5 Zg + 0.5 Zl`.
- Threshold: Known Validation P95, NumPy `method=higher`; Unknown is positive.
- Inference batch: 64, matching the Stage 14C Native validation loader.
- Four project-owned GPU workers used physical GPUs `0,2,3,4`; no optimizer, backward, training, or checkpoint update occurred.
- 15/15 Native Validation metric-parity checks passed; 15 checkpoint hashes remained unchanged.

## Core results

| Scope | Method | AUROC mean ± std | AUPRC mean ± std | UFAR mean | Known FRR mean | Known Macro-F1 mean |
|---|---|---:|---:|---:|---:|---:|
| Low | M0 | 0.873644 ± 0.082165 | 0.816955 ± 0.130406 | 0.568802 | 0.045767 | 0.792093 |
| Low | M1 | 0.852855 ± 0.115234 | 0.819125 ± 0.167711 | 0.582839 | 0.046481 | 0.775320 |
| Low | M2 | 0.850350 ± 0.131341 | 0.823039 ± 0.180468 | 0.554040 | 0.046240 | 0.775320 |
| Medium | M0 | 0.804528 ± 0.125601 | 0.895331 ± 0.093052 | 0.651804 | 0.046496 | 0.817843 |
| Medium | M1 | 0.846212 ± 0.060854 | 0.913170 ± 0.072536 | 0.665543 | 0.045121 | 0.805771 |
| Medium | M2 | 0.850156 ± 0.068707 | 0.917105 ± 0.076955 | 0.629824 | 0.048113 | 0.805771 |
| High | M0 | 0.834433 ± 0.106579 | 0.909919 ± 0.081649 | 0.548904 | 0.047468 | 0.878881 |
| High | M1 | 0.856052 ± 0.056147 | 0.922603 ± 0.076503 | 0.549458 | 0.046483 | 0.860158 |
| High | M2 | 0.858747 ± 0.065776 | 0.924952 ± 0.080593 | 0.523876 | 0.049725 | 0.860158 |
| Overall | M0 | 0.837535 ± 0.102664 | 0.874068 ± 0.104990 | 0.589837 | 0.046577 | 0.829606 |
| Overall | M1 | 0.851706 ± 0.075966 | 0.884966 ± 0.116405 | 0.599280 | 0.046029 | 0.813750 |
| Overall | M2 | 0.853085 ± 0.086780 | 0.888365 ± 0.123087 | 0.569247 | 0.048026 | 0.813750 |

Overall paired comparisons:

- M1−M0: ΔAUROC `+0.014171`, 95% CI `[-0.027038, +0.064536]`, AUROC wins `6/15`; ΔAUPRC `+0.010898`, ΔUFAR `+0.009443`, ΔKnown-FRR `−0.000548`.
- M2−M1: ΔAUROC `+0.001378`, 95% CI `[-0.006574, +0.009444]`, AUROC wins `9/15`; ΔAUPRC `+0.003399`, ΔUFAR `−0.030033`, ΔKnown-FRR `+0.001998`.
- M2−M0: ΔAUROC `+0.015549`, 95% CI `[-0.028604, +0.068649]`, AUROC wins `6/15`; ΔAUPRC `+0.014297`, ΔUFAR `−0.020590`, ΔKnown-FRR `+0.001449`.

## Preserved evidence

- Required aggregate CSVs and report are in this directory.
- `artifacts/<protocol>/` contains latent outputs, score arrays, per-sample scores, per-class analysis, result metadata, and SUCCESS marker for all 15 protocols.
- `checkpoint_freeze.json` and `completion_verification.json` preserve before/after integrity evidence.
- `preliminary_batch512_artifacts/` preserves the excluded first attempt and its High-2025 parity failure; no preliminary score enters formal tables.
- Formal execution evidence: `.tmux-task/s14d_formal_gpu{0,2,3,4}_20260918_1412/` and `.tmux-task/s14d_final_aggregate_20260918_1442/`.

## Limitations

- Natural frozen sample prevalence is used; AUPRC therefore differs with Unknown-class composition and should be interpreted through paired protocol comparisons.
- Protocols are 15 frozen class compositions from one dataset, not 15 independent datasets; bootstrap resamples paired protocols.
- Stage 14B Known splits are flow-random and not capture-disjoint, as already frozen and disclosed.
- M1's positive overall mean is dominated by a few protocols: it improves only 6/15 and decreases on Low on average.
- With a hard class (`rsync/scp/sftp`) Unknown, M1 improves 3/12 protocols; with no hard class Unknown, it improves 3/3. This composition dependence limits generalization.

## Conclusion and next step

Final conclusion: `DECOUPLING_ONLY_PARTIALLY_CONFIRMED`.

Representation–Support Decoupling has a positive overall mean but is not stable across settings/seeds and its bootstrap CI crosses zero. Global+Local DES-v1 lowers UFAR on average relative to DES-v0, but its AUROC improvement is small, occurs in 9/15 protocols, and its CI also crosses zero. The evidence does not justify promoting DES-v0 or DES-v1 as the paper's main method; retain `Open-Detect baseline` as the formal baseline. No next stage or tuning experiment was started.
