# Experiment results: stage15b-known-only-hybrid-detector-20260918-v1

- Status: `success`
- Experiment type: `method_development`
- Claim scope: `development-only diagnostic`
- Created (UTC): `2026-09-18T14:59:00Z`
- Objective: evaluate preregistered Known-Validation-calibrated H1/H2/H3 hybrids on frozen VNAT/USTC development protocols, with retrospective ISCX checks.

## Data and split

- Primary development datasets: VNAT and USTC-TFC2016, 15 frozen protocols each.
- Supplementary retrospective datasets: ISCX-VPN and ISCXTor2016, 15 frozen protocols each.
- VNAT and USTC are not treated as untouched external validation.
- Existing Known Validation/Test and Unknown Test roles were preserved; Unknown/Test calibration samples were both zero.
- 60 protocols, 300 protocol-method metric rows, 938 failure-regime rows, and 60 compressed per-sample score bundles were preserved.
- All 255 frozen source files had identical SHA256 before and after execution.

## Configuration and execution

- ECDF: `count(Known-Val score <= x) / n_val`, fitted separately for each signal using Known Validation only.
- H1: `max(A_OD, A_DES0)`.
- H2: `max(A_OD, A_DES1)` where frozen DES-v1 exists.
- H3: `max(A_proto, A_DES0, A_Vpost)`.
- Threshold: each method's Known-Validation P95, `numpy method=higher`.
- No continuous weights, extra signals, k values, or gating models were searched.
- No encoder training, fine-tuning, or inference was run; only frozen latent/scores/checkpoints were read.
- CPU-only tmux execution: `s15b_hybrid_eval_20260918_1736`, exit code 0.
- Unit tests: 4/4 passed.

## Core results

| Dataset | H0 AUROC | H1 AUROC | H2 AUROC | H3 AUROC | H1−H0 | H2−H0 | H3−H0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VNAT | 0.837535 | 0.858118 | 0.859116 | 0.849366 | +0.020583 | +0.021581 | +0.011831 |
| USTC-TFC2016 | 0.939528 | 0.962698 | 0.972491 | 0.953250 | +0.023170 | +0.032963 | +0.013722 |
| ISCX-VPN | 0.553024 | 0.561670 | unavailable | 0.521592 | +0.008646 | unavailable | −0.031433 |
| ISCXTor2016 | 0.524614 | 0.572999 | unavailable | 0.555676 | +0.048385 | unavailable | +0.031062 |

- Pooled primary H1 ΔAUROC: `+0.021876`, paired bootstrap 95% CI `[+0.004168,+0.041910]`, wins `18/30`.
- Pooled primary H2 ΔAUROC: `+0.027272`, CI `[+0.007297,+0.048631]`, wins `19/30`.
- Pooled primary H3 ΔAUROC: `+0.012776`, CI `[−0.003043,+0.030354]`, wins `16/30`.
- All candidates improved mean AUPRC and UFAR on both primary datasets.
- DES-v0 had 12 negative primary protocols; H1/H2/H3 had `12/11/14`. None met the preregistered reduction of at least two.
- H1/H2/H3 worst primary protocol ΔAUROC: `−0.060650/−0.056453/−0.057898`.
- H1 did not repair rsync: mean/worst ΔAUROC `−0.105989/−0.145269`; scp remained `−0.029232/−0.047784`.
- H2 improved aggregate means but worsened rsync/scp relative to H1.
- H3 did not rescue posterior-uncertainty failure regimes and was less stable; on ISCX-VPN it degraded AUROC by `−0.031433`.
- Final Gate: `HYBRID_PARTIAL`.
- Preregistered single-formula recommendation: `H1`, as a partial development candidate only.

## Preserved evidence

- `stage15b_hybrid_results.csv`
- `stage15b_protocol_comparison.csv`
- `stage15b_failure_regime_analysis.csv`
- `stage15b_hybrid_report.md`
- `candidate_gate.json`
- `completion_verification.json`
- `source_hashes_before.json` / `source_hashes_after.json`
- `artifacts/*/hybrid_scores.npz` and per-protocol metadata for all 60 protocols
- `config.json`, analysis source, and unit tests
- tmux log: `.tmux-task/s15b_hybrid_eval_20260918_1736/output.log`

## Limitations

- This is development, not untouched external validation.
- ISCX datasets have no frozen DES-v1, so H2 cannot be checked there.
- VNAT candidate-level bootstrap intervals cross zero despite positive means.
- H1 is recommended by the frozen simplicity/stability rule, not because it passed the full Gate.
- Development Unknown outcomes were used only to compare the preregistered formulas and apply the Gate, never for ECDF/threshold/weight fitting.

## Conclusion and next step

`HYBRID_PARTIAL`: fixed percentile-max fusion improves average AUROC/AUPRC and UFAR, but does not solve protocol instability or the rsync/scp support-overlap regime. H1 is the only recommended continuation candidate because H2's modest aggregate gain comes with worse hard-class behavior and H3 is less stable. It must not be described as a confirmed universally superior detector, and no next stage was started automatically.
