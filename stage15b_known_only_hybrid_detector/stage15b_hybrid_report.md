# Stage 15B — Known-Only Hybrid Detector Development

## Scope and controls

All encoders and source scores are frozen. Empirical CDFs and thresholds use Known Validation only. Unknown/Test calibration count is zero. No weights, k, signals, or gating model were searched. VNAT and USTC are development datasets, not untouched external validation; ISCX results are supplementary retrospective checks.

## Mean results

| Dataset | Method | AUROC | AUPRC | UFAR | Known FRR |
|---|---|---:|---:|---:|---:|
| VNAT | H0 | 0.837535 | 0.874068 | 0.589837 | 0.046577 |
| VNAT | DES0 | 0.851706 | 0.884966 | 0.599280 | 0.046029 |
| VNAT | DES1 | 0.853085 | 0.888365 | 0.569247 | 0.048026 |
| VNAT | H1 | 0.858118 | 0.890088 | 0.537804 | 0.047479 |
| VNAT | H2 | 0.859116 | 0.893043 | 0.524487 | 0.048159 |
| VNAT | H3 | 0.849366 | 0.886613 | 0.543704 | 0.045753 |
| USTC-TFC2016 | H0 | 0.939528 | 0.950011 | 0.139198 | 0.044650 |
| USTC-TFC2016 | DES0 | 0.970982 | 0.971302 | 0.081800 | 0.047144 |
| USTC-TFC2016 | DES1 | 0.979954 | 0.975291 | 0.072059 | 0.049180 |
| USTC-TFC2016 | H1 | 0.962698 | 0.965614 | 0.096819 | 0.044406 |
| USTC-TFC2016 | H2 | 0.972491 | 0.970216 | 0.085047 | 0.044567 |
| USTC-TFC2016 | H3 | 0.953250 | 0.954498 | 0.110663 | 0.048792 |
| ISCX-VPN | H0 | 0.553024 | 0.702165 | 0.964023 | 0.051647 |
| ISCX-VPN | DES0 | 0.576476 | 0.714070 | 0.957451 | 0.050632 |
| ISCX-VPN | H1 | 0.561670 | 0.706639 | 0.960380 | 0.050885 |
| ISCX-VPN | H3 | 0.521592 | 0.686597 | 0.968148 | 0.050716 |
| ISCXTor2016 | H0 | 0.524614 | 0.741977 | 0.971589 | 0.055094 |
| ISCXTor2016 | DES0 | 0.594347 | 0.778673 | 0.952806 | 0.053543 |
| ISCXTor2016 | H1 | 0.572999 | 0.766987 | 0.960517 | 0.053787 |
| ISCXTor2016 | H3 | 0.555676 | 0.772122 | 0.935456 | 0.058626 |

## Candidate gate

| Candidate | VNAT ΔAUROC | USTC ΔAUROC | pooled ΔAUROC | negatives | worst ΔAUROC | rsync mean/worst | scp mean/worst | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H1 | 0.020583 | 0.023170 | 0.021876 | 12 | -0.060650 | -0.105989/-0.145269 | -0.029232/-0.047784 | PARTIAL |
| H2 | 0.021581 | 0.032963 | 0.027272 | 11 | -0.056453 | -0.125066/-0.173188 | -0.035960/-0.054568 | PARTIAL |
| H3 | 0.011831 | 0.013722 | 0.012776 | 14 | -0.057898 | -0.108114/-0.186310 | -0.050397/-0.077741 | PARTIAL |

## Paired AUROC comparison against H0

| Scope | Candidate | mean delta | positive/total | paired bootstrap 95% CI |
|---|---|---:|---:|---:|
| USTC-TFC2016 | H1 | 0.023170 | 12/15 | [0.010808, 0.036963] |
| USTC-TFC2016 | H2 | 0.032963 | 12/15 | [0.014655, 0.053603] |
| USTC-TFC2016 | H3 | 0.013722 | 10/15 | [0.004590, 0.023505] |
| VNAT | H1 | 0.020583 | 6/15 | [-0.011622, 0.059801] |
| VNAT | H2 | 0.021581 | 7/15 | [-0.011707, 0.061969] |
| VNAT | H3 | 0.011831 | 6/15 | [-0.017422, 0.047027] |
| PRIMARY_POOLED | H1 | 0.021876 | 18/30 | [0.004168, 0.041910] |
| PRIMARY_POOLED | H2 | 0.027272 | 19/30 | [0.007297, 0.048631] |
| PRIMARY_POOLED | H3 | 0.012776 | 16/30 | [-0.003043, 0.030354] |

## Mechanism answers

1. **Q1 — H1 only partially preserves OD and DES advantages.** It improves over H0 by 0.020583 on VNAT and 0.023170 on USTC, but retains 12 negative protocols, exactly the same as DES-v0's 12, and is below DES-v0 on USTC. Worst protocol delta is -0.060650.
2. **Q2 — H2 adds real but modest ranking value, not only an operating-point change.** Mean H2−H1 AUROC is 0.000998 on VNAT and 0.009793 on USTC; negatives fall from 12 to 11 and UFAR improves further. However, H2 worsens both rsync and scp relative to H1, so the local term is not a stable failure-regime repair.
3. **Q3 — H3 does not repair posterior-uncertainty support-overlap failures.** rsync mean/worst ΔAUROC remains -0.108114/-0.186310; scp remains -0.050397/-0.077741. H3 also has more negative protocols than DES-v0.
4. **Q4 — no Hybrid passes the worst-case requirement.** DES-v0 has 12 negative primary protocols. H1/H2/H3 have 12/11/14; worst deltas are -0.060650/-0.056453/-0.057898. H2 reduces the count by only one, below the preregistered reduction of two.

## VNAT failure regimes

| Unknown | Method | AUROC | ΔAUROC vs H0 | negative units | large failures | worst delta |
|---|---|---:|---:|---:|---:|---:|
| rsync | DES0 | 0.383843 | -0.161182 | 3 | 3 | -0.276327 |
| rsync | H0 | 0.545025 | 0.000000 | 0 | 0 | 0.000000 |
| rsync | H1 | 0.439035 | -0.105989 | 4 | 3 | -0.145269 |
| rsync | H2 | 0.419958 | -0.125066 | 4 | 3 | -0.173188 |
| rsync | H3 | 0.436911 | -0.108114 | 4 | 3 | -0.186310 |
| scp | DES0 | 0.771014 | -0.045337 | 4 | 1 | -0.109834 |
| scp | H0 | 0.816351 | 0.000000 | 0 | 0 | 0.000000 |
| scp | H1 | 0.787118 | -0.029232 | 5 | 0 | -0.047784 |
| scp | H2 | 0.780390 | -0.035960 | 5 | 0 | -0.054568 |
| scp | H3 | 0.765953 | -0.050397 | 5 | 0 | -0.077741 |
| sftp | DES0 | 0.736672 | 0.202935 | 2 | 0 | -0.047918 |
| sftp | H0 | 0.533738 | 0.000000 | 0 | 0 | 0.000000 |
| sftp | H1 | 0.683226 | 0.149488 | 3 | 0 | -0.073669 |
| sftp | H2 | 0.680609 | 0.146871 | 3 | 0 | -0.068687 |
| sftp | H3 | 0.662227 | 0.128490 | 3 | 0 | -0.047729 |

## Supplementary retrospective check

| Dataset | Candidate | mean ΔAUROC vs H0 | positive/total | bootstrap 95% CI |
|---|---|---:|---:|---:|
| ISCX-VPN | H1 | 0.008646 | 13/15 | [0.003619, 0.012991] |
| ISCX-VPN | H3 | -0.031433 | 1/15 | [-0.042656, -0.018849] |
| ISCXTor2016 | H1 | 0.048385 | 14/15 | [0.033964, 0.062690] |
| ISCXTor2016 | H3 | 0.031062 | 13/15 | [0.016549, 0.045525] |

## Final decision

**HYBRID_PARTIAL**

Recommended single formula for any controlled continuation: **H1**.

This is a partial, not confirmed, recommendation. It follows the preregistered cross-dataset stability/simplicity rule: H2's small aggregate advantage is offset by worse rsync/scp behavior, while H3 is less stable. Full-gate candidates: none. No dataset-specific formula is selected.

## Limitations

- This is method development on VNAT and USTC, not untouched external validation.
- ISCX-VPN/ISCXTor2016 lack frozen DES-v1, so H2 is unavailable there.
- Percentile fusion changes score scale and its P95 operating point; AUROC and operating metrics must be interpreted separately.
- Candidate selection uses development Unknown outcomes only for the preregistered Gate; Unknown never enters calibration or parameter fitting.
