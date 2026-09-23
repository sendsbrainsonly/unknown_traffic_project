# Stage 15F-1A — Packet Window Sufficiency and Class-Conditional Feature Benchmark

## Scope and controls

All 15 formal runs use only frozen Known Train/Validation samples. Known Test and Unknown Test usage are both zero. The feature formula, seed, optimizer, 100-epoch budget, model topology, loss, and checkpoint rule are fixed; only the maximum packet window changes.

Stage 15R T8 used ordinary BatchNorm1d, which allowed structural padding to influence batch moments. This stage therefore preserves the legacy checkpoint only for inference parity and compares T8/T16/T32 using one preregistered mask-safe BatchNorm implementation. Invalid slots are zeroed and excluded from BatchNorm and masked pooling. The padding-invariance tests pass.

## Protocol-level results

| Dataset | Protocol | Window | Accuracy | Macro-F1 | Weighted-F1 | Best epoch | Runtime (s) | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iscx_tor | medium_seed2022 | T8 | 0.477868 | 0.419935 | 0.458324 | 83 | 75.0 | 21.4 |
| iscx_tor | medium_seed2022 | T16 | 0.479675 | 0.435929 | 0.466414 | 56 | 72.1 | 26.1 |
| iscx_tor | medium_seed2022 | T32 | 0.474255 | 0.438157 | 0.461855 | 67 | 72.6 | 35.7 |
| iscx_vpn | medium_seed2022 | T8 | 0.400372 | 0.364561 | 0.383996 | 48 | 91.1 | 21.4 |
| iscx_vpn | medium_seed2022 | T16 | 0.398510 | 0.372374 | 0.380919 | 88 | 91.8 | 26.1 |
| iscx_vpn | medium_seed2022 | T32 | 0.397890 | 0.365499 | 0.378481 | 96 | 88.2 | 35.7 |
| ustc | A-2 | T8 | 0.859892 | 0.879057 | 0.857514 | 99 | 1509.2 | 21.4 |
| ustc | A-2 | T16 | 0.869655 | 0.884775 | 0.867759 | 99 | 1483.7 | 26.1 |
| ustc | A-2 | T32 | 0.882855 | 0.894669 | 0.881592 | 98 | 1504.0 | 35.7 |
| vnat | medium_seed2025 | T8 | 0.885714 | 0.739700 | 0.877260 | 100 | 104.6 | 21.4 |
| vnat | medium_seed2025 | T16 | 0.884694 | 0.750873 | 0.877820 | 74 | 99.1 | 26.1 |
| vnat | medium_seed2025 | T32 | 0.881633 | 0.683690 | 0.869481 | 95 | 100.2 | 35.7 |
| vnat | medium_seed2026 | T8 | 0.976975 | 0.883760 | 0.975618 | 68 | 100.4 | 21.4 |
| vnat | medium_seed2026 | T16 | 0.979592 | 0.919876 | 0.979094 | 54 | 98.9 | 26.1 |
| vnat | medium_seed2026 | T32 | 0.962847 | 0.891900 | 0.960858 | 54 | 99.1 | 35.7 |

## Historical E2, mask-safe T8, and Native boundary

| Dataset | Protocol | Historical E2 Macro-F1 | Formal mask-safe T8 Macro-F1 | Δ mask-safe T8 - historical E2 | Native Macro-F1 |
|---|---|---:|---:|---:|---:|
| iscx_vpn | medium_seed2022 | 0.384810 | 0.364561 | -0.020249 | 0.769169 |
| iscx_tor | medium_seed2022 | 0.425010 | 0.419935 | -0.005075 | 0.676900 |
| vnat | medium_seed2025 | 0.761427 | 0.739700 | -0.021727 | 0.739755 |
| vnat | medium_seed2026 | 0.856865 | 0.883760 | +0.026895 | 0.983318 |
| ustc | A-2 | 0.887379 | 0.879057 | -0.008322 | 0.977545 |

Legacy inference parity is exact at the classification/confusion level. The historical-to-formal T8 delta is not a window effect: formal runs replace padding-sensitive BatchNorm with the preregistered mask-safe implementation and retrain the model. Window claims use only formal T16-T8 and T32-T8 deltas.


## Paired window deltas

| Dataset | Protocol | Comparison | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | Runtime ratio |
|---|---|---|---:|---:|---:|---:|
| iscx_vpn | medium_seed2022 | T16-T8 | -0.001862 | +0.007813 | -0.003077 | 1.007 |
| iscx_vpn | medium_seed2022 | T32-T8 | -0.002483 | +0.000938 | -0.005515 | 0.968 |
| iscx_vpn | medium_seed2022 | T32-T16 | -0.000621 | -0.006875 | -0.002438 | 0.961 |
| iscx_tor | medium_seed2022 | T16-T8 | +0.001807 | +0.015995 | +0.008090 | 0.962 |
| iscx_tor | medium_seed2022 | T32-T8 | -0.003613 | +0.018222 | +0.003531 | 0.968 |
| iscx_tor | medium_seed2022 | T32-T16 | -0.005420 | +0.002227 | -0.004560 | 1.007 |
| vnat | medium_seed2025 | T16-T8 | -0.001020 | +0.011172 | +0.000560 | 0.947 |
| vnat | medium_seed2025 | T32-T8 | -0.004082 | -0.056010 | -0.007779 | 0.958 |
| vnat | medium_seed2025 | T32-T16 | -0.003061 | -0.067183 | -0.008339 | 1.011 |
| vnat | medium_seed2026 | T16-T8 | +0.002616 | +0.036117 | +0.003477 | 0.985 |
| vnat | medium_seed2026 | T32-T8 | -0.014129 | +0.008140 | -0.014760 | 0.987 |
| vnat | medium_seed2026 | T32-T16 | -0.016745 | -0.027977 | -0.018237 | 1.002 |
| ustc | A-2 | T16-T8 | +0.009762 | +0.005718 | +0.010245 | 0.983 |
| ustc | A-2 | T32-T8 | +0.022963 | +0.015612 | +0.024078 | 0.997 |
| ustc | A-2 | T32-T16 | +0.013201 | +0.009894 | +0.013833 | 1.014 |

## Gate

Final Gate: `CLASS_CONDITIONAL_WINDOW_BENEFIT`

The stronger longer-window candidate is T16: mean ΔMacro-F1 +0.015363, positive protocols 5/5, worst protocol +0.005718.

Protocol-level numerical gate: PASS. Class-breadth constraint from the task specification: FAIL (23 positive, 23 negative, 6 unchanged out of 52; median ΔF1 +0.000000). A protocol-level pass is not promoted to a global window bottleneck when class breadth fails.

Qualifying class-conditional gains:

- iscx_tor / medium_seed2022 / Hangouts: ΔF1 +0.105713, net rescue +6.
- vnat / medium_seed2026 / netflix: ΔF1 +0.254545, net rescue +5.

Focus-class regressions below -0.05:

- vnat / medium_seed2025 / netflix: ΔF1 -0.068627.

## Native versus temporal complementarity

| Dataset | Protocol | Native Macro-F1 | T16 Macro-F1 | Native-only correct | T16-only correct | Both correct | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| iscx_vpn | medium_seed2022 | 0.769169 | 0.372374 | 763 | 138 | 504 | 206 |
| iscx_tor | medium_seed2022 | 0.676900 | 0.435929 | 358 | 129 | 402 | 218 |
| vnat | medium_seed2025 | 0.739755 | 0.750873 | 79 | 55 | 1679 | 147 |
| vnat | medium_seed2026 | 0.983318 | 0.919876 | 31 | 1 | 1871 | 8 |
| ustc | A-2 | 0.977545 | 0.884775 | 5301 | 657 | 37026 | 347 |

The two off-diagonal counts are the direct sample-level test of complementary information. A temporal-only rescue demonstrates information not used by Native; a simultaneous large Native-only count shows that the temporal branch is not a replacement for Native.

## Focus-class results

| Dataset | Protocol | Class | T8 F1 | T16 F1 | T32 F1 | Best longer Δ vs T8 |
|---|---|---|---:|---:|---:|---:|
| iscx_vpn | medium_seed2022 | Netflix | 0.344615 | 0.325000 | 0.320755 | -0.019615 |
| iscx_vpn | medium_seed2022 | Spotify | 0.322034 | 0.300000 | 0.274194 | -0.022034 |
| iscx_vpn | medium_seed2022 | Vimeo | 0.442308 | 0.445545 | 0.445545 | +0.003237 |
| iscx_tor | medium_seed2022 | FTP | 0.549372 | 0.514286 | 0.511688 | -0.035086 |
| iscx_tor | medium_seed2022 | Facebook | 0.377049 | 0.366667 | 0.403361 | -0.010383 |
| iscx_tor | medium_seed2022 | Hangouts | 0.216867 | 0.322581 | 0.322581 | +0.105713 |
| iscx_tor | medium_seed2022 | Vimeo | 0.415584 | 0.405405 | 0.392694 | -0.010179 |
| iscx_tor | medium_seed2022 | YouTube | 0.480663 | 0.478723 | 0.457766 | -0.001940 |
| vnat | medium_seed2025 | netflix | 0.833333 | 0.764706 | 0.777778 | -0.068627 |
| vnat | medium_seed2025 | rsync | 0.319149 | 0.348123 | 0.266160 | +0.028974 |
| vnat | medium_seed2025 | scp | 0.628366 | 0.621572 | 0.634315 | -0.006794 |
| vnat | medium_seed2025 | youtube | 0.875000 | 0.825397 | 0.833333 | -0.049603 |
| vnat | medium_seed2026 | netflix | 0.533333 | 0.787879 | 0.705882 | +0.254545 |
| vnat | medium_seed2026 | rsync | 0.949333 | 0.923913 | 0.830409 | -0.025420 |
| vnat | medium_seed2026 | vimeo | 0.943548 | 0.967213 | 0.958678 | +0.023665 |
| ustc | A-2 | Weibo | 0.726662 | 0.768514 | 0.830545 | +0.041852 |
| ustc | A-2 | WorldOfWarcraft | 0.998095 | 0.998095 | 0.998095 | +0.000000 |

## VNAT medium-2025 rsync confusion

| Method | rsync support | predicted rsync | predicted scp | predicted sftp |
|---|---:|---:|---:|---:|
| Native | 191 | 1 | 188 | 0 |
| T8 | 191 | 45 | 142 | 0 |
| T16 | 191 | 51 | 136 | 0 |
| T32 | 191 | 35 | 152 | 0 |

## Coverage versus performance

- T8 class truncation ratio versus T16-T8 class F1: Spearman rho=0.031212, p=0.826140, n=52.
- T8 class truncation ratio versus T32-T8 class F1: Spearman rho=0.119850, p=0.397388, n=52.

Observed T8 truncation is not a useful class-level predictor of the gain from T16 or T32 in these pilots. Lower truncation at longer windows therefore does not by itself establish a performance bottleneck.


## Direct answers

1. **Is the first-8-packet E2 globally window-limited?** No under the preregistered global criterion. The best longer window changes mean Macro-F1 by +0.015363 across five protocols.
2. **Do 16/32 packets improve ISCX-VPN and ISCXTor2016?** Best observed Macro-F1 deltas versus T8 are +0.007813 and +0.018222, respectively.
3. **Are benefits class-conditional?** 2 preregistered focus class-protocol units meet ΔF1>=0.05 with positive net rescue; 1 focus units regress below -0.05 for the selected longer window.
4. **Does a longer window repair VNAT rsync/scp?** medium-2025 rsync F1 is T8=0.319149, T16=0.348123, T32=0.266160; the confusion counts above show whether scp errors are actually removed rather than redistributed.
5. **Do temporal and Native inputs carry complementary information?** Yes only where both Native-only and temporal-only correct counts are non-zero; the protocol table reports the exact counts without using test data.
6. **Why was historical E2 weak?** Window insufficiency alone is not sufficient to explain the gap. Formal T8/T16/T32 isolate window effects, while the persistent Native gap and two-way rescue quantify feature/model limitations separately.

## T64 decision

The preregistered T64 gate is NOT MET. T64 was not run in this stage, as required.

## Recommendation

Retain T16 only as a class-conditional temporal candidate; do not promote it to the default standalone representation. Do not retain T32 as the default and do not enter T64. A later preregistered benchmark should prioritize richer flow statistics, burst structure, or structured byte inputs because the Native gap remains much larger than the window-only gains. No next-stage experiment was started.

## Interpretation boundary

Any difference between the formal T8 and the historical Stage 15R E2 score includes the preregistered padding-safe BatchNorm correction; only formal T16-T8 and T32-T8 comparisons identify window effects. Native-versus-temporal complementarity is descriptive and uses exactly the same Known Validation sample IDs.
