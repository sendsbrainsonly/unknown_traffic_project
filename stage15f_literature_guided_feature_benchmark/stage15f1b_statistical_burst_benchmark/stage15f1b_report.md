# Stage 15F-1B — Statistical & Burst Feature Sufficiency Benchmark

## Scope and integrity

All experiments use only frozen Known Train and Known Validation rows. Known Test and Unknown Test feature values are not loaded. FULL_FLOW S12 reuses and exactly reproduces Stage 15R E3; all other runs retain the same LightGBM configuration and change only the preregistered feature set or observation horizon.

## Pilot Macro-F1

| Dataset | Protocol | S12 | S-A | S-AB | S-ABC | S-ABCD | S-Burst | Early16 S-Burst |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iscx_vpn | medium_seed2022 | 0.466724 | 0.467165 | 0.475349 | 0.480547 | 0.485081 | 0.495390 | 0.490843 |
| iscx_tor | medium_seed2022 | 0.487714 | 0.500626 | 0.511148 | 0.511119 | 0.533966 | 0.541617 | 0.538750 |
| vnat | medium_seed2025 | 0.774562 | 0.774044 | 0.789367 | 0.796711 | 0.784849 | 0.798648 | 0.817538 |
| vnat | medium_seed2026 | 0.846497 | 0.853558 | 0.857619 | 0.881219 | 0.851487 | 0.842826 | 0.940423 |
| ustc | A-2 | 0.926336 | 0.929230 | 0.931011 | 0.931980 | 0.943816 | 0.951185 | 0.942296 |

## Gate

Conclusions: `STATISTICAL_FEATURE_BENEFIT, BURST_COMPLEMENTARITY_CONFIRMED, CLASS_CONDITIONAL_BEHAVIOR_BENEFIT`

S-ABCD - S12 mean ΔMacro-F1=+0.019473, positive=5/5, worst=+0.004990.
S-Burst - S-ABCD mean ΔMacro-F1=+0.006094, positive=4/5, worst=-0.008661.
S-Burst - S12 mean ΔMacro-F1=+0.025567, positive=4/5, worst=-0.003671.

## VNAT medium-2025 rsync/scp audit

| Method | rsync correct/191 | rsync→scp | scp→rsync | rsync Recall | rsync F1 | scp Recall | scp F1 |
|---|---:|---:|---:|---:|---:|---:|
| Native | 1/191 | 188 | 0 | 0.005236 | 0.010417 | 0.982533 | 0.700935 |
| T16 | 51/191 | 136 | 51 | 0.267016 | 0.348123 | 0.742358 | 0.621572 |
| S12 | 59/191 | 132 | 52 | 0.308901 | 0.390728 | 0.759825 | 0.649254 |
| S-ABCD | 64/191 | 127 | 56 | 0.335079 | 0.411576 | 0.751092 | 0.651515 |
| S-Burst | 67/191 | 124 | 57 | 0.350785 | 0.424051 | 0.746725 | 0.652672 |

## Incremental feature-group ablation

| Increment | Meaning | Mean ΔMacro-F1 | Positive pilots | Worst | Best |
|---|---|---:|---:|---:|---:|
| S-A-S12 | A rates | +0.004558 | 4/5 | -0.000518 | +0.012912 |
| S-AB-S-A | B length distribution | +0.007974 | 5/5 | +0.001781 | +0.015323 |
| S-ABC-S-AB | C bidirectional | +0.007416 | 4/5 | -0.000029 | +0.023600 |
| S-ABCD-S-ABC | D IAT distribution | -0.000475 | 3/5 | -0.029732 | +0.022847 |
| S-Burst-S-ABCD | E burst | +0.006094 | 4/5 | -0.008661 | +0.013799 |

## Early-16 versus full-flow

Positive values mean full-flow is better; negative values mean Early-16 is better.

| Dataset | Protocol | Feature set | Full Macro-F1 | Early-16 Macro-F1 | Full − Early |
|---|---|---|---:|---:|---:|
| iscx_vpn | medium_seed2022 | S12 | 0.466724 | 0.468593 | -0.001870 |
| iscx_tor | medium_seed2022 | S12 | 0.487714 | 0.498673 | -0.010959 |
| vnat | medium_seed2025 | S12 | 0.774562 | 0.792303 | -0.017741 |
| vnat | medium_seed2026 | S12 | 0.846497 | 0.932908 | -0.086411 |
| ustc | A-2 | S12 | 0.926336 | 0.922405 | +0.003931 |
| iscx_vpn | medium_seed2022 | S-ABCD | 0.485081 | 0.490611 | -0.005530 |
| iscx_tor | medium_seed2022 | S-ABCD | 0.533966 | 0.537631 | -0.003666 |
| vnat | medium_seed2025 | S-ABCD | 0.784849 | 0.796256 | -0.011408 |
| vnat | medium_seed2026 | S-ABCD | 0.851487 | 0.938790 | -0.087303 |
| ustc | A-2 | S-ABCD | 0.943816 | 0.941596 | +0.002220 |
| iscx_vpn | medium_seed2022 | S-Burst | 0.495390 | 0.490843 | +0.004548 |
| iscx_tor | medium_seed2022 | S-Burst | 0.541617 | 0.538750 | +0.002867 |
| vnat | medium_seed2025 | S-Burst | 0.798648 | 0.817538 | -0.018891 |
| vnat | medium_seed2026 | S-Burst | 0.842826 | 0.940423 | -0.097597 |
| ustc | A-2 | S-Burst | 0.951185 | 0.942296 | +0.008889 |

## Focus-class behavior

Only classes that are Known in the frozen pilot are shown. VNAT `sftp` is Unknown in both pilots, and `scp` is Unknown in medium-2026, so their Known-Validation features are deliberately unavailable.

| Dataset | Protocol | Class | S12 F1 | S-ABCD F1 | S-Burst F1 | Best Δ vs S12 |
|---|---|---|---:|---:|---:|---:|
| iscx_vpn | medium_seed2022 | Netflix | 0.436090 | 0.461538 | 0.457565 | +0.025448 |
| iscx_vpn | medium_seed2022 | Spotify | 0.407643 | 0.456376 | 0.463576 | +0.055933 |
| iscx_vpn | medium_seed2022 | Vimeo | 0.540816 | 0.540816 | 0.568528 | +0.027712 |
| iscx_tor | medium_seed2022 | FTP | 0.509138 | 0.511749 | 0.511082 | +0.002611 |
| iscx_tor | medium_seed2022 | Facebook | 0.450000 | 0.461538 | 0.484848 | +0.034848 |
| iscx_tor | medium_seed2022 | Hangouts | 0.263736 | 0.391753 | 0.380000 | +0.128016 |
| iscx_tor | medium_seed2022 | Vimeo | 0.452107 | 0.476190 | 0.498024 | +0.045916 |
| iscx_tor | medium_seed2022 | YouTube | 0.501377 | 0.563050 | 0.568047 | +0.066670 |
| vnat | medium_seed2025 | netflix | 0.871795 | 0.871795 | 0.926829 | +0.055034 |
| vnat | medium_seed2025 | rsync | 0.390728 | 0.411576 | 0.424051 | +0.033322 |
| vnat | medium_seed2025 | scp | 0.649254 | 0.651515 | 0.652672 | +0.003418 |
| vnat | medium_seed2025 | youtube | 0.849315 | 0.898551 | 0.925373 | +0.076058 |
| vnat | medium_seed2026 | netflix | 0.606061 | 0.685714 | 0.628571 | +0.079654 |
| vnat | medium_seed2026 | rsync | 0.994764 | 0.997389 | 0.997389 | +0.002625 |
| vnat | medium_seed2026 | vimeo | 0.935484 | 0.951220 | 0.947368 | +0.015736 |
| ustc | A-2 | Weibo | 0.772153 | 0.823071 | 0.849797 | +0.077644 |
| ustc | A-2 | WorldOfWarcraft | 0.998095 | 0.997462 | 0.997462 | -0.000633 |

## System-level error complementarity

Counts use exactly the same Known-Validation samples. `lost` is baseline-correct/candidate-wrong; `rescued` is baseline-wrong/candidate-correct.

| Dataset | Protocol | Comparison | Lost | Rescued | Net rescue |
|---|---|---|---:|---:|---:|
| iscx_vpn | medium_seed2022 | S-Burst − Native | 606 | 170 | -436 |
| iscx_vpn | medium_seed2022 | S-Burst − T16 | 158 | 347 | +189 |
| iscx_vpn | medium_seed2022 | S-Burst − S-ABCD | 17 | 27 | +10 |
| iscx_tor | medium_seed2022 | S-Burst − Native | 321 | 157 | -164 |
| iscx_tor | medium_seed2022 | S-Burst − T16 | 41 | 106 | +65 |
| iscx_tor | medium_seed2022 | S-Burst − S-ABCD | 12 | 17 | +5 |
| vnat | medium_seed2025 | S-Burst − Native | 60 | 73 | +13 |
| vnat | medium_seed2025 | S-Burst − T16 | 61 | 98 | +37 |
| vnat | medium_seed2025 | S-Burst − S-ABCD | 15 | 20 | +5 |
| vnat | medium_seed2026 | S-Burst − Native | 16 | 7 | -9 |
| vnat | medium_seed2026 | S-Burst − T16 | 9 | 30 | +21 |
| vnat | medium_seed2026 | S-Burst − S-ABCD | 2 | 1 | -1 |
| ustc | A-2 | S-Burst − Native | 1913 | 758 | -1155 |
| ustc | A-2 | S-Burst − T16 | 818 | 4307 | +3489 |
| ustc | A-2 | S-Burst − S-ABCD | 319 | 829 | +510 |

## Gain-based feature importance

Importance is diagnostic, not causal. Values below are group-normalized gain for full-flow S-Burst.

| Dataset | Protocol | A | B | C | D | E | S12 | Top three individual features |
|---|---|---:|---:|---:|---:|---:|---:|---|
| iscx_vpn | medium_seed2022 | 0.037 | 0.255 | 0.223 | 0.226 | 0.104 | 0.155 | packet_length_min (0.165), forward_reverse_byte_ratio (0.112), iat_max_seconds (0.054) |
| iscx_tor | medium_seed2022 | 0.026 | 0.122 | 0.233 | 0.297 | 0.181 | 0.141 | iat_median_seconds (0.073), forward_reverse_byte_ratio (0.062), forward_iat_mean_seconds (0.060) |
| vnat | medium_seed2025 | 0.012 | 0.404 | 0.358 | 0.044 | 0.083 | 0.100 | forward_length_mean (0.330), packet_length_min (0.231), packet_length_p75 (0.113) |
| vnat | medium_seed2026 | 0.004 | 0.460 | 0.276 | 0.017 | 0.129 | 0.115 | packet_length_min (0.268), forward_length_mean (0.208), packet_length_p25 (0.084) |
| ustc | A-2 | 0.014 | 0.321 | 0.117 | 0.170 | 0.132 | 0.246 | packet_length_median (0.127), packet_length_min (0.081), duration_seconds (0.068) |

## Answers to the 12 required questions

1. **Are the original 12 statistics insufficient?** Yes, at pilot scope. S-ABCD improves S12 by +0.019473 mean Macro-F1, is positive on 5/5 pilots, and has no negative pilot (worst +0.004990). This establishes a measurable S12 bottleneck, not sufficiency of behavior features as a replacement for Native bytes.

2. **Most valuable new group?** Group B, packet-length distribution, is the most stable incremental block: mean +0.007974, positive 5/5, worst +0.001781. Its gain importance is also largest in VPN/VNAT/USTC S-Burst models.

3. **Independent bidirectional contribution?** Yes. Adding C after A+B yields mean +0.007416, positive 4/5, with only a negligible worst change of -0.000029. In VNAT, `forward_length_mean` alone has normalized gain 0.330 (2025) and 0.208 (2026), so this benefit carries capture/domain-fingerprint risk and is not causal evidence.

4. **Independent IAT-distribution contribution?** Not globally stable. Adding D after A+B+C has mean -0.000475, positive 3/5, and worst -0.029732; it helps VPN, Tor and USTC but hurts both VNAT pilots.

5. **Does Burst add information?** Yes, modestly and conditionally. S-Burst − S-ABCD is +0.006094 mean, positive 4/5, worst -0.008661; the preregistered Burst gate passes. The only focus-class incremental gain above +0.05 is VNAT-2025 Netflix, so this is not a universal large effect.

6. **Early-16 versus full-flow?** Mean full-minus-Early Macro-F1 is -0.022610 for S12, -0.021137 for S-ABCD and -0.020037 for S-Burst. Full-flow is slightly better for S-Burst on VPN (+0.004548), Tor (+0.002867) and USTC (+0.008889), but worse on both VNAT pilots (-0.018891 and -0.097597). Therefore more packets are not a global behavior-feature advantage.

7. **ISCX-VPN improvement?** Yes within behavior features: S-Burst − S12 is +0.028667 Macro-F1, including Spotify F1 0.407643→0.463576 and Vimeo 0.540816→0.568528. It still trails Native by 0.273779 Macro-F1, so it is complementary evidence rather than a replacement.

8. **ISCXTor2016 improvement?** Yes: S-Burst − S12 is +0.053903. Hangouts improves 0.263736→0.380000 and YouTube 0.501377→0.568047. It remains 0.135283 Macro-F1 below Native.

9. **Does VNAT rsync/scp truly improve?** Only modestly. In medium-2025, rsync correct rises 59→64→67 (S12→S-ABCD→S-Burst), rsync→scp falls 132→127→124, and rsync F1 rises 0.390728→0.411576→0.424051. But scp→rsync increases 52→56→57 while scp F1 is nearly flat (0.649254→0.651515→0.652672). The confusion is mitigated, not solved. `sftp` is Unknown in both pilots and was correctly not read.

10. **Damage to Native-good classes?** Relative to S12, S-Burst has only one negative pilot and the worst loss is -0.003671. Relative to Native, however, standalone S-Burst loses more Native-correct samples than it rescues on 4/5 pilots; only VNAT-2025 has positive net rescue (+13). It must not replace Native wholesale.

11. **Which signals complement Native/T16?** Length-distribution B, bidirectional C, selected IAT D and Burst E all contribute. S-Burst rescues Native errors on every pilot (170, 157, 73, 7 and 758 samples), although its net balance is usually negative. Against T16 it has positive net rescue on all five pilots (189, 65, 37, 21, 3489). This is real sample-level complementarity, not overall dominance.

12. **Next representation direction?** Prioritize a preregistered Byte–Behavior multi-view encoder over another standalone behavior-only model: Native remains much stronger on VPN, Tor, VNAT-2026 and USTC, while behavior features rescue non-overlapping errors and improve the hard VNAT-2025 composition. This is a recommendation only; no multi-view model or Stage 15F-1C was started.


## Interpretation

Feature importance is diagnostic only. No single feature dominates all datasets, but VNAT relies heavily on `forward_length_mean` and `packet_length_min`; that concentration may encode capture environment and requires future cross-capture checks. Full-flow versus Early-16 differences combine observation amount and representation effects and are not attributed to one feature family. System-level S-Burst-versus-Native/T16 differences also combine different models and observation budgets.

## Stopping rule

Full15 gate met: `False`; full15 status: `NOT_RUN_STAGE_SCOPE`. Stage 15F-1C, open-set evaluation, DES/H1 modification, and Byte-Behavior model development were not started.
