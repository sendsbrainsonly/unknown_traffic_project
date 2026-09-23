# Stage 14C-6 — Cleaned Pipeline Full 15-Run Closed-Set Validation

## Final conclusion

```text
CLOSED_SET_PASS_WITH_HARD_PROTOCOLS
```

The frozen cleaned F2 pipeline completed all 15 Stage 14B protocols without a crash, NaN, early stop, Test/Unknown access, or DES execution. Its overall Known-Validation performance is effectively tied with Native Open-Detect on Accuracy and Weighted-F1 and is lower by `0.009875` Macro-F1 on average. The remaining variance is concentrated in frozen class compositions rather than an observed training implementation failure.

## Integrity

- Runs completed: `15/15`; every run completed `100/100` epochs.
- Verified best checkpoints and SHA256 values: `15/15`.
- Stage 14B freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Known Test samples used: `0`.
- Unknown Test samples used: `0`.
- DES executed: `false`.
- Weak runs under the frozen rule (`Macro-F1 < 0.50` or `Accuracy < 0.60` or NaN/crash): `0`.
- Medium-2025/2026 exactly reproduce the Stage 14C-4 selected metrics to `1e-12`, confirming wrapper/configuration reproducibility.

## All 15 formal runs

| protocol_id     |   best_epoch |   train_loss |   val_loss |   val_accuracy |   val_macro_f1 |   val_weighted_f1 | weak_run   | checkpoint_sha256                                                |
|:----------------|-------------:|-------------:|-----------:|---------------:|---------------:|------------------:|:-----------|:-----------------------------------------------------------------|
| low_seed2022    |           90 |     0.18946  |   0.199534 |       0.938776 |       0.883119 |          0.935065 | False      | 22616e14a4d587234a26107b3e5fdea0b12e5f0f0d4a8e8151af49f02b865920 |
| low_seed2023    |          100 |     0.287226 |   0.309332 |       0.855971 |       0.794229 |          0.855866 | False      | 79bed1da072eb0056a2d05698aa4c594bbee003aa265e8435364f86c8c7fcc43 |
| low_seed2024    |          100 |     0.322133 |   0.392962 |       0.857503 |       0.872996 |          0.849163 | False      | aabe30abe541ba20f69f2304f856dbaced1024cb0d6c5d0f2abb8234482a65c9 |
| low_seed2025    |           95 |     0.392167 |   0.371664 |       0.822672 |       0.581333 |          0.82046  | False      | d4a6ae3cae1ae20c2d2bd8a8b015ba09320866dc3ff84030c128087f08c3ce6a |
| low_seed2026    |           68 |     0.220822 |   0.194881 |       0.907009 |       0.82412  |          0.875821 | False      | 04c3ce58e66cb4aa8449fe4bfeb3e452cb2d38d07a6f1d9f456a3c9560bb98a0 |
| medium_seed2022 |           85 |     0.136639 |   0.198307 |       0.939929 |       0.883628 |          0.937962 | False      | 76123de083366b25473ef17d034f7eead735bd6163e3a1a74565023e143e778e |
| medium_seed2023 |           54 |     0.787606 |   0.796464 |       0.643198 |       0.742117 |          0.596543 | False      | 68a72e872741ebcc987f124565ee71013be4726d4258a8fc56bf14b92f52c4fc |
| medium_seed2024 |           91 |     0.67273  |   0.542334 |       0.731429 |       0.677427 |          0.716204 | False      | 4c3c10b5cd4752d292f9689ea3037440a3fe3d0a31d3d75828eb0b0bbf7d35ce |
| medium_seed2025 |           93 |     0.187101 |   0.186996 |       0.892857 |       0.727247 |          0.864414 | False      | 0d6d30984e2a033fad325536d1feb6d57279fdce82854d322a632a77b8f77f8c |
| medium_seed2026 |           53 |     0.039323 |   0.04515  |       0.994767 |       0.980668 |          0.994803 | False      | 4a2c736d97a39805c034b2d5f1cfe32e0df41744d575b800f063ad5c46e3a6ad |
| high_seed2022   |           83 |     0.118732 |   0.137324 |       0.950025 |       0.898354 |          0.948038 | False      | 18f12041d4038ba390b9af57d57694a61245b5575aa15bb1867a2e5434cb43fe |
| high_seed2023   |           89 |     0.384209 |   0.494818 |       0.806801 |       0.878377 |          0.798973 | False      | ddac4e74bb70637723dca063766a819b0fa484874061cb223a5d8bba22e2189c |
| high_seed2024   |           82 |     0.468583 |   0.5381   |       0.80829  |       0.824032 |          0.800256 | False      | 25ab4639362d7997a18c4145df9ff5ae4a6c79190720574e47d37caa2962f2fc |
| high_seed2025   |           82 |     0.173726 |   0.172422 |       0.883697 |       0.7644   |          0.883095 | False      | b5d86bb7d617b89c9dc7c0c7d8c9c82bb9f4977ecaa74af98c12aecb6e3b7ec2 |
| high_seed2026   |           65 |     0.023233 |   0.072706 |       0.994756 |       0.98338  |          0.994711 | False      | ca9f34bd296367452b3f24fbcfdf7a99608c73b964de1c6516962cdb84c0123b |

## Low / Medium / High summary

Values are mean and sample standard deviation across five frozen protocols.

| setting   |   runs |   val_accuracy_mean |   val_accuracy_std |   val_macro_f1_mean |   val_macro_f1_std |   val_weighted_f1_mean |   val_weighted_f1_std |   weak_runs |
|:----------|-------:|--------------------:|-------------------:|--------------------:|-------------------:|-----------------------:|----------------------:|------------:|
| Low       |      5 |            0.876386 |           0.046067 |            0.79116  |           0.122757 |               0.867275 |              0.042774 |           0 |
| Medium    |      5 |            0.840436 |           0.147691 |            0.802217 |           0.125775 |               0.821985 |              0.163654 |           0 |
| High      |      5 |            0.888714 |           0.083975 |            0.869709 |           0.082148 |               0.885014 |              0.087458 |           0 |
| Overall   |     15 |            0.868512 |           0.096451 |            0.821029 |           0.109748 |               0.858092 |              0.105427 |           0 |

Overall cleaned-pipeline mean ± std:

- Accuracy: `0.868512 ± 0.096451`
- Macro-F1: `0.821029 ± 0.109748`
- Weighted-F1: `0.858092 ± 0.105427`

## Paired comparison with existing Native Open-Detect

Native models were not retrained. Each row compares the same frozen protocol and Known-Validation samples. Each method retains its previously frozen training-seed policy, so the comparison is paired by protocol/data but is not a single-seed architecture ablation.

| protocol_id     |   native_accuracy |   val_accuracy |   delta_accuracy_our_minus_native |   native_macro_f1 |   val_macro_f1 |   delta_macro_f1_our_minus_native |   native_weighted_f1 |   val_weighted_f1 |   delta_weighted_f1_our_minus_native |
|:----------------|------------------:|---------------:|----------------------------------:|------------------:|---------------:|----------------------------------:|---------------------:|------------------:|-------------------------------------:|
| low_seed2022    |          0.935453 |       0.938776 |                          0.003322 |          0.884986 |       0.883119 |                         -0.001866 |             0.934994 |          0.935065 |                             7e-05    |
| low_seed2023    |          0.851413 |       0.855971 |                          0.004558 |          0.786118 |       0.794229 |                          0.008111 |             0.850082 |          0.855866 |                             0.005784 |
| low_seed2024    |          0.857503 |       0.857503 |                          0        |          0.819716 |       0.872996 |                          0.05328  |             0.849318 |          0.849163 |                            -0.000154 |
| low_seed2025    |          0.841957 |       0.822672 |                         -0.019285 |          0.681934 |       0.581333 |                         -0.100601 |             0.836487 |          0.82046  |                            -0.016027 |
| low_seed2026    |          0.904206 |       0.907009 |                          0.002804 |          0.780423 |       0.82412  |                          0.043697 |             0.871694 |          0.875821 |                             0.004127 |
| medium_seed2022 |          0.933872 |       0.939929 |                          0.006058 |          0.867653 |       0.883628 |                          0.015975 |             0.930069 |          0.937962 |                             0.007893 |
| medium_seed2023 |          0.587112 |       0.643198 |                          0.056086 |          0.704279 |       0.742117 |                          0.037837 |             0.587748 |          0.596543 |                             0.008795 |
| medium_seed2024 |          0.781429 |       0.731429 |                         -0.05     |          0.853112 |       0.677427 |                         -0.175685 |             0.772234 |          0.716204 |                            -0.05603  |
| medium_seed2025 |          0.896939 |       0.892857 |                         -0.004082 |          0.739755 |       0.727247 |                         -0.012508 |             0.862802 |          0.864414 |                             0.001613 |
| medium_seed2026 |          0.99529  |       0.994767 |                         -0.000523 |          0.983318 |       0.980668 |                         -0.00265  |             0.995223 |          0.994803 |                            -0.000421 |
| high_seed2022   |          0.946966 |       0.950025 |                          0.00306  |          0.89161  |       0.898354 |                          0.006744 |             0.943689 |          0.948038 |                             0.004349 |
| high_seed2023   |          0.774343 |       0.806801 |                          0.032457 |          0.867425 |       0.878377 |                          0.010952 |             0.76748  |          0.798973 |                             0.031493 |
| high_seed2024   |          0.815199 |       0.80829  |                         -0.006908 |          0.854692 |       0.824032 |                         -0.03066  |             0.808604 |          0.800256 |                            -0.008347 |
| high_seed2025   |          0.903427 |       0.883697 |                         -0.01973  |          0.765535 |       0.7644   |                         -0.001134 |             0.885933 |          0.883095 |                            -0.002838 |
| high_seed2026   |          0.995805 |       0.994756 |                         -0.001049 |          0.983001 |       0.98338  |                          0.000379 |             0.995812 |          0.994711 |                            -0.001102 |

### Paired summary

| setting   |   runs |   delta_accuracy_mean |   our_wins_accuracy |   delta_macro_f1_mean |   our_wins_macro_f1 |   delta_weighted_f1_mean |   our_wins_weighted_f1 |
|:----------|-------:|----------------------:|--------------------:|----------------------:|--------------------:|-------------------------:|-----------------------:|
| Low       |      5 |             -0.00172  |                   3 |              0.000524 |                   3 |                -0.00124  |                      3 |
| Medium    |      5 |              0.001508 |                   2 |             -0.027406 |                   2 |                -0.00763  |                      3 |
| High      |      5 |              0.001566 |                   2 |             -0.002744 |                   3 |                 0.004711 |                      2 |
| Overall   |     15 |              0.000451 |                   7 |             -0.009875 |                   8 |                -0.001386 |                      8 |

Overall Our-Cleaned − Native:

- Δ Accuracy: `+0.000451`; Our-Cleaned wins `7/15`, ties `1/15`.
- Δ Macro-F1: `−0.009875`; Our-Cleaned wins `8/15`.
- Δ Weighted-F1: `−0.001386`; Our-Cleaned wins `8/15`.

This is best described as **comparable**, not a consistent superiority claim. Medium-2024 (`Δ Macro-F1 = −0.175685`) and Low-2025 (`−0.100601`) account for most of the aggregate Macro-F1 deficit. Positive examples include Low-2024 (`+0.053280`), Low-2026 (`+0.043697`) and Medium-2023 (`+0.037837`).

## Per-class result

| class   |   runs |   recall_mean |   recall_std |   f1_mean |   f1_std |
|:--------|-------:|--------------:|-------------:|----------:|---------:|
| netflix |     11 |      0.781818 |     0.212453 |  0.795681 | 0.210299 |
| rdp     |     11 |      0.818182 |     0.196561 |  0.834127 | 0.195043 |
| rsync   |     11 |      0.542599 |     0.401645 |  0.515161 | 0.359634 |
| scp     |     10 |      0.778603 |     0.216196 |  0.64606  | 0.137234 |
| sftp    |     10 |      0.451807 |     0.074433 |  0.542609 | 0.086861 |
| skype   |     10 |      0.990476 |     0.013898 |  0.98848  | 0.006156 |
| ssh     |     10 |      0.999115 |     0.001031 |  0.999042 | 0.000922 |
| vimeo   |     11 |      0.978212 |     0.012948 |  0.984482 | 0.006254 |
| youtube |     11 |      0.898396 |     0.15913  |  0.879783 | 0.129678 |
| zoiper  |     10 |      0.996774 |     0.007258 |  0.995176 | 0.006907 |

### Difficult classes versus Native

| class   |   runs |   our_recall_mean |   native_recall_mean |   delta_recall_mean |   our_f1_mean |   native_f1_mean |   delta_f1_mean |
|:--------|-------:|------------------:|---------------------:|--------------------:|--------------:|-----------------:|----------------:|
| rdp     |     11 |          0.818182 |             0.704545 |            0.113636 |      0.834127 |         0.761111 |        0.073016 |
| rsync   |     11 |          0.542599 |             0.545455 |           -0.002856 |      0.515161 |         0.51766  |       -0.002498 |
| scp     |     10 |          0.778603 |             0.744541 |            0.034061 |      0.64606  |         0.634753 |        0.011308 |
| sftp    |     10 |          0.451807 |             0.471084 |           -0.019277 |      0.542609 |         0.544429 |       -0.00182  |

- `sftp` remains the lowest-recall class (`0.451807`), but is only `−0.019277` below Native Recall.
- `rsync` remains highly composition-sensitive (Recall std `0.401645`) and is nearly tied with Native on its cross-protocol mean.
- `scp` and `rdp` improve over Native on mean Recall by `+0.034061` and `+0.113636` respectively.
- High-support stable classes remain `skype`, `ssh`, `vimeo`, and `zoiper`.

## Hard protocols and composition

- **Medium-2025 remains a hard composition:** Macro-F1 `0.727247`; rsync Recall `0.031414` because rsync and scp are simultaneously Known. It exactly reproduces Stage 14C-4 and is only `−0.012508` below Native, so this is reproducible composition difficulty rather than a new run failure.
- **Low-2025:** Macro-F1 `0.581333`; it contains rsync, scp, and sftp simultaneously. The largest losses versus Native occur on netflix and rsync, partly offset by higher scp/rdp performance.
- **Medium-2024:** Macro-F1 `0.677427`; its deficit versus Native is concentrated in netflix, youtube, rdp, and sftp.
- Across the 15 frozen protocols, simultaneous `rsync/scp` presence is descriptively associated with mean Macro-F1 `0.738908` versus `0.875776` when they are not both Known. This association is confounded by other class-composition changes and is not treated as a causal estimate.

## Required answers

1. **Stable across 15/15?** Operationally yes: 15/15 completed, no crash/NaN, all hashes and frozen-data gates passed. Performance variance remains substantial across protocols.
2. **Overall performance?** Accuracy `0.868512`, Macro-F1 `0.821029`, Weighted-F1 `0.858092`.
3. **Versus Native Open-Detect?** Comparable overall: Accuracy `+0.000451`, Macro-F1 `−0.009875`, Weighted-F1 `−0.001386`; Macro-F1 wins on 8/15 protocols.
4. **Composition effect?** Yes, the largest repeatable weaknesses are concentrated in protocols containing difficult combinations such as rsync/scp/sftp and small-support rdp, although pairwise composition statistics are descriptive rather than causal.
5. **Remaining implementation issue?** None observed under the requested checks. Medium-2025/2026 exact metric parity with Stage 14C-4 provides a direct reproducibility check.
6. **Can the cleaned pipeline be frozen?** Yes, with hard-protocol behavior explicitly documented.
7. **Can it enter open-set evaluation?** Yes from the closed-set and integrity gates. This stage did not read Unknown Test or start open-set evaluation.

## Stage 14C-4 reproducibility check

| protocol_id     |   delta_accuracy |   delta_macro_f1 |   delta_weighted_f1 |   best_epoch_old |   best_epoch_new | exact_metric_parity_1e_12   |
|:----------------|-----------------:|-----------------:|--------------------:|-----------------:|-----------------:|:----------------------------|
| medium_seed2025 |                0 |                0 |                   0 |               93 |               93 | True                        |
| medium_seed2026 |                0 |                0 |                   0 |               53 |               53 | True                        |
