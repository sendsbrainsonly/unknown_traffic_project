# Stage 14C.5 — VNAT Feature Representation Audit

## Data boundary

- Stage 14B freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Model-visible samples: frozen Known Train and Known Validation only.
- Unknown Test samples used: **0**.
- Known Test samples used: **0**.
- Unknown Detection executed: **NO**.

## Mean ± std across seeds

| feature   | setting   |   runs |   val_accuracy_mean |   val_accuracy_std |   val_macro_f1_mean |   val_macro_f1_std |   val_weighted_f1_mean |   val_weighted_f1_std |   minority_recall_mean |   minority_recall_std |   weak_run_count_macro_f1_lt_0_35 |   worst_val_macro_f1 |   worst_val_weighted_f1 |
|:----------|:----------|-------:|--------------------:|-------------------:|--------------------:|-------------------:|-----------------------:|----------------------:|-----------------------:|----------------------:|----------------------------------:|---------------------:|------------------------:|
| f0        | Low       |      5 |            0.817794 |           0.095868 |            0.594534 |           0.149471 |               0.784763 |              0.121641 |               0.598236 |              0.247062 |                                 0 |             0.445140 |                0.576959 |
| f0        | Medium    |      5 |            0.759880 |           0.184836 |            0.559701 |           0.269197 |               0.718804 |              0.218550 |               0.443413 |              0.341923 |                                 2 |             0.284444 |                0.509103 |
| f0        | High      |      5 |            0.821455 |           0.156604 |            0.690267 |           0.160183 |               0.808909 |              0.155275 |               0.671226 |              0.245778 |                                 0 |             0.429056 |                0.585317 |
| f0        | Overall   |     15 |            0.799710 |           0.142290 |            0.614834 |           0.194124 |               0.770825 |              0.162225 |               0.570958 |              0.278870 |                                 2 |             0.284444 |                0.509103 |
| f1        | Low       |      5 |            0.814307 |           0.103311 |            0.577190 |           0.128887 |               0.788426 |              0.101574 |               0.577046 |              0.162083 |                                 0 |             0.395209 |                0.629250 |
| f1        | Medium    |      5 |            0.695891 |           0.335511 |            0.533973 |           0.312628 |               0.653724 |              0.383057 |               0.456943 |              0.312044 |                                 1 |             0.042109 |                0.050952 |
| f1        | High      |      5 |            0.832921 |           0.139320 |            0.726361 |           0.179247 |               0.809310 |              0.178518 |               0.743434 |              0.190110 |                                 0 |             0.477349 |                0.539525 |
| f1        | Overall   |     15 |            0.781039 |           0.211432 |            0.612508 |           0.221650 |               0.750486 |              0.243044 |               0.592474 |              0.245839 |                                 1 |             0.042109 |                0.050952 |
| f2        | Low       |      5 |            0.836132 |           0.070688 |            0.647453 |           0.129567 |               0.824961 |              0.069874 |               0.655897 |              0.220001 |                                 0 |             0.444189 |                0.726319 |
| f2        | Medium    |      5 |            0.796258 |           0.181189 |            0.621443 |           0.151463 |               0.762937 |              0.211520 |               0.489486 |              0.288355 |                                 0 |             0.373376 |                0.513728 |
| f2        | High      |      5 |            0.827819 |           0.145739 |            0.690012 |           0.182564 |               0.800738 |              0.167106 |               0.691911 |              0.240677 |                                 0 |             0.529717 |                0.611565 |
| f2        | Overall   |     15 |            0.820070 |           0.131119 |            0.652969 |           0.147410 |               0.796212 |              0.151177 |               0.612431 |              0.249930 |                                 0 |             0.373376 |                0.513728 |
| f3        | Low       |      5 |            0.832539 |           0.059278 |            0.670362 |           0.111480 |               0.824775 |              0.055629 |               0.704320 |              0.208797 |                                 0 |             0.542322 |                0.745185 |
| f3        | Medium    |      5 |            0.774405 |           0.181130 |            0.557654 |           0.205882 |               0.737841 |              0.207974 |               0.395695 |              0.310739 |                                 1 |             0.327121 |                0.507927 |
| f3        | High      |      5 |            0.830173 |           0.142153 |            0.686600 |           0.180852 |               0.796863 |              0.170905 |               0.679588 |              0.235542 |                                 0 |             0.443717 |                0.607484 |
| f3        | Overall   |     15 |            0.812372 |           0.130094 |            0.638205 |           0.168907 |               0.786493 |              0.151640 |               0.593201 |              0.277313 |                                 1 |             0.327121 |                0.507927 |

Macro-F1 gives every application equal weight and remains the primary decision metric. Weighted-F1 weights each application by Known-Validation support and is reported alongside it to reveal whether aggregate performance is dominated by large classes.

## Medium-2024/2025 weak-run check

| feature   |   seed |   val_accuracy |   val_macro_f1 |   val_weighted_f1 |   minority_recall |
|:----------|-------:|---------------:|---------------:|------------------:|------------------:|
| f0        |   2024 |       0.627143 |       0.334134 |          0.516484 |          0.000000 |
| f0        |   2025 |       0.740816 |       0.284444 |          0.681317 |          0.166667 |
| f1        |   2024 |       0.172857 |       0.042109 |          0.050952 |          0.000000 |
| f1        |   2025 |       0.868878 |       0.481513 |          0.833517 |          0.261765 |
| f2        |   2024 |       0.638571 |       0.373376 |          0.557790 |          0.019608 |
| f2        |   2025 |       0.893367 |       0.653313 |          0.864173 |          0.605882 |
| f3        |   2024 |       0.618571 |       0.327121 |          0.507927 |          0.000000 |
| f3        |   2025 |       0.867347 |       0.538861 |          0.843631 |          0.343137 |

## Stability

| feature   |   mean_within_setting_seed_std |   overall_mean_macro_f1 |   worst_macro_f1 |
|:----------|-------------------------------:|------------------------:|-----------------:|
| f0        |                       0.192951 |                0.614834 |         0.284444 |
| f1        |                       0.206921 |                0.612508 |         0.042109 |
| f2        |                       0.154531 |                0.652969 |         0.373376 |
| f3        |                       0.166071 |                0.638205 |         0.327121 |

## Pre-registered decision

- Best non-F0 by overall mean Macro-F1: **F2**.
- Overall ΔMacro-F1 versus F0: **+0.038136**.
- Paired Macro-F1 wins versus F0: **9/15**.
- Overall minority-recall delta: **+0.041473**.
- Settings with lower seed std: **2/3**.
- Most stable by mean within-setting seed std: **F2**.
- Feature representation is the primary bottleneck: **NO**.
- Recommend freezing a new feature configuration before formal encoder retraining: **NO**.

The primary-bottleneck conclusion follows the thresholds frozen in README.md before training. A NO does not mean features are irrelevant; it means the observed improvement did not satisfy all four pre-registered conditions.
