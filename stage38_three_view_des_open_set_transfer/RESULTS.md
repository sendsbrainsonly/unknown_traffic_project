# Stage 38 — Three-view DES open-set transfer

## Data and split

Development datasets: VNAT Stage14B medium_seed2025, ISCX-VPN and ISCXTor2016 Stage12 medium_seed2022. Known/Unknown membership and sample IDs use those frozen manifests. ISCX Known-only training uses the preregistered capture-derived coarse category mapping; it is not authoritative per-flow service ground truth.

## Configuration and execution

VNAT uses the frozen Stage33 six-class three-view encoder. ISCX-VPN/Tor use newly trained Stage31/32-style TrafficFormer, graph and YaTC branches, 64-dimensional adapters, and a fixed equal-weight feature fusion. Training and checkpoint selection use Known Train/Validation only. Scores are C0 MSP, C1 Energy, D0 empirical centroid distance, and D1 fixed global/local DES-v1 (k=10, 0.5/0.5). Each threshold is Known-Val P95 (higher).

## Core results

Known Test closed-set metrics:

| Dataset | Accuracy | Macro-F1 | Weighted-F1 | Known Test |
|---|---:|---:|---:|---:|
| VNAT | 0.963776 | 0.943109 | 0.963750 | 1960 |
| ISCX-VPN | 0.967921 | 0.975277 | 0.967844 | 1621 |
| ISCXTor2016 | 0.942498 | 0.941121 | 0.942448 | 1113 |

Open-set scores (higher AUROC/AUPRC, lower UFAR/FRR are better):

| Dataset | Score | AUROC | AUPRC | UFAR | Known FRR |
|---|---|---:|---:|---:|---:|
| VNAT | C0 | 0.888421 | 0.925579 | 0.611765 | 0.050000 |
| VNAT | C1 | 0.851933 | 0.918982 | 0.608366 | 0.051020 |
| VNAT | D0 | 0.752023 | 0.841602 | 0.712941 | 0.053571 |
| VNAT | D1 | 0.897053 | 0.924377 | 0.623791 | 0.053061 |
| ISCX-VPN | C0 | 0.583954 | 0.836184 | 0.846500 | 0.053054 |
| ISCX-VPN | C1 | 0.565430 | 0.832767 | 0.848500 | 0.052437 |
| ISCX-VPN | D0 | 0.530237 | 0.821181 | 0.863167 | 0.048735 |
| ISCX-VPN | D1 | 0.559509 | 0.830727 | 0.855500 | 0.054287 |
| ISCXTor2016 | C0 | 0.901799 | 0.958798 | 0.633000 | 0.050314 |
| ISCXTor2016 | C1 | 0.912840 | 0.967270 | 0.536250 | 0.045822 |
| ISCXTor2016 | D0 | 0.898750 | 0.963052 | 0.488250 | 0.050314 |
| ISCXTor2016 | D1 | 0.924869 | 0.965810 | 0.634750 | 0.043127 |

Paired differences on the same frozen representation:

| Dataset | Comparison | ΔAUROC | ΔAUPRC | ΔUFAR | ΔKnown FRR |
|---|---|---:|---:|---:|---:|
| VNAT | D1−D0 | +0.145030 | +0.082775 | -0.089150 | -0.000510 |
| VNAT | D0−C0 | -0.136398 | -0.083977 | +0.101176 | +0.003571 |
| VNAT | D1−C0 | +0.008632 | -0.001202 | +0.012026 | +0.003061 |
| VNAT | D1−C1 | +0.045120 | +0.005395 | +0.015425 | +0.002041 |
| ISCX-VPN | D1−D0 | +0.029272 | +0.009546 | -0.007667 | +0.005552 |
| ISCX-VPN | D0−C0 | -0.053717 | -0.015003 | +0.016667 | -0.004318 |
| ISCX-VPN | D1−C0 | -0.024445 | -0.005457 | +0.009000 | +0.001234 |
| ISCX-VPN | D1−C1 | -0.005921 | -0.002040 | +0.007000 | +0.001851 |
| ISCXTor2016 | D1−D0 | +0.026119 | +0.002758 | +0.146500 | -0.007188 |
| ISCXTor2016 | D0−C0 | -0.003049 | +0.004254 | -0.144750 | +0.000000 |
| ISCXTor2016 | D1−C0 | +0.023070 | +0.007012 | +0.001750 | -0.007188 |
| ISCXTor2016 | D1−C1 | +0.012029 | -0.001460 | +0.098500 | -0.002695 |

## Preserved evidence

`stage38_dataset_results.csv`, `stage38_paired_comparison.csv`, `stage38_unknown_application_results.csv`, `stage38_closed_set_results.csv`, each dataset's `detection/sample_scores.csv`, calibration, representations and checkpoint hashes, named tmux logs, and `completion_verification.json` preserve the run and its replay.

The first aggregate packaging validation failed after all calculations because the finalizer wrote the unsupported manifest status `complete`. The code and manifest were normalized to the schema-supported `success`; the original finalizer failure record remains preserved, and the repaired bundle was re-inventoried and hash-validated without rerunning models or changing metrics.

## Limitations

One seed per dataset; the VNAT taxonomy followed earlier exposed Test results. ISCX service labels are capture-derived and may not identify each individual flow's activity. These are development diagnostics, not untouched external validation. C0/C1 are classifier-uncertainty baselines and are not Open-Detect Native. Across-encoder historical comparisons cannot isolate the DES support mechanism.

## Conclusion and next step

Final Gate: `MIXED_OR_INCONCLUSIVE`. The paired D1−D0 comparison tests the global/local support contribution within one representation. The C0/C1 comparison measures operating performance on the same encoder. Any further method choice needs a new frozen multi-setting evaluation; this run makes no method or threshold update.
