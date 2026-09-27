# Stage41 matched-flow Open-Detect vs three-view DES-v1

Single seed 2022, same USTC flow IDs and source Train/Val/Test partitions for both methods.
The 1:1 Test subset was frozen by flow ID before score inspection. Primary operating point
is Known Validation P95; Test-Youden is explicitly retrospective/oracle, not deployable.

| Scenario | Known/Unknown classes | Balanced Test Known/Unknown | OD AUROC | Ours AUROC | ΔAUROC | OD F1@P95 | Ours F1@P95 | ΔF1 | OD UFAR | Ours UFAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-1 | 19/1 | 85/85 | 0.9907 | 0.9769 | -0.0138 | 0.9659 | 0.9605 | -0.0055 | 0.0000 | 0.0000 |
| A-2 | 17/3 | 558/558 | 0.9290 | 0.9406 | +0.0116 | 0.5038 | 0.6741 | +0.1703 | 0.6434 | 0.4606 |
| A-3 | 15/5 | 1031/1031 | 0.9782 | 0.9609 | -0.0173 | 0.9191 | 0.9231 | +0.0039 | 0.1125 | 0.0921 |

The full natural Test prevalence and retrospective oracle-threshold
metrics are in `comparison_run_results.csv`. This local 10% flow subset is
not the authors' exact five-fold dataset/protocol. Our method includes
external pretrained TrafficFormer/YaTC branches; OD is trained from scratch.
A-1 has only 85 Unknown Test flows, so its single-seed estimate is fragile.
