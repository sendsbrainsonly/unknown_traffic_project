# Experiment results: stage40-cic-independent-resume-v2-20260927

- Status: `success`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-27T10:09:45Z`; completed (UTC): `2026-09-27T14:32:36Z`
- Objective: Complete the two frozen CIC-IDS-2017 Known-only open-set pilots and independently replay their metrics.

## Data and split

The Stage 40 frozen role manifests and protocol were reused without regeneration. Single seed 2022; each task has BENIGN plus the non-held-out attack as Known, and one whole held-out attack as Unknown:

| Unit | Known classes | Unknown class | Known Train / Val / Test | Unknown Test |
|---|---|---|---:|---:|
| `unknown_portscan` | BENIGN, Slowhttptest | PortScan | 7,818 / 2,110 / 264 | 1,136 |
| `unknown_slowhttptest` | BENIGN, PortScan | Slowhttptest | 27,698 / 1,802 / 2,272 | 132 |

The role/source hashes were unchanged: protocol `b16647c7c9059abb2d526bc33ea65392b336586a84d203824dea6efd78e485b0`; PortScan roles `ca0eb9bae57662b41342ec44e5e25006c8dd6105b74da95a4895018123043c89`; Slowhttptest roles `6b6d78e3abd34eab2a3a2797d1fe9376b2a9a364078fa8ce677e2a6a876dd99a`.

## Configuration and execution

- The frozen three-view Stage 40 recipe was trained on Known Train and selected/calibrated using Known Validation only. No Unknown samples were used for fitting, and no Test samples were used for fitting or threshold selection.
- Scores: MSP, Energy, empirical centroid distance, and DES-v1 (`k=10`, equal global/local weights); operating threshold: Known Validation P95.
- Maximum concurrent GPU workloads: 1. The v2 coordinator finished with exit code 0. Each role's audit records `unknown_fit_count=0`, `test_fit_count=0`, and unchanged checkpoint hashes.
- Independent replay PASS for both roles: PortScan 1,400 score rows and Slowhttptest 2,404 score rows; all four methods replayed and both role/source hash checks passed.

## Core results

Metrics are reported at Known-Val P95. `UFAR` is the fraction of Unknown Test accepted as Known; `Known FRR` is the fraction of Known Test rejected.

### Unknown PortScan

Known Test 264; Unknown Test 1,136. Closed-set Known Test Accuracy/Macro-F1/Weighted-F1: `1.0000 / 1.0000 / 1.0000`.

| Score | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.847244 | 0.921874 | 0.951585 | 0.041667 | 0.091514 |
| Energy | 0.719677 | 0.867625 | 0.963028 | 0.018939 | 0.071006 |
| Centroid | 0.285411 | 0.714805 | 0.956866 | 0.162879 | 0.079805 |
| DES-v1 | 0.463178 | 0.761719 | 0.948063 | 0.166667 | 0.095238 |

### Unknown Slowhttptest

Known Test 2,272; Unknown Test 132. Closed-set Known Test Accuracy/Macro-F1/Weighted-F1: `0.993838 / 0.993838 / 0.993838`.

| Score | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.954846 | 0.646372 | 0.045455 | 0.049296 | 0.681081 |
| Energy | 0.950461 | 0.619815 | 0.075758 | 0.048856 | 0.668493 |
| Centroid | 0.962588 | 0.630503 | 0.045455 | 0.054577 | 0.659686 |
| DES-v1 | 0.997753 | 0.955759 | 0.000000 | 0.071303 | 0.619718 |

## Preserved evidence

- Per-role scores, sample-level predictions, calibration, evaluation audits, Known Test closed-set summaries, SUCCESS markers, and independent replay JSON are under `../unknown_portscan/detection/` and `../unknown_slowhttptest/detection/`.
- This attempt's coordinator progress and logs are retained at `progress.json` and project-local `.tmux-task/stage40_cic_independent_v2_0927/`.
- The earlier failed/partial attempts remain intact; this result bundle documents the successful v2 continuation.

## Limitations

- Single-seed diagnostic pilots with only two Known classes each; do not generalize to the full CIC label space.
- Attack, day, and capture are confounded in CIC-IDS-2017; group-disjoint splitting prevents same-group leakage but does not establish attack-independent generalization.
- Slowhttptest Unknown Test has only 132 flows, so its estimates are comparatively uncertain.
- High AUROC on one role and poor AUROC on the other show substantial role dependence. These pilots do not establish broad superiority of a detector.

## Conclusion and next step

Both frozen CIC pilot roles completed and passed independent score replay. The results are diagnostic only: DES-v1 is weak for Unknown PortScan (`AUROC 0.463178`, `UFAR 0.948063`) but ranks Unknown Slowhttptest strongly (`AUROC 0.997753`, `UFAR 0.000000`, with Known FRR `0.071303`). Do not pool the roles as if they were the same Unknown distribution or use these results to retune the frozen protocol. No further experiment was started.
