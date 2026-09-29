# Stage 44 Results — VNAT Coarse Service-Level LOSO Open Set

Status: `success / independently replayed PASS` (2026-09-29 UTC). Claim scope: single-seed diagnostic on previously developed VNAT.

## Data and split

- Source: Stage 14B clean VNAT pool, 23,449 unique flows; four frozen service-level leave-one-service-out folds, seed 2022.
- The Known Train/Validation/Test roles are disjoint by group and capture. The held-out service is wholly Unknown Test; its Train/Validation usage is zero.
- Per-fold Known Train/Validation/Test and Unknown Test counts: communication `17,352/2,509/1,380/2,208`; file_transfer `14,604/2,021/954/5,870`; remote_access `7,705/1,129/1,008/13,607`; streaming `17,693/2,531/1,461/1,764`.

## Configuration and execution

- Three-view TrafficFormer + graph + YaTC equal-feature fusion, followed by service classification. Each fold trains its own Known-only model with the fixed Stage 31 recipes. Single physical GPU 5, serial queue, 20/20 jobs successful.
- Unknown scores: MSP, Energy, empirical centroid and DES-v1. Primary operating point: Known Validation P95, fixed before Known/Unknown Test evaluation. Natural-prevalence and fixed 1:1 evaluation views, plus P90/P99 sensitivity rows, are retained.
- Executed in tmux session `stage44_single_gpu_queue_v5_20260928`; process exit status `0`. The saved-score independent replay ran in `stage44_independent_replay_20260929` and also exited `0`.

## Core results

## Closed-set Known Test

| Unknown Service | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| communication | 1.000000 | 1.000000 | 1.000000 |
| file_transfer | 0.996855 | 0.996908 | 0.996862 |
| remote_access | 1.000000 | 1.000000 | 1.000000 |
| streaming | 0.998631 | 0.997935 | 0.998633 |

## Open-set natural prevalence, Known-Val P95

| Unknown Service | Method | AUROC | AUPRC | Binary F1 | UFAR | Known FRR |
|---|---|---:|---:|---:|---:|---:|
| communication | centroid | 0.982437 | 0.986873 | 0.950319 | 0.055707 | 0.068841 |
| communication | des_v1 | 0.995450 | 0.995545 | 0.984828 | 0.000453 | 0.048551 |
| communication | energy | 0.771250 | 0.895933 | 0.754603 | 0.378170 | 0.042029 |
| communication | msp | 0.778326 | 0.899532 | 0.779634 | 0.344656 | 0.041304 |
| file_transfer | centroid | 0.994079 | 0.997638 | 0.994136 | 0.003578 | 0.050314 |
| file_transfer | des_v1 | 0.994760 | 0.997727 | 0.993447 | 0.005622 | 0.046122 |
| file_transfer | energy | 0.994121 | 0.997653 | 0.987437 | 0.015843 | 0.056604 |
| file_transfer | msp | 0.992544 | 0.997684 | 0.986149 | 0.017547 | 0.061845 |
| remote_access | centroid | 0.944667 | 0.996002 | 0.951049 | 0.090468 | 0.042659 |
| remote_access | des_v1 | 0.987627 | 0.999101 | 0.979799 | 0.037554 | 0.028770 |
| remote_access | energy | 0.946625 | 0.995994 | 0.916971 | 0.151025 | 0.036706 |
| remote_access | msp | 0.944866 | 0.995851 | 0.916283 | 0.152201 | 0.036706 |
| streaming | centroid | 0.975841 | 0.971145 | 0.948346 | 0.073696 | 0.032854 |
| streaming | des_v1 | 0.991060 | 0.987620 | 0.977827 | 0.000000 | 0.054757 |
| streaming | energy | 0.957053 | 0.973423 | 0.939255 | 0.070862 | 0.059548 |
| streaming | msp | 0.957382 | 0.973388 | 0.940063 | 0.070862 | 0.057495 |

## Preserved evidence

- `stage44_run_results.csv`: all 96 method × view × threshold × fold rows; `stage44_method_summary.csv` and `stage44_closed_set_results.csv`: aggregate and closed-set metrics.
- `protocols/*/evaluation/sample_scores.csv`: 36,442 saved Known Validation, Known Test and Unknown Test score rows; each fold also contains closed-set, calibration and per-application outputs.
- `vnat_loso_protocol.json`, `protocols/*/role_manifest.csv`, `split_audit.csv`, model checkpoints, training histories, queue logs and exit status are retained.
- `completion_verification.json` and `independent_replay_verification.json`: four folds PASS; 96/96 metric rows replayed with maximum absolute difference `0`; 20 checkpoint hashes verified; Unknown/Test fitting and threshold samples `0`.

## Limitations

- All normalization, supports and thresholds use Known Train/Validation only.
- This is a single-seed, previously developed VNAT diagnostic, not untouched external validation.
- Group/capture-disjoint splitting causes real imbalance, especially Remote-Access and Streaming; see `split_audit.csv`.
- Natural and fixed 1:1 results are both retained; Binary F1 is prevalence-dependent.

## Conclusion and next step

The three-view closed-set classifier scores Macro-F1 `0.996908–1.000000` across the four Known Test folds. DES-v1 is the strongest of the four tested rejection scores by mean natural-prevalence AUROC (`0.992224`) and improves on centroid AUROC in all four folds. Its mean UFAR at Known-Val P95 is `0.010907`, with mean Known FRR `0.044550`; Remote-Access has the highest DES-v1 UFAR (`0.037554`). These are diagnostic results under the frozen coarse-service protocol, not evidence of generalization to an untouched dataset. Stage 44 is complete; no follow-on experiment is launched here.
