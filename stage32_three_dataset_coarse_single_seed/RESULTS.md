# Experiment results: stage32-three-dataset-coarse-single-seed-20260925-v1

- Status: `success / THREE_DATASET_SINGLE_SEED_DIAGNOSTIC_COMPLETE`
- Experiment type: `benchmark`; claim scope: `diagnostic`
- Created (UTC): `2026-09-25T11:03:06Z`; completed (UTC): `2026-09-25T11:26:49.426071+00:00`

## Data and split

Frozen Stage20 ISCX-VPN/Tor full-flow Known Train/Validation/Test and Stage14B VNAT `medium_seed2025` Known roles. Stage25 fixed VPN4/Tor5 semantic map; VNAT original-paper application→category map with four observed Known categories. USTC and all Unknown samples excluded. Exact ID/label/source preflight passed for 3/3 units; 19 protected hashes unchanged.

## Configuration and execution

One training seed 2022 per dataset; TrafficFormer + FIG/TAGCN + YaTC frozen three-view representations, Stage30/31 T0 fixed 1/3 feature fusion, Known-Train-only standardization, 30 adapter + 30 head epochs, Known-Val checkpoint selection. Training tmux `codex_stage32_three_coarse_train_20260925` exited 0; gated Test tmux `codex_stage32_three_coarse_test_20260925` exited 0. Three heads were frozen before Test values opened. No Unknown fitting or Test parameter selection.

## Core results

| Dataset | Known Test flows | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| iscx_vpn | 1093 | 0.915828 | 0.919350 | 0.914487 |
| iscx_tor | 1117 | 0.896150 | 0.899375 | 0.896717 |
| vnat | 1960 | 1.000000 | 1.000000 | 1.000000 |

Paired differences and bootstrap intervals are in `stage32_paired_comparison.csv`; the full per-dataset interpretation is in `stage32_report.md`. VNAT historical F2 pairs on Known Validation only, not Known Test.

## Preserved evidence

`preflight.json`, frozen hashes before/after, `progress.json`, three run directories (60 training epochs and both best checkpoints each), per-flow validation/Test predictions, Test logits and per-class metrics, `stage32_test_results.csv`, `stage32_paired_comparison.csv`, `stage32_report.md`, `completion_verification.json`, `manifest.json`, and named tmux logs.

## Limitations

One seed cannot establish stability. The Test sets were exposed in historical work, so these are development results. The prefit ISCX historical Test CSV was scanned for ID/truth coverage, without using its metrics to fit or select; this is not a strict untouched Test. VNAT's flow-random split and capture-derived labels can leak capture signals. A coarse-label score rise is not an improvement on the old fine-label task, and no open-set conclusion follows.

## Conclusion and next step

Stop Stage32 after this completed diagnostic. Do not tune on Test or automatically launch open-set evaluation.
