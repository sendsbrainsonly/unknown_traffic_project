# Experiment results: stage15r-representation-bottleneck-audit-20260919-v1

- Status: `success`
- Experiment type: `audit_and_controlled_experiment`
- Claim scope: `diagnostic; Known Train/Validation only`
- Created (UTC): `2026-09-19`
- Final Gate: `DATA_OR_PROTOCOL_BOTTLENECK_IDENTIFIED`

## Objective

Determine whether the weak Known classification observed on ISCX-VPN, ISCXTor2016, and hard VNAT protocols is primarily caused by the current byte-image features, Open-Detect's joint objective, model architecture, or dataset/protocol properties.

## Data and split

Five preregistered frozen pilots were used: ISCX-VPN `medium_seed2022`, ISCXTor `medium_seed2022`, VNAT `medium_seed2025/2026`, and USTC `A-2`. E0 reused frozen Native Open-Detect results. E1/E2/E3 used only Known Train and Known Validation. Known Test and Unknown Test feature values were not opened for training, validation, feature selection, normalization, or checkpoint selection.

## Configuration and execution

- E1: current 32×32 Open-Detect image, ResNet18 + CE.
- E2: first-8 signed packet length + IAT + mask, train-only median/IQR, 1D-CNN + CE.
- E3: preregistered full-flow statistics, LightGBM 4.6.0.
- Neural controls: seed 2022, batch 128, 100 epochs, Adam 0.001, MultiStepLR `[50,80]`, no early stopping.
- E3: up to 500 rounds with Known-Validation log-loss early stopping 50.

## Core results

| Candidate | Mean ΔMacro-F1 vs E0 | Positive / 5 | Worst Δ | Gate |
|---|---:|---:|---:|---|
| E1 | -0.072320 | 1 | -0.148221 | FAIL |
| E2 | -0.166239 | 1 | -0.384359 | FAIL |
| E3 | -0.128971 | 1 | -0.302445 | FAIL |

E0 validation Macro-F1 was `0.769169` on ISCX-VPN, `0.676900` on ISCXTor, `0.739755/0.983318` on VNAT hard/normal, and `0.977545` on USTC. No new candidate passed the preregistered clear or diagnostic gate, so full-15 was not run and `full15_closed_set_results.csv` was intentionally not created.

## Input and flow audit

- Stage 15R-0 status: PASS; protocol rows 600, lineage rows 32, packet-stat rows 245.
- ISCX-VPN cache: 22,142/22,142 flows.
- ISCXTor cache: 15,096/15,096 flows using exact Stage 12 sessionization semantics.
- USTC cache: 489,101/489,101 flows.
- VNAT development cache: 23,447/23,449 clean-pool flows. The two omitted rows are Known-Test-only ssh flows and their feature values were deliberately not opened; this is not flow loss.
- Protected Stage 12/14/15 input hashes are recorded before and after execution.

## Preserved evidence

- `stage15r_representation_report.md`: full audit, results, ten required answers, and decision.
- `pilot_closed_set_results.csv`: all 20 run-level results.
- `paired_vs_native.csv`: paired E1/E2/E3 minus E0 metrics.
- `per_class_results.csv` and `confusion_analysis.csv`: class-level diagnostics.
- `dataset_protocol_audit.csv`, `input_shape_and_lineage_audit.csv`, `packet_flow_statistics.csv`, `encryption_regime_audit.csv`: data and preprocessing audit.
- `pilot_runs/`: checkpoints, histories, run configs, confusion matrices, and raw metrics.
- `failed_attempts/`: preserved AF_UNIX, interrupted USTC workers-0, and partial tshark evidence.

## Limitations

- Pilot results do not estimate full protocol distributions.
- E1 is not a perfectly isomorphic loss-only ablation.
- E2 is limited to the same first-8-packet horizon as E0.
- Different datasets retain different flow construction and capture-split semantics.
- No new open-set, DES, H1, or Unknown-Test result was produced.

## Conclusion and next step

The evidence does not support freezing CE-only, signed length/IAT, or simple flow statistics as a replacement representation. The primary next action is a stricter capture/group and class-overlap protocol audit. Only after that should a pretrained byte+temporal hierarchical representation be evaluated under the same frozen Known splits. Stage 15R stops here.
