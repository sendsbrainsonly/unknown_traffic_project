# Experiment results: stage22-pretrained-trafficformer-e3-closed-set-20260923-v1

- Status: `success / CLOSED_SET_DIAGNOSTIC_COMPLETE`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Objective: compare official-pretrained TrafficFormer and the E3 fusion on identical frozen Stage20 samples.

## Data and split

- Reused Stage20 `closed_service_manifest.csv` without modification.
- ISCX-VPN: 10,955 flows; Train/Validation/Test = 8,764/1,098/1,093; six Services.
- ISCXTor2016: 11,181 flows; Train/Validation/Test = 8,946/1,118/1,117; seven Services.
- Seeds: 2022 and 2023. Test samples used for checkpoint selection: 0.

## Configuration and execution

- Official pretrained SHA256: `be9dcc1e0c6b68db005e506adf13a2d937a2f3982539a2c8a58f78c55af0d5ce`.
- TrafficFormer: official 120k pretrained initialization, 20 epochs, batch 64, LR 6e-5, first pooling, sequence length 320.
- TAGCN: unchanged Stage21 settings, 50 epochs, batch 64, LR 1e-3, hidden 128, K=2.
- E3: Known-Train-only branch z-score, 768+128 concatenation, linear head, 30 epochs.
- All checkpoints selected by Known Validation Macro-F1; Known Test was evaluated only after selection.

## Core results

| Dataset | Encoder | Accuracy mean±std | Macro-F1 mean±std | Weighted-F1 mean±std |
|---|---|---:|---:|---:|
| iscx_vpn | E1 | 0.853156±0.004117 | 0.860607±0.003961 | 0.847449±0.004264 |
| iscx_vpn | E3 | 0.849954±0.008234 | 0.859493±0.007015 | 0.846448±0.007424 |
| iscx_tor | E1 | 0.831244±0.005819 | 0.821259±0.006376 | 0.832378±0.005145 |
| iscx_tor | E3 | 0.832140±0.006714 | 0.822178±0.007643 | 0.833903±0.005991 |

### E3 minus TrafficFormer

| Dataset | Mean ΔAccuracy | Mean ΔMacro-F1 | Mean ΔWeighted-F1 | Positive Macro-F1 seeds |
|---|---:|---:|---:|---:|
| iscx_vpn | -0.003202 | -0.001115 | -0.001001 | 1/2 |
| iscx_tor | +0.000895 | +0.000918 | +0.001526 | 1/2 |

## Preserved evidence

- Four run directories with E1/E2/E3 checkpoints, training histories, representations and validation/test predictions.
- `run_metrics.csv`, `aggregate_metrics.csv`, `per_class_metrics.csv`, `confusion_matrices.csv` and paired-delta tables.
- `preflight.json`, protected input hashes, queue logs and `completion_verification.json`.

## Limitations

- The public TrafficFormer README does not establish the official pretraining corpus membership; this experiment is not a strict Unknown-Free open-set result.
- Service labels are capture-derived weak labels and Stage20 is not uniformly capture-disjoint.
- Only two seeds are included, so standard deviations are descriptive.
- Stage21-to-Stage22 differences combine official pretraining and a 3-to-20 epoch budget increase; they are not a pure pretraining ablation.
- This stage reports closed-set performance only and does not run unknown detection.

## Conclusion and next step

- Interpret E3 versus E1 only from the paired same-run deltas above. Do not compare these scores directly with historical TrafficFormer runs that used different flow populations.
- Stop after the closed-set comparison; any open-set evaluation requires a separate frozen protocol decision about external pretraining exposure.
