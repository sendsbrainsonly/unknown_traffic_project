# Stage 30 — True three-view entropy fusion

Known-only development experiment: frozen TrafficFormer 768-D, FIG/TAGCN 128-D, YaTC 192-D as **three independent views**. No encoder updates, Known Test or Unknown values.

## Known Validation: matched new heads

| Dataset | Method | Macro-F1 mean ± SD | Accuracy mean | Weighted-F1 mean |
|---|---|---:|---:|---:|
| iscx_vpn | T0_equal | 0.890366 ± 0.005437 | 0.880510 | 0.880544 |
| iscx_vpn | T1_three_entropy | 0.888327 ± 0.004822 | 0.878233 | 0.878391 |
| iscx_tor | T0_equal | 0.887543 ± 0.002867 | 0.891771 | 0.891917 |
| iscx_tor | T1_three_entropy | 0.877244 ± 0.002732 | 0.882290 | 0.882499 |

## Preregistered paired gate

| Dataset | T1−T0 mean ΔMacro-F1 | Positive / 10 | Worst Δ | ΔAccuracy | ΔWeighted-F1 | Pass |
|---|---:|---:|---:|---:|---:|---|
| iscx_vpn | -0.002039 | 2/10 | -0.005042 | -0.002277 | -0.002153 | False |
| iscx_tor | -0.010300 | 0/10 | -0.016319 | -0.009481 | -0.009418 | False |

## Frozen Stage27 historical reference (not the causal control)

| Dataset | Method | Known-Val Macro-F1 mean across 2 encoder seeds |
|---|---|---:|
| iscx_vpn | F1_LinearConcat | 0.883861 |
| iscx_vpn | F2_EqualProjected | 0.884465 |
| iscx_vpn | F3_FeatureGate | 0.877283 |
| iscx_tor | F1_LinearConcat | 0.866764 |
| iscx_tor | F2_EqualProjected | 0.867376 |
| iscx_tor | F3_FeatureGate | 0.847794 |

## Mechanism diagnostics

- T1−its own forced-equal head: mean ΔMacro-F1 -0.003166; positive 8/20.
- Selected λ means across the 20 runs: TrafficFormer 0.503138, graph 0.501012, YaTC 0.500393. Mean view weights: 0.327289 / 0.346326 / 0.326385; per-flow weight SD averages 0.056082 / 0.057232 / 0.058015. Thus the weights vary by flow but remain close to one-third on average; the measured variation does not help classification.
- Maximum batch-size logit difference: 8.5830688e-06.
- Mean maximum-view weight-collapse ratio: 0.000000.
- T1 rescues/hurts T0: 223/354 decisions; repeated flows across head seeds.

## Interpretation and limits

- Three-view Gaussian adapter and λ rule are explicitly inspired by ER-CMGI, not an exact paper reproduction. Stage27 F3 uses a different softmax gate.
- Five head seeds per encoder pair reuse Known-Val flows; units are not independent captures. Stage20 Test was previously exposed in earlier work, but was not opened here.
- Closed-set Known-Val results do not establish open-set or external Test benefit.
- Preregistered gate: **FAIL**. Do not promote or retune this entropy branch using Test/Unknown.
