# Stage 29 — Native E3-only entropy fusion

This is a Known-only development diagnostic on the original two E3 encoders, not YaTC or end-to-end E3 retraining.

## Scope and provenance

- Stage22 TrafficFormer 768-D + FIG/TAGCN 128-D, exact Stage20 Known Train/Validation Service flows.
- Frozen encoders and original E3 head; 20 paired new-head units. No YaTC, Known Test or Unknown values were loaded.
- Entropy fusion has a single shared λ so its two-view high/low formula can move off the Stage28 cancellation line.
- All 40 N0/N1 checkpoints independently replayed on CPU; frozen source hashes unchanged.

## Known Validation results

| Dataset | Method | Macro-F1 mean ± SD | Accuracy mean | Weighted-F1 mean |
|---|---|---:|---:|---:|
| iscx_vpn | E3_original | 0.851877 ± 0.000321 | 0.839253 | 0.838955 |
| iscx_vpn | N0_equal | 0.837604 ± 0.008823 | 0.822769 | 0.823513 |
| iscx_vpn | N1_shared_lambda_entropy | 0.837617 ± 0.009027 | 0.822860 | 0.823572 |
| iscx_tor | E3_original | 0.844107 ± 0.006295 | 0.846601 | 0.848857 |
| iscx_tor | N0_equal | 0.849594 ± 0.004005 | 0.852326 | 0.854146 |
| iscx_tor | N1_shared_lambda_entropy | 0.849822 ± 0.003971 | 0.852415 | 0.854213 |

## Paired dynamic-fusion gate

| Dataset | N1−N0 mean ΔMacro-F1 | Positive / 10 | Worst Δ | N1−original E3 | Pass |
|---|---:|---:|---:|---:|---|
| iscx_vpn | +0.000013 | 2/10 | -0.000855 | -0.014260 | False |
| iscx_tor | +0.000228 | 4/10 | +0.000000 | +0.005715 | False |

## Mechanism checks

- N1−forced-equal mean Macro-F1: +0.000441; positive in 8/20 paired units.
- Batch-invariance maximum absolute logit difference: 7.6293945e-06.
- Learned λ mean across 20 units: 0.502877; mean TrafficFormer weight: 0.500582.
- Sample-level N1 rescues/hurts N0: 13/11 decisions (head seeds reuse the same flows).

## Limitations and conclusion

- The Stage20 Test was previously exposed in unrelated earlier work but was not opened here; these numbers are Known-Val development evidence only.
- Two encoder seeds and repeated head seeds share validation flows; do not treat 10 units per dataset as 10 independent captures.
- Shared-λ is a documented correction to the two-view algebra, not the paper's exact dynamic-fusion implementation.
- Preregistered gate: **FAIL**. Do not promote N1 or tune using Test; stop the entropy-fusion route under this protocol.
