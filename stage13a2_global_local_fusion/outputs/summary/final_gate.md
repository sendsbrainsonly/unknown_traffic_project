# Stage 13A-2 Final Gate

## Decision: GO

## GL-Fusion versus centroid

| Scenario | Mean ΔAUROC | Positive seeds | Mean ΔUFAR | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|
| A1 | -0.000695 | 2/5 | +0.000000 | +0.009000 |
| A2 | +0.026021 | 5/5 | -0.019333 | -0.000667 |
| A3 | +0.001590 | 5/5 | -0.009889 | -0.002225 |

- Overall positive paired seeds: `12/15`.

## GO conditions

- a1_preserved: `PASS`
- a2_material_improvement: `PASS`
- a3_preserved: `PASS`
- at_least_10_of_15_positive_seeds: `PASS`
- known_frr_safe: `PASS`
- a2_ufar_decreased: `PASS`

## Conditional conditions

- a1_not_materially_harmed: `PASS`
- a2_positive: `PASS`
- a3_not_materially_harmed: `PASS`
- at_least_8_of_15_positive_seeds: `PASS`
- known_frr_safe: `PASS`
- a2_ufar_decreased: `PASS`

- New encoder training: `NO`.
- External Test datasets read: `NO`.
- Frozen experiment modified: `NO`.
