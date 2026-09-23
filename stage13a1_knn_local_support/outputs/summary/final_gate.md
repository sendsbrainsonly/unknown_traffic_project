# Stage 13A-1 Final Gate

## Decision: CONDITIONAL_GO

## Primary kNN-10 versus centroid

| Scenario | Mean ΔAUROC | Positive seeds | Mean ΔUFAR | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|
| A1 | -0.011600 | 0/5 | +0.014000 | +0.010000 |
| A2 | +0.036064 | 4/5 | -0.039333 | -0.004000 |
| A3 | +0.001138 | 3/5 | -0.003461 | -0.001483 |

## Gate conditions

- at_least_two_material_scenarios: `FAIL`
- no_clear_harm_scenario: `PASS`
- overall_majority_positive_seeds: `FAIL`
- known_frr_safe: `PASS`
- ufar_lower_in_each_material_scenario: `PASS`

- More stable sensitivity setting: `KNN_10`.
- Fusion study worthwhile: `True`.
- New encoder training: `NO`.
- External Test datasets read: `NO`.
- Frozen experiment modified: `NO`.
