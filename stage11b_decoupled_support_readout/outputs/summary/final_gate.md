# Stage 11B Final Gate

## Decision

**NO_GO**

## Primary R2 versus R0

| Scenario | Mean ΔAUROC | R2 better seeds | Mean ΔKnown Macro-F1 | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|
| A1 | -0.024100 | 0/5 | 0.014308 | 0.011000 |
| A2 | 0.108791 | 5/5 | 0.001215 | -0.000667 |
| A3 | 0.023048 | 5/5 | -0.001102 | -0.000000 |

## Fixed interpretations

- Mechanism: **D5**.
- R1 stable over R0: `False`.
- R2 stable over R0: `False`.
- Covariance: **COVARIANCE_BENEFIT_CONTEXT_DEPENDENT**.
- K2: **MULTICOMPONENT_SECONDARY**.

## Scope

`DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. No encoder training, no CipherSpectrum Stage9 sample-level Test access, and no next stage.
