# Stage 11A Final Gate

## Decision

**NO_GO**

## Primary evidence

| Scenario | Mean ΔAUROC | Median ΔAUROC | DAP better seeds | Gap-lower seeds | Mean ΔKnown Macro-F1 | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|---:|---:|
| A1 | -0.167800 | -0.153725 | 0/5 | 5/5 | -0.060538 | 0.010000 |
| A2 | -0.068712 | -0.046122 | 0/5 | 5/5 | -0.036661 | 0.005667 |
| A3 | -0.036409 | -0.043832 | 1/5 | 5/5 | -0.094302 | 0.001978 |


## Fixed gate checks

- Prototype alignment decreases 5/5 in every scenario: `True`.
- DAP-Native scenario mean AUROC is non-negative versus B0 in all scenarios: `False`.
- Scenarios meeting material `+0.02` AUROC: `0`.
- Known Test Macro-F1 has no scenario mean drop of 0.02 or more: `False`.
- Mean Known FRR does not rise by 0.02 or more: `True`.
- DAP AUROC improves in at least 4/5 paired seeds per scenario: `False`.

## Covariance diagnostics

- DAP K1 minus Native mean AUROC: A1=-0.001560, A2=0.078767, A3=0.034826.
- Verdict: **COVARIANCE_REQUIREMENT_NOT_STABLE**.
- DAP K2 minus K1 mean AUROC: A1=-0.067470, A2=0.013793, A3=0.003361.
- Verdict: **MULTICOMPONENT_SECONDARY**.

## Scope boundary

`DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`. CipherSpectrum Stage 9 sample-level Test data were not opened, read, or rescored. No next stage was started.
