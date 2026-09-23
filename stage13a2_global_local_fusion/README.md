# Stage 13A-2 — DES Global + Local Support Fusion

This diagnostic evaluates whether the frozen DES-v0 centroid score and the
Stage 13A-1 class-conditional kNN-10 score are complementary on the USTC
development campaign (A1/A2/A3, seeds 2022–2026).

## Frozen inputs

- Exactly 15 successful Stage 13A-1 runs are read-only inputs.
- Their frozen Open-Detect checkpoint, Stage 11B representation bundle,
  centroid score, kNN-10 score, split identity, and hashes are verified before
  and after this stage.
- No encoder inference or training is performed.
- ISCX-VPN Test, ISCXTor2016 Test, and CipherSpectrum Test are not read.

## Fixed method

For each run and each source score independently, the median and unscaled MAD
are computed from Known Validation only:

```text
Zc = (Sc - median(Sc_val)) / (MAD(Sc_val) + 1e-12)
Zk = (Sk - median(Sk_val)) / (MAD(Sk_val) + 1e-12)
S_GL = 0.5 * Zc + 0.5 * Zk
```

There is no weight search, learned fusion, log-variance, margin, covariance, or
K2 term. Each method uses its own Known Validation P95 threshold with NumPy
`method="higher"`. USTC Test is evaluation-only.

## Preregistered decision rule

`GO` requires all user-specified conditions:

- A1 mean GL−centroid ΔAUROC > -0.005;
- A2 mean ΔAUROC >= +0.02;
- A3 mean ΔAUROC > -0.005;
- at least 10/15 paired runs have positive ΔAUROC;
- no scenario mean ΔKnown FRR exceeds +0.02;
- A2 mean ΔUFAR is negative.

Before results are evaluated, `CONDITIONAL_GO` is fixed as: A1 and A3 both
remain above -0.02, A2 remains positive, at least 8/15 runs are positive,
Known FRR remains safe, and A2 UFAR decreases. Any other outcome is `NO_GO`.
This secondary rule does not relax or redefine the `GO` criteria.

## Outputs

- `artifacts/<scenario>/fold<fold>_seed<seed>/`: per-run configuration,
  normalization parameters, scores, metrics, hashes, and evidence records.
- `outputs/summary/run_level_results.csv`
- `outputs/summary/scenario_summary.csv`
- `outputs/summary/gl_vs_centroid.csv`
- `outputs/summary/final_gate.md`
## Recorded Final Result

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

