# Stage 3 Openness Trend

This is a three-point descriptive comparison only; no trend model is fitted.

| Setting | Known/Unknown classes | Single UFAR | Multi UFAR | Multi-Single gap |
|---|---:|---:|---:|---:|
| A-1 | 19/1 | 0.522353 | 0.623529 | +0.101176 |
| A-2 | 17/3 | 0.618211 | 0.620900 | +0.002689 |
| A-3 | 15/5 | 0.116437 | 0.093538 | -0.022899 |

- Single UFAR: non-monotonic / unstable from A-1 to A-3.
- Multi UFAR: monotonically decreases from A-1 to A-3.
- Multi-Single gap: monotonically decreases; negative values favor Multi.
- Because the held-out class composition changes together with openness, these points do not identify a causal openness effect.
