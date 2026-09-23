# Calibration Sample Sufficiency Audit

This is a class-count and hypothetical component-weight stress audit. No CSTNET encoder, scaler, PCA, GMM, component assignment, score, or threshold was produced.

## Known Validation Distribution

| Fold | Classes | min | P05 | P25 | median | P75 | max | n<20 | n<30 | n<50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 113 | 7 | 11.60 | 25.00 | 44.00 | 49.00 | 63 | 18 | 32 | 86 |
| medium | 101 | 8 | 11.00 | 24.00 | 42.00 | 48.00 | 66 | 19 | 29 | 81 |
| high | 89 | 8 | 12.00 | 25.00 | 39.00 | 46.00 | 61 | 16 | 25 | 78 |

## Component Fallback Stress

| Fold | Weight | Classes expected n<30 | Rate |
|---|---|---:|---:|
| low | 50/50 | 109/113 | 96.46% |
| low | 80/20 | 113/113 | 100.00% |
| low | 90/10 | 113/113 | 100.00% |
| low | 95/5 | 113/113 | 100.00% |
| medium | 50/50 | 99/101 | 98.02% |
| medium | 80/20 | 101/101 | 100.00% |
| medium | 90/10 | 101/101 | 100.00% |
| medium | 95/5 | 101/101 | 100.00% |
| high | 50/50 | 88/89 | 98.88% |
| high | 80/20 | 89/89 | 100.00% |
| high | 90/10 | 89/89 | 100.00% |
| high | 95/5 | 89/89 | 100.00% |

Frozen minimum-count=100 recommendation: **REVISE_BEFORE_TRAINING**.

If revision is required, Stage 5.5 does not choose between: (A) increasing the eligibility minimum; or (B) keeping the class set while defining a larger Known-only calibration pool. Either change requires a separately frozen protocol v2 before training.
