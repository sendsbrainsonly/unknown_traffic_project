# Latent Quality Report

## Gate

- Status: PASS
- Frozen checkpoint SHA-256 verified: `f8e53d36bd6dc8333451b2f159300bc5a2cd875fccaff01a2c6dc43356951b36`
- Latent dimension: 128
- Train rows: 391280 (expected 391280)
- Validation rows: 48910 (expected 48910)
- Train flow IDs unique: yes
- Validation flow IDs unique: yes
- Train/validation overlap: 0
- NaN: 0
- Inf: 0
- Collapsed train dimensions (variance <= 1e-12): 0/128
- Near-collapsed train dimensions (variance <= 1e-8): 0/128
- Train per-dimension variance min/median/max: 0.000959563049 / 0.00524916987 / 3.82755382
- Four-class Open-Detect/TrafficFormer train and validation flow-ID sets: 8/8 exact

## Representative-class latent quality

| class | split | n | mean mu norm | mean within-class distance | p95 within-class distance | total within-class variance |
|---|---|---:|---:|---:|---:|---:|
| FTP | train | 80830 | 5.954262 | 0.191552 | 0.347151 | 0.049854 |
| Cridex | train | 13108 | 5.075682 | 0.363935 | 0.525625 | 0.166052 |
| Miuref | train | 10782 | 4.223025 | 0.453006 | 0.864177 | 0.305621 |
| Outlook | train | 6019 | 4.559489 | 0.334156 | 0.725317 | 0.276054 |
| FTP | val | 10103 | 5.951658 | 0.193786 | 0.359941 | 0.049908 |
| Cridex | val | 1638 | 5.082842 | 0.354725 | 0.525997 | 0.145015 |
| Miuref | val | 1348 | 4.226073 | 0.460918 | 0.873097 | 0.328275 |
| Outlook | val | 753 | 4.553043 | 0.371734 | 0.963580 | 0.378221 |

## Open-Detect prototype compactness (train)

| class | n | assigned mean | assigned median | assigned p95 | within-class variance | nearest-wrong mean | mean margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| FTP | 80830 | 0.974919 | 0.980521 | 1.097564 | 0.049854 | 6.586681 | 5.611762 |
| Cridex | 13108 | 0.913145 | 0.897352 | 1.070214 | 0.166052 | 5.161216 | 4.248071 |
| Miuref | 10782 | 1.394938 | 1.410479 | 1.534080 | 0.305621 | 3.697485 | 2.302548 |
| Outlook | 6019 | 1.852590 | 1.875140 | 2.079304 | 0.276054 | 3.909477 | 2.056886 |

Distances are descriptive checks of the learned compactness constraint. They do not establish that a single Gaussian is adequate and are not semantic-mode evidence.
