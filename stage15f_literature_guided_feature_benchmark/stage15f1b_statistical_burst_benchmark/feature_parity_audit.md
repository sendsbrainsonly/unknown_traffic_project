# S12 / Stage 15R E3 parity audit

Status: `PASS`

All five FULL_FLOW S12 runs reuse the exact Stage 15R E3 LightGBM checkpoint and reconstruct the original feature matrix/order. Validation metrics and confusion matrices match exactly within the preregistered numerical tolerance.

| Dataset | Protocol | Accuracy | Macro-F1 | Weighted-F1 | Metrics | Confusion |
|---|---|---:|---:|---:|---|---|
| iscx_vpn | medium_seed2022 | 0.500931 | 0.466724 | 0.484721 | PASS | PASS |
| iscx_tor | medium_seed2022 | 0.497742 | 0.487714 | 0.492444 | PASS | PASS |
| vnat | medium_seed2025 | 0.898980 | 0.774562 | 0.893513 | PASS | PASS |
| vnat | medium_seed2026 | 0.989534 | 0.846497 | 0.988444 | PASS | PASS |
| ustc | A-2 | 0.916503 | 0.926336 | 0.916104 | PASS | PASS |

The parity baseline does not use the newly extracted FULL_FLOW behavior cache. Consequently, data repairs, restored VNAT direction fields, or new feature definitions cannot be counted as S12 gain.
