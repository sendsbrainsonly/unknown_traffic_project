# RoNeTC vs project method H1 — completed protocols only

- Status: `PARTIAL_COMPARISON_FOUR_COMPLETED_PROTOCOLS`
- H1 definition: `max(A_OD, A_DES0)`, calibrated from Known Validation only.
- Match check: protocol ID, Known/Unknown classes, split membership and sample counts match.
- Excluded: USTC `A-2`; RoNeTC was stopped at 5/100 epochs and never evaluated on Known Test or Unknown Test.

| Protocol | RoNeTC AUROC | H1 AUROC | RoNeTC AUPRC | H1 AUPRC | RoNeTC UFAR | H1 UFAR | RoNeTC FRR | H1 FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ISCXTor medium-2022 | **0.609545** | 0.604153 | **0.813783** | 0.796046 | **0.949000** | 0.981500 | **0.052111** | 0.055705 |
| ISCX-VPN medium-2022 | 0.583144 | **0.587177** | **0.812776** | 0.810527 | **0.947833** | 0.959000 | 0.058606 | **0.057372** |
| VNAT medium-2025 | **0.864428** | 0.806226 | **0.885988** | 0.871320 | 0.767059 | **0.662222** | 0.062755 | **0.038265** |
| VNAT medium-2026 | 0.639538 | **0.759437** | 0.754789 | **0.809419** | **0.881833** | 0.936264 | 0.047619 | **0.046049** |
| Unweighted mean | 0.674164 | **0.689248** | 0.816834 | **0.821828** | 0.886431 | **0.884747** | 0.055273 | **0.049348** |

AUROC/AUPRC are higher-is-better; UFAR/FRR are lower-is-better. H1 leads the four-protocol unweighted mean on all four metrics, while the AUROC protocol win count is tied 2–2. This does not include USTC and must not be presented as a complete four-dataset comparison.
