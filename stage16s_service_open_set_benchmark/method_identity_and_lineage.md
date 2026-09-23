# Method identity and lineage

| Method | Shared encoder/classifier | Frozen score implementation |
|---|---|---|
| OD-Native | Per-protocol corrected Open-Detect checkpoint | minimum learned-prototype KL, including clamped posterior variance |
| DES-v0 | Same checkpoint and native Known classifier | minimum squared Euclidean distance from posterior mean to Known-Train empirical class centroid |
| DES-v1 | Same checkpoint and native Known classifier | 0.5 global centroid z-score + 0.5 predicted-centroid-class kNN-10 z-score; median/MAD fit on Known Validation |
| H1 | Same checkpoint and native Known classifier | max of Known-Val empirical percentiles for OD-Native and DES-v0 |

Code lineage is reused from Stage 14D (`stage14d_common.py`,
`evaluate_protocol.py`) and Stage 15B (`run_stage15b.py`). No detector formula,
k, fusion weight, score direction, or threshold rule was searched or changed.
DQ-3F establishes that the task model is corrected Open-Detect; Stage 16S does
not relabel that encoder as a new method.
