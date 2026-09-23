# Stage 16 capture dependency audit

All labels remain `WEAK_CAPTURE_LABEL`; the table is descriptive only.

| Service | Train flows | Validation flows | Total flows | Independent captures |
|---|---:|---:|---:|---:|
| Chat | 132 | 17 | 149 | 6 |
| Email | 145 | 23 | 168 | 2 |
| File-Transfer | 332 | 43 | 375 | 4 |
| P2P | 804 | 100 | 904 | 1 |
| Streaming | 854 | 104 | 958 | 4 |
| VoIP | 463 | 48 | 511 | 3 |

P2P has one independent capture, so `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE` remains in force. The current random flow split cannot be interpreted as capture-disjoint generalization.
