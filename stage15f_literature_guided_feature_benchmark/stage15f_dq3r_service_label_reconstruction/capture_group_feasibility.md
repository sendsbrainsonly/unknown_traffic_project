# Capture-Group Feasibility

| Service | Train flows | Validation flows | Independent captures | Existing Train/Val capture overlap | Group split feasible? |
|---|---:|---:|---:|---:|---|
| Chat | 132 | 17 | 6 | 4 | True |
| Email | 145 | 23 | 2 | 2 | True |
| File-Transfer | 332 | 43 | 4 | 4 | True |
| P2P | 804 | 100 | 1 | 1 | False |
| Streaming | 854 | 104 | 4 | 4 | True |
| VoIP | 463 | 48 | 3 | 3 | True |

## Findings

- Every Service has non-zero frozen flow-level Train and Validation support.
- The current frozen membership is flow-disjoint but not capture-disjoint: each listed overlap means the same capture contributes flows to both Train and Validation.
- P2P is supplied only by `vpn_bittorrent.pcap`; it cannot appear in both sides of a capture-disjoint closed-set split.
- Labels are constant within each capture by construction, so Service is fully tied to capture identity. A random flow split can reward capture-specific signatures.
- Full six-service capture-group Train/Validation feasibility: **FALSE**.

Therefore DQ-3F can be run as a controlled same-membership diagnostic, but this 31-PCAP subset alone cannot validate group-generalized six-Service classification. Excluding P2P merely to make the split feasible is not preregistered and would change the task.
