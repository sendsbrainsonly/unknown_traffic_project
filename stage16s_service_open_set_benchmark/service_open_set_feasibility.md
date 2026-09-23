# Service-level open-set feasibility

All six LOSO rounds are executable as **flow-level weak-label diagnostics**.

| Unknown Service | Unknown flows | Unknown captures | Status |
|---|---:|---:|---|
| Chat | 149 | 6 | FEASIBLE_FLOW_LEVEL |
| Email | 168 | 2 | FEASIBLE_FLOW_LEVEL |
| File-Transfer | 375 | 4 | FEASIBLE_FLOW_LEVEL |
| P2P | 904 | 1 | PROTOCOL_LIMITED |
| Streaming | 958 | 4 | FEASIBLE_FLOW_LEVEL |
| VoIP | 511 | 3 | FEASIBLE_FLOW_LEVEL |

P2P has one capture and is retained only as `PROTOCOL_LIMITED`; it cannot support a cross-capture claim.
All rounds share captures across flow-level roles, so `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE` applies globally.
