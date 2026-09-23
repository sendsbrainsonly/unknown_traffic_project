# Stage 4 Provenance Snapshot

- Frozen Stage 3 commit: `ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc`
- Protocol canonical SHA-256: `1fff4ed33211de0b123cf4c0ec6b203cc33be683ac315b610ffd8f0194550ae1`
- Frozen asset operations: `transform`, `score_samples`, `_estimate_weighted_log_prob`; estimator fitting operations: `0`.

| Setting | package SHA-256 before | after | files | unchanged |
|---|---|---|---:|---|
| A-1 | `0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9` | `0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9` | 26 | PASS |
| A-2 | `5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6` | `5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6` | 37 | PASS |
| A-3 | `1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be` | `1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be` | 26 | PASS |

- Stage 3 prediction replay: `PASS`, 146,733 setting-sample rows across Known and Unknown.
