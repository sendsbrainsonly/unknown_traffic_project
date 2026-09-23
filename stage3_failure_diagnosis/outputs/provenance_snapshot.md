# Stage 3 Provenance Snapshot

- Frozen Stage 3 commit: `ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc`
- Commit present and ancestor of analysis HEAD: `PASS`
- Protocol canonical SHA-256: `1fff4ed33211de0b123cf4c0ec6b203cc33be683ac315b610ffd8f0194550ae1`
- Protocol raw SHA-256: `86718b1930c71ef6a904f16f37a199b8ceeea6600d3ceba65e7592048f54677a`
- Execution plan SHA-256: `197e36be2e3559c20dddf4375d117855b24cab1c1ab0f89d76f08a7f164d13de`

| Setting | frozen package SHA-256 recorded before diagnosis | audit-run tree SHA-256 before | after | files | unchanged |
|---|---|---|---|---:|---|
| A-1 | `0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9` | `0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9` | `0f8908d8602f704ba17db1f9cc4b1349bfd7b0dcc5babf5703680788fdad1be9` | 26 | PASS |
| A-2 | `5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6` | `5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6` | `5e0aa24845c391b2c71099df0f17db35bb6041f97af72290b5f773ad3242dfa6` | 37 | PASS |
| A-3 | `1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be` | `1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be` | `1328d3b3e81f9f6fe079d1d47a09859d0596c11f12e2835eecdc76475a3c22be` | 26 | PASS |

- Frozen manifest files re-hashed successfully: `33`.
- Diagnosis wrote only under `stage3_failure_diagnosis/`; official Stage 3 output trees are byte-stable across the run.
