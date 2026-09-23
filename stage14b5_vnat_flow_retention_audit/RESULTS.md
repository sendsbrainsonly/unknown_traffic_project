# Experiment results: stage14b5-vnat-flow-retention-audit-20260917-v1

- Status: `success`
- Experiment type: `dataset-audit`
- Claim scope: `diagnostic`
- Verdict: `PASS_WITH_EXPLAINED_DIFFERENCE`
- Completed (UTC): `2026-09-17T12:31:54Z`

## Data and frozen inputs

- Dataset: local VNAT release, 165 PCAPs plus official connection/feature HDF5 artifacts.
- Frozen Stage 14B clean pool: 23,449 flows.
- Frozen protocol hash before/after: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` — unchanged.
- Frozen protocol JSON SHA256: `5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced`.
- Frozen split manifest SHA256: `66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e`.

## Configuration and execution

- Read-only source audit; no Stage 14B regeneration, model training, Open-Detect, DES, split change, seed change, or Unknown-class change.
- Fixed environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`.
- Core audit: `python stage14b5_vnat_flow_retention_audit/scripts/audit_flow_retention.py`.
- Independent verification: `python stage14b5_vnat_flow_retention_audit/scripts/verify_stage14b5.py`.
- One initial report-generation attempt failed on an f-string brace; its tmux log is preserved. The corrected run completed and passed verification.

## Core results

- PCAPs: 165 total, 163 fully readable, 2 truncated.
- Packets in successful PCAPs: 24,901,394; IP-eligible 24,859,370; non-IP excluded 42,024.
- Stage 14A flow construction: 23,454 flows.
- Post-construction removals: 1 flow from exact duplicate `vpn_skype-chat_capture4`; 4 cross-application duplicate rows (`rsync`/`sftp`); no other flow filters.
- Final clean pool: `23,454 - 1 - 4 = 23,449`.
- Truncated-PCAP readable-prefix diagnostic: SCP 10,554 flows and Skype 1 flow; official H5 contains 10,555 and 1 connections respectively. These diagnostic counts were not inserted into Stage 14B.
- Raw-file diagnostic over all complete packet rows: `34,009 - 10,555 whole-corrupt-PCAP flows - 1 duplicate-PCAP flow - 4 cross-app rows = 23,449`; total absent from final under this diagnostic ledger is 10,560.
- Official H5: 33,711 connections. On the same 163 readable filenames, official=23,155 versus local=23,454; the -299 difference is attributed to non-equivalent connection/session definitions, not a hidden filter.

## Preserved evidence

- `stage14b5_flow_audit.md`: human-readable full audit and required answers.
- `stage14b5_flow_lineage.csv`: complete processing lineage and quantity conservation.
- `stage14b5_filter_breakdown.csv`: applied and explicitly absent rules with code locations and parameters.
- `stage14b5_class_retention.csv`: per-application/VPN retention and official comparison.
- `artifacts/official_connection_comparison.csv`: 165-capture official/local count comparison.
- `artifacts/corrupt_prefix_flow_counts.json`: diagnostic-only readable-prefix scan.
- `artifacts/freeze_hash_before.json`, `artifacts/freeze_hash_after.json`: immutable-input evidence.
- `artifacts/audit_summary.json`: machine-readable core result.

## Limitations

- A truncated PCAP's intended missing tail is unavailable, so the exact full-file counterfactual under the local tshark grouping rule cannot be known.
- Official H5 connections and local tshark streams are different definitions and were compared by count only, not treated as interchangeable samples.
- Application labels are inherited from filenames/capture metadata rather than an independent per-flow ground truth source.

## Conclusion

`PASS_WITH_EXPLAINED_DIFFERENCE`: the frozen 23,449-flow pool is fully and exactly explained from the 23,454 successfully constructed Stage 14A flows. The two whole-PCAP integrity exclusions and official/local connection-definition mismatch are explicitly quantified and remain relevant fairness caveats for VNAT external validation.
