# Experiment results: STAGE15F-0

- Status: `success`
- Experiment type: `audit`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-19T14:06:53Z`
- Objective: Literature-guided feature registry, four-dataset recoverability/window/encryption audit, and a frozen next-stage matrix without model training.

## Data and split

- USTC-TFC2016: union of Known Train/Validation in frozen A-1/A-2/A-3; 432,537 cached flows.
- VNAT: union of Known Train/Validation across 15 frozen Stage 14B protocols; 23,447 cached flows.
- ISCX-VPN: union of Known Train/Validation in frozen low/medium/high Stage 12 protocols; 18,121 cached flows.
- ISCXTor2016: union of Known Train/Validation in frozen low/medium/high Stage 12 protocols; 11,783 cached flows.
- Known Test feature values used: `0`.
- Unknown Test feature values used: `0`.

## Configuration and execution

- Fixed environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`.
- All commands ran in project-local named tmux sessions.
- Packet windows audited: `8, 16, 32, 64`.
- USTC flow semantics: retained Stage-0 bidirectional IPv4 TCP/UDP five-tuples, no timeout.
- ISCX flow semantics: bidirectional IPv4 TCP/UDP, 60 s timeout, TCP SYN/FIN/RST boundaries.
- VNAT flow semantics: capture-local `tcp.stream`/`udp.stream` joined to frozen `flow_uid`.
- No GPU workload, model training, checkpoint selection, detector fitting, or Test metric computation.

## Core results

1. Primary-source registry completed for 19/19 requested works, with explicit evidence tags and paper/code discrepancies.
2. Thirteen B/T/S/M/P feature configurations are defined with shapes, windows, normalization, masks, costs, and leakage risks.
3. Four datasets passed field recoverability and window-cache parity checks; unmatched protocol membership rows = 0.
4. Mean protocol-level N=8 truncation: USTC 31.47%, VNAT 17.42%, ISCX-VPN 11.82%, ISCXTor 28.52%.
5. Mean protocol-level N=32 truncation: USTC 14.49%, VNAT 7.59%, ISCX-VPN 5.45%, ISCXTor 16.90%.
6. N=8 insufficiency is strongly class-dependent: examples include USTC WorldOfWarcraft 98.77%, USTC Weibo 95.94%, VNAT vimeo 97.46%, and ISCXTor FTP 50.67% truncated.
7. Coverage shows a real early-flow horizon but cannot establish that window length caused the prior Macro-F1 gap; all performance questions remain NOT_RUN.
8. Strict Unknown-Free audit PASS: Known Test usage = 0 and Unknown Test usage = 0.
9. Protected Stage 12--15R plus formal input hashes: 193/193 unchanged.

## Preserved evidence

- `stage15f_report.md`: complete scientific report and answers to all 15 questions.
- `literature_primary_source_notes.md`: primary-source evidence ledger.
- `literature_feature_registry.csv`: machine-readable 19-work registry.
- `feature_group_definitions.json`: 13 feature definitions.
- `dataset_feature_availability.csv`: actual field/source lineage.
- `packet_window_coverage.csv`: 2,928 dataset/protocol/class/role/window aggregates.
- `encryption_visibility_audit.csv`: application/transport versus tunnel visibility.
- `feature_lineage_audit.md`: extraction and semantic lineage.
- `stage15f1_preregistered_matrix.csv`: frozen pilot/full15 decision matrix; all rows NOT_RUN.
- `window_cache_parity.json`: exact cache parity evidence.
- `completion_verification.json`: final machine-readable checks.
- `failed_attempts/iscx_vpn_scapy_slow/`: preserved stopped slow-path evidence.

## Limitations

- Packet-window coverage is descriptive; no causal feature-performance claim is made.
- Protocol-level means weight protocols equally and overlapping memberships are not unique dataset counts.
- VPN/Tor packet bytes and headers are outer-tunnel observations.
- USTC encryption state remains undetermined without frozen per-flow evidence.
- Public pretrained representation P1 is blocked pending provenance, license, preprocessing-parity, and overlap audit.
- Stage 15F-1 through Stage 15F-5 are `NOT_RUN`; future result CSVs are intentionally absent.

## Conclusion and next step

`STAGE15F0_PASS`. The audit supports preregistered controlled comparison of 8/16/32 packet horizons (with 64 retained as a declared endpoint), structured byte families, and full-flow statistics/bursts. It does not yet select a feature family or encoder.
