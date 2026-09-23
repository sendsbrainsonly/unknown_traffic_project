# Experiment results: stage14a-vnat-data-audit-20260917-v1

- Status: `success_with_quality_issues`
- Experiment type: `dataset-audit`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-17T05:46:33Z`
- Completed (UTC): `2026-09-17T08:13:00Z`
- Objective: audit the read-only VNAT dataset for a strict group-aware, Unknown-Free open-set protocol without model execution.

## Data and grouping

- Input: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT` (read-only).
- Dataset inventory: 172 files, including 165 PCAPs and 7 metadata/HDF/script files.
- Flow definition: per-PCAP tshark TCP/UDP stream; other IP traffic grouped by canonical bidirectional protocol/endpoints.
- Semantic class: `application`; VPN and non-VPN are variants of the same class.
- `strict_base_group_id`: application + capture variant + capture number, conservatively shared across VPN/non-VPN.
- Final `group_id`: strict base group plus union of exact duplicate PCAP groups.
- No Train/Val/Test split, Unknown class, or Low/Medium/High setting was generated.

## Core results

| application | usable/total PCAPs | flows | effective groups | status |
|---|---:|---:|---:|---|
| netflix | 3/3 | 205 | 2 | excluded |
| rdp | 8/8 | 44 | 5 | eligible |
| rsync | 5/5 | 1,914 | 4 | eligible with duplicate-flow quarantine |
| scp | 4/5 | 2,290 | 3 | eligible with corrupt-capture exclusion; minimally supported |
| sftp | 9/9 | 1,670 | 7 | eligible with duplicate-flow quarantine |
| skype | 109/110 | 1,270 | 56 | eligible with corrupt-capture exclusion and duplicate-PCAP union |
| ssh | 10/10 | 13,563 | 5 | eligible |
| vimeo | 2/2 | 1,218 | 1 | excluded |
| youtube | 7/7 | 341 | 4 | eligible |
| zoiper | 6/6 | 939 | 3 | eligible; minimally supported |

- Total parsed flows: 23,454.
- All ten applications contain both VPN and non-VPN data.
- Eligible applications: `rdp`, `rsync`, `scp`, `sftp`, `skype`, `ssh`, `youtube`, `zoiper`.
- Excluded applications: `netflix` (2 usable groups) and `vimeo` (1 usable group); neither can support non-empty group-disjoint Train/Val/Test.
- Final readiness: `CONDITIONAL YES` for Stage 14B Protocol Freeze, restricted to eligible classes and the safeguards below.

## Data-quality findings

- Two PCAPs are truncated and fail full-read validation: `nonvpn_scp_long_capture1.pcap` and `vpn_skype-chat_capture6.pcap`.
- One exact duplicate PCAP group: `vpn_skype-chat_capture3` and `vpn_skype-chat_capture4`; final `group_id` unifies them.
- Six duplicate-flow rows were found; four rows (two packet sequences) cross the `rsync`/`sftp` application labels and must be quarantined before a Known/Unknown protocol is frozen.
- There are 1,263 repeated cross-capture 5-tuple rows; they are not classified as exact duplicates without identical packet-byte sequences.
- Filename parsing agrees with the supplied 165-row metadata manifest: 0 mismatches.
- Labels remain filename-derived capture labels, not authoritative per-flow labels; background/auxiliary flows may inherit the capture label.
- The feature HDF has 15,093 rows, five coarse labels, and no capture/group column, so it cannot independently support the requested ten-application group-aware protocol.
- Strong imbalance remains: RDP has 44 flows, SSH has 13,563; Skype has 110 PCAPs while the smallest classes have 2–3.

## Preserved evidence

- Requested outputs: `vnat_manifest.csv`, `vnat_class_statistics.csv`, `eligible_classes.json`, `excluded_classes.json`, `stage14a_vnat_audit.md`.
- Supporting tables: `outputs/audit_details.json`, `outputs/duplicate_files.csv`, `outputs/duplicate_flows.csv`.
- Per-PCAP scan cache and stderr: `artifacts/pcap_scans/` (165 JSON records plus read logs).
- Reproducible implementation and checks: `scripts/`, `tests/`.
- Machine-readable bundle inventory: `manifest.json`.

## Verification

- Final generation session `stage14a-final-generate3`: exit code 0.
- `python -m pytest stage14a_vnat_data_audit/tests/test_audit_vnat.py -q`: 5 passed.
- `python stage14a_vnat_data_audit/scripts/verify_stage14a.py`: PASS.
- Verification confirmed 172 manifest rows, 163/165 fully readable PCAPs, 23,454 flows, all required columns, exact eligible/excluded coverage, duplicate-aware grouping, and all forbidden-action flags false.

## Limitations and next step

No model was trained or evaluated, and no Unknown detection result was accessed. Stage 14B may proceed only conditionally: exclude both truncated PCAPs, quarantine cross-application exact duplicate flows, preserve duplicate-aware groups, and keep each application's VPN/non-VPN samples on the same semantic Known/Unknown side. This run did not start Stage 14B.
